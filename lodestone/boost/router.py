"""
Boost router: picks a free inference source for each call and falls back across sources.

    surface   who is asking: chat | courage | loops | workspaces (each switched on in config.json boost.surfaces)
    declared  the caller's content class (general | code | home); egress.classify can only raise it
    pinned    "<provider>/<model>" or "<provider>" to force one source (still egress-checked)

Order: edge nodes first for home content, then boost.priority (default groq, cloudflare, gemini, zen,
openrouter), each skipped when it has no key, is out of quota, cooling down after a 429, can't take tools,
or may not see this content. Last comes the local coordinator (:8001) unless the caller says allow_local=False.
"""

import json
import os
import re
import threading
import time
from lodestone import endpoints
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from . import egress
from .providers import DEFAULT_PRIORITY, Provider, load_providers
from .quota import QuotaBook

SURFACES = ("chat", "courage", "loops", "workspaces")
MODEL_CACHE_S = 3600
SKIP_MODEL_WORDS = ("tts", "image", "audio", "embedding", "embed", "live", "vision-only", "guard", "whisper")


class Http:
    """Thin urllib wrapper; tests replace it."""

    def request(self, method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None,
                timeout: float = 60.0) -> Tuple[int, Dict[str, str], Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
                return r.status, {k.lower(): v for k, v in r.headers.items()}, _json_or_text(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, _json_or_text(raw)

    def open_stream(self, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float = 60.0):
        """Open a streaming POST. Returns (status, headers, response-or-error-body)."""
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, _json_or_text(raw)


def _json_or_text(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _err_text(body: Any) -> str:
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            return str(e.get("message") or e)[:300]
        if e:
            return str(e)[:300]
    if isinstance(body, list) and body and isinstance(body[0], dict):  # Gemini wraps errors in a list
        return _err_text(body[0])
    return str(body)[:300]


def _retry_after(headers: Dict[str, str], body: Any) -> float:
    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    text = _err_text(body).lower()
    if "per day" in text or "daily" in text or "rpd" in text or "quota" in text:
        return 3600.0
    return 60.0


def _est_tokens(messages: List[Dict[str, Any]], max_tokens: int) -> int:
    return sum(len(t) for t in egress._texts(messages)) // 4 + int(max_tokens or 0)


class BoostRouter:
    def __init__(self, config_fn: Callable[[], Dict[str, Any]], data_dir: Optional[str] = None,
                 http: Optional[Http] = None, env: Optional[Dict[str, str]] = None, clock: Callable[[], float] = time.time,
                 home_terms: Union[Tuple[str, ...], Callable[[], Iterable[str]]] = ()):
        self.config_fn = config_fn
        self.http = http or Http()
        self.env = env
        self.clock = clock
        # Callable: re-read on every call, so a profile added at runtime counts as home content at once
        self.base_home_terms = home_terms if callable(home_terms) else tuple(home_terms)
        qpath = os.path.join(data_dir, "boost_quota.json") if data_dir else None
        self.quota = QuotaBook(qpath, loop_share=self.settings().get("loop_share", 0.5), clock=clock)
        self._models: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        # The trace (courage/trace.py): one record per call, METADATA ONLY. Boost traffic can hold home text, so no
        # message content is recorded, and home terms appear only as their kind ('home:term', never the name).
        self.on_call: Optional[Callable[[Dict[str, Any]], None]] = None

    # ---- settings ------------------------------------------------------------
    def settings(self) -> Dict[str, Any]:
        cfg = self.config_fn() or {}
        b = cfg.get("boost") or {}
        surfaces = {s: bool((b.get("surfaces") or {}).get(s, False)) for s in SURFACES}
        return {
            "enabled": bool(b.get("enabled", False)),
            "surfaces": surfaces,
            "priority": list(b.get("priority") or DEFAULT_PRIORITY),
            "loop_share": float(b.get("loop_share", 0.5)),
            "home_terms": list(b.get("home_terms") or []),
            "local_url": endpoints.url("coordinator", cfg),
            "timeout_s": float(b.get("timeout_s", 60)),
        }

    def enabled(self, surface: str) -> bool:
        s = self.settings()
        return s["enabled"] and s["surfaces"].get(surface, False)

    def providers(self) -> Dict[str, Provider]:
        return load_providers(self.config_fn() or {}, self.env)

    def local_provider(self) -> Provider:
        return Provider(id="local", label="Local coordinator", base_url=self.settings()["local_url"].rstrip("/"),
                        tier="local", key_env=None, models=["coordinator"], list_models=False)

    def home_terms(self) -> List[str]:
        base = self.base_home_terms
        if callable(base):
            try:
                self._last_home_terms = tuple(base())
            except Exception:
                pass    # lookup failed: keep the last good list (never fall back to an empty one)
            base = getattr(self, "_last_home_terms", ())
        return list(base) + self.settings()["home_terms"]

    # ---- models --------------------------------------------------------------
    def live_models(self, p: Provider, refresh: bool = False) -> List[Dict[str, Any]]:
        """Free models the provider lists right now ([] when it has no /models or it failed)."""
        if not p.list_models or not p.has_key or not p.base_url:
            return []
        now = self.clock()
        cached = self._models.get(p.id)
        if cached and not refresh and now - cached[0] < MODEL_CACHE_S:
            return cached[1]
        try:
            status, _, body = self.http.request("GET", p.base_url.rstrip("/") + "/models", p.headers(), timeout=10)
        except Exception:
            status, body = 0, None
        models: List[Dict[str, Any]] = []
        if status == 200 and isinstance(body, dict):
            for m in body.get("data") or body.get("models") or []:
                mid = str(m.get("id") or m.get("name") or "")
                mid = mid[len("models/"):] if mid.startswith("models/") else mid
                if not mid or any(w in mid.lower() for w in SKIP_MODEL_WORDS):
                    continue
                if p.free_filter and not re.search(p.free_filter, mid):
                    continue
                params = m.get("supported_parameters")
                models.append({"id": mid, "ctx": m.get("context_length") or m.get("context_window"),
                               "tools": ("tools" in params) if isinstance(params, list) else None})
            # biggest context first, then full models before -lite, then newest version (ids sort by version)
            models.sort(key=lambda x: (x["ctx"] or 0, "lite" not in x["id"], x["id"]), reverse=True)
            self._models[p.id] = (now, models)
        elif cached:
            return cached[1]
        return models

    def choose_model(self, p: Provider, need_tools: bool, pinned_model: Optional[str] = None) -> Optional[str]:
        if pinned_model:
            return pinned_model
        live = self.live_models(p)
        live_ids = {m["id"] for m in live}
        for mid in p.models:  # configured preference, if the provider still offers it
            if not live or mid in live_ids:
                return mid
        for m in live:
            if need_tools and m["tools"] is False:
                continue
            return m["id"]
        return p.models[0] if p.models else (p.fallback_models[0] if p.fallback_models else None)

    # ---- selection -----------------------------------------------------------
    def candidates(self, surface: str, verdict: Dict[str, Any], need_tools: bool = False, est_tokens: int = 0,
                   pinned: Optional[str] = None, allow_local: bool = True) -> Tuple[List[Tuple[Provider, str]], List[str]]:
        provs = self.providers()
        self.quota.loop_share = self.settings()["loop_share"]
        skipped: List[str] = []
        pin_pid, pin_model = None, None
        if pinned:
            pin_pid, _, pin_model = pinned.partition("/")
            pin_model = pin_model or None
        edges = [pid for pid in provs if pid.startswith("edge:")]
        order = (edges if verdict["class"] in ("home", "secret") else []) + self.settings()["priority"] + \
                ([] if verdict["class"] in ("home", "secret") else edges)
        if pin_pid:
            order = [pin_pid]
        out: List[Tuple[Provider, str]] = []
        seen = set()
        for pid in order:
            if pid in seen:
                continue
            seen.add(pid)
            p = provs.get(pid)
            if p is None:
                skipped.append(f"{pid}: unknown")
                continue
            if not p.enabled:
                skipped.append(f"{pid}: disabled")
                continue
            if not p.has_key:
                skipped.append(f"{pid}: no key")
                continue
            if need_tools and not p.tools_ok:
                skipped.append(f"{pid}: no tool calls")
                continue
            why = egress.refusal_reason(p.tier, verdict)
            if why:
                skipped.append(f"{pid}: {why}")
                continue
            why = self.quota.check(pid, p.limits, surface, est_tokens)
            if why:
                skipped.append(f"{pid}: {why}")
                continue
            model = self.choose_model(p, need_tools, pin_model)
            if not model:
                skipped.append(f"{pid}: no free model listed")
                continue
            out.append((p, model))
        if allow_local:
            loc = self.local_provider()
            out.append((loc, "coordinator"))
        return out, skipped

    # ---- calls -----------------------------------------------------------------
    def _body(self, model: str, messages, tools, max_tokens, temperature, stream, extra=None) -> Dict[str, Any]:
        body = {"model": model, "messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
                "stream": bool(stream)}
        if tools:
            body["tools"] = tools
        for k, v in (extra or {}).items():
            if k not in body and v is not None:
                body[k] = v
        return body

    def _note_failure(self, p: Provider, status: int, headers: Dict[str, str], body: Any) -> str:
        err = f"HTTP {status}: {_err_text(body)}" if status else f"unreachable: {body}"
        if p.id == "local":
            return err
        if status == 429:
            self.quota.cool_down(p.id, _retry_after(headers, body), err)
        elif status in (401, 403):
            self.quota.cool_down(p.id, 3600, err)       # bad key: don't hammer
        elif status == 0 or status >= 500:
            self.quota.cool_down(p.id, 120, err)
        else:
            self.quota.note_error(p.id, err)
        return err

    def _report(self, t0: float, surface: str, declared: str, verdict: Dict[str, Any], stream: bool,
                res: Dict[str, Any]) -> Dict[str, Any]:
        """Send one trace record for this call (if tracing is on) and hand `res` back unchanged."""
        if self.on_call:
            ok = bool(res.get("ok"))
            usage = ((res.get("data") or {}).get("usage") or {}) if ok else {}
            triggers = []
            if not ok:
                triggers.append("all_failed")
            elif res.get("tried"):
                triggers.append("fell_through")               # an earlier source failed first
            if ok and res.get("provider") == "local":
                triggers.append("local_fallback")
            rec = {"kind": "boost", "at": round(t0, 3), "surface": surface, "declared": declared,
                   "class": verdict["class"], "found": sorted({":".join(h.split(":")[:2]) for h in verdict.get("hits", [])}),
                   "images": bool(verdict.get("images")), "stream": stream, "outcome": "answered" if ok else "failed",
                   "provider": res.get("provider"), "model": res.get("model"), "tier": res.get("tier"),
                   "ms": round((self.clock() - t0) * 1000), "tokens": usage.get("total_tokens"),
                   "tried": [t[:160] for t in res.get("tried") or []], "skipped": len(res.get("skipped") or []),
                   "triggers": triggers}
            try:
                self.on_call(rec)
            except Exception:
                pass  # tracing must never break a call
        return res

    def complete(self, messages: List[Dict[str, Any]], surface: str, declared: str = "general",
                 tools: Optional[List[Dict[str, Any]]] = None, max_tokens: int = 800, temperature: float = 0.4,
                 pinned: Optional[str] = None, allow_local: bool = True, extra: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
        """Non-streaming completion. Returns {ok, provider, model, tier, data | error, tried, skipped}."""
        t0 = self.clock()
        verdict = egress.classify(declared, messages, self.home_terms())
        cands, skipped = self.candidates(surface, verdict, bool(tools), _est_tokens(messages, max_tokens), pinned, allow_local)
        tried: List[str] = []
        timeout = self.settings()["timeout_s"]
        for p, model in cands:
            body = self._body(model, messages, tools, max_tokens, temperature, False, extra)
            try:
                status, headers, data = self.http.request("POST", p.base_url.rstrip("/") + "/chat/completions",
                                                          p.headers(), body, timeout=timeout)
            except Exception as e:
                status, headers, data = 0, {}, f"{type(e).__name__}: {e}"
            if status == 200 and isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if p.id != "local":
                    self.quota.record(p.id, usage.get("total_tokens") or _est_tokens(messages, 0), surface)
                return self._report(t0, surface, declared, verdict, False,
                                    {"ok": True, "provider": p.id, "label": p.label, "model": model, "tier": p.tier,
                                     "data": data, "class": verdict["class"], "tried": tried, "skipped": skipped})
            tried.append(f"{p.id}: {self._note_failure(p, status, headers, data)}")
        return self._report(t0, surface, declared, verdict, False,
                            {"ok": False, "error": "no free source could take this call", "class": verdict["class"],
                             "tried": tried, "skipped": skipped})

    def open_stream(self, messages: List[Dict[str, Any]], surface: str, declared: str = "general",
                    tools: Optional[List[Dict[str, Any]]] = None, max_tokens: int = 800, temperature: float = 0.4,
                    pinned: Optional[str] = None, allow_local: bool = True, extra: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
        """Streaming: try sources until one answers 200, then hand back its open response.

        Returns {ok, provider, model, tier, response, ...}; the caller iterates `response` and closes it.
        Usage is recorded here with an estimate (streams rarely report token counts). The trace's ms is the time to
        the first answering source, not to the end of the stream."""
        t0 = self.clock()
        verdict = egress.classify(declared, messages, self.home_terms())
        est = _est_tokens(messages, max_tokens)
        cands, skipped = self.candidates(surface, verdict, bool(tools), est, pinned, allow_local)
        tried: List[str] = []
        timeout = self.settings()["timeout_s"]
        for p, model in cands:
            body = self._body(model, messages, tools, max_tokens, temperature, True, extra)
            try:
                status, headers, resp = self.http.open_stream(p.base_url.rstrip("/") + "/chat/completions",
                                                              p.headers(), body, timeout=timeout)
            except Exception as e:
                status, headers, resp = 0, {}, f"{type(e).__name__}: {e}"
            if status == 200:
                if p.id != "local":
                    self.quota.record(p.id, est, surface)
                return self._report(t0, surface, declared, verdict, True,
                                    {"ok": True, "provider": p.id, "label": p.label, "model": model, "tier": p.tier,
                                     "response": resp, "class": verdict["class"], "tried": tried, "skipped": skipped})
            tried.append(f"{p.id}: {self._note_failure(p, status, headers, resp)}")
        return self._report(t0, surface, declared, verdict, True,
                            {"ok": False, "error": "no free source could take this call", "class": verdict["class"],
                             "tried": tried, "skipped": skipped})

    # ---- status ----------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        s = self.settings()
        provs = self.providers()
        rows = []
        for pid in list(dict.fromkeys(s["priority"] + list(provs))):
            p = provs.get(pid)
            if not p:
                continue
            row = p.public()
            row["usage"] = self.quota.usage(pid, p.limits)
            row["live_models"] = [m["id"] for m in self._models.get(pid, (0, []))[1]][:40]
            rows.append(row)
        return {"enabled": s["enabled"], "surfaces": s["surfaces"], "priority": s["priority"],
                "loop_share": s["loop_share"], "providers": rows}


def iter_sse(resp) -> Iterator[str]:
    """Yield decoded SSE lines from an open streaming response and close it."""
    try:
        for line in resp:
            decoded = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
            if decoded.strip():
                yield decoded if decoded.endswith("\n") else decoded + "\n"
    finally:
        try:
            resp.close()
        except Exception:
            pass
