"""
Live system profile: the one place StoneSage learns what hardware, engines and models it runs on.

Every label the UI, CLI, prompts and telemetry show ("Qwen3 14B on RX 6750 XT, 6k x 2") must come from
here, never from a hardcoded string. Sources:
  - each llama-server's /props (model file, context per slot, slots) via the URLs in config.json
  - `hw_probe.py` on the inference host over SSH (GPU names, VRAM, which engine sits on which GPU,
    draft model, KV cache type)
Cached for CACHE_S; `get_profile(fresh=True)` re-probes (call it after changing an engine).
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
from urllib.parse import urlparse

CACHE_S = 30
ROLES = [  # role, config key, what it is for
    ("coordinator", "coordinator_url", "Chat, reasoning and Courage's tool calls"),
    ("worker", "worker_url", "Fast drafts, code and JSON"),
    ("embedder", "embedder_url", "Embeddings for memory search"),
    ("vision", "vision_url", "Camera and image understanding"),
]
PROBE_CMD = "sudo -n python3 /opt/cluster-bridge/hw_probe.py"
QUANT = re.compile(r"[-_.]((?:i?q\d(?:_[a-z0-9]+)*)|bf16|f16|f32)$", re.I)
UPPER = {"bge": "BGE", "vl": "VL", "moe": "MoE", "gguf": "", "en": "", "instruct": "", "chat": "", "it": ""}

_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "profile": None}


def model_names(path: Optional[str]) -> Dict[str, Optional[str]]:
    """'/opt/models/qwen2.5-coder-3b-instruct-q5_k_m.gguf' -> name 'Qwen2.5 Coder 3B', quant 'Q5_K_M', params '3B'."""
    if not path:
        return {"file": None, "name": None, "quant": None, "params": None}
    file = os.path.basename(path)
    stem = re.sub(r"\.gguf$", "", file, flags=re.I)
    quant = QUANT.search(stem)
    if quant:
        stem = stem[:quant.start()]
    words = []
    for tok in re.split(r"[-_]+", stem):
        low = tok.lower()
        if low in UPPER:
            if UPPER[low]:
                words.append(UPPER[low])
        elif re.fullmatch(r"\d+(\.\d+)?[bm]", low) or re.fullmatch(r"a\d+(\.\d+)?b", low):
            words.append(tok.upper())
        elif re.fullmatch(r"v\d+(\.\d+)*", low):
            words.append(low)
        else:
            words.append(tok[:1].upper() + tok[1:])
    params = next((w for w in words if re.fullmatch(r"\d+(\.\d+)?[BM]", w)), None)
    return {"file": file, "name": " ".join(words) or file, "quant": quant.group(1).upper() if quant else None,
            "params": params}


def gpu_short(name: str) -> str:
    """'AMD Radeon RX 6750 XT' -> 'RX 6750 XT'; 'NVIDIA GeForce RTX 4090' -> 'RTX 4090'."""
    return re.sub(r"^(AMD Radeon|NVIDIA GeForce|NVIDIA|Intel\(R\)|Intel)\s+", "", name or "").strip() or name


def _get_json(url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _base(url: str) -> str:
    url = (url or "").rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def run_probe(host: str, user: str, timeout: float = 10.0) -> Dict[str, Any]:
    if os.environ.get("STONESAGE_OFFLINE"):
        return {"error": "offline"}
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", f"{user}@{host}", PROBE_CMD],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return {"error": (out.stderr or f"exit {out.returncode}").strip()[:200]}
        return json.loads(out.stdout)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _engine(role: str, purpose: str, url: str, probe_engines: Dict[int, Dict[str, Any]],
            gpus_by_pci: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = _base(url)
    port = urlparse(base).port
    props = _get_json(f"{base}/props") or {}
    online = bool(props) or bool(_get_json(f"{base}/health"))
    live = probe_engines.get(port, {})
    model_path = props.get("model_path") or live.get("model_path")
    names = model_names(model_path)
    ctx_slot = (props.get("default_generation_settings") or {}).get("n_ctx")
    slots = props.get("total_slots") or live.get("slots")
    vram = {pci: mb for pci, mb in (live.get("vram_mb_by_gpu") or {}).items() if mb}
    gpu = gpus_by_pci.get(max(vram, key=vram.get)) if vram else None
    ctx_total = live.get("ctx_total") or (ctx_slot * slots if ctx_slot and slots else None)
    eng = {
        "role": role, "purpose": purpose, "url": url, "port": port, "online": online,
        "model": names["name"], "model_file": names["file"], "quant": names["quant"], "params": names["params"],
        "draft_model": model_names(live.get("draft_model_path"))["name"],
        "draft_model_file": model_names(live.get("draft_model_path"))["file"],
        "mmproj": model_names(live.get("mmproj_path"))["file"],
        "ctx_per_slot": ctx_slot or (ctx_total // slots if ctx_total and slots else None),
        "ctx_total": ctx_total, "slots": slots, "kv_cache": live.get("kv_cache_type"),
        "flash_attn": live.get("flash_attn"),
        "unit": (live.get("unit") or "").removesuffix(".service") or None,  # systemd unit on the inference host
        "gpu": {"index": gpu["index"], "name": gpu["name"], "short": gpu["short"]} if gpu else None,
        "vram_mb": sum(vram.values()) or None,
        "offloaded": bool(live) and not vram,  # running but nothing resident in VRAM: CPU/RAM
    }
    where = eng["gpu"]["short"] if eng["gpu"] else ("CPU" if eng["offloaded"] else None)
    eng["label"] = " · ".join(x for x in (eng["model"] or role.title(), where) if x)
    return eng


def _ssh_user(cfg: Dict[str, Any]) -> str:
    from lodestone import endpoints     # (endpoints imports this module: look it up when used)
    return endpoints.ssh_user(cfg)


def build_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cl = cfg.get("cluster", {})
    urls = {role: cl.get(key) for role, key, _ in ROLES if cl.get(key)}
    host = urlparse(_base(next(iter(urls.values()), ""))).hostname
    probe = run_probe(host, _ssh_user(cfg)) if host else {"error": "no engine URLs configured"}
    gpus = []
    for i, g in enumerate(probe.get("gpus") or []):
        gpus.append(dict(g, index=i, short=gpu_short(g.get("name", "")),
                         vram_total_gb=round(g.get("vram_total_mb", 0) / 1024, 1),
                         vram_used_gb=round(g.get("vram_used_mb", 0) / 1024, 1)))
    by_pci = {g["pci"]: g for g in gpus}
    live = {e["port"]: e for e in probe.get("engines") or []}
    purposes = {role: purpose for role, _, purpose in ROLES}
    with ThreadPoolExecutor(max_workers=len(urls) or 1) as pool:
        futs = {role: pool.submit(_engine, role, purposes[role], url, live, by_pci) for role, url in urls.items()}
        engines = {role: f.result() for role, f in futs.items()}
    for g in gpus:
        g["engines"] = [r for r, e in engines.items() if e["gpu"] and e["gpu"]["index"] == g["index"]]
    return {"updated_at": time.time(), "host": probe.get("host") or {"hostname": host},
            "gpus": gpus, "engines": engines, "probe_error": probe.get("error")}


CONFIG_PATH = os.environ.get("STONESAGE_CONFIG_FILE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")   # the backend's config.json


def load_cfg() -> Dict[str, Any]:
    """config.json for modules that don't get it passed in (never raises)."""
    try:
        with open(CONFIG_PATH, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def get_profile(cfg: Dict[str, Any], fresh: bool = False) -> Dict[str, Any]:
    with _lock:
        if not fresh and _cache["profile"] and time.time() - _cache["at"] < CACHE_S:
            return _cache["profile"]
    profile = build_profile(cfg)
    with _lock:
        _cache.update(at=time.time(), profile=profile)
    return profile


def cached_profile() -> Optional[Dict[str, Any]]:
    """The last profile that was read, or None: for callers that must not wait on a probe (an SSH round trip)."""
    with _lock:
        return _cache["profile"]


def engine_label(cfg: Dict[str, Any], role: str, default: str = "") -> str:
    """Short label for prompts and logs, e.g. 'Qwen2.5 Coder 3B · RX 6750 XT'. Never raises."""
    try:
        eng = get_profile(cfg)["engines"].get(role)
        return (eng or {}).get("label") or default or role
    except Exception:
        return default or role


def live_mode(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The running engine layout as one 'cluster mode' entry (replaces the old hardcoded mode presets)."""
    engines = profile.get("engines") or {}
    coord, worker = engines.get("coordinator") or {}, engines.get("worker") or {}
    vram = ", ".join(f"{g['short']} {g['vram_used_gb']}/{g['vram_total_gb']} GB" for g in profile.get("gpus") or [])
    ctx = " / ".join(f"{e['ctx_per_slot']}×{e['slots'] or 1}" for e in (coord, worker) if e.get("ctx_per_slot"))
    return {
        "id": "live", "recommended": True,
        "name": " + ".join(e["label"] for e in (coord, worker) if e.get("label")) or "No engines reachable",
        "short": coord.get("params") or coord.get("model") or "offline",
        "vram": vram or "unknown", "context": ctx,
        "coordinator_model": coord.get("model_file"), "worker_model": worker.get("model_file"),
        "description": "The engines as they are running now (read live, not a preset).",
    }


def loaded_model_files(profile: Dict[str, Any]) -> set:
    """Model files the running engines have loaded (main, draft, mmproj): these must never be deleted."""
    files = set()
    for e in (profile.get("engines") or {}).values():
        files.update(f for f in (e.get("model_file"), e.get("mmproj")) if f)
        if e.get("draft_model_file"):
            files.add(e["draft_model_file"])
    return files


_models_cache: Dict[str, Any] = {"at": 0.0, "models": None}
MODELS_CACHE_S = 300


def get_models(cfg: Dict[str, Any], fresh: bool = False) -> list:
    """Every GGUF on the inference host with metadata read from its header (architecture, trained context,
    layers, KV heads, size label). Marks files loaded by a running engine. Empty list when unreachable."""
    with _lock:
        if not fresh and _models_cache["models"] is not None and time.time() - _models_cache["at"] < MODELS_CACHE_S:
            return _models_cache["models"]
    cl = cfg.get("cluster", {})
    host = urlparse(_base(cl.get("coordinator_url", ""))).hostname
    if not host or os.environ.get("STONESAGE_OFFLINE"):
        return []
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", f"{_ssh_user(cfg)}@{host}",
                              PROBE_CMD + " --models"], capture_output=True, text=True, timeout=60)
        items = json.loads(out.stdout).get("models", []) if out.returncode == 0 else []
    except Exception:
        items = []
    loaded = loaded_model_files(get_profile(cfg))
    for m in items:
        m.update(model_names(m.get("path")))  # name, quant, params from the filename
        m["params"] = m.get("size_label") or m.get("params")
        m["is_loaded"] = m.get("file") in loaded
        m["is_projector"] = m.get("architecture") == "clip"
    with _lock:
        _models_cache.update(at=time.time(), models=items)
    return items
