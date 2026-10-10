"""One OpenAI-compatible address in front of the local engines (docs/LAYERS.md M4).

An app or a third-party agent should need one base URL and one key, not to know that chat runs on one port, vision on a
second, embeddings on a third and speech on a server of its own. The gateway names them once (lodestone.endpoints) and
routes by what the request says:

    GET  /v1/models                  union of every engine's model list; each model also under its role alias
    POST /v1/chat/completions        by the request's "model": a role alias (coordinator | worker | vision) or a model id
    POST /v1/completions             some engine lists; no "model" = coordinator
    POST /v1/embeddings             -> embedder
    POST /v1/audio/transcriptions   -> voice (body and Content-Type untouched: multipart passes through byte for byte)
    POST /v1/audio/speech           -> voice
    GET  /health                     which upstreams answer (never needs the key: it holds no model data)

    python -m lodestone.gateway [--host H] [--port P]      config gateway.{host,port,api_key,timeout_s}

Why it relays instead of redirecting: the engines listen on private addresses, and only the gateway has to be exposed.
Why streams are relayed chunk by chunk: a token-by-token reply that waits for the whole answer is a different product.
Keys: gateway.api_key (or env LODESTONE_API_KEY) turns on `Authorization: Bearer`; it is compared in constant time and
never logged. Without one the default bind is loopback only; binding wider with no key is logged as a warning.
Not here yet: per-key quotas, Boost routing, upstream keys (see docs/LAYERS.md).
"""

import argparse
import http.client
import hmac
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lodestone import endpoints
from lodestone.system_profile import load_cfg

CHAT_ROLES = ("coordinator", "worker", "vision")
LIST_ROLES = CHAT_ROLES + ("embedder",)
DEFAULTS = {"host": "127.0.0.1", "port": 8000, "timeout_s": 120, "probe_timeout_s": 3, "models_ttl_s": 10,
            "max_body_mb": 100, "log_requests": True}
CHUNK = 8192


class GatewayError(Exception):
    """Becomes {"error": {"message", "type", "code"}} with this HTTP status (the shape OpenAI clients parse)."""

    def __init__(self, status: int, message: str, etype: str = "invalid_request_error", code: str = ""):
        super().__init__(message)
        self.status, self.message, self.etype, self.code = status, message, etype, code


def _join(base: str, path: str) -> str:
    """base + a '/v1/...' path. Engines are configured with the /v1 included, the voice server without it."""
    base = base.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        base = base[:-3]
    return base + path


class Gateway:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._models: Dict[str, Tuple[float, Optional[List[Dict[str, Any]]]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------------- config ----
    def setting(self, key: str) -> Any:
        value = (self.cfg.get("gateway") or {}).get(key)
        return DEFAULTS.get(key) if value in (None, "") else value

    @property
    def api_key(self) -> str:
        return str(self.setting("api_key") or os.environ.get("LODESTONE_API_KEY", "")).strip()

    def authorised(self, header: str) -> bool:
        key = self.api_key
        if not key:
            return True
        scheme, _, token = (header or "").strip().partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode())

    # ------------------------------------------------------------------------------------------------ upstream ----
    def open(self, name: str, path: str, data: Optional[bytes] = None, content_type: str = "application/json",
             timeout: Optional[float] = None):
        """An open upstream response (an error status is a response too: the engine's own error body is relayed)."""
        base = endpoints.url(name, self.cfg)
        if not base:
            raise GatewayError(502, f"upstream '{name}' is not configured", "api_error", "upstream_unavailable")
        headers = {"Content-Type": content_type} if data is not None else {}
        req = Request(_join(base, path), data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            return urlopen(req, timeout=timeout or float(self.setting("timeout_s")))
        except HTTPError as e:
            return e
        except (socket.timeout, TimeoutError):
            raise GatewayError(504, f"upstream '{name}' timed out", "api_error", "upstream_timeout")
        except URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise GatewayError(504, f"upstream '{name}' timed out", "api_error", "upstream_timeout")
            raise GatewayError(502, f"upstream '{name}' is not answering", "api_error", "upstream_unavailable")
        except (OSError, http.client.HTTPException):   # refused, reset, or not HTTP at all
            raise GatewayError(502, f"upstream '{name}' is not answering", "api_error", "upstream_unavailable")

    def list_models(self, role: str, fresh: bool = False) -> Optional[List[Dict[str, Any]]]:
        """One engine's /v1/models entries, or None when it does not answer. Cached briefly: routing asks per request."""
        ttl = float(self.setting("models_ttl_s"))
        with self._lock:
            hit = self._models.get(role)
        if hit and not fresh and time.monotonic() - hit[0] < ttl:
            return hit[1]
        entries: Optional[List[Dict[str, Any]]] = None
        try:
            resp = self.open(role, "/v1/models", timeout=float(self.setting("probe_timeout_s")))
            with resp:
                if resp.getcode() == 200:
                    body = json.loads(resp.read().decode("utf-8", "replace"))
                    data = body.get("data") if isinstance(body, dict) else None
                    entries = [m for m in (data or []) if isinstance(m, dict) and m.get("id")]
        except (GatewayError, ValueError, OSError, http.client.HTTPException):
            entries = None
        with self._lock:
            self._models[role] = (time.monotonic(), entries)
        return entries

    def all_models(self) -> Dict[str, Optional[List[Dict[str, Any]]]]:
        with ThreadPoolExecutor(max_workers=len(LIST_ROLES)) as pool:
            return dict(zip(LIST_ROLES, pool.map(self.list_models, LIST_ROLES)))

    # --------------------------------------------------------------------------------------------------- routes ----
    def models_response(self) -> Dict[str, Any]:
        seen, data = set(), []
        for role, entries in self.all_models().items():
            if not entries:
                continue          # an engine that is down must not fail the whole list
            for m in entries:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    data.append(dict(m, object="model", role=role, owned_by=m.get("owned_by") or role))
            if role not in seen:  # the alias: stable across model swaps, which is the point of asking for a role
                seen.add(role)
                data.append({"id": role, "object": "model", "created": 0, "owned_by": "lodestone", "role": role,
                             "root": entries[0]["id"]})
        return {"object": "list", "data": data}

    def resolve_chat(self, model: Any) -> Tuple[str, bool]:
        """(role, alias_used) for a request's model; GatewayError 404 when no engine has it."""
        if model in (None, ""):
            return "coordinator", True
        if not isinstance(model, str):
            raise GatewayError(400, "'model' must be a string")
        if model in CHAT_ROLES:
            return model, True
        for role in CHAT_ROLES:
            if any(m["id"] == model for m in self.list_models(role) or []):
                return role, False
        raise GatewayError(404, f"The model '{model}' does not exist on any engine", "invalid_request_error",
                           "model_not_found")

    def health(self) -> Tuple[int, Dict[str, Any]]:
        probe = float(self.setting("probe_timeout_s"))

        def check(name: str) -> bool:
            try:
                with self.open(name, "/v1/models" if name != "voice" else "/health", timeout=probe) as r:
                    return r.getcode() < 500
            except (GatewayError, OSError, http.client.HTTPException):
                return False

        names = list(LIST_ROLES) + ["voice"]
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            up = dict(zip(names, pool.map(check, names)))
        status = "ok" if all(up.values()) else "degraded" if any(up.values()) else "down"
        return (503 if status == "down" else 200), {"status": status, "upstreams": up}


# -------------------------------------------------------------------------------------------------------- HTTP ----
def _make_handler(gw: Gateway):
    class Handler(BaseHTTPRequestHandler):
        server_version = "lodestone-gateway"

        def log_message(self, fmt, *args):   # method, path and status only: never headers, so never the key
            if gw.setting("log_requests") is not False:
                sys.stderr.write("[gateway] %s %s\n" % (self.address_string(), fmt % args))

        # -- replies
        def _json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, e: GatewayError) -> None:
            err = {"message": e.message, "type": e.etype}
            if e.code:
                err["code"] = e.code
            self._json(e.status, {"error": err})

        def _relay(self, resp, stream: bool) -> None:
            status, ctype = resp.getcode(), resp.headers.get("Content-Type") or "application/json"
            with resp:
                if stream and status < 400:
                    self.send_response(status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()           # no Content-Length: the connection closing ends the stream
                    try:
                        while True:
                            chunk = resp.read1(CHUNK)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (OSError, http.client.HTTPException):
                        pass                      # client left, or the engine died mid-reply: nothing left to say
                    return
                try:
                    raw = resp.read()
                except (OSError, http.client.HTTPException):
                    raise GatewayError(502, "upstream closed the connection mid-reply", "api_error", "upstream_unavailable")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        # -- requests
        def _body(self) -> bytes:
            self._read = True
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise GatewayError(400, "bad Content-Length")
            if n < 0 or n > int(gw.setting("max_body_mb")) * 1024 * 1024:
                raise GatewayError(413, "request body too large", "invalid_request_error", "body_too_large")
            return self.rfile.read(n) if n else b""

        def _json_body(self, raw: bytes) -> Dict[str, Any]:
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except (ValueError, UnicodeDecodeError):
                raise GatewayError(400, "request body is not valid JSON")
            if not isinstance(body, dict):
                raise GatewayError(400, "request body must be a JSON object")
            return body

        def _route(self, method: str) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health" and method == "GET":
                status, body = gw.health()
                return self._json(status, body)
            if not gw.authorised(self.headers.get("Authorization", "")):
                raise GatewayError(401, "Invalid or missing API key", "invalid_request_error", "invalid_api_key")
            if method == "GET" and path == "/v1/models":
                return self._json(200, gw.models_response())
            if method != "POST":
                raise GatewayError(404, f"Unknown route {method} {path}", "invalid_request_error", "unknown_route")
            if path in ("/v1/chat/completions", "/v1/completions"):
                raw = self._body()
                body = self._json_body(raw)
                role, alias = gw.resolve_chat(body.get("model"))
                if alias:    # engines only know their own model id; a role alias is the gateway's
                    ids = [m["id"] for m in gw.list_models(role) or []]
                    if ids:
                        body["model"] = ids[0]
                        raw = json.dumps(body).encode()
                return self._relay(gw.open(role, path, raw), body.get("stream") is True)
            if path == "/v1/embeddings":
                raw = self._body()
                self._json_body(raw)
                return self._relay(gw.open("embedder", path, raw), False)
            if path in ("/v1/audio/transcriptions", "/v1/audio/speech"):
                raw = self._body()
                ctype = self.headers.get("Content-Type") or "application/octet-stream"
                return self._relay(gw.open("voice", path, raw, ctype), False)
            raise GatewayError(404, f"Unknown route {method} {path}", "invalid_request_error", "unknown_route")

        def _drain(self) -> None:
            """Read an unread request body before replying: closing on unread data resets the connection on some
            systems, and the client then sees a dropped connection instead of our 401/404."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= int(gw.setting("max_body_mb")) * 1024 * 1024:
                    self.rfile.read(n)
            except (ValueError, OSError):
                pass

        def _dispatch(self, method: str) -> None:
            self._read = False
            try:
                self._route(method)
            except GatewayError as e:
                if not self._read:
                    self._drain()
                self._error(e)
            except (BrokenPipeError, ConnectionError):
                pass
            except Exception as e:   # a bug must come back as a JSON error, not a dropped connection
                sys.stderr.write("[gateway] internal error: %s: %s\n" % (type(e).__name__, e))
                try:
                    self._error(GatewayError(500, "internal gateway error", "api_error"))
                except OSError:
                    pass

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def make_server(cfg: Dict[str, Any], host: Optional[str] = None, port: Optional[int] = None) -> ThreadingHTTPServer:
    """A bound, not yet serving, gateway (port 0 = pick a free one: the tests do)."""
    gw = Gateway(cfg)
    host = host if host is not None else str(gw.setting("host"))
    srv = ThreadingHTTPServer((host, int(gw.setting("port") if port is None else port)), _make_handler(gw))
    srv.daemon_threads = True
    srv.gateway = gw  # type: ignore[attr-defined]
    if not gw.api_key and host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write("[gateway] WARNING: listening on %s with no API key (set gateway.api_key)\n" % host)
    return srv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lodestone.gateway", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)
    srv = make_server(load_cfg(), args.host, args.port)
    print("[gateway] serving on http://%s:%d (key %s)" % (srv.server_address[0], srv.server_address[1],
                                                           "required" if srv.gateway.api_key else "off"))  # type: ignore[attr-defined]
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
