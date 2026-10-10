"""Offline tests for the OpenAI-compatible gateway (lodestone/gateway.py): fake engines on ephemeral loopback ports,
the gateway on another, a config that points at them. Nothing here leaves 127.0.0.1."""

import json
import os
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import gateway  # noqa: E402


class Fake:
    """A fake upstream. name shows in replies so a test can tell which engine answered."""

    def __init__(self, name, model_ids=(), voice=False):
        self.name, self.model_ids, self.seen = name, list(model_ids), []
        self.first_chunk_seen = threading.Event()     # set by the client side of the streaming test
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body, ctype="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, json.dumps({"object": "list", "data": [
                        {"id": i, "object": "model"} for i in fake.model_ids]}).encode())
                elif self.path == "/health":
                    self._send(200, b'{"status":"ok"}')
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.seen.append((self.path, self.headers.get("Content-Type"), raw, self.headers.get("Authorization")))
                if self.path.startswith("/v1/audio/") or self.path == "/v1/embeddings":
                    return self._send(200, raw, self.headers.get("Content-Type") or "application/octet-stream")
                body = json.loads(raw)
                if body.get("model") == "boom":
                    return self._send(429, json.dumps({"error": {"message": "slow down", "type": "rate"}}).encode())
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"n":1}\n\n')
                    self.wfile.flush()
                    fake.first_chunk_released = fake.first_chunk_seen.wait(5)   # the client must see chunk 1 first
                    self.wfile.write(b'data: {"n":2}\n\ndata: [DONE]\n\n')
                    return
                self._send(200, json.dumps({"engine": fake.name, "model": body.get("model")}).encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, args=(0.01,), daemon=True).start()
        self.first_chunk_released = False

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def dead_port():
    """A port whose owner hangs up on every connection: an engine that is down (a refused connect takes ~2 s on
    Windows, which would make these tests slow)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)

    def hang_up():
        while True:
            try:
                s.accept()[0].close()
            except OSError:
                return

    threading.Thread(target=hang_up, daemon=True).start()
    DEAD.append(s)
    return s.getsockname()[1]


DEAD = []


def call(base, path, body=None, headers=None, raw=None):
    """(status, headers, bytes) for a GET (no body) or POST (dict -> JSON, or raw bytes)."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = dict(headers or {})
    if data is not None and "Content-Type" not in hdrs:
        hdrs["Content-Type"] = "application/json"
    try:
        with urlopen(Request(base + path, data=data, headers=hdrs), timeout=10) as r:
            return r.status, r.headers, r.read()
    except HTTPError as e:
        return e.code, e.headers, e.read()


class GatewayTest(unittest.TestCase):
    gw_extra = {}

    def setUp(self):
        self.coord = Fake("coordinator", ["big-chat-model"])
        self.worker = Fake("worker", ["small-code-model"])
        self.vision = Fake("vision", ["eyes-model"])
        self.embed = Fake("embedder", ["vec-model"])
        self.voice = Fake("voice")
        self.fakes = [self.coord, self.worker, self.vision, self.embed, self.voice]
        self.start({})

    def start(self, gw_cfg, **overrides):
        if getattr(self, "srv", None):
            self.srv.shutdown()
            self.srv.server_close()
        url = lambda f, v1=True: "http://127.0.0.1:%d%s" % (f.port, "/v1" if v1 else "")   # noqa: E731
        cfg = {"cluster": {"coordinator_url": url(self.coord), "worker_url": url(self.worker),
                           "vision_url": url(self.vision), "embedder_url": url(self.embed)},
               "voice": {"stt_url": url(self.voice, False)},
               "gateway": dict({"models_ttl_s": 0, "probe_timeout_s": 2, "timeout_s": 10, "log_requests": False}, **gw_cfg)}
        for section, vals in overrides.items():
            cfg[section].update(vals)
        self.srv = gateway.make_server(cfg, "127.0.0.1", 0)
        threading.Thread(target=self.srv.serve_forever, args=(0.01,), daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        for f in self.fakes:
            f.stop()
        while DEAD:
            DEAD.pop().close()

    # ------------------------------------------------------------------------------------------------- models ----
    def test_models_union_with_role_aliases(self):
        status, _, body = call(self.base, "/v1/models")
        ids = [m["id"] for m in json.loads(body)["data"]]
        self.assertEqual(status, 200)
        for i in ("big-chat-model", "small-code-model", "eyes-model", "vec-model", "coordinator", "worker", "vision"):
            self.assertIn(i, ids)
        self.assertEqual(len(ids), len(set(ids)))
        alias = next(m for m in json.loads(body)["data"] if m["id"] == "worker")
        self.assertEqual(alias["root"], "small-code-model")

    def test_models_skips_a_down_engine(self):
        self.start({}, cluster={"worker_url": "http://127.0.0.1:%d/v1" % dead_port()})
        status, _, body = call(self.base, "/v1/models")
        ids = [m["id"] for m in json.loads(body)["data"]]
        self.assertEqual(status, 200)
        self.assertIn("big-chat-model", ids)
        self.assertNotIn("small-code-model", ids)
        self.assertNotIn("worker", ids)

    # ---------------------------------------------------------------------------------------------- chat route ----
    def chat(self, model=None, **extra):
        body = dict({"messages": [{"role": "user", "content": "hi"}]}, **extra)
        if model is not None:
            body["model"] = model
        status, _, raw = call(self.base, "/v1/chat/completions", body)
        return status, json.loads(raw)

    def test_default_and_alias_route_and_alias_is_rewritten_to_the_engines_model(self):
        self.assertEqual(self.chat(), (200, {"engine": "coordinator", "model": "big-chat-model"}))
        self.assertEqual(self.chat("worker"), (200, {"engine": "worker", "model": "small-code-model"}))
        self.assertEqual(self.chat("vision")[1]["engine"], "vision")

    def test_model_id_routes_to_the_engine_that_lists_it(self):
        self.assertEqual(self.chat("small-code-model"), (200, {"engine": "worker", "model": "small-code-model"}))
        self.assertEqual(self.chat("eyes-model")[1]["engine"], "vision")

    def test_completions_route_the_same_way(self):
        status, _, raw = call(self.base, "/v1/completions", {"model": "worker", "prompt": "x"})
        self.assertEqual((status, json.loads(raw)["engine"]), (200, "worker"))

    def test_unknown_model_is_404_in_openai_shape(self):
        status, body = self.chat("no-such-model")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "model_not_found")
        self.assertIn("no-such-model", body["error"]["message"])
        self.assertIn("type", body["error"])
        self.assertEqual([len(f.seen) for f in self.fakes], [0] * 5)

    def test_embedding_model_is_not_a_chat_model(self):
        self.assertEqual(self.chat("vec-model")[0], 404)

    def test_upstream_down_is_502(self):
        self.start({}, cluster={"coordinator_url": "http://127.0.0.1:%d/v1" % dead_port()})
        status, body = self.chat("coordinator")
        self.assertEqual(status, 502)
        self.assertEqual(body["error"]["type"], "api_error")

    def test_upstream_error_status_and_body_are_relayed(self):
        self.coord.model_ids = ["boom"]
        status, body = self.chat("boom")
        self.assertEqual((status, body["error"]["message"]), (429, "slow down"))

    def test_bad_json_is_400(self):
        for raw in (b"{not json", b"[1,2]", b""):
            status, _, body = call(self.base, "/v1/chat/completions", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertIn("message", json.loads(body)["error"])
        self.assertEqual(call(self.base, "/v1/embeddings", raw=b"{oops")[0], 400)

    def test_unknown_route_is_404(self):
        self.assertEqual(call(self.base, "/v1/nothing", {"a": 1})[0], 404)
        self.assertEqual(call(self.base, "/v1/chat/completions")[0], 404)   # GET on a POST route

    # -------------------------------------------------------------------------------------------- streaming ----
    def test_stream_is_relayed_chunk_by_chunk(self):
        body = {"model": "coordinator", "stream": True, "messages": []}
        req = Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as r:
            self.assertEqual(r.headers.get("Content-Type"), "text/event-stream")
            first = r.readline()
            # the engine is still blocked waiting for us: if the gateway buffered the reply we would never get here
            self.assertFalse(self.coord.first_chunk_released)
            self.assertEqual(first, b'data: {"n":1}\n')
            self.coord.first_chunk_seen.set()
            rest = r.read()
        self.assertIn(b'"n":2', rest)
        self.assertTrue(self.coord.first_chunk_released)     # the engine was released by us, not by its own timeout
        self.assertTrue(rest.rstrip().endswith(b"[DONE]"))

    # ------------------------------------------------------------------------------------------- passthrough ----
    def test_embeddings_go_to_the_embedder_unchanged(self):
        payload = {"model": "vec-model", "input": ["a", "b"]}
        status, _, raw = call(self.base, "/v1/embeddings", payload)
        self.assertEqual((status, json.loads(raw)), (200, payload))
        self.assertEqual(len(self.embed.seen), 1)
        self.assertEqual(len(self.coord.seen), 0)

    def test_audio_body_and_content_type_pass_through_byte_for_byte(self):
        blob = bytes(range(256)) * 40 + b"\r\n--x--\r\n"
        ctype = "multipart/form-data; boundary=x"
        status, hdrs, raw = call(self.base, "/v1/audio/transcriptions", raw=blob, headers={"Content-Type": ctype})
        self.assertEqual((status, raw, hdrs.get("Content-Type")), (200, blob, ctype))
        path, seen_type, seen_raw, _ = self.voice.seen[-1]
        self.assertEqual((path, seen_type, seen_raw), ("/v1/audio/transcriptions", ctype, blob))
        status, hdrs, raw = call(self.base, "/v1/audio/speech", {"input": "hello"})
        self.assertEqual((status, self.voice.seen[-1][0]), (200, "/v1/audio/speech"))

    def test_health_reports_each_upstream(self):
        status, _, raw = call(self.base, "/health")
        self.assertEqual((status, json.loads(raw)["status"]), (200, "ok"))
        self.start({}, cluster={"worker_url": "http://127.0.0.1:%d/v1" % dead_port()})
        status, _, raw = call(self.base, "/health")
        body = json.loads(raw)
        self.assertEqual((status, body["status"], body["upstreams"]["worker"], body["upstreams"]["coordinator"]),
                         (200, "degraded", False, True))

    # ---------------------------------------------------------------------------------------------------- auth ----
    def test_no_key_configured_means_open(self):
        os.environ.pop("LODESTONE_API_KEY", None)
        self.assertEqual(call(self.base, "/v1/models")[0], 200)

    def test_key_required_when_configured(self):
        self.start({"api_key": "test-key-123"})
        status, _, raw = call(self.base, "/v1/models")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw)["error"]["code"], "invalid_api_key")
        self.assertEqual(call(self.base, "/v1/models", headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(call(self.base, "/v1/models", headers={"Authorization": "test-key-123"})[0], 401)
        self.assertEqual(call(self.base, "/v1/models", headers={"Authorization": "Bearer test-key-123"})[0], 200)
        self.assertEqual(call(self.base, "/v1/embeddings", {"input": "x"})[0], 401)
        self.assertEqual(call(self.base, "/health")[0], 200)          # health holds no model data and needs no key
        self.assertEqual(len(self.embed.seen), 0)

    def test_key_from_environment_and_never_forwarded(self):
        os.environ["LODESTONE_API_KEY"] = "env-key-456"
        try:
            self.assertEqual(call(self.base, "/v1/models")[0], 401)
            hdr = {"Authorization": "Bearer env-key-456"}
            status, _, _ = call(self.base, "/v1/chat/completions", {"messages": []}, headers=hdr)
            self.assertEqual(status, 200)
            self.assertIsNone(self.coord.seen[-1][3])               # the engines never see the gateway's key
        finally:
            os.environ.pop("LODESTONE_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
