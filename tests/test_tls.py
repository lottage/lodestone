"""TLS verification for the https services the server calls (lodestone/tls.py; Proxmox, Immich, probes): on by default,
`ca_file` to trust a self-signed certificate, `verify_tls: false` only when asked for. Offline: the handshake tests talk to a
server on 127.0.0.1 with a certificate made on the spot by the openssl program (skipped where there is none)."""

import http.server
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import tls  # noqa: E402
from lodestone.proxmox_client import ProxmoxClient  # noqa: E402

OPENSSL = shutil.which("openssl")


class TestSettings(unittest.TestCase):
    def test_the_default_is_to_verify(self):
        self.assertEqual(tls.settings(), {"verify": True, "ca_file": ""})
        self.assertEqual(tls.settings({}, None, {"url": "x"}), {"verify": True, "ca_file": ""})

    def test_only_an_explicit_false_turns_it_off(self):
        for off in (False, "false", "False", "no", "off", "0"):
            self.assertFalse(tls.settings({"verify_tls": off})["verify"], off)
        for on in (True, "true", "yes", 1, 0, None, "", [], "maybe", {}):
            self.assertTrue(tls.settings({"verify_tls": on})["verify"], repr(on))   # anything odd keeps the safe default

    def test_a_later_section_wins_and_ca_file_is_kept(self):
        s = tls.settings({"verify_tls": False, "ca_file": "/etc/a.pem"}, {"verify_tls": True})
        self.assertEqual(s, {"verify": True, "ca_file": "/etc/a.pem"})
        self.assertEqual(tls.settings({"ca_file": "/a"}, {"ca_file": " /b "})["ca_file"], "/b")

    def test_the_old_verify_ssl_key_means_nothing(self):
        """It sat in example configs and was never read: it must not silently switch checking off."""
        self.assertTrue(tls.settings({"verify_ssl": False})["verify"])


class TestContext(unittest.TestCase):
    def test_default_context_checks_the_certificate_and_the_name(self):
        ctx = tls.context({}, name="svc")
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)

    def test_verify_false_is_explicit_and_warned_about_once(self):
        tls._warned.discard("warn-once-svc")
        with self.assertLogs("stonesage.tls", level="WARNING") as logs:
            ctx = tls.context({"verify_tls": False}, name="warn-once-svc")
            tls.context({"verify_tls": False}, name="warn-once-svc")
        self.assertEqual(ctx.verify_mode, ssl.CERT_NONE)
        self.assertFalse(ctx.check_hostname)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("warn-once-svc", logs.output[0])

    def test_a_ca_file_that_cannot_be_read_is_an_error_not_a_quiet_fall_back(self):
        with self.assertRaises(tls.TlsConfigError) as cm:
            tls.context({"ca_file": os.path.join(tempfile.gettempdir(), "no-such-ca-file.pem")}, name="svc")
        self.assertIn("ca_file", str(cm.exception))
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
            f.write("this is not a certificate\n")
        try:
            with self.assertRaises(tls.TlsConfigError):
                tls.context({"ca_file": f.name}, name="svc")
        finally:
            os.unlink(f.name)

    def test_audit_lists_where_checking_is_off(self):
        cfg = {"proxmox": {"verify_tls": False, "pve_node": {"verify_tls": "off"}, "app_node": {"verify_tls": True}},
               "immich": {"url": "https://x"}, "services": [{"name": "a"}, {"name": "b", "verify_tls": False}], "other": {"verify_tls": "maybe"}}
        self.assertEqual(sorted(tls.audit(cfg)), ["proxmox", "proxmox.pve_node", "services[1]"])
        self.assertEqual(tls.audit({}), [])

    def test_explain_adds_the_fix_to_a_certificate_failure_only(self):
        err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        text = tls.explain(err, "Proxmox")
        self.assertIn("ca_file", text)
        self.assertIn("verify_tls", text)
        self.assertEqual(tls.explain(OSError("connection refused"), "Proxmox"), "connection refused")


class TestNoClientTurnsCheckingOffByItself(unittest.TestCase):
    """Source ratchet: certificate checking is switched off in one place only, lodestone/tls.py, behind `verify_tls: false`."""
    SHIPPED = [os.path.join(ROOT, "StoneSage", "backend"), os.path.join(ROOT, "lodestone"), os.path.join(ROOT, "harness"),
               os.path.join(ROOT, "server setup", "alexa-skill"), os.path.join(ROOT, "server setup", "voice-server")]
    PATTERN = re.compile(r"CERT_NONE|check_hostname\s*=\s*False|_create_unverified_context|verify\s*=\s*False")

    def test_only_tls_py_may_switch_verification_off(self):
        offenders = []
        for base in self.SHIPPED:
            for dirpath, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "node_modules", "out", "venv")]
                for f in files:
                    if not f.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, f)
                    if path.endswith(os.path.join("lodestone", "tls.py")):
                        continue
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        for n, line in enumerate(fh, 1):
                            if self.PATTERN.search(line) and not line.lstrip().startswith("#"):
                                offenders.append(f"{os.path.relpath(path, ROOT)}:{n}")
        self.assertEqual(offenders, [], "a client turns TLS checking off by itself: use lodestone.tls.context")

    def test_the_example_config_has_no_verify_off_switch(self):
        example = os.path.join(ROOT, "StoneSage", "backend", "config.example.json")
        if not os.path.exists(example):
            self.skipTest("no example config in this tree")
        with open(example, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("verify_ssl", text)
        self.assertNotRegex(text, r'"verify_tls"\s*:\s*false')


class TestClientsUseTheContext(unittest.TestCase):
    def test_proxmox_node_settings_win_over_the_section(self):
        client = ProxmoxClient({"verify_tls": False, "pve_node": {"name": "pve", "verify_tls": True}})
        self.assertEqual(client._ssl_ctx(client.node_configs["pve_node"]).verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(client._ssl_ctx({}).verify_mode, ssl.CERT_NONE)           # the section said so, knowingly
        self.assertEqual(ProxmoxClient({})._ssl_ctx({}).verify_mode, ssl.CERT_REQUIRED)

    @unittest.skipUnless(importlib.util.find_spec("hearth"), "the Home app is not installed")
    def test_immich_verifies_by_default(self):
        ImmichClient = importlib.import_module("hearth.immich_client").ImmichClient
        self.assertEqual(ImmichClient({"url": "https://192.0.2.9"}, tempfile.gettempdir())._ctx().verify_mode, ssl.CERT_REQUIRED)
        off = ImmichClient({"url": "https://192.0.2.9", "verify_tls": False}, tempfile.gettempdir())
        self.assertEqual(off._ctx().verify_mode, ssl.CERT_NONE)


class _Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": {"cpu": 0.5, "memory": {"used": 1, "total": 2}, "rootfs": {"used": 1, "total": 2}, "uptime": 5,
                                    "kversion": "x"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(OPENSSL, "the openssl program is needed to make a throwaway certificate")
class TestHandshake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="tls-test-")
        cls.cert, cls.key = os.path.join(cls.dir, "c.pem"), os.path.join(cls.dir, "k.pem")
        r = subprocess.run([OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", cls.key, "-out", cls.cert, "-days", "2",
                            "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost"],
                           capture_output=True, text=True)
        if r.returncode:
            raise unittest.SkipTest("openssl could not make a certificate: " + r.stderr[-200:])
        sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        sctx.load_cert_chain(cls.cert, cls.key)
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
        cls.httpd.socket = sctx.wrap_socket(cls.httpd.socket, server_side=True)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"https://127.0.0.1:{cls.port}/"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def fetch(self, *sections):
        with urllib.request.urlopen(self.url, context=tls.context(*sections, name="handshake"), timeout=5) as r:
            return r.status

    def test_a_self_signed_certificate_is_refused_by_default(self):
        with self.assertRaises(urllib.error.URLError) as cm:
            self.fetch({})
        self.assertIsInstance(cm.exception.reason, ssl.SSLCertVerificationError)

    def test_ca_file_trusts_that_certificate_and_still_verifies(self):
        self.assertEqual(self.fetch({"ca_file": self.cert}), 200)

    def test_a_wrong_ca_file_is_still_refused(self):
        other = os.path.join(self.dir, "other.pem")
        subprocess.run([OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", os.path.join(self.dir, "ok.pem"),
                        "-out", other, "-days", "2", "-subj", "/CN=elsewhere"], capture_output=True, check=True)
        with self.assertRaises(urllib.error.URLError):
            self.fetch({"ca_file": other})

    def test_verify_false_accepts_it(self):
        self.assertEqual(self.fetch({"verify_tls": False}), 200)

    def test_proxmox_client_default_fails_and_says_what_to_set(self):
        node = {"name": "pve", "host": "127.0.0.1", "port": self.port, "token_id": "u@pam!t", "token_secret": "s"}
        res = ProxmoxClient({"pve_node": node}).get_node_status("pve_node")
        self.assertIn("error", res)
        self.assertIn("ca_file", res["error"])

    def test_proxmox_client_with_ca_file_or_explicit_off_works(self):
        node = {"name": "pve", "host": "127.0.0.1", "port": self.port, "token_id": "u@pam!t", "token_secret": "s"}
        for extra in ({"ca_file": self.cert}, {"verify_tls": False}):
            with self.subTest(extra=extra):
                res = ProxmoxClient({"pve_node": dict(node, **extra)}).get_node_status("pve_node")
                self.assertNotIn("error", res)
                self.assertEqual(res["cpu"], 0.5)
        # the setting may also sit on the whole "proxmox" section
        self.assertNotIn("error", ProxmoxClient({"ca_file": self.cert, "pve_node": node}).get_node_status("pve_node"))

    @unittest.skipUnless(importlib.util.find_spec("hearth"), "the Home app is not installed")
    def test_immich_default_fails_and_ca_file_works(self):
        ImmichClient = importlib.import_module("hearth.immich_client").ImmichClient
        base = {"url": self.url.rstrip("/")}
        self.assertFalse(ImmichClient(base, self.dir).is_online())
        self.assertTrue(ImmichClient(dict(base, ca_file=self.cert), self.dir).is_online())
        self.assertTrue(ImmichClient(dict(base, verify_tls=False), self.dir).is_online())

    @unittest.skipUnless(importlib.util.find_spec("stonesage"), "the platform is not installed")
    def test_the_node_probe_follows_the_same_rule(self):
        server = importlib.import_module("stonesage.server")
        url = self.url.rstrip("/")
        bad = server.ping_harness_instance(url, timeout=3)
        self.assertFalse(bad["is_online"])
        self.assertIn("ca_file", bad["error"])
        self.assertTrue(server.ping_harness_instance(url, timeout=3, tls_section={"ca_file": self.cert})["is_online"])
        self.assertTrue(server.ping_harness_instance(url, timeout=3, tls_section={"verify_tls": False})["is_online"])


if __name__ == "__main__":
    unittest.main()
