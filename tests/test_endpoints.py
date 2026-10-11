"""lodestone/endpoints.py: every service address from config, env, or a localhost default (docs/LAYERS.md M2)."""

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import endpoints  # noqa: E402

CFG = {"cluster": {"coordinator_url": "http://10.0.0.5:8001/v1/", "valkey_host": "10.0.0.5", "ssh_user": "ops"},
       "qdrant": {"url": "http://10.0.0.9:6333"}, "voice": {"host": "10.0.0.7"},
       "proxmox": {"host": "10.0.0.2"}}


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k not in {s[1] for s in endpoints.SERVICES.values()}}
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_config_wins_and_is_trimmed(self):
        self.assertEqual(endpoints.url("coordinator", CFG), "http://10.0.0.5:8001/v1")
        self.assertEqual(endpoints.host("coordinator", CFG), "10.0.0.5")

    def test_second_path_is_read(self):
        self.assertEqual(endpoints.url("qdrant", CFG), "http://10.0.0.9:6333")        # qdrant.url, no cluster.qdrant_url
        both = {"cluster": {"qdrant_url": "http://a:1"}, "qdrant": {"url": "http://b:2"}}
        self.assertEqual(endpoints.url("qdrant", both), "http://a:1")                 # first path first

    def test_bare_host_borrows_scheme_and_port(self):
        self.assertEqual(endpoints.url("valkey", CFG), "redis://10.0.0.5:6379")
        self.assertEqual((endpoints.host("valkey", CFG), endpoints.port("valkey", CFG)), ("10.0.0.5", 6379))
        self.assertEqual(endpoints.url("voice_legacy", CFG), "http://10.0.0.7")
        self.assertEqual(endpoints.url("proxmox", CFG), "https://10.0.0.2:8006")

    def test_env_then_default(self):
        self.assertEqual(endpoints.url("worker", {}), "http://127.0.0.1:8002/v1")
        with mock.patch.dict(os.environ, {"WORKER_URL": "http://w:9/v1"}):
            self.assertEqual(endpoints.url("worker", {}), "http://w:9/v1")
            self.assertEqual(endpoints.url("worker", {"cluster": {"worker_url": "http://c:1/v1"}}), "http://c:1/v1")

    def test_optional_service_is_empty(self):
        self.assertEqual(endpoints.url("immich", {}), "")
        self.assertEqual(endpoints.host("immich", {}), "")

    def test_missing_lists_defaults_only(self):
        gone = endpoints.missing(CFG)
        self.assertIn("embedder", gone)                                               # core engine left on its default
        self.assertNotIn("coordinator", gone)
        self.assertNotIn("immich", gone)                                              # optional: no default to fall to

    def test_optional_services_are_not_warned_about_unless_used(self):
        gone = endpoints.missing({"cluster": {"coordinator_url": "http://c:1/v1"}})
        self.assertEqual(sorted(gone), ["embedder", "homeassistant"])                 # Hearth (the default app) needs HA
        for name in ("worker", "vision", "mcp", "qdrant", "valkey", "couchdb", "prometheus", "frontier", "voice",
                     "gemini_web", "assembly", "stonesage"):
            self.assertIn(name, endpoints.OPTIONAL)
            self.assertNotIn(name, gone)
        quiet = endpoints.missing({"apps": [], "cluster": {"coordinator_url": "http://c:1/v1"}})
        self.assertEqual(quiet, ["embedder"])                                         # no Hearth, no HA warning
        info = endpoints.missing({"apps": [], "cluster": {}}, optional=True)
        self.assertIn("worker", info)
        self.assertIn("voice", info)
        self.assertNotIn("coordinator", info)
        self.assertNotIn("immich", info)

    def test_an_optional_service_with_its_own_config_section_is_needed(self):
        cfg = {"apps": [], "voice": {"stt_engine": "parakeet"}, "couchdb": {"username": "u"}}
        gone = endpoints.missing(cfg)
        self.assertIn("voice", gone)                                                  # configured around, URL forgotten
        self.assertIn("couchdb", gone)
        self.assertNotIn("voice", endpoints.missing(cfg, optional=True))              # never in both lists
        self.assertNotIn("voice", endpoints.missing(dict(cfg, voice={"stt_url": "http://v:1"})))

    def test_server_and_boost_sections_do_not_make_stonesage_needed(self):
        cfg = {"apps": [], "server": {"port": 8888}, "boost": {"enabled": False}}
        self.assertNotIn("stonesage", endpoints.missing(cfg))

    def test_coordinator_on_cpu(self):
        self.assertFalse(endpoints.coordinator_on_cpu({}))
        self.assertTrue(endpoints.coordinator_on_cpu({"cluster": {"coordinator_cpu": True}}))
        with mock.patch.dict(os.environ, {"COORDINATOR_CPU": "yes"}):
            self.assertTrue(endpoints.coordinator_on_cpu({}))
            self.assertFalse(endpoints.coordinator_on_cpu({"cluster": {"coordinator_cpu": False}}))

    def test_no_lan_address_in_defaults(self):
        for name, (_, _, default) in endpoints.SERVICES.items():
            self.assertNotRegex(default, r"192\.168\.|10\.\d+\.|172\.(1[6-9]|2\d|3[01])\.", name)

    def test_ssh(self):
        self.assertEqual(endpoints.ssh_target(CFG), "ops@10.0.0.5")
        cmd = endpoints.ssh_cmd(CFG, "uptime", connect_timeout=5)
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("StrictHostKeyChecking=accept-new", cmd)
        self.assertIn("ConnectTimeout=5", cmd)
        self.assertEqual(cmd[-3:], ["--", "ops@10.0.0.5", "uptime"])
        strict = dict(CFG, ssh={"host_key_checking": "yes"})
        self.assertIn("StrictHostKeyChecking=yes", endpoints.ssh_cmd(strict, "x", target="root@h", no_stdin=True))
        self.assertEqual(endpoints.ssh_cmd(strict, "x", target="root@h", no_stdin=True)[1], "-n")
        # "no" (trust any host) is not offered, even if a config says it
        lab = dict(CFG, ssh={"host_key_checking": "no"})
        self.assertIn("StrictHostKeyChecking=accept-new", endpoints.ssh_cmd(lab, "x", target="root@h"))
        self.assertNotIn("StrictHostKeyChecking=no", endpoints.ssh_cmd(lab, "x", target="root@h"))

    def test_ssh_target_cannot_carry_an_option(self):
        """A user or host that starts with '-' would be read by ssh as an option (-oProxyCommand=...): refused, and `--`
        ends the options before the target anyway."""
        self.assertEqual(endpoints.ssh_cmd(CFG, "uptime")[-3:], ["--", "ops@10.0.0.5", "uptime"])
        for bad in ("-oProxyCommand=x@h", "root@-oProxyCommand=x", "a b@h", "root@h;id"):
            with self.assertRaises(ValueError, msg=bad):
                endpoints.ssh_cmd(CFG, "x", target=bad)
        with self.assertRaises(ValueError):
            endpoints.ssh_target(dict(CFG, cluster=dict(CFG.get("cluster", {}), ssh_user="-oProxyCommand=x")))


if __name__ == "__main__":
    unittest.main()
