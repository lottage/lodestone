"""Offline tests for the Phase 6 idle gate (backend/idle_gate.py): per GPU from Prometheus, engines per GPU from the
live profile, and the shared conditions (approvals, a recent Computer turn, the household)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import idle_gate as ig  # noqa: E402

PROFILE = {"gpus": [{"pci": "0000:01:00.0", "short": "RX 6750 XT", "engines": ["coordinator", "worker"]},
                    {"pci": "0000:02:00.0", "short": "RX 6600", "engines": ["embedder", "vision"]}]}
NOW = 1_800_000_000.0


class Rig:
    def __init__(self):
        self.busy = {"0000:01:00.0": 3.0, "0000:02:00.0": 4.0}
        self.idle = {"coordinator": 1, "worker": 1, "embedder": 1, "vision": 1}
        self.house = {"quiet": True, "reasons": []}
        self.pending, self.last_turn, self.prom_down = 0, None, False
        self.gate = ig.IdleGate(lambda: {}, lambda: PROFILE, self.prom, lambda: self.house, lambda: self.pending,
                                lambda: self.last_turn, clock=lambda: NOW)

    def prom(self, expr):
        if self.prom_down:
            raise OSError("connection refused")
        if expr.startswith("stonesage:gpu_busy"):
            return [{"metric": {"pci": k}, "value": [NOW, str(v)]} for k, v in self.busy.items()]
        return [{"metric": {"engine": k}, "value": [NOW, str(v)]} for k, v in self.idle.items()]

    def check(self):
        return self.gate.check(fresh=True)


class TestIdleGate(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def test_open_when_everything_is_quiet(self):
        c = self.r.check()
        self.assertTrue(c["open"])
        self.assertTrue(c["gpus"]["RX 6750 XT"]["open"] and c["gpus"]["RX 6600"]["open"])

    def test_busy_engine_closes_only_its_gpu(self):
        self.r.idle["vision"] = 0                   # the sentry just used the VLM
        c = self.r.check()
        self.assertFalse(c["gpus"]["RX 6600"]["open"])
        self.assertIn("vision served a request in the last 5 min", c["gpus"]["RX 6600"]["reasons"])
        self.assertTrue(c["gpus"]["RX 6750 XT"]["open"])
        self.assertTrue(c["open"])

    def test_busy_gpu_and_unscraped_engine(self):
        self.r.busy["0000:01:00.0"] = 40
        del self.r.idle["embedder"]
        c = self.r.check()
        self.assertIn("GPU 40 % busy", c["gpus"]["RX 6750 XT"]["reasons"])
        self.assertIn("embedder: no metrics", c["gpus"]["RX 6600"]["reasons"])
        self.assertFalse(c["open"])

    def test_shared_conditions_close_every_gpu(self):
        for setup, why in ((lambda: setattr(self.r, "pending", 1), "1 approval(s) waiting"),
                           (lambda: setattr(self.r, "last_turn", NOW - 300), "Computer answered 5 min ago"),
                           (lambda: setattr(self.r, "house", {"quiet": False, "reasons": ["Alex is awake (using their phone)"]}),
                            "Alex is awake (using their phone)")):
            self.r = Rig()
            setup()
            c = self.r.check()
            self.assertFalse(c["open"], why)
            self.assertIn(why, c["shared_reasons"])

    def test_prometheus_down_is_closed_not_open(self):
        self.r.prom_down = True
        c = self.r.check()
        self.assertFalse(c["open"])
        self.assertTrue(any("Prometheus" in r for r in c["shared_reasons"]))

    def test_metrics_lines(self):
        self.r.idle["vision"] = 0
        text = "\n".join(self.r.gate.metrics())
        self.assertIn('stonesage_idle_gate_open{gpu="RX 6600"} 0', text)
        self.assertIn('stonesage_idle_gate_open{gpu="RX 6750 XT"} 1', text)
        self.assertIn("stonesage_household_quiet 1", text)



class TestIdleGateCockpit(unittest.TestCase):
    def rig(self):
        r = Rig()
        self.act = {"reason": None}
        r.gate.activity_fn = lambda: self.act["reason"]
        return r

    def test_it_is_a_shared_reason_that_closes_every_gpu(self):
        r = self.rig()
        self.assertTrue(r.check()["open"])
        self.act["reason"] = "someone is using StoneSage (4 s ago)"
        c = r.check()
        self.assertFalse(c["open"])
        self.assertIn("someone is using StoneSage (4 s ago)", c["shared_reasons"])
        self.assertEqual(c["cockpit"], "someone is using StoneSage (4 s ago)")
        for g in c["gpus"].values():
            self.assertFalse(g["open"])
            self.assertTrue(g["open_ignoring_cockpit"])

    def test_a_broken_signal_does_not_close_the_gate(self):
        r = self.rig()
        r.gate.activity_fn = lambda: 1 / 0
        self.assertTrue(r.check()["open"])


if __name__ == "__main__":
    unittest.main()
