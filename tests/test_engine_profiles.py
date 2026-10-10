"""Offline tests for Engine Profiles: gating, library-status checks, and the keep-going-on-failure
orchestration in _run_profile. model_loader is stubbed out entirely, so this never touches the LAN."""

import os
import sys
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import engine_profiles as ep  # noqa: E402


class TestGate(unittest.TestCase):
    def test_no_gate_is_always_met(self):
        self.assertTrue(ep._gate_met({}, {"project_phase": 0}))

    def test_gate_blocks_below_min_phase(self):
        self.assertFalse(ep._gate_met({"min_phase": 4}, {"project_phase": 2}))

    def test_gate_allows_at_or_above_min_phase(self):
        self.assertTrue(ep._gate_met({"min_phase": 4}, {"project_phase": 4}))
        self.assertTrue(ep._gate_met({"min_phase": 2}, {"project_phase": 4}))


class TestListProfiles(unittest.TestCase):
    def test_status_reflects_live_library(self):
        cfg = {
            "project_phase": 2,
            "engine_profiles": {
                "p1": {"targets": {"engine:coordinator": {"model": "a.gguf"},
                                    "engine:worker": {"model": "missing.gguf"},
                                    "lmstudio:ally": {}}},
            },
        }
        with mock.patch.object(ep, "_library_keys", return_value={"a.gguf"}), \
             mock.patch.object(ep.model_loader, "get_state", return_value={"nodes": []}):
            out = ep.list_profiles(cfg)
        t = out["profiles"][0]["targets"]
        self.assertEqual(t["engine:coordinator"]["status"], "available")
        self.assertEqual(t["engine:worker"]["status"], "model_missing")
        self.assertEqual(t["lmstudio:ally"]["status"], "unspecified")

    def test_running_model_is_reported_active(self):
        cfg = {"engine_profiles": {"p1": {"targets": {"engine:coordinator": {"model": "/opt/models/a.gguf"}}}}}
        state = {"nodes": [{"targets": [{"id": "engine:coordinator", "model_file": "a.gguf"}]}]}
        with mock.patch.object(ep, "_library_keys", return_value=set()), \
             mock.patch.object(ep.model_loader, "get_state", return_value=state):
            out = ep.list_profiles(cfg)
        self.assertEqual(out["profiles"][0]["targets"]["engine:coordinator"]["status"], "active")


class TestEngineRequest(unittest.TestCase):
    LIVE = {"id": "engine:coordinator", "model_file": "a.gguf", "ctx_per_slot": 6144, "slots": 2}

    def test_same_model_and_nothing_else_is_skipped(self):
        self.assertIsNone(ep._engine_request("engine:coordinator", {"model": "/opt/models/a.gguf"}, self.LIVE))

    def test_model_swap_keeps_live_sizing(self):
        req = ep._engine_request("engine:coordinator", {"model": "/opt/models/b.gguf"}, self.LIVE)
        self.assertEqual((req["ctx_per_slot"], req["slots"]), (6144, 2))
        self.assertEqual(req["model"], "/opt/models/b.gguf")

    def test_profile_sizing_wins_over_live(self):
        want = {"model": "/opt/models/a.gguf", "ctx_per_slot": 8192}
        req = ep._engine_request("engine:coordinator", want, self.LIVE)
        self.assertEqual((req["ctx_per_slot"], req["slots"]), (8192, 2))


class TestRunProfileKeepsGoing(unittest.TestCase):
    """The design call was: a failing target does not stop the rest of the profile."""

    def test_all_targets_are_attempted_and_status_is_partial(self):
        targets = {"engine:coordinator": {"model": "a.gguf"}, "engine:worker": {"model": "b.gguf"},
                   "engine:vision": {"model": "c.gguf"}}

        def fake_submit(target_id, want):
            if target_id == "engine:worker":
                return {"ok": False, "error": "model not found"}  # fails immediately, no job
            return {"ok": True, "job": f"job-{target_id}"}

        def fake_wait(job_id, timeout=300):
            return {"status": "done"} if job_id == "job-engine:coordinator" else {"status": "failed"}

        with mock.patch.object(ep, "_submit_target", side_effect=fake_submit), \
             mock.patch.object(ep, "_wait_target_job", side_effect=fake_wait):
            ep._run_profile("job1", "p1", targets)

        job = ep.get_profile_job("job1")
        self.assertEqual(set(job["results"].keys()), set(targets.keys()))  # every target was attempted
        self.assertEqual(job["results"]["engine:coordinator"]["status"], "done")
        self.assertEqual(job["results"]["engine:worker"]["error"], "model not found")
        self.assertEqual(job["results"]["engine:vision"]["status"], "failed")
        self.assertEqual(job["status"], "partial")

    def test_all_success_reports_done(self):
        targets = {"engine:coordinator": {"model": "a.gguf"}}
        with mock.patch.object(ep, "_submit_target", return_value={"ok": True, "skipped": "already set"}):
            ep._run_profile("job2", "p1", targets)
        self.assertEqual(ep.get_profile_job("job2")["status"], "done")


if __name__ == "__main__":
    unittest.main()
