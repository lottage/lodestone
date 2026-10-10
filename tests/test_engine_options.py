"""Offline tests: llama-server --help parsing, model/hardware refinement, and option edits on a unit."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import engine_options as eo  # noqa: E402
from lodestone import model_loader as ml  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_model_loader import UNIT  # noqa: E402

HELP = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures_llama_server_help.txt"), encoding="utf-8").read()


class TestHelpParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opts = eo.parse_llama_help(HELP)
        cls.by = {o["flag"]: o for o in cls.opts}

    def test_covers_the_build(self):
        self.assertGreater(len(self.opts), 200)
        self.assertEqual({o["section"] for o in self.opts}, {"common", "sampling", "speculative", "example-specific"})

    def test_kinds(self):
        self.assertEqual(self.by["--flash-attn"]["choices"], ["on", "off", "auto"])
        self.assertEqual(self.by["--split-mode"]["choices"], ["none", "layer", "row", "tensor"])
        self.assertEqual((self.by["--poll"]["min"], self.by["--poll"]["max"]), (0, 100))
        self.assertEqual(self.by["--kv-offload"]["kind"], "tristate")
        self.assertEqual(self.by["--kv-offload"]["off_flag"], "--no-kv-offload")
        self.assertEqual(self.by["--swa-full"]["kind"], "switch")
        self.assertEqual(self.by["--temperature"]["kind"], "float")
        self.assertEqual(self.by["--seed"]["kind"], "int")
        self.assertIn("q4_0", self.by["--cache-type-k"]["choices"])
        self.assertEqual(self.by["--reasoning-format"]["kind"], "enum")
        self.assertEqual(self.by["--prio"]["choices"], ["-1", "0", "1", "2", "3"])

    def test_removed_hidden_and_managed(self):
        flags = {n for o in self.opts for n in o["names"]}
        self.assertNotIn("--draft-max", flags)      # "has been removed"
        self.assertNotIn("--mlock", flags)          # DEPRECATED
        self.assertNotIn("--port", flags)           # set by the loader, never by hand
        self.assertTrue(self.by["--ctx-size"]["managed"])
        self.assertTrue(self.by["--metrics"]["required_on"])

    def test_refine_to_model_and_hardware(self):
        dense = {"layers": 40, "trained_ctx": 40960, "arch": "qwen3", "params": "14B"}
        moe = {"layers": 41, "trained_ctx": 262144, "arch": "qwen35moe", "params": "35B-A3B", "experts": 256}
        r = {o["flag"]: o for o in eo.refine(self.opts, dense, {"cpu_threads": 10, "gpu_count": 1}, "llama", "coordinator")}
        self.assertEqual((r["--threads"]["min"], r["--threads"]["max"]), (-1, 10))
        self.assertTrue(r["--n-cpu-moe"]["hidden"])
        self.assertTrue(r["--split-mode"]["hidden"])
        self.assertTrue(r["--mmproj-offload"]["hidden"])
        r = {o["flag"]: o for o in eo.refine(self.opts, moe, {"cpu_threads": 10, "gpu_count": 2}, "llama", "vision")}
        self.assertFalse(r["--n-cpu-moe"].get("hidden"))
        self.assertEqual(r["--n-cpu-moe"]["max"], 41)
        self.assertFalse(r["--split-mode"].get("hidden"))
        self.assertFalse(r["--mmproj-offload"].get("hidden"))


class TestOptionEdits(unittest.TestCase):
    def setUp(self):
        self.opts = eo.parse_llama_help(HELP)
        self.by = {o["flag"]: o for o in self.opts}
        self.vf = ml._value_flags(self.opts)

    def edit(self, **changes):
        u = ml.parse_unit(UNIT, self.vf)
        for flag, val in changes.items():
            ml.set_option(u, self.by[flag], val)
        return ml.render_unit(u)

    def test_current_values(self):
        cur = ml.current_options(ml.parse_unit(UNIT, self.vf), self.opts)
        self.assertEqual(cur["--ctx-size"], "12288")
        self.assertTrue(cur["--metrics"])

    def test_tristate_on_off_default(self):
        self.assertIn("--no-kv-offload", self.edit(**{"--kv-offload": False}))
        self.assertIn("    --kv-offload", self.edit(**{"--kv-offload": True}))
        out = self.edit(**{"--kv-offload": False})
        u = ml.parse_unit(out, self.vf)
        ml.set_option(u, self.by["--kv-offload"], None)
        self.assertNotIn("kv-offload", ml.render_unit(u))

    def test_keeps_existing_spelling_and_position(self):
        out = self.edit(**{"--ubatch-size": "512", "--cache-type-k": "q8_0"})
        self.assertIn("    -ctk q8_0 \\", out)          # -ctk stays -ctk, same line
        self.assertLess(out.index("-ctk q8_0"), out.index("-ctv q4_0"))

    def test_required_flags_cannot_be_dropped(self):
        out = self.edit(**{"--metrics": None, "--props": False})
        self.assertIn("--metrics", out)
        self.assertIn("--props", out)


if __name__ == "__main__":
    unittest.main()
