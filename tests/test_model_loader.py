"""Offline tests for the Model Loader's unit editing and memory math (no SSH, no engines)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import model_loader as ml  # noqa: E402

UNIT = """[Unit]
Description=Llama-Server Qwen3-14B Coordinator + 0.6B Draft Speculative (RX 6750 XT - Vulkan)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/models
Environment="GGML_VK_VISIBLE_DEVICES=0"
ExecStart=/usr/local/bin/llama-server \\
    --model /opt/models/qwen3-14b-q4_k_m.gguf \\
    --spec-draft-model /opt/models/qwen3-0.6b-q8_0.gguf \\
    --spec-draft-n-max 12 \\
    --spec-draft-ngl 99 \\
    --host 0.0.0.0 \\
    --port 8001 \\
    --device Vulkan0 \\
    -ngl 99 \\
    -c 12288 \\
    -np 2 \\
    --flash-attn on \\
    -ctk q4_0 \\
    -ctv q4_0 \\
    --alias coordinator \\
    --metrics \\
    --slots \\
    --props

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


class TestUnitEditing(unittest.TestCase):
    def test_round_trip_is_byte_identical(self):
        self.assertEqual(ml.render_unit(ml.parse_unit(UNIT)), UNIT)

    def test_parse_reads_flags_and_env(self):
        u = ml.parse_unit(UNIT)
        self.assertEqual(ml.flag_get(u, "--model"), "/opt/models/qwen3-14b-q4_k_m.gguf")
        self.assertEqual(ml.flag_get(u, "-c"), "12288")
        self.assertEqual(ml.flag_get(u, "--spec-draft-model"), "/opt/models/qwen3-0.6b-q8_0.gguf")
        self.assertEqual(u["env"]["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertIn(["--metrics", None], u["args"])

    def test_edit_changes_only_the_touched_lines(self):
        u = ml.parse_unit(UNIT)
        ml.flag_set(u, "-c", "16384")
        out = ml.render_unit(u)
        changed = [(a, b) for a, b in zip(UNIT.splitlines(), out.splitlines()) if a != b]
        self.assertEqual(changed, [("    -c 12288 \\", "    -c 16384 \\")])

    def test_new_flag_goes_before_alias_and_removal_works(self):
        u = ml.parse_unit(UNIT)
        ml.flag_set(u, "--temp", "0.6")
        for f in ("--spec-draft-model", "--spec-draft-n-max", "--spec-draft-ngl"):
            ml.flag_set(u, f, None, present=False)
        out = ml.render_unit(u)
        self.assertNotIn("spec-draft", out)
        self.assertLess(out.index("--temp 0.6"), out.index("--alias coordinator"))
        self.assertTrue(out.rstrip().splitlines()[-1] != "\\")  # no dangling continuation
        self.assertEqual(ml.render_unit(ml.parse_unit(out)), out)

    def test_description_and_gpu_env(self):
        out = ml.render_unit(ml.parse_unit(UNIT), description="llama-server coordinator: X on GPU", env={"GGML_VK_VISIBLE_DEVICES": "1"})
        self.assertIn("Description=llama-server coordinator: X on GPU\n", out)
        self.assertIn('Environment="GGML_VK_VISIBLE_DEVICES=1"\n', out)
        self.assertNotIn('GGML_VK_VISIBLE_DEVICES=0', out)

    def test_paths_with_spaces_are_quoted(self):
        u = ml.parse_unit(UNIT)
        ml.flag_set(u, "--model", "/home/alex/.lmstudio/models/My Model/x.gguf")
        out = ml.render_unit(u)
        self.assertIn("--model '/home/alex/.lmstudio/models/My Model/x.gguf'", out)
        self.assertEqual(ml.flag_get(ml.parse_unit(out), "--model"), "/home/alex/.lmstudio/models/My Model/x.gguf")


class TestMemoryMath(unittest.TestCase):
    def test_kv_from_gguf_geometry(self):
        # Qwen3 14B: 40 layers, 8 KV heads, 5120 embedding / 40 heads = 128 head dim; 12288 tokens of q4_0
        m = {"layers": 40, "kv_heads": 8, "heads": 40, "embedding": 5120}
        gb, how = ml._kv_gb(m, 12288, "q4_0")
        self.assertEqual(how, "gguf")
        self.assertAlmostEqual(gb, 2 * 40 * 8 * 128 * 12288 * (18 / 32) / 1024 ** 3, places=6)
        self.assertAlmostEqual(ml._kv_gb(m, 12288, "f16")[0] / gb, 2 / (18 / 32), places=6)

    def test_kv_without_geometry_is_approximate(self):
        gb, how = ml._kv_gb({"params": "14B", "size_gb": 8}, 8192, "f16")
        self.assertEqual(how, "approx")
        self.assertTrue(1.0 < gb < 2.0)  # ~0.16 MB/token for a 14B GQA model

    def test_param_parsing(self):
        self.assertEqual(ml._num("14B"), 14.0)
        self.assertEqual(ml._num("35B-A3B"), 35.0)
        self.assertAlmostEqual(ml._num("600M"), 0.6)
        self.assertIsNone(ml._num(None))


if __name__ == "__main__":
    unittest.main()
