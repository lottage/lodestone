"""Offline tests for model provenance (Phase 6): the VM-side scanner on tiny hand-written GGUF files and fake build
trees, and StoneSage's Provenance (loader block, CouchDB upsert keeping human fields)."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
def provenance_module_file() -> str:
    from lodestone import provenance
    return provenance.__file__


SCANNER = os.path.join(os.path.dirname(provenance_module_file()), "provenance_scan.py")   # lodestone/, however installed

from lodestone import provenance as pv  # noqa: E402


def _s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensor_types):
    """A GGUF v3 header: string/uint32 key-values and one tensor info per type (no tensor data: never read)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensor_types), len(kv))
    for k, v in kv.items():
        out += _s(k) + (struct.pack("<I", 8) + _s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    for i, t in enumerate(tensor_types):
        out += _s(f"blk.{i}.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 16, 16) + struct.pack("<IQ", t, 0)
    with open(path, "wb") as f:
        f.write(out)


def write_build(root, types, archs):
    os.makedirs(os.path.join(root, "ggml", "include"))
    os.makedirs(os.path.join(root, "src"))
    enum = "\n".join(f"        GGML_TYPE_{n} = {i}," for i, n in types.items())
    with open(os.path.join(root, "ggml", "include", "ggml.h"), "w") as f:
        f.write("enum ggml_type {\n" + enum + "\n        // GGML_TYPE_Q4_2 = 4, support has been removed\n"
                "        GGML_TYPE_COUNT = 999,\n    };\n")
    with open(os.path.join(root, "src", "llama-arch.cpp"), "w") as f:
        f.write("static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n"
                + "\n".join(f'    {{ LLM_ARCH_{a.upper()}, "{a}" }},' for a in archs) + "\n};\n")


MAIN = {0: "F32", 12: "Q4_K", 14: "Q6_K"}
PRISM = {**MAIN, 142: "PQ2_0", 143: "PTQ1_0"}


class TestScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        t = cls.tmp
        write_build(os.path.join(t, "main"), MAIN, ["qwen3", "llama"])
        write_build(os.path.join(t, "prism"), PRISM, ["qwen3"])
        os.makedirs(os.path.join(t, "models"))
        write_gguf(os.path.join(t, "models", "qwen.gguf"),
                   {"general.architecture": "qwen3", "general.name": "Qwen3 14B", "general.base_model.count": 1,
                    "general.base_model.0.name": "Qwen3 14B Base", "general.base_model.0.repo_url": "https://hf.co/Qwen/Qwen3-14B-Base",
                    "general.quantized_by": "Unsloth", "qwen3.context_length": 40960}, [0, 12, 14])
        write_gguf(os.path.join(t, "models", "bonsai.gguf"), {"general.architecture": "qwen3"}, [0, 143])
        write_gguf(os.path.join(t, "models", "old-llama.gguf"), {"general.architecture": "llama"}, [0, 12])
        with open(os.path.join(t, "models", "broken.gguf"), "wb") as f:
            f.write(b"nope")
        try:
            os.symlink(os.path.join(t, "models", "qwen.gguf"), os.path.join(t, "models", "alias.gguf"))
            cls.symlinks = True
        except (OSError, NotImplementedError):
            cls.symlinks = False                        # Windows without the privilege
        cls.job = {"builds": [{"id": "mainline", "binary": sys.executable, "source": os.path.join(t, "main")},
                              {"id": "prism", "binary": sys.executable, "source": os.path.join(t, "prism")}],
                   "globs": [os.path.join(t, "models", "*.gguf")], "hash": True}
        env = dict(os.environ, HOME=t, USERPROFILE=t)
        res = subprocess.run([sys.executable, SCANNER], input=json.dumps(cls.job), capture_output=True, text=True, env=env)
        assert res.returncode == 0, res.stderr
        cls.out = json.loads(res.stdout)
        cls.by_file = {m["file"]: m for m in cls.out["models"]}

    def test_builds_parse_types_and_archs(self):
        b = {x["id"]: x for x in self.out["builds"]}
        self.assertEqual(b["prism"]["types"]["143"], "PTQ1_0")
        self.assertNotIn("4", b["mainline"]["types"])          # commented-out, removed type
        self.assertEqual(b["mainline"]["archs"], ["llama", "qwen3"])

    def test_loadable_by_types_and_architecture(self):
        self.assertEqual(self.by_file["qwen.gguf"]["loadable_by"], ["mainline", "prism"])
        self.assertEqual(self.by_file["bonsai.gguf"]["loadable_by"], ["prism"])
        self.assertEqual(self.by_file["bonsai.gguf"]["tensor_types"], {"F32": 1, "PTQ1_0": 1})
        self.assertEqual(self.by_file["old-llama.gguf"]["loadable_by"], ["mainline"])   # prism lacks the arch
        self.assertIn("error", self.by_file["broken.gguf"])

    def test_lineage_and_hash(self):
        q = self.by_file["qwen.gguf"]
        self.assertEqual(len(q["sha256"]), 64)
        lin = pv.lineage(q["gguf"])
        self.assertEqual(lin["base_models"], [{"name": "Qwen3 14B Base", "repo_url": "https://hf.co/Qwen/Qwen3-14B-Base"}])
        self.assertEqual(lin["quantized_by"], "Unsloth")

    def test_symlink_is_one_model_with_two_paths(self):
        if not self.symlinks:
            self.skipTest("no symlinks here")
        q = self.by_file["qwen.gguf"]
        self.assertTrue(any(p.endswith("alias.gguf") for p in q["paths"]))
        self.assertNotIn("alias.gguf", self.by_file)


UNIT_MAIN = "[Service]\nExecStart=/usr/local/bin/llama-server --model /m/qwen.gguf --port 8001\n"
UNIT_PRISM = "[Service]\nExecStart=/opt/llama-prism/current/llama-server --model /m/qwen.gguf\n"
CFG = {"provenance": {"builds": [{"id": "mainline", "binary": "/usr/local/bin/llama-server"},
                                 {"id": "prism", "binary": "/opt/llama-prism/current/llama-server"}], "globs": ["/m/*.gguf"]}}
SCAN = {"builds": [{"id": "mainline", "types": {}, "archs": []}, {"id": "prism", "types": {}, "archs": []}], "models": [
    {"real_path": "/m/qwen.gguf", "paths": ["/m/qwen.gguf"], "file": "qwen.gguf", "sha256": "a" * 64,
     "gguf": {"general.architecture": "qwen3"}, "tensor_types": {"Q4_K": 1}, "loadable_by": ["mainline", "prism"]},
    {"real_path": "/m/bonsai.gguf", "paths": ["/m/bonsai.gguf", "/opt/models/bonsai-link.gguf"], "file": "bonsai.gguf",
     "sha256": "b" * 64, "gguf": {"general.architecture": "qwen35"}, "tensor_types": {"F32": 1, "PTQ1_0": 1},
     "loadable_by": ["prism"]}]}


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.tmp = tempfile.mkdtemp()

        def run(cmd, stdin, timeout):
            job = json.loads(stdin)
            self.calls.append(job)
            models = [m for m in SCAN["models"] if not job.get("paths") or m["real_path"] in job["paths"]]
            if job.get("paths") and not models:
                models = [{"real_path": job["paths"][0], "paths": job["paths"], "file": "x.gguf", "error": "OSError: gone"}]
            return SimpleNamespace(returncode=0, stdout=json.dumps({"builds": SCAN["builds"], "models": models}), stderr="")

        self.p = pv.Provenance(lambda: CFG, run, None, os.path.join(self.tmp, "prov.json"))

    def test_block_a_fork_model_on_the_mainline_build(self):
        self.p.scan()
        why = self.p.check("/opt/models/bonsai-link.gguf", UNIT_MAIN)       # found by its symlink path too
        self.assertIn("cannot run on the mainline build", why)
        self.assertIn("PTQ1_0", why)
        self.assertIn("needs: prism", why)
        self.assertIsNone(self.p.check("/m/bonsai.gguf", UNIT_PRISM))
        self.assertIsNone(self.p.check("/m/qwen.gguf", UNIT_MAIN))

    def test_unscanned_model_is_scanned_on_the_spot_and_unreadable_blocks(self):
        self.assertIsNone(self.p.check("/m/qwen.gguf", UNIT_MAIN))
        self.assertEqual(self.calls[-1]["paths"], ["/m/qwen.gguf"])
        self.assertIn("could not read the model header", self.p.check("/m/gone.gguf", UNIT_MAIN))

    def test_unknown_build_blocks(self):
        self.assertIn("unknown llama-server build", self.p.check("/m/qwen.gguf", "ExecStart=/tmp/llama-server -m x\n"))

    def test_publish_keeps_human_fields(self):
        self.p.scan()
        store = {"gguf:" + "a" * 64: {"_id": "gguf:" + "a" * 64, "_rev": "1-x", "first_seen": "2026-09-01",
                                      "notes": "Courage's brain", "recommended_quant": "Q4_K_M", "file": "old"}}
        written = []

        def couch(method, path, body=None):
            if method == "PUT":
                return {"ok": True}
            if path.endswith("_all_docs?include_docs=true"):
                return {"rows": [{"id": k, "doc": store[k]} for k in body["keys"] if k in store]}
            written.extend(body["docs"])
            return [{"ok": True} for _ in body["docs"]]

        self.p._couch = couch
        self.assertEqual(self.p.publish(), {"written": 4, "failed": 0})      # 2 models + 2 builds
        q = next(d for d in written if d["_id"] == "gguf:" + "a" * 64)
        self.assertEqual((q["_rev"], q["first_seen"], q["notes"], q["recommended_quant"], q["file"]),
                         ("1-x", "2026-09-01", "Courage's brain", "Q4_K_M", "qwen.gguf"))


if __name__ == "__main__":
    unittest.main()
