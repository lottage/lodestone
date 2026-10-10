#!/usr/bin/env python3
"""
Model provenance scan (Phase 6, 2026-09-27). Runs on the inference host (VM 102) over SSH; StoneSage's
backend/provenance.py sends the job as JSON on stdin and stores the result (CouchDB model_provenance + a local copy).

stdin:  {"builds": [{"id": "mainline", "binary": "/usr/local/bin/llama-server", "source": "/opt/llama.cpp-rocm"}, ...],
         "globs": ["/opt/models/**/*.gguf", ...], "paths": [optional: only these files], "hash": true}
stdout: {"builds": [{id, binary, version, types: {id: name}}], "models": [{sha256, real_path, paths, size_bytes,
         gguf: {...lineage...}, tensor_types: {name: count}, unknown_types: [ids], loadable_by: [build ids]}]}

Which build can load a model is decided by tensor types and architecture, not names: each build's ggml.h lists the GGML types it knows
(2026-09-27: the Prism fork adds PQ2_0 = 142 and PTQ1_0 = 143; every shared id means the same type), and a GGUF's
tensor infos say which types it uses; src/llama-arch.cpp lists the architectures. A model is loadable by a build that
knows its architecture and all its types.
sha256 of the whole file (30-60 s for 9 GB, once): cached by real path + size + mtime in ~/.cache/stonesage/.
Symlinks (/opt/models -> ~/.lmstudio/models) are one model with several paths.
"""

import glob
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

CACHE = os.path.expanduser("~/.cache/stonesage/gguf_sha256.json")
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
LINEAGE = re.compile(r"^general\.(name|basename|finetune|size_label|organization|quantized_by|license|url|repo_url|"
                     r"source\.url|source\.repo_url|file_type|quantization_version|architecture|"
                     r"base_model\.count|base_model\.\d+\.(name|organization|repo_url|version))$")


def read_gguf(path):
    """GGUF header: the lineage keys, the trained context, and how many tensors of each GGML type id."""
    kv, types = {}, {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            raise ValueError(f"GGUF v{version} not supported")

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t in SCALAR:
                fmt = SCALAR[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et in SCALAR:
                    f.seek(struct.calcsize(SCALAR[et]) * n, 1)
                else:
                    for _ in range(n):
                        rval(et)
                return None
            raise ValueError(f"unknown GGUF value type {t}")

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            val = rval(t)
            if LINEAGE.match(key) or key.endswith(".context_length"):
                kv[key] = val
        for _ in range(n_tensors):
            rstr()                                              # tensor name
            (n_dims,) = struct.unpack("<I", f.read(4))
            f.seek(8 * n_dims, 1)                               # dims
            (ttype,) = struct.unpack("<I", f.read(4))
            f.seek(8, 1)                                        # offset
            types[ttype] = types.get(ttype, 0) + 1
    return kv, types


def build_types(source):
    """{type id: name} from a build's ggml/include/ggml.h (commented-out, removed types excluded)."""
    out = {}
    with open(os.path.join(source, "ggml", "include", "ggml.h"), encoding="utf-8", errors="replace") as f:
        text = f.read()
    body = text[text.index("enum ggml_type {"):]
    body = body[:body.index("};")]
    for line in body.splitlines():
        m = re.match(r"\s*GGML_TYPE_([A-Z0-9_]+)\s*=\s*(\d+)", line)
        if m and m.group(1) != "COUNT":
            out[int(m.group(2))] = m.group(1)
    return out


def build_archs(source):
    """Model architecture names a build knows, from src/llama-arch.cpp (2026-09-27: mainline 150, Prism 149)."""
    with open(os.path.join(source, "src", "llama-arch.cpp"), encoding="utf-8", errors="replace") as f:
        return sorted(set(re.findall(r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"([^"]+)"\s*\}', f.read())))


def build_info(b):
    info = {"id": b["id"], "binary": b["binary"], "source": b.get("source")}
    try:
        r = subprocess.run([b["binary"], "--version"], capture_output=True, text=True, timeout=20)
        info["version"] = next((ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("version")), "")
    except (OSError, subprocess.TimeoutExpired) as e:
        info["version_error"] = str(e)[:200]
    tag = os.path.join(os.path.dirname(os.path.realpath(b["binary"])), "BUILD_TAG")
    if os.path.exists(tag):
        info["build_tag"] = open(tag).read().strip()
    try:
        info["types"] = build_types(b["source"]) if b.get("source") else {}
        info["archs"] = build_archs(b["source"]) if b.get("source") else []
    except (OSError, ValueError) as e:
        info["types"], info["archs"], info["types_error"] = {}, [], str(e)[:200]
    return info


def sha256(path, cache):
    st = os.stat(path)
    hit = cache.get(path)
    if hit and hit["size"] == st.st_size and hit["mtime"] == st.st_mtime:
        return hit["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    cache[path] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}
    return cache[path]["sha256"]


def main():
    job = json.load(sys.stdin)
    builds = [build_info(b) for b in job.get("builds") or []]
    paths = job.get("paths") or sorted({p for g in job.get("globs") or [] for p in glob.glob(g, recursive=True)})
    try:
        cache = json.load(open(CACHE))
    except (OSError, ValueError):
        cache = {}
    by_real = {}
    for p in paths:
        real = os.path.realpath(p)
        by_real.setdefault(real, []).append(p)
    models = []
    for real, seen in sorted(by_real.items()):
        rec = {"real_path": real, "paths": sorted(set(seen) | {real}), "file": os.path.basename(real)}
        try:
            st = os.stat(real)
            kv, types = read_gguf(real)
        except (OSError, ValueError, struct.error, UnicodeDecodeError) as e:
            models.append(dict(rec, error=f"{type(e).__name__}: {e}"[:200]))
            continue
        known = {}
        for b in builds:
            known.update(b["types"])
        rec.update(size_bytes=st.st_size, mtime=st.st_mtime, gguf=kv,
                   tensor_types={known.get(t, f"type{t}"): n for t, n in sorted(types.items())},
                   unknown_types=sorted(t for t in types if t not in known),
                   loadable_by=[b["id"] for b in builds if b["types"] and all(t in b["types"] for t in types)
                                and kv.get("general.architecture") in b["archs"]])
        if job.get("hash"):
            try:
                rec["sha256"] = sha256(real, cache)
            except OSError as e:
                rec["hash_error"] = str(e)[:200]
        elif real in cache and cache[real]["size"] == st.st_size and cache[real]["mtime"] == st.st_mtime:
            rec["sha256"] = cache[real]["sha256"]
        models.append(rec)
    if job.get("hash"):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE + ".tmp", "w") as f:
            json.dump(cache, f)
        os.replace(CACHE + ".tmp", CACHE)
    for b in builds:
        b["types"] = {str(k): v for k, v in b["types"].items()}
    json.dump({"builds": builds, "models": models}, sys.stdout)


if __name__ == "__main__":
    main()
