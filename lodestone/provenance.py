"""
Model provenance (Phase 6, 2026-09-27): what every GGUF on the inference host is, where it came from, and which llama.cpp
build can load it. The scan runs on the engine host (`lodestone/provenance_scan.py`, stdlib only, copied there; JSON job on stdin);
builds and model folders come from config.json `provenance`, never code.

Records: CouchDB database `provenance.db` (default model_provenance, on the LiveSync CouchDB but a separate database),
one document per model file content: _id "gguf:<sha256>" with paths, size, GGUF lineage (general.* keys: name, base
models, source/repo URLs, quantized_by, license, file_type), tensor types and `loadable_by`. Fields a person adds
(notes, recommended_quant, lineage_note) survive rescans. Builds: "build:<id>" (version, tag, #types, #archs).
A local copy (data/model_provenance.json) serves the Model Loader without CouchDB.

The Model Loader's preview refuses (hard, no override) a model the target unit's llama-server build cannot load:
2026-09-27 the two Ternary Bonsai files need the Prism fork (PQ2_0 / PTQ1_0 tensors), every other model loads on both.
A model not scanned yet is scanned on the spot (header only, ~0.1 s); an unreadable header blocks too.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

SCRIPT = "/opt/cluster-bridge/model_provenance.py"   # where provenance_scan.py is installed on the engine host;
                                                    # config provenance.script overrides
HUMAN_FIELDS = ("notes", "recommended_quant", "lineage_note")
RESCAN_S = 6 * 3600


def lineage(gguf: Dict[str, Any]) -> Dict[str, Any]:
    """The GGUF general.* keys as a small lineage record."""
    bases = []
    for i in range(int(gguf.get("general.base_model.count") or 0)):
        b = {k: gguf.get(f"general.base_model.{i}.{k}") for k in ("name", "organization", "repo_url", "version")}
        if any(b.values()):
            bases.append({k: v for k, v in b.items() if v})
    out = {"name": gguf.get("general.name"), "basename": gguf.get("general.basename"),
           "finetune": gguf.get("general.finetune"), "size_label": gguf.get("general.size_label"),
           "organization": gguf.get("general.organization"), "quantized_by": gguf.get("general.quantized_by"),
           "license": gguf.get("general.license"),
           "source": gguf.get("general.source.repo_url") or gguf.get("general.source.url")
           or gguf.get("general.repo_url") or gguf.get("general.url"),
           "base_models": bases}
    return {k: v for k, v in out.items() if v}


def unit_binary(unit_text: str) -> Optional[str]:
    m = re.search(r"^ExecStart=\s*(\S+)", unit_text or "", re.M)
    return m.group(1) if m else None


class Provenance:
    def __init__(self, cfg_fn: Callable[[], Dict[str, Any]], run: Callable[[str, str, float], Any],
                 couch: Optional[Dict[str, Any]], path: str):
        """run(cmd, stdin, timeout) -> CompletedProcess on the inference host; couch: {url, headers} or None."""
        self.cfg_fn, self.run, self.couch, self.path = cfg_fn, run, couch, path
        self.data: Dict[str, Any] = {"builds": [], "models": [], "scanned_at": None, "hashed_at": None}
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        try:
            with open(path, encoding="utf-8") as f:
                self.data.update(json.load(f))
        except (OSError, ValueError):
            pass

    def _pcfg(self) -> Dict[str, Any]:
        return self.cfg_fn().get("provenance") or {}

    # ------------------------------------------------------------------ scan ----
    def scan(self, hash_files: bool = False, paths: Optional[List[str]] = None, timeout: float = 3600) -> Dict[str, Any]:
        pcfg = self._pcfg()
        job = {"builds": pcfg.get("builds") or [], "globs": pcfg.get("globs") or [], "hash": hash_files}
        if paths:
            job["paths"] = paths
        res = self.run(f"python3 {self._pcfg().get('script') or SCRIPT}", json.dumps(job), timeout)
        if res.returncode != 0:
            raise RuntimeError((res.stderr or "scan failed").strip()[-300:])
        out = json.loads(res.stdout)
        with self._lock:
            if paths:                               # one file: merge it into what we have
                by_real = {m["real_path"]: m for m in self.data["models"]}
                by_real.update({m["real_path"]: m for m in out["models"]})
                self.data["models"] = sorted(by_real.values(), key=lambda m: m["real_path"])
            else:
                self.data["models"] = out["models"]
                self.data["scanned_at"] = time.time()
                if hash_files:
                    self.data["hashed_at"] = time.time()
            self.data["builds"] = out["builds"]
            self._save()
        if self.couch and not paths:
            self.publish()
        return self.summary()

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- lookup ----
    def lookup(self, path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((m for m in self.data["models"] if path in (m.get("paths") or []) or path == m.get("real_path")), None)

    def build_for(self, binary: str) -> Optional[Dict[str, Any]]:
        """The configured build a unit's llama-server binary belongs to (exact path, or the same directory)."""
        for b in self._pcfg().get("builds") or []:
            if binary == b["binary"] or os.path.dirname(binary) == os.path.dirname(b["binary"]):
                return b
        return None

    def check(self, model_path: str, unit_text: str) -> Optional[str]:
        """Why this unit's llama-server cannot load this model (hard block), or None when it can."""
        binary = unit_binary(unit_text)
        build = self.build_for(binary or "")
        if not build:
            return f"unknown llama-server build {binary!r}: add it to config.json provenance.builds"
        m = self.lookup(model_path)
        if m is None or "loadable_by" not in m:
            try:
                self.scan(paths=[model_path], timeout=60)
            except Exception as e:
                return f"could not read the model header ({e})"[:300]
            m = self.lookup(model_path)
        if not m or m.get("error"):
            return f"could not read the model header ({(m or {}).get('error', 'not found')})"
        if build["id"] in (m.get("loadable_by") or []):
            return None
        need = m.get("loadable_by") or []
        arch = (m.get("gguf") or {}).get("general.architecture")
        why = ", ".join(m.get("tensor_types") or {})
        return (f"{m['file']} ({arch}; tensor types {why}) cannot run on the {build['id']} build "
                f"({binary}); it needs: {', '.join(need) or 'no installed build'}")

    # --------------------------------------------------------------- couchdb ----
    def _couch(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        req = urllib.request.Request(self.couch["url"].rstrip("/") + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers=self.couch["headers"])
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def publish(self) -> Dict[str, int]:
        """Upsert one document per hashed model and per build; keep fields a person added."""
        db = urllib.parse.quote(self._pcfg().get("db") or "model_provenance")
        try:
            self._couch("PUT", f"/{db}")
        except urllib.error.HTTPError as e:
            if e.code != 412:                   # 412 = exists
                raise
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        docs = []
        for m in self.data["models"]:
            if not m.get("sha256"):
                continue
            docs.append({"_id": f"gguf:{m['sha256']}", "type": "gguf", "sha256": m["sha256"], "file": m["file"],
                         "paths": m.get("paths"), "size_bytes": m.get("size_bytes"),
                         "architecture": (m.get("gguf") or {}).get("general.architecture"),
                         "lineage": lineage(m.get("gguf") or {}), "gguf": m.get("gguf"),
                         "tensor_types": m.get("tensor_types"), "loadable_by": m.get("loadable_by"), "last_seen": now})
        for b in self.data["builds"]:
            docs.append({"_id": f"build:{b['id']}", "type": "build", "binary": b.get("binary"), "version": b.get("version"),
                         "build_tag": b.get("build_tag"), "types": len(b.get("types") or {}),
                         "archs": len(b.get("archs") or []), "last_seen": now})
        existing = {r["id"]: r.get("doc") for r in self._couch("POST", f"/{db}/_all_docs?include_docs=true",
                                                                {"keys": [d["_id"] for d in docs]})["rows"] if r.get("doc")}
        for d in docs:
            old = existing.get(d["_id"])
            if old:
                d["_rev"] = old["_rev"]
                d["first_seen"] = old.get("first_seen") or now
                d.update({k: old[k] for k in HUMAN_FIELDS if k in old})
            else:
                d["first_seen"] = now
        res = self._couch("POST", f"/{db}/_bulk_docs", {"docs": docs})
        failed = [r for r in res if r.get("error")]
        if failed:
            self.last_error = f"CouchDB: {len(failed)} doc(s) failed, e.g. {failed[0]}"[:300]
        return {"written": len(res) - len(failed), "failed": len(failed)}

    # ----------------------------------------------------------------- views ----
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            models = [{"file": m["file"], "paths": m.get("paths"), "sha256": m.get("sha256"),
                       "size_gb": round((m.get("size_bytes") or 0) / 1e9, 2),
                       "architecture": (m.get("gguf") or {}).get("general.architecture"),
                       "loadable_by": m.get("loadable_by"), "tensor_types": m.get("tensor_types"),
                       "lineage": lineage(m.get("gguf") or {}), "error": m.get("error")} for m in self.data["models"]]
            builds = [{k: b.get(k) for k in ("id", "binary", "version", "build_tag")} |
                      {"types": len(b.get("types") or {}), "archs": len(b.get("archs") or [])} for b in self.data["builds"]]
        return {"builds": builds, "models": models, "scanned_at": self.data.get("scanned_at"),
                "hashed_at": self.data.get("hashed_at"), "hashed": sum(1 for m in models if m["sha256"]),
                "last_error": self.last_error}

    def start(self) -> None:
        """Header scan now, then a hashing scan every RESCAN_S (cached hashes make it quick unless files changed)."""
        if self._thread and self._thread.is_alive():
            return

        def loop():
            first = True
            while True:
                try:
                    self.scan(hash_files=not first)
                    self.last_error = None
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"[:300]
                time.sleep(1800 if first else RESCAN_S)   # first hashing pass after 30 min: a restart never piles
                first = False                              # a second 234 GB read onto one already running

        self._thread = threading.Thread(target=loop, name="provenance", daemon=True)
        self._thread.start()
