"""
Client for the frontier worker on the engine host (lodestone/boost/frontier_worker.py, stdlib only, copied there): whole tasks run by
The owner's Claude Code / Gemini CLI subscriptions, in a scratch clone, returning an answer and a diff.

Only `general`/`code` content may go there (consumer subscription terms), so every task passes the egress
scan first; anything that looks like home data or a secret is refused before it leaves LXC 120.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from . import egress


class FrontierClient:
    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.token = token if token is not None else os.environ.get("FRONTIER_WORKER_TOKEN", "")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except Exception:
                return {"ok": False, "error": f"HTTP {e.code}"}
        except Exception as e:
            return {"ok": False, "error": f"frontier worker unreachable: {e}"}

    def health(self) -> Dict[str, Any]:
        return self._call("GET", "/health")

    def submit(self, engine: str, task: str, repo: Optional[str] = None, ref: Optional[str] = None,
               home_terms=()) -> Dict[str, Any]:
        verdict = egress.classify("code", [{"role": "user", "content": task}], home_terms)
        if not egress.egress_allowed("training", verdict["class"]):
            return {"ok": False, "error": f"refused: the task contains {verdict['class']} content ({', '.join(verdict['hits'])})"}
        if not self.token:
            return {"ok": False, "error": "FRONTIER_WORKER_TOKEN is not set on StoneSage"}
        return self._call("POST", "/jobs", {"engine": engine, "task": task, "repo": repo, "ref": ref})

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/jobs/{job_id}")

    def jobs(self) -> Dict[str, Any]:
        return self._call("GET", "/jobs")
