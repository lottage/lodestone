#!/usr/bin/env python3
"""
Frontier worker (VM 102, :8770): runs whole tasks through the owner's own subscription CLIs.

    claude -p   Claude Code, logged in with the owner's Claude Pro account
    gemini -p   Gemini CLI, logged in with the owner's Google account (Google AI Pro)

Each CLI is logged in by the owner, as the `frontier` user. This service never sees those credentials.

Rules (Boost plan, 2026-09-24):
- Tasks are general or code only. The CLIs get no MCP servers and no home tools, and the child process
  environment is scrubbed of every secret.
- A job never touches a live tree: it runs in a fresh clone under JOBS_DIR. The result is the CLI's answer
  plus a `git diff`, and the owner decides whether to apply it.
- One job per engine at a time, FRONTIER_DAILY_JOBS per engine per day (so loops can't burn the
  subscriptions), and a FRONTIER_TIMEOUT_S timeout.
- Optional CLI flags are used only if the installed version's --help lists them (same idea as engine_options.py).

HTTP (Bearer FRONTIER_WORKER_TOKEN, except /health):
    GET  /health         {ok, engines: {claude: {installed, jobs_today, cap, busy}}, ...}
    POST /jobs           {engine: claude|gemini, task, repo?: git url or path on this host, ref?} -> {id}
    GET  /jobs           recent jobs (no output)
    GET  /jobs/<id>      {id, engine, state: queued|running|done|failed, answer, diff, error, ...}
"""

import datetime
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

BIND = os.environ.get("FRONTIER_BIND", "127.0.0.1")
PORT = int(os.environ.get("FRONTIER_PORT", "8770"))
TOKEN = os.environ.get("FRONTIER_WORKER_TOKEN", "")
JOBS_DIR = os.environ.get("FRONTIER_JOBS_DIR", "/srv/frontier/jobs")
STATE_FILE = os.environ.get("FRONTIER_STATE", "/srv/frontier/state.json")
DAILY_JOBS = int(os.environ.get("FRONTIER_DAILY_JOBS", "10"))
TIMEOUT_S = int(os.environ.get("FRONTIER_TIMEOUT_S", "1200"))
KEEP_JOBS = 50
MAX_TASK_CHARS = 8000
MAX_DIFF_CHARS = 200_000

ENGINES = ("claude", "gemini")
# Claude Code: file tools plus read-only git and tests. No web, no MCP.
CLAUDE_ALLOWED = ["Read", "Edit", "Write", "Grep", "Glob", "Bash(git diff:*)", "Bash(git status:*)",
                  "Bash(git log:*)", "Bash(python -m pytest:*)", "Bash(python3 -m pytest:*)", "Bash(ls:*)"]
SAFE_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "NODE_OPTIONS")
PREAMBLE = ("You are working in a scratch clone of a repository. Make the change requested, keep it minimal, "
            "and finish with a short summary of what you changed and why. Do not push, and do not touch anything "
            "outside this directory.\n\nTask:\n")

_help_cache: Dict[str, str] = {}
_jobs: Dict[str, Dict[str, Any]] = {}
_engine_locks = {e: threading.Lock() for e in ENGINES}
_state_lock = threading.Lock()


# ---- CLI flags -------------------------------------------------------------------
def cli_help(binary: str) -> str:
    if binary not in _help_cache:
        try:
            out = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30, env=child_env())
            _help_cache[binary] = (out.stdout or "") + (out.stderr or "")
        except Exception:
            _help_cache[binary] = ""
    return _help_cache[binary]


def installed(binary: str) -> bool:
    return shutil.which(binary) is not None


def build_command(engine: str, task: str, help_text: str) -> List[str]:
    """argv for one job. Flags the installed CLI doesn't list are dropped (older/newer versions differ)."""
    def has(flag: str) -> bool:
        return re.search(rf"(^|\s|,){re.escape(flag)}(\s|,|=|$)", help_text, re.M) is not None

    prompt = PREAMBLE + task
    if engine == "claude":
        cmd = ["claude", "-p", prompt]
        if has("--output-format"):
            cmd += ["--output-format", "json"]
        if has("--allowedTools") or has("--allowed-tools"):
            cmd += ["--allowedTools", ",".join(CLAUDE_ALLOWED)]
        if has("--permission-mode"):
            cmd += ["--permission-mode", "acceptEdits"]
        if has("--strict-mcp-config"):
            cmd += ["--strict-mcp-config"]          # ignore any MCP config: no home tools
        if has("--max-turns"):
            cmd += ["--max-turns", "40"]
        return cmd
    if engine == "gemini":
        cmd = ["gemini", "-p", prompt]
        if has("--output-format"):
            cmd += ["--output-format", "json"]
        if has("--approval-mode"):
            cmd += ["--approval-mode", "auto_edit"]    # edits yes, shell commands no
        if has("--allowed-mcp-server-names"):
            cmd += ["--allowed-mcp-server-names", "none"]
        return cmd
    raise ValueError(f"unknown engine {engine}")


def child_env() -> Dict[str, str]:
    """Only harmless variables reach the CLIs: no tokens, no HASS_*, no FRONTIER_WORKER_TOKEN."""
    return {k: v for k, v in os.environ.items() if k in SAFE_ENV}


def parse_answer(engine: str, stdout: str) -> str:
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return str(data.get("result") or data.get("response") or data.get("text") or stdout)
    except Exception:
        pass
    return stdout


# ---- daily caps --------------------------------------------------------------------
def _today() -> str:
    return datetime.date.today().isoformat()


def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st if st.get("day") == _today() else {"day": _today(), "counts": {}}
    except Exception:
        return {"day": _today(), "counts": {}}


def jobs_today(engine: str) -> int:
    with _state_lock:
        return int(_load_state()["counts"].get(engine, 0))


def take_slot(engine: str, cap: int = DAILY_JOBS) -> bool:
    with _state_lock:
        st = _load_state()
        n = int(st["counts"].get(engine, 0))
        if n >= cap:
            return False
        st["counts"][engine] = n + 1
        os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f)
        return True


# ---- jobs ---------------------------------------------------------------------------
def _git(args: List[str], cwd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=child_env())


def prepare_workdir(job_id: str, repo: Optional[str], ref: Optional[str]) -> str:
    wd = os.path.join(JOBS_DIR, job_id)
    os.makedirs(JOBS_DIR, exist_ok=True)
    if repo:
        out = subprocess.run(["git", "clone", "--depth", "50"] + (["--branch", ref] if ref else []) + [repo, wd],
                             capture_output=True, text=True, timeout=600, env=child_env())
        if out.returncode != 0:
            raise RuntimeError(f"git clone failed: {out.stderr.strip()[:300]}")
    else:
        os.makedirs(wd)
        _git(["init", "-q"], wd)
    _git(["-c", "user.name=frontier", "-c", "user.email=frontier@localhost", "commit", "-q", "--allow-empty", "-m", "base"], wd)
    return wd


def collect_diff(wd: str) -> str:
    _git(["add", "-A", "-N"], wd)               # show new files in the diff without staging content
    diff = _git(["diff", "HEAD"], wd).stdout
    return diff if len(diff) <= MAX_DIFF_CHARS else diff[:MAX_DIFF_CHARS] + "\n… (diff truncated)"


def run_job(job: Dict[str, Any]) -> None:
    engine = job["engine"]
    with _engine_locks[engine]:
        job.update(state="running", started=time.time())
        try:
            wd = prepare_workdir(job["id"], job.get("repo"), job.get("ref"))
            cmd = build_command(engine, job["task"], cli_help(engine))
            out = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=TIMEOUT_S, env=child_env())
            job["answer"] = parse_answer(engine, out.stdout)[:50_000]
            job["diff"] = collect_diff(wd)
            job["exit_code"] = out.returncode
            if out.returncode != 0:
                job.update(state="failed", error=(out.stderr or out.stdout).strip()[-1500:])
            else:
                job["state"] = "done"
        except subprocess.TimeoutExpired:
            job.update(state="failed", error=f"timed out after {TIMEOUT_S} s")
        except Exception as e:
            job.update(state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            job["finished"] = time.time()
            _prune_jobs()


def _prune_jobs() -> None:
    done = sorted((j for j in _jobs.values() if j.get("finished")), key=lambda j: j["finished"])
    for j in done[:-KEEP_JOBS]:
        _jobs.pop(j["id"], None)
        shutil.rmtree(os.path.join(JOBS_DIR, j["id"]), ignore_errors=True)


def submit(body: Dict[str, Any]) -> Dict[str, Any]:
    engine = body.get("engine")
    task = str(body.get("task") or "").strip()
    if engine not in ENGINES:
        return {"ok": False, "error": f"engine must be one of {ENGINES}", "status": 400}
    if not task or len(task) > MAX_TASK_CHARS:
        return {"ok": False, "error": f"task must be 1-{MAX_TASK_CHARS} characters", "status": 400}
    if not installed(engine):
        return {"ok": False, "error": f"{engine} CLI is not installed for this user", "status": 503}
    if not take_slot(engine):
        return {"ok": False, "error": f"daily cap reached ({DAILY_JOBS} {engine} jobs)", "status": 429}
    job = {"id": uuid.uuid4().hex[:12], "engine": engine, "task": task, "repo": body.get("repo") or None,
           "ref": body.get("ref") or None, "state": "queued", "created": time.time()}
    _jobs[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True, name=f"job-{job['id']}").start()
    return {"ok": True, "id": job["id"]}


def health() -> Dict[str, Any]:
    return {"ok": True, "engines": {e: {"installed": installed(e), "jobs_today": jobs_today(e), "cap": DAILY_JOBS,
                                        "busy": _engine_locks[e].locked()} for e in ENGINES},
            "timeout_s": TIMEOUT_S}


# ---- HTTP ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if TOKEN and self.headers.get("Authorization", "") == f"Bearer {TOKEN}":
            return True
        self._send({"ok": False, "error": "unauthorized"}, 401)
        return False

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            return self._send(health())
        if not self._authed():
            return
        if path == "/jobs":
            return self._send({"ok": True, "jobs": [{k: j.get(k) for k in ("id", "engine", "state", "created", "finished", "error")}
                                                    for j in sorted(_jobs.values(), key=lambda j: -j["created"])]})
        if path.startswith("/jobs/"):
            job = _jobs.get(path.split("/")[-1])
            return self._send({"ok": True, **job} if job else {"ok": False, "error": "no such job"}, 200 if job else 404)
        self._send({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        if not self._authed():
            return
        if self.path.rstrip("/") != "/jobs":
            return self._send({"ok": False, "error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
        except Exception:
            return self._send({"ok": False, "error": "bad json"}, 400)
        res = submit(body)
        self._send(res, res.pop("status", 200))

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    if not TOKEN:
        raise SystemExit("FRONTIER_WORKER_TOKEN is not set")
    os.makedirs(JOBS_DIR, exist_ok=True)
    print(f"frontier worker on {BIND}:{PORT}, engines: {health()['engines']}", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
