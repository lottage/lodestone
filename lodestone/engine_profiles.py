"""
Engine Profiles: named bundles of "which model loads on which node", applied through the Model
Loader's existing per-target pipeline (backup, token-checked apply, health check, auto-rollback).

This replaces the retired switch_cluster_mode.py idea (see STATE.md "Known issues") without its two
failure modes: every model is checked against the target's live library before anything is touched,
and a profile only ever rewrites the targets it actually lists, never nodes it doesn't mention.
A profile can be gated behind a minimum project phase (config.json "project_phase") so, e.g., a
profile that hands the Ally over to free-use agents can't fire before the garden actually opens.
"""

import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from lodestone import model_loader
from lodestone import system_profile

_profile_jobs: Dict[str, Dict[str, Any]] = {}
_profile_jobs_lock = threading.Lock()


def _cfg() -> Dict[str, Any]:
    return system_profile.load_cfg()


def _library_keys(node_id: str) -> set:
    return {m["key"] for m in model_loader.get_library(node_id)["models"]}


def _gate_met(spec: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    min_phase = spec.get("min_phase")
    if min_phase is None:
        return True
    return cfg.get("project_phase", 0) >= min_phase


# ------------------------------------------------------------------ listing ----
def list_profiles(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every configured profile, with each target's model checked against what that node actually has."""
    cfg = cfg or _cfg()
    try:
        live = {t["id"]: t for n in model_loader.get_state().get("nodes", []) for t in n.get("targets", [])}
    except Exception:
        live = {}
    out = []
    for name, spec in (cfg.get("engine_profiles") or {}).items():
        targets = spec.get("targets", {})
        status = {}
        for target_id, want in targets.items():
            node = "host:inference" if target_id.startswith("engine:") else target_id
            if not want.get("model"):
                status[target_id] = "unspecified"
                continue
            if os.path.basename(want["model"]) == (live.get(target_id) or {}).get("model_file"):
                status[target_id] = "active"
                continue
            try:
                status[target_id] = "available" if want["model"] in _library_keys(node) else "model_missing"
            except Exception as e:
                status[target_id] = f"error: {type(e).__name__}: {e}"
        out.append({
            "name": name, "description": spec.get("description", ""), "min_phase": spec.get("min_phase"),
            "unlocked": _gate_met(spec, cfg), "targets": {t: {"model": w.get("model"), "status": status[t]}
                                                           for t, w in targets.items()},
        })
    return {"ok": True, "profiles": out, "project_phase": cfg.get("project_phase", 0)}


# --------------------------------------------------------------- per-target ----
def _live_engine(target_id: str) -> Dict[str, Any]:
    """What an engine target runs right now (model_loader.get_state's per-target entry), or {}."""
    for node in model_loader.get_state().get("nodes", []):
        for t in node.get("targets", []):
            if t.get("id") == target_id:
                return t
    return {}


def _engine_request(target_id: str, want: Dict[str, Any], live: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the Model Loader request for an engine target, or None if it already runs the wanted model
    and the profile asks for nothing else. Sizing the profile leaves out is carried over from the live
    engine: model_loader.preview() defaults to 4096 x 1, which would silently shrink the coordinator's
    6144 x 2 just because a profile only named the model."""
    extra = {k for k in want if k != "model"}
    if not extra and os.path.basename(want["model"]) == live.get("model_file"):
        return None
    req = {"target": target_id, **want}
    for key in ("ctx_per_slot", "slots"):
        if req.get(key) is None and live.get(key):
            req[key] = live[key]
    return req


def _submit_target(target_id: str, want: Dict[str, Any]) -> Dict[str, Any]:
    """Preview+apply one profile target. Mirrors what the Model Loader UI does for a single change:
    review first, apply only the reviewed token. Returns model_loader.apply()'s {"ok","job"} shape,
    or {"ok": False, "error": ...} if it never got that far (model missing, preview rejected it)."""
    node = "host:inference" if target_id.startswith("engine:") else target_id
    if not want.get("model"):
        return {"ok": True, "skipped": "no model specified for this target"}
    try:
        have = _library_keys(node)
    except Exception as e:
        return {"ok": False, "error": f"could not read {node}'s library: {type(e).__name__}: {e}"}
    if want["model"] not in have:
        return {"ok": False, "error": f"model '{want['model']}' not found on {node}"}
    if target_id.startswith("engine:"):
        req = _engine_request(target_id, want, _live_engine(target_id))
        if req is None:
            return {"ok": True, "skipped": "already running this model"}
    else:
        req = {"target": target_id, **want}
    pv = model_loader.preview(req)
    if not pv.get("ok"):
        return {"ok": False, "error": pv.get("error", "preview failed")}
    if pv["kind"] == "unit" and not pv["changed"]:
        return {"ok": True, "skipped": "already set to this model"}
    req["token"] = pv["token"]
    return model_loader.apply(req)


def _wait_target_job(job_id: str, timeout: float = 900) -> Dict[str, Any]:
    """Poll a model_loader job (restart + /health + /props check happens inside it) until it settles.
    900 s covers model_loader's worst case: 300 s health wait, then a rollback with its own 240 s wait.
    A shorter timeout would move on while the job still holds _apply_lock, and the next target would
    fail with "Another engine change is still running"."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = model_loader.get_job(job_id)
        if job.get("status") not in (None, "running"):
            return job
        time.sleep(2)
    return {"status": "timeout", "error": "target did not finish in time"}


def _profile_job(job_id: str, **upd) -> None:
    with _profile_jobs_lock:
        job = _profile_jobs.setdefault(job_id, {"id": job_id, "results": {}, "created": time.time()})
        job.update(upd)


def get_profile_job(job_id: str) -> Dict[str, Any]:
    with _profile_jobs_lock:
        return dict(_profile_jobs.get(job_id) or {"error": "unknown job"})


def _run_profile(job_id: str, name: str, targets: Dict[str, Dict[str, Any]]) -> None:
    """Apply every target in the profile, one at a time (model_loader's own _apply_lock only allows
    one engine change cluster-wide anyway). A failing target does not stop the rest - a bad vision
    swap in courage_default shouldn't block worker from applying - so this always visits every
    target and reports "partial" if any of them failed rather than aborting early."""
    results: Dict[str, Any] = {}
    any_failed = False
    for target_id, want in targets.items():
        _profile_job(job_id, step=f"Applying {target_id}")
        outcome = _submit_target(target_id, want)
        if outcome.get("ok") and outcome.get("job"):
            outcome = _wait_target_job(outcome["job"])  # job dict: status is "done"/"failed"/"rolled_back"/"timeout"
            ok = outcome.get("status") == "done"
        else:
            ok = bool(outcome.get("ok"))  # immediate result: ok=True means applied or intentionally skipped
        if not ok:
            any_failed = True
        results[target_id] = outcome
        _profile_job(job_id, results=dict(results))
    _profile_job(job_id, status="partial" if any_failed else "done",
                 step="Finished with failures" if any_failed else "Finished: every target applied")


def apply_profile(name: str) -> Dict[str, Any]:
    cfg = _cfg()
    spec = (cfg.get("engine_profiles") or {}).get(name)
    if not spec:
        return {"ok": False, "error": f"unknown profile: {name}"}
    if not _gate_met(spec, cfg):
        return {"ok": False, "error": f"'{name}' needs project_phase >= {spec.get('min_phase')} "
                                       f"(currently {cfg.get('project_phase', 0)})"}
    targets = spec.get("targets") or {}
    if not targets:
        return {"ok": False, "error": "profile has no targets"}
    job_id = uuid.uuid4().hex[:10]
    _profile_job(job_id, status="running", profile=name)
    threading.Thread(target=_run_profile, args=(job_id, name, targets), daemon=True).start()
    return {"ok": True, "job": job_id}
