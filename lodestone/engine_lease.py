"""
Engine lease (docs/plan-2026-09-26-local-loop/DESIGN.md section B; built 2026-09-28): lend a GPU's engines to another
layout (Bonsai 2 27B on the Prism fork, :8005) for a while, with a dead man's switch.

Layouts come from config.json engine_layouts (unit, evicts, url, when, optional stand_in, optional disabled reason);
no unit or model names in code. The layout units on the engine host carry `Conflicts=` on what they evict, so systemd itself
keeps one layout at a time: starting the layout stops the evicted engines, and starting the evicted engines again
(release, expiry, revert) stops the layout.

acquire(layout, ttl_s, reason, approver, force)
  refuses when a lease is held, the layout is disabled, its `when` gate fails ("idle" = the idle gate (idle_gate.py)
  is open for every GPU an evicted engine sits on; "any" = no gate), or an evicted engine is in use (a Computer turn
  within IN_USE_S for the coordinator/worker, a patrol sweep for vision) unless force. Then starts the unit, waits for
  /health and /props (HEALTH_TIMEOUT_S), and reverts at once if it never comes up.
Lease time: ttl (default 60 min, max MAX_TTL_S). At CHECKIN_BEFORE_S before the end a check-in goes to the owner's phone
  with Another hour / Stop buttons; no answer means it ends. Only a person renews: the phone button. The HTTP renew
  route refuses (a LAN API can't tell the owner from Claude Code or the loop worker); nothing automatic ever renews.
release / expiry / StoneSage restarting with a lease on disk (data/engine_lease.json): start the evicted units again
  and check they answer. A failed revert is traced (revert_failed) and pushed to the phone.
While leased: `on_loan(role)` tells callers who is away until when, so Computer answers "on loan until HH:MM" instead of
  failing (and never climbs the escalation ladder for it), camera looks fall back to Frigate's objects, patrol and
  commentary skip vision.
Stand-in (DESIGN.md B2, 2026-09-28): a layout with `stand_in` (config engine_stand_ins: unit, url, replaces, eval_passed)
  starts it BEFORE evicting the engine it replaces, when its eval_passed is set; brain(role) then tells Computer to use it
  (lite mode: every action asks, short replies, it says it's on the spare brain until HH:MM). It stops only after the
  real engine answers again; if that revert fails, the stand-in keeps answering and the lease stays (revert_failed)
  until a person releases it. A stand-in that doesn't come up means Computer says "on loan" instead.
yield_to_computer (2026-10-13): background work (the garden) leases with this flag. It is gated on hardware demand only (the idle
  gate's demand_open: GPU busy, engines idle, no approval, no recent Computer turn; never the household signal), plus the cockpit
  signal (every garden lease yields to Computer) (a lease a person asks for keeps the full gate and ignores the cockpit signal), and a lease that
  evicts the coordinator while the stand-in answers can be taken back on demand (stonesage/courage/takeback.py: release, the
  coordinator is back in ~30-50 s). brain() reports the flag; yielding() returns such a lease.
Races (2026-10-14, after a takeback released a lease that was still `starting`, and the start sequence went on to evict the
  coordinator with nothing left to revert it): one long-operation lock (`_op`) serialises acquire, release/revert/expiry,
  recover and reconcile. release() on a `starting` lease only sets a cancel flag (it never touches the engines itself) and waits,
  bounded (`cancel_wait_s`), for the acquiring thread, which checks the flag before every step (stand-in start, stand-in health
  wait, before the layout unit evicts anything, after the eviction, during the layout's health wait, before the lease turns
  active), undoes exactly what it started (stand-in stop; or the full revert once the layout unit was started) and traces
  `cancelled_while_starting` with the stage and what it undid. It never reports "released" while anything can still be evicted.
  reconcile(): no lease held but a layout unit is active, or an evicted engine unit is inactive (seen twice, `RECONCILE_CONFIRM_S`
  apart): put it back to normal, trace `reconcile`, one phone note per incident. A held lease is never touched.
Not built yet: hands-off / night grants and autopilot (they belong to loop jobs, which don't exist yet).
Every step is traced as kind "lease".
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

DEFAULT_TTL_S = 3600
MAX_TTL_S = 10 * 3600
CHECKIN_BEFORE_S = 600
IN_USE_S = 60
HEALTH_TIMEOUT_S = 300
CANCEL_WAIT_S = 150            # how long release() waits for a cancelled start to put the engines back
RECONCILE_CONFIRM_S = 20       # a stray unit must be seen twice this far apart before reconcile acts (a restart in progress is not a stray)
STAND_IN_TIMEOUT_S = 120     # Courage-lite on CPU answers in ~6 s; a stand-in that takes this long is not starting
RENEW, STOP = "LEASE_RENEW_", "LEASE_STOP_"


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


class _Start:
    """The start sequence of one lease, seen from release(): a cancel flag, and a done event set when the sequence has ended
    (and, if it was cancelled, after everything it started was put back)."""

    def __init__(self, lease_id: str):
        self.lease_id = lease_id
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.layout_started = False            # the layout unit's start was issued: from then on engines may have been evicted
        self.stand_in_unit: Optional[str] = None
        self.result: Dict[str, Any] = {}


class EngineLease:
    def __init__(self, cfg_fn: Callable[[], Dict[str, Any]], systemctl: Callable[[str, str], Dict[str, Any]],
                 http_ok: Callable[[str], bool], unit_roles: Callable[[], Dict[str, Dict[str, Any]]],
                 idle_check: Callable[[], Dict[str, Any]], in_use: Callable[[List[str]], Optional[str]],
                 push: Optional[Callable[[str, str, Optional[List[Dict[str, str]]], str], Any]] = None,
                 trace: Optional[Callable[[Dict[str, Any]], None]] = None, path: Optional[str] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        """systemctl(verb, unit) -> {ok, error}; http_ok(url) -> bool; unit_roles() -> {unit: {role, gpu, url}};
        idle_check() -> idle_gate.check(); in_use(roles) -> why an evicted engine is busy, or None;
        push(title, message, buttons, tag)."""
        self.cfg_fn, self.systemctl, self.http_ok, self.unit_roles = cfg_fn, systemctl, http_ok, unit_roles
        self.idle_check, self.in_use, self.push, self.trace = idle_check, in_use, push, trace
        self.path, self.clock, self.sleep = path, clock, sleep
        self.lease: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()          # short: guards self.lease / self._starting
        self._op = threading.Lock()            # long: one of acquire / revert / recover / reconcile at a time
        self._starting: Optional[_Start] = None
        self.cancel_wait_s = CANCEL_WAIT_S
        self.reconcile_confirm_s = RECONCILE_CONFIRM_S
        self._suspect: Optional[Dict[str, Any]] = None
        self._incident: Optional[str] = None
        self.last_reconcile: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ config ----
    def layouts(self) -> Dict[str, Dict[str, Any]]:
        return self.cfg_fn().get("engine_layouts") or {}

    def _roles_of(self, units: List[str]) -> List[Dict[str, Any]]:
        known = self.unit_roles()
        return [dict(known.get(u) or {}, unit=u) for u in units]

    def layout_info(self, layout: str) -> Optional[Dict[str, Any]]:
        """What a layout would evict, resolved against the live profile, for callers that decide before they acquire (the garden
        scheduler): the evicted engines' roles and GPUs, and whether every one of them could be resolved."""
        spec = self.layouts().get(layout)
        if not spec:
            return None
        evicted = self._roles_of(spec.get("evicts") or [])
        roles = [e["role"] for e in evicted if e.get("role")]
        stand_in = spec.get("stand_in")
        si = (self.cfg_fn().get("engine_stand_ins") or {}).get(stand_in or "") or {}
        return {"layout": layout, "unit": spec.get("unit"), "url": spec.get("url"), "when": spec.get("when", "idle"),
                "disabled": spec.get("disabled"), "roles": roles,
                "gpus": sorted({e["gpu"] for e in evicted if e.get("gpu")}),
                "resolved": all(e.get("role") and e.get("gpu") for e in evicted),
                # a stand-in that replaces an engine this layout evicts AND has passed its eval (what _start_stand_in needs)
                "stand_in": stand_in, "stand_in_ready": bool(stand_in and si.get("replaces") in roles and si.get("eval_passed"))}

    # ------------------------------------------------------------------- state ----
    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.lease, f)
        os.replace(tmp, self.path)

    def _record(self, event: str, **fields: Any) -> Dict[str, Any]:
        rec = {"kind": "lease", "at": round(self.clock(), 3), "event": event, "outcome": event,
               "triggers": ["revert_failed"] if event == "revert_failed" else [], **fields}
        if self.trace:
            try:
                self.trace(rec)
            except Exception:
                pass
        return rec

    def on_loan(self, role: str) -> Optional[float]:
        """When `role` (coordinator, worker, embedder, vision) is lent out: the lease's end time, else None."""
        lease = self.lease
        if lease and role in (lease.get("evicted_roles") or []):
            return lease["expires"]
        return None

    def status(self) -> Dict[str, Any]:
        lease = dict(self.lease) if self.lease else None
        if lease:
            lease["minutes_left"] = round((lease["expires"] - self.clock()) / 60, 1)
            lease["until"] = _hhmm(lease["expires"])
        return {"lease": lease, "layouts": {k: {"unit": v.get("unit"), "evicts": v.get("evicts"), "when": v.get("when"),
                                                "disabled": v.get("disabled")} for k, v in self.layouts().items()}}

    # ------------------------------------------------------------------ acquire ----
    def acquire(self, layout: str, ttl_s: float = DEFAULT_TTL_S, reason: str = "", approver: str = "",
                force: bool = False, yield_to_computer: bool = False) -> Dict[str, Any]:
        """yield_to_computer: background work. The cockpit signal closes the gate for it, and Computer may take the lease back."""
        if not self._op.acquire(blocking=False):
            held = self.lease
            return self._refuse(layout, f"a lease is already held ({held['layout']} until {_hhmm(held['expires'])})" if held
                                else "another lease operation is in progress")
        try:
            return self._acquire(layout, ttl_s, reason, approver, force, yield_to_computer)
        finally:
            self._op.release()

    def _acquire(self, layout: str, ttl_s: float, reason: str, approver: str, force: bool, yield_to_computer: bool) -> Dict[str, Any]:
        with self._lock:
            if self.lease:
                return self._refuse(layout, f"a lease is already held ({self.lease['layout']} until "
                                            f"{_hhmm(self.lease['expires'])})")
            spec = self.layouts().get(layout)
            if not spec:
                return self._refuse(layout, "unknown layout")
            if spec.get("disabled"):
                return self._refuse(layout, f"layout disabled: {spec['disabled']}")
            evicted = self._roles_of(spec.get("evicts") or [])
            roles = [e.get("role") for e in evicted if e.get("role")]
            if spec.get("when", "idle") == "idle":
                gate = self.idle_check()
                gpus = {e.get("gpu") for e in evicted if e.get("gpu")}
                cockpit = None if yield_to_computer else gate.get("cockpit")       # a person's own lease ignores his being on the page
                shared = [r for r in gate.get("shared_reasons") or [] if r != cockpit]
                # Background work (yield_to_computer) is gated on hardware demand only (idle_gate.py, 2026-10-14): the household
                # signal never closes it. The cockpit signal counts too: every background lease yields.
                key = None
                if yield_to_computer:
                    key = "demand_open_with_cockpit"
                    shared = list(gate.get("demand_shared_reasons") or []) + ([gate["cockpit"]] if gate.get("cockpit") else [])

                def is_open(g: Dict[str, Any]) -> bool:
                    if key and key in g:
                        return bool(g[key])
                    return bool(g.get("open_ignoring_cockpit", g.get("open"))) if cockpit else bool(g.get("open"))

                closed = [f"{g}: {', '.join(gate['gpus'].get(g, {}).get('reasons') or shared or ['closed'])}"
                          for g in sorted(gpus) if not is_open(gate.get("gpus", {}).get(g) or {})]
                if closed or not gpus:
                    return self._refuse(layout, "idle gate closed: " + ("; ".join(closed) or "evicted engines' GPUs unknown"))
            if not force:
                busy = self.in_use(roles)
                if busy:
                    return self._refuse(layout, f"in use: {busy} (force to override)")
            ttl = max(60.0, min(float(ttl_s or DEFAULT_TTL_S), MAX_TTL_S))
            lease = {"id": uuid.uuid4().hex[:8], "layout": layout, "unit": spec["unit"], "url": spec.get("url"),
                     "evicts": spec.get("evicts") or [], "evicted_roles": roles, "reason": reason[:200],
                     # role / GPU / URL of each evicted engine, recorded now: while leased they aren't running, so the
                     # live profile no longer lists them, and the revert must still know where to check /health
                     "evicted": evicted,
                     "approver": approver or "unknown", "started": self.clock(), "expires": self.clock() + ttl,
                     "ttl_s": ttl, "checked_in": False, "state": "starting", "yield_to_computer": bool(yield_to_computer)}
            self.lease = lease
            ctl = self._starting = _Start(lease["id"])
        self._save()
        try:
            return self._run_start(ctl, lease, spec, roles, ttl, layout, force, yield_to_computer)
        finally:
            with self._lock:
                self._starting = None
            ctl.done.set()

    def _run_start(self, ctl: _Start, lease: Dict[str, Any], spec: Dict[str, Any], roles: List[str], ttl: float, layout: str,
                   force: bool, yield_to_computer: bool) -> Dict[str, Any]:
        """The start sequence. release() can set ctl.cancel at any moment; it is checked before every step that has an effect."""
        if ctl.cancel.is_set():
            return self._cancel_start(ctl, lease, "before_stand_in")
        lease["stand_in"] = self._start_stand_in(spec, roles, ctl)   # before the coordinator goes: Computer never goes dark
        self._save()
        if ctl.cancel.is_set():                                        # also covers a stand-in wait that was cut short
            return self._cancel_start(ctl, lease, "after_stand_in")    # nothing is evicted yet
        ctl.layout_started = True                                      # set first: the start itself evicts
        res = self.systemctl("start", spec["unit"])
        if ctl.cancel.is_set():
            return self._cancel_start(ctl, lease, "after_eviction")
        up = res.get("ok") and self._wait_up(spec.get("url"), cancel=ctl.cancel)
        with self._lock:
            if self.lease is not lease:
                return {"ok": False, "error": "the lease was released while starting"}
            if not ctl.cancel.is_set():
                self._starting = None                                  # from here release() takes the ordinary path
                if up:
                    lease["state"] = "active"
                    lease["expires"] = self.clock() + ttl              # the clock starts once it answers
                    self._save()
                    self._record("acquire", lease_id=lease["id"], layout=layout, evicts=lease["evicts"], ttl_s=ttl,
                                 reason=lease["reason"], approver=lease["approver"], forced=force,
                                 yield_to_computer=bool(yield_to_computer))
                    return {"ok": True, "lease_id": lease["id"], "url": lease["url"], "expires": lease["expires"],
                            "until": _hhmm(lease["expires"])}
        if ctl.cancel.is_set():
            return self._cancel_start(ctl, lease, "during_health_wait")
        self._revert(f"{spec['unit']} did not come up: {res.get('error') or 'no /health within timeout'}")
        return {"ok": False, "error": f"{spec['unit']} did not come up; engines reverted"}

    def _cancel_start(self, ctl: _Start, lease: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """release() asked for the lease back while it was starting: undo what this sequence started, then report."""
        undone: List[str] = []
        failed: List[str] = []
        if ctl.layout_started:
            out = self._revert("cancelled while starting", event="cancelled_while_starting", stage=stage)
            undone = ["layout stopped", "evicted engines started"] + (["stand-in stopped"] if (lease.get("stand_in") or {}).get("up") else [])
            failed = out.get("failed") or []
        else:
            if ctl.stand_in_unit:
                self.systemctl("stop", ctl.stand_in_unit)
                undone.append("stand-in stopped")
            with self._lock:
                self.lease = None
                self._save()
            self._record("cancelled_while_starting", lease_id=lease["id"], layout=lease.get("layout"), stage=stage, undone=undone,
                         why="released while starting", failed=None)
        ctl.result = {"stage": stage, "undone": undone, "failed": failed}
        return {"ok": False, "cancelled": True, "error": "the lease was released while starting", "stage": stage, "undone": undone}

    def _start_stand_in(self, spec: Dict[str, Any], roles: List[str], ctl: Optional[_Start] = None) -> Optional[Dict[str, Any]]:
        """Start the layout's stand-in (config engine_stand_ins, DESIGN.md B2) when it replaces an engine this lease
        evicts and has passed its eval (eval_passed set). None when there is none, or it did not come up."""
        name = spec.get("stand_in")
        si = (self.cfg_fn().get("engine_stand_ins") or {}).get(name or "") or {}
        if not name or si.get("replaces") not in roles:
            return None
        if not si.get("eval_passed"):
            self._record("stand_in_skipped", stand_in=name, why="eval_passed is not set")
            return None
        if ctl is not None:
            ctl.stand_in_unit = si["unit"]
        res = self.systemctl("start", si["unit"])
        if res.get("ok") and self._wait_up(si.get("url"), STAND_IN_TIMEOUT_S, ctl.cancel if ctl is not None else None):
            self._record("stand_in_up", stand_in=name, unit=si["unit"])
            return {"name": name, "unit": si["unit"], "url": si.get("url"), "replaces": si["replaces"], "up": True}
        self.systemctl("stop", si["unit"])
        if ctl is not None and ctl.cancel.is_set():
            return None                                        # cut short by release(), not a failure of the stand-in
        self._record("stand_in_failed", stand_in=name, why=res.get("error") or "no /health", triggers=["stand_in_failed"])
        return None

    def brain(self, role: str = "coordinator") -> Optional[Dict[str, Any]]:
        """What answers for `role` right now: None (the engine itself), {"mode": "lite", url, until} (a stand-in),
        or {"mode": "on_loan", until} (nothing: Computer says so)."""
        lease = self.lease
        if not lease or role not in (lease.get("evicted_roles") or []):
            return None
        si = lease.get("stand_in") or {}
        if si.get("up") and si.get("replaces") == role:
            return {"mode": "lite", "url": si["url"], "until": lease["expires"], "state": lease.get("state"),
                    "yield_to_computer": bool(lease.get("yield_to_computer"))}
        return {"mode": "on_loan", "until": lease["expires"], "yield_to_computer": bool(lease.get("yield_to_computer"))}

    def yielding(self, role: str = "coordinator") -> Optional[Dict[str, Any]]:
        """A copy of the held lease when it is background work (yield_to_computer), is starting or active, and has `role` lent
        out; else None. What Computer may take back (stonesage/courage/takeback.py)."""
        lease = self.lease
        if lease and lease.get("yield_to_computer") and lease.get("state") in ("starting", "active")                 and role in (lease.get("evicted_roles") or []):
            return dict(lease)
        return None

    def _refuse(self, layout: str, why: str) -> Dict[str, Any]:
        self._record("refused", layout=layout, why=why)
        return {"ok": False, "error": why}

    def _wait_up(self, url: Optional[str], timeout: float = HEALTH_TIMEOUT_S, cancel: Optional[threading.Event] = None) -> bool:
        if not url:
            return True
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            if cancel is not None and cancel.is_set():
                return False
            if self.http_ok(url.rstrip("/") + "/health") and self.http_ok(url.rstrip("/") + "/props"):
                return True
            self.sleep(5)
        return False

    # ------------------------------------------------------------ renew / end ----
    def renew(self, lease_id: str, approver: str, surface: str, ttl_s: float = DEFAULT_TTL_S) -> Dict[str, Any]:
        with self._lock:
            if not self.lease or self.lease["id"] != lease_id:
                return {"ok": False, "error": "no such lease (it may have ended)"}
            ttl = max(60.0, min(float(ttl_s or DEFAULT_TTL_S), MAX_TTL_S))
            self.lease.update(expires=self.clock() + ttl, checked_in=False)
            self._save()
            self._record("renew", lease_id=lease_id, approver=approver or "unknown", surface=surface, ttl_s=ttl)
            return {"ok": True, "expires": self.lease["expires"], "until": _hhmm(self.lease["expires"])}

    def release(self, lease_id: Optional[str] = None, why: str = "released", approver: str = "") -> Dict[str, Any]:
        """End the lease. A lease that is still starting is cancelled instead: this sets the cancel flag and waits (bounded) for the
        start sequence to put back whatever it started; {"in_progress": True} means that has not finished yet. Never "released"
        while the start sequence could still evict something."""
        with self._lock:
            ctl, held = self._starting, self.lease
            if not held or (lease_id and held["id"] != lease_id):
                return {"ok": False, "error": "no such lease"}
            cancelling = ctl if (ctl is not None and ctl.lease_id == held["id"]) else None
            if cancelling is not None:
                cancelling.cancel.set()
        if cancelling is not None:
            if not cancelling.done.wait(self.cancel_wait_s):
                return {"ok": False, "in_progress": True, "cancelling": True,
                        "error": "the lease is still starting; its start is being cancelled and the engines put back"}
            r = cancelling.result
            if not r:                                           # the start ended on its own first (failed and reverted)
                return self.release(lease_id, why=why, approver=approver) if self.lease else {"ok": True, "reverted": []}
            return {"ok": not r.get("failed"), "cancelled_while_starting": True, "stage": r.get("stage"), "undone": r.get("undone"),
                    "failed": r.get("failed") or []}
        with self._op:
            with self._lock:
                if not self.lease or (lease_id and self.lease["id"] != lease_id):
                    return {"ok": False, "error": "no such lease"}
            return self._revert(why, approver=approver)

    def _revert(self, why: str, approver: str = "", event: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        """Start the evicted engines again (Conflicts= stops the layout) and check they answer. Caller holds _op."""
        lease = self.lease or {}
        failed = []
        for unit in lease.get("evicts") or []:
            res = self.systemctl("start", unit)
            if not res.get("ok"):
                failed.append(f"{unit}: {res.get('error')}")
        for e in lease.get("evicted") or []:
            if e.get("url") and not self._wait_up(e["url"]):
                failed.append(f"{e['unit']}: no /health")
        if lease.get("unit"):
            self.systemctl("stop", lease["unit"])      # normally already stopped by Conflicts=; this makes sure
        si = lease.get("stand_in") or {}
        if si.get("up") and not failed:
            self.systemctl("stop", si["unit"])         # the real engine answers again: the stand-in can go
        if failed:
            event_out = "revert_failed"
        else:
            event_out = event or ("expire" if why == "expired" else "release")
        self._record(event_out, lease_id=lease.get("id"), layout=lease.get("layout"), why=why, approver=approver or None,
                     held_min=round((self.clock() - lease.get("started", self.clock())) / 60, 1), failed=failed or None, **extra)
        if failed and self.push:
            try:
                self.push("Computer: engines did not come back", f"After the {lease.get('layout')} lease: "
                          + "; ".join(failed)[:300], None, "engine-lease")
            except Exception:
                pass
        if failed and si.get("up"):
            # The coordinator did not come back: keep the stand-in answering (Computer stays on the spare brain, not
            # dark) and keep the lease record, so nothing new is leased; a person retries with release.
            lease["state"] = "revert_failed"
            with self._lock:
                self.lease = lease
        else:
            with self._lock:
                self.lease = None
        self._save()
        return {"ok": not failed, "reverted": lease.get("evicts"), "failed": failed}

    # ------------------------------------------------------------------ timers ----
    def tick(self) -> None:
        """Check-in before the end, end when the time is up."""
        lease = self.lease
        if not lease or lease.get("state") != "active":
            return
        with self._op:
            lease = self.lease
            if not lease or lease.get("state") != "active":
                return
            left = lease["expires"] - self.clock()
            if left <= 0:
                self._revert("expired")
                return
            if left <= CHECKIN_BEFORE_S and not lease.get("checked_in"):
                lease["checked_in"] = True
                self._save()
                if self.push:
                    try:
                        self.push("Computer: engine lease ending",
                                  f"{lease['layout']} ({lease.get('reason') or 'no reason given'}) ends at "
                                  f"{_hhmm(lease['expires'])}. Keep it another hour?",
                                  [{"action": f"{RENEW}{lease['id']}", "title": "Another hour"},
                                   {"action": f"{STOP}{lease['id']}", "title": "Stop"}], f"lease-{lease['id']}")
                    except Exception:
                        pass
                self._record("checkin", lease_id=lease["id"], minutes_left=round(left / 60, 1))

    def handle_action(self, action: str) -> Optional[str]:
        """A tapped check-in button on the phone (hooked into push_approvals). None when it isn't ours."""
        if action.startswith(RENEW):
            res = self.renew(action[len(RENEW):], "owner", "phone")
            return f"lease renewed until {res['until']}" if res.get("ok") else res["error"]
        if action.startswith(STOP):
            res = self.release(action[len(STOP):], why="stopped from the phone", approver="owner")
            return "lease stopped, engines back" if res.get("ok") else f"stop: {res.get('error') or res.get('failed')}"
        return None

    def recover(self) -> Optional[Dict[str, Any]]:
        """At start: a lease left on disk by a previous StoneSage run is an orphan; revert it."""
        if not self.path:
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                self.lease = json.load(f)
        except (OSError, ValueError):
            self.lease = None
        if not self.lease:
            return None
        with self._op:
            out = self._revert("orphaned by a StoneSage restart")
        return out

    # --------------------------------------------------------------- reconcile ----
    def _active(self, unit: str) -> Optional[bool]:
        """Is the unit active? None when it cannot be told (the host did not answer: an error text comes back)."""
        res = self.systemctl("is-active", unit)
        if res.get("ok"):
            return True
        return None if res.get("error") else False

    def reconcile(self, immediate: bool = False) -> Optional[Dict[str, Any]]:
        """Self-healing: with NO lease held, a layout unit must be stopped and every engine a layout can evict must be running.
        If not (a start that was cut short, a crash, a lease record lost), put it back: stop the strays (and the stand-ins they
        used), start the engines, check /health. A stray is acted on only when seen twice `reconcile_confirm_s` apart (a unit
        mid-restart is not a stray) unless `immediate`. Never touches a held lease or a lease operation in progress.
        Returns what it did, or None."""
        if self.lease or self._starting is not None:
            self._suspect = self._incident = None
            return None
        if not self._op.acquire(blocking=False):
            return None
        try:
            if self.lease:
                return None
            layouts = self.layouts()
            layout_units = sorted({v["unit"] for v in layouts.values() if v.get("unit")})
            engine_units = sorted({u for v in layouts.values() for u in (v.get("evicts") or [])})
            stray = [u for u in layout_units if self._active(u) is True]
            down = [u for u in engine_units if self._active(u) is False]
            if not stray and not down:
                self._suspect = self._incident = None
                return None
            sig = "|".join(stray) + ">" + "|".join(down)
            now = self.clock()
            if not immediate and not (self._suspect and self._suspect["sig"] == sig
                                      and now - self._suspect["at"] >= self.reconcile_confirm_s):
                if not (self._suspect and self._suspect["sig"] == sig):
                    self._suspect = {"sig": sig, "at": now}
                return None
            stand_ins = self.cfg_fn().get("engine_stand_ins") or {}
            stray_specs = [v for v in layouts.values() if v.get("unit") in stray]
            si_units = sorted({stand_ins[v["stand_in"]]["unit"] for v in stray_specs
                               if v.get("stand_in") in stand_ins and stand_ins[v["stand_in"]].get("unit")})
            failed: List[str] = []
            for u in stray:
                res = self.systemctl("stop", u)
                if not res.get("ok"):
                    failed.append(f"{u}: {res.get('error')}")
            for u in sorted(set(down) | set(e for v in stray_specs for e in (v.get("evicts") or []))):
                res = self.systemctl("start", u)
                if not res.get("ok"):
                    failed.append(f"{u}: {res.get('error')}")
            for e in self._roles_of(sorted(set(down) | set(e for v in stray_specs for e in (v.get("evicts") or [])))):
                if e.get("url") and not self._wait_up(e["url"]):
                    failed.append(f"{e['unit']}: no /health")
            if not failed:
                for u in si_units:
                    self.systemctl("stop", u)               # the real engine answers again: the spare brain can go
            rec = self._record("reconcile", stray_layouts=stray, engines_down=down, stand_ins_stopped=[] if failed else si_units,
                               failed=failed or None, triggers=["revert_failed"] if failed else [])
            self.last_reconcile = rec
            self._suspect = None
            if self.push and not self._incident:            # one note until things are clean again
                self._incident = sig
                try:
                    self.push("Computer: engines were out of place",
                              ("Found no lease but %s. Put them back%s." % (
                                  "; ".join(filter(None, [("running: " + ", ".join(stray)) if stray else "",
                                                          ("stopped: " + ", ".join(down)) if down else ""])),
                                  (" -- but " + "; ".join(failed)[:200]) if failed else "")), None, "engine-lease")
                except Exception:
                    pass
            return rec
        finally:
            self._op.release()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def first() -> None:
            self.recover()
            try:
                self.reconcile(immediate=True)
            except Exception:
                pass

        threading.Thread(target=first, name="engine-lease-recover", daemon=True).start()

        def loop():
            while True:
                try:
                    self.tick()
                    self.reconcile()
                except Exception:
                    pass
                self.sleep(30)

        self._thread = threading.Thread(target=loop, name="engine-lease", daemon=True)
        self._thread.start()
