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
STAND_IN_TIMEOUT_S = 120     # Courage-lite on CPU answers in ~6 s; a stand-in that takes this long is not starting
RENEW, STOP = "LEASE_RENEW_", "LEASE_STOP_"


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


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
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ config ----
    def layouts(self) -> Dict[str, Dict[str, Any]]:
        return self.cfg_fn().get("engine_layouts") or {}

    def _roles_of(self, units: List[str]) -> List[Dict[str, Any]]:
        known = self.unit_roles()
        return [dict(known.get(u) or {}, unit=u) for u in units]

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
                force: bool = False) -> Dict[str, Any]:
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
                closed = [f"{g}: {', '.join(gate['gpus'].get(g, {}).get('reasons') or gate.get('shared_reasons') or ['closed'])}"
                          for g in sorted(gpus) if not (gate.get("gpus", {}).get(g) or {}).get("open")]
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
                     "ttl_s": ttl, "checked_in": False, "state": "starting"}
            self.lease = lease
            self._save()
        lease["stand_in"] = self._start_stand_in(spec, roles)   # before the coordinator goes: Computer never goes dark
        self._save()
        res = self.systemctl("start", spec["unit"])
        up = res.get("ok") and self._wait_up(spec.get("url"))
        with self._lock:
            if self.lease is not lease:                # released while it was still starting
                return {"ok": False, "error": "the lease was released while starting"}
            if not up:
                self._revert(f"{spec['unit']} did not come up: {res.get('error') or 'no /health within timeout'}")
                return {"ok": False, "error": f"{spec['unit']} did not come up; engines reverted"}
            lease["state"] = "active"
            lease["expires"] = self.clock() + ttl          # the clock starts once it answers
            self._save()
            self._record("acquire", lease_id=lease["id"], layout=layout, evicts=lease["evicts"], ttl_s=ttl,
                         reason=lease["reason"], approver=lease["approver"], forced=force)
            return {"ok": True, "lease_id": lease["id"], "url": lease["url"], "expires": lease["expires"],
                    "until": _hhmm(lease["expires"])}

    def _start_stand_in(self, spec: Dict[str, Any], roles: List[str]) -> Optional[Dict[str, Any]]:
        """Start the layout's stand-in (config engine_stand_ins, DESIGN.md B2) when it replaces an engine this lease
        evicts and has passed its eval (eval_passed set). None when there is none, or it did not come up."""
        name = spec.get("stand_in")
        si = (self.cfg_fn().get("engine_stand_ins") or {}).get(name or "") or {}
        if not name or si.get("replaces") not in roles:
            return None
        if not si.get("eval_passed"):
            self._record("stand_in_skipped", stand_in=name, why="eval_passed is not set")
            return None
        res = self.systemctl("start", si["unit"])
        if res.get("ok") and self._wait_up(si.get("url"), STAND_IN_TIMEOUT_S):
            self._record("stand_in_up", stand_in=name, unit=si["unit"])
            return {"name": name, "unit": si["unit"], "url": si.get("url"), "replaces": si["replaces"], "up": True}
        self.systemctl("stop", si["unit"])
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
            return {"mode": "lite", "url": si["url"], "until": lease["expires"], "state": lease.get("state")}
        return {"mode": "on_loan", "until": lease["expires"]}

    def _refuse(self, layout: str, why: str) -> Dict[str, Any]:
        self._record("refused", layout=layout, why=why)
        return {"ok": False, "error": why}

    def _wait_up(self, url: Optional[str], timeout: float = HEALTH_TIMEOUT_S) -> bool:
        if not url:
            return True
        deadline = self.clock() + timeout
        while self.clock() < deadline:
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
        with self._lock:
            if not self.lease or (lease_id and self.lease["id"] != lease_id):
                return {"ok": False, "error": "no such lease"}
            return self._revert(why, approver=approver)

    def _revert(self, why: str, approver: str = "") -> Dict[str, Any]:
        """Start the evicted engines again (Conflicts= stops the layout) and check they answer. Caller holds the lock."""
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
        event = "revert_failed" if failed else ("expire" if why == "expired" else "release")
        self._record(event, lease_id=lease.get("id"), layout=lease.get("layout"), why=why, approver=approver or None,
                     held_min=round((self.clock() - lease.get("started", self.clock())) / 60, 1), failed=failed or None)
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
            self.lease = lease
        else:
            self.lease = None
        self._save()
        return {"ok": not failed, "reverted": lease.get("evicts"), "failed": failed}

    # ------------------------------------------------------------------ timers ----
    def tick(self) -> None:
        """Check-in before the end, end when the time is up."""
        with self._lock:
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
        with self._lock:
            return self._revert("orphaned by a StoneSage restart")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        threading.Thread(target=self.recover, name="engine-lease-recover", daemon=True).start()

        def loop():
            while True:
                try:
                    self.tick()
                except Exception:
                    pass
                self.sleep(30)

        self._thread = threading.Thread(target=loop, name="engine-lease", daemon=True)
        self._thread.start()
