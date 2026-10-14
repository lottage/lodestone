"""
Idle gate (Phase 6, 2026-09-27): may a GPU be taken away from the house right now?

"Idle is not free" (docs/handoff-2026-09-25-stack-limits/REVIEW.md): both GPUs are full of loaded models, so using one
for anything else (a fine-tune, the local loop's engine lease in docs/plan-2026-09-26-local-loop/DESIGN.md) stops the
engines on it, and Computer or vision goes offline for the run. The gate is open for a GPU only when all hold:

  GPU        busy < busy_max % averaged over 5 min (Prometheus stonesage:gpu_busy_percent:avg5m, by PCI slot)
  engines    every engine on that GPU served no request for 5 min (stonesage:engine_idle; not scraped = closed)
  approvals  no approval is waiting for a yes/no
  Computer   no turn in the last recent_turn_min
  household  nobody awake who might want the house brain (commentary.Commentary.household: phone in use, watch
             steps, just got home or seen -> awake; home and unknown -> the house quiet for idle_min)
  cockpit    nobody is using the StoneSage page (stonesage/activity.py: a heartbeat from the page, or any signed-in
             request that is not a poll, within the window; default 10 min). A shared reason like the others, kept
             separately as `cockpit` and as each GPU's `open_ignoring_cockpit`: a lease a PERSON asks for from that
             very page must not be refused because he is on it (engine_lease.acquire); background work (the garden) is.

Two gates since 2026-10-14 (the owner: "no user activity = open; the garden may run for the whole movie"):
  DEMAND gate (`demand_open`, `demand_shared_reasons`)  hardware demand only: GPU busy, engines idle, no approval waiting, no
             Computer turn in computer_turn_min (default 2). No household, no schedule, no human presence. The garden
             and every background lease (yield_to_computer) use this one. The cockpit signal is kept apart (`cockpit`,
             `demand_open_with_cockpit`) because it applies to garden leases (all of which yield to Computer) but not to a lease a person asks for.
  FULL gate (`open`, `shared_reasons`)  the demand rules with recent_turn_min plus the household signal: what a lease a person
             asks for, and the metric, have always used. The household signal (awake/asleep/quiet) is for human-facing
             features (commentary: whether to speak unprompted), not for lending hardware.

Which engines sit on which GPU comes from the live system profile (fdinfo), never from code. No clock anywhere
(the owner: no set sleep schedule). The gate only reports; nothing acts on it yet (the engine lease will).
"""

import time
from typing import Any, Callable, Dict, List, Optional

DEFAULTS = {"busy_max": 10, "recent_turn_min": 15, "computer_turn_min": 2}
CACHE_S = 20
GATE_VERSION = 2          # 1 = the old rule (household signals closed the gate); 2 = hardware demand only for background work


class IdleGate:
    def __init__(self, cfg_fn: Callable[[], Dict[str, Any]], profile_fn: Callable[[], Dict[str, Any]],
                 prom_query: Callable[[str], List[Dict[str, Any]]], household_fn: Callable[[], Dict[str, Any]],
                 pending_fn: Callable[[], int], last_turn_fn: Callable[[], Optional[float]],
                 clock: Callable[[], float] = time.time, activity_fn: Optional[Callable[[], Optional[str]]] = None):
        """activity_fn() -> "someone is using StoneSage (N s ago)" while a person is on the cockpit, else None."""
        self.activity_fn = activity_fn
        self.cfg_fn, self.profile_fn, self.prom_query = cfg_fn, profile_fn, prom_query
        self.household_fn, self.pending_fn, self.last_turn_fn, self.clock = household_fn, pending_fn, last_turn_fn, clock
        self._cache: Optional[Dict[str, Any]] = None

    def _by(self, expr: str, label: str) -> Dict[str, float]:
        out = {}
        for r in self.prom_query(expr) or []:
            try:
                out[r["metric"][label]] = float(r["value"][1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return out

    def check(self, fresh: bool = False) -> Dict[str, Any]:
        now = self.clock()
        if not fresh and self._cache and now - self._cache["checked_at"] < CACHE_S:
            return self._cache
        cfg = {**DEFAULTS, **(self.cfg_fn().get("idle_gate") or {})}
        shared: List[str] = []
        try:
            busy = self._by("stonesage:gpu_busy_percent:avg5m", "pci")
            idle = self._by("stonesage:engine_idle", "engine")
        except Exception as e:
            busy, idle = {}, {}
            shared.append(f"no metrics from Prometheus ({type(e).__name__})")
        pending = self.pending_fn()
        if pending:
            shared.append(f"{pending} approval(s) waiting")
        last = self.last_turn_fn()
        if last and now - last < float(cfg["recent_turn_min"]) * 60:
            shared.append(f"Computer answered {round((now - last) / 60)} min ago")
        demand_shared = [r for r in shared if not r.startswith("Computer answered")]
        if last and now - last < float(cfg["computer_turn_min"]) * 60:
            demand_shared.append(f"Computer answered {round((now - last) / 60)} min ago")
        try:
            house = self.household_fn()
        except Exception:
            house = {"quiet": True, "reasons": []}
        shared += house.get("reasons") or []
        cockpit = None
        if self.activity_fn:
            try:
                cockpit = self.activity_fn()
            except Exception:
                cockpit = None
        shared_core = list(shared)
        if cockpit:
            shared.append(cockpit)
        gpus = {}
        for g in self.profile_fn().get("gpus") or []:
            name, reasons = g.get("short") or g.get("name") or g.get("pci"), []
            b = busy.get(g.get("pci"))
            if b is None:
                reasons.append("GPU busy % not reported")
            elif b >= float(cfg["busy_max"]):
                reasons.append(f"GPU {b:.0f} % busy")
            for eng in g.get("engines") or []:
                if eng not in idle:
                    reasons.append(f"{eng}: no metrics")
                elif idle[eng] < 1:
                    reasons.append(f"{eng} served a request in the last 5 min")
            gpus[name] = {"open": not reasons and not shared, "engines": g.get("engines") or [],
                          "busy_percent": b, "reasons": reasons, "open_ignoring_cockpit": not reasons and not shared_core,
                          "demand_open": not reasons and not demand_shared,
                          "demand_open_with_cockpit": not reasons and not demand_shared and not cockpit}
        self._cache = {"open": any(v["open"] for v in gpus.values()), "demand_open": any(v["demand_open"] for v in gpus.values()),
                       "demand_shared_reasons": demand_shared, "gate_version": GATE_VERSION,
                       "gpus": gpus, "shared_reasons": shared, "cockpit": cockpit, "household": house, "pending_approvals": pending, "checked_at": now}
        return self._cache

    def metrics(self) -> List[str]:
        """Prometheus lines: 1 when that GPU could be lent out now."""
        c = self.check()
        lines = ["# HELP stonesage_idle_gate_open 1 when the idle gate would lend this GPU out now.",
                 "# TYPE stonesage_idle_gate_open gauge"]
        for name, g in sorted(c["gpus"].items()):
            lines.append(f'stonesage_idle_gate_open{{gpu="{name}"}} {1 if g["open"] else 0}')
        lines += ["# HELP stonesage_idle_gate_demand_open 1 when hardware demand alone (no household signal) would lend this GPU out.",
                  "# TYPE stonesage_idle_gate_demand_open gauge"]
        for name, g in sorted(c["gpus"].items()):
            lines.append(f'stonesage_idle_gate_demand_open{{gpu="{name}"}} {1 if g["demand_open"] else 0}')
        lines += ["# HELP stonesage_household_quiet 1 when nobody seems awake who might want the house brain.",
                  "# TYPE stonesage_household_quiet gauge",
                  f"stonesage_household_quiet {1 if c['household'].get('quiet') else 0}"]
        return lines
