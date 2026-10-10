"""
Per-provider free-tier accounting: requests per minute, requests and tokens per rolling day, 429 cooldowns,
and the background share (loops/workspaces may use at most `loop_share` of each day's requests, so Courage
and chat always keep headroom).

Rolling 24 h windows are conservative (providers reset at midnight UTC or on their own schedule).
State survives restarts in data/boost_quota.json.
"""

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

BACKGROUND_SURFACES = ("loops", "workspaces")
DAY = 86400.0


class QuotaBook:
    def __init__(self, path: Optional[str] = None, loop_share: float = 0.5, clock: Callable[[], float] = time.time):
        self.path = path
        self.loop_share = loop_share
        self.clock = clock
        self._lock = threading.Lock()
        self._events: Dict[str, List[List[Any]]] = {}   # provider -> [[ts, tokens, surface], ...] within the last day
        self._cooldown: Dict[str, float] = {}           # provider -> unix time it may be used again
        self._last_error: Dict[str, str] = {}
        self._dirty_at = 0.0
        self._load()

    # ---- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._events = {k: [list(e) for e in v] for k, v in (data.get("events") or {}).items()}
            self._cooldown = {k: float(v) for k, v in (data.get("cooldown") or {}).items()}
        except Exception:
            self._events, self._cooldown = {}, {}

    def _save(self, force: bool = False) -> None:
        if not self.path:
            return
        now = self.clock()
        if not force and now - self._dirty_at < 10:
            return
        self._dirty_at = now
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"events": self._events, "cooldown": self._cooldown}, f)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- accounting ----------------------------------------------------------
    def _prune(self, pid: str, now: float) -> List[List[Any]]:
        ev = [e for e in self._events.get(pid, []) if now - e[0] < DAY]
        self._events[pid] = ev
        return ev

    def record(self, pid: str, tokens: int, surface: str) -> None:
        with self._lock:
            now = self.clock()
            self._prune(pid, now).append([now, int(tokens or 0), surface])
            self._last_error.pop(pid, None)
            self._save()

    def cool_down(self, pid: str, seconds: float, reason: str = "") -> None:
        with self._lock:
            self._cooldown[pid] = max(self._cooldown.get(pid, 0.0), self.clock() + max(5.0, seconds))
            if reason:
                self._last_error[pid] = reason
            self._save(force=True)

    def note_error(self, pid: str, reason: str) -> None:
        with self._lock:
            self._last_error[pid] = reason

    def check(self, pid: str, limits: Dict[str, int], surface: str, est_tokens: int = 0) -> Optional[str]:
        """None if the provider may take this call now, else the reason it may not."""
        with self._lock:
            now = self.clock()
            until = self._cooldown.get(pid, 0.0)
            if until > now:
                return f"cooling down {int(until - now)} s"
            ev = self._prune(pid, now)
            rpm, rpd, tpd = limits.get("rpm") or 0, limits.get("rpd") or 0, limits.get("tpd") or 0
            if rpm and sum(1 for e in ev if now - e[0] < 60) >= rpm:
                return "per-minute limit"
            if rpd and len(ev) >= rpd:
                return "daily request limit"
            if tpd and sum(e[1] for e in ev) + est_tokens > tpd:
                return "daily token limit"
            if rpd and surface in BACKGROUND_SURFACES:
                bg = sum(1 for e in ev if e[2] in BACKGROUND_SURFACES)
                if bg >= int(rpd * self.loop_share):
                    return f"background share ({int(self.loop_share * 100)}%) used"
            return None

    def usage(self, pid: str, limits: Dict[str, int]) -> Dict[str, Any]:
        with self._lock:
            now = self.clock()
            ev = self._prune(pid, now)
            until = self._cooldown.get(pid, 0.0)
            return {
                "requests_today": len(ev),
                "tokens_today": sum(e[1] for e in ev),
                "background_today": sum(1 for e in ev if e[2] in BACKGROUND_SURFACES),
                "last_minute": sum(1 for e in ev if now - e[0] < 60),
                "rpd": limits.get("rpd") or None, "tpd": limits.get("tpd") or None, "rpm": limits.get("rpm") or None,
                "cooldown_s": int(until - now) if until > now else 0,
                "last_error": self._last_error.get(pid),
            }

    def flush(self) -> None:
        with self._lock:
            self._save(force=True)
