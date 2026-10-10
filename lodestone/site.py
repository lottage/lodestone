"""This installation's place in the world: its time zone (docs/LAYERS.md M2: no time zone written in code).

The zone is config.json `server.timezone` (an IANA name such as "Europe/London"), else the TZ environment variable
(a service can set it in its unit, e.g. a systemd drop-in), else the machine's current offset. The first two
give a real zone that follows daylight saving; the last is a fixed offset, right until the clocks change.
"""

import logging
import os
from datetime import datetime, tzinfo
from typing import Any, Dict, Optional

from lodestone.system_profile import load_cfg

logger = logging.getLogger("stonesage.site")


def tz_name(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The IANA name in use, or "" when only the machine's offset is known."""
    cfg = load_cfg() if cfg is None else cfg
    return ((cfg.get("server") or {}).get("timezone") or os.environ.get("TZ") or "").strip().lstrip(":")


def local_tz(cfg: Optional[Dict[str, Any]] = None) -> tzinfo:
    name = tz_name(cfg)
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception as e:     # unknown name, or no tz database on this machine
            logger.warning("time zone %r not usable (%s); using the machine's offset", name, e)
    return datetime.now().astimezone().tzinfo


def obsidian_vault(cfg: Optional[Dict[str, Any]] = None) -> str:
    """The user's Obsidian vault: config.json obsidian.user_vault_path, else OBSIDIAN_VAULT / OBSIDIAN_VAULT_PATH,
    else "" (no vault configured: callers skip it)."""
    cfg = load_cfg() if cfg is None else cfg
    return ((cfg.get("obsidian") or {}).get("user_vault_path") or os.environ.get("OBSIDIAN_VAULT")
            or os.environ.get("OBSIDIAN_VAULT_PATH") or "")


def describe(cfg: Optional[Dict[str, Any]] = None) -> str:
    """'Europe/London (BST)' for status pages."""
    abbr = datetime.now(local_tz(cfg)).strftime("%Z")
    name = tz_name(cfg)
    return f"{name} ({abbr})" if name and abbr and abbr != name else (name or abbr)
