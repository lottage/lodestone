"""Where each service lives: one table, so no code names a LAN address (docs/LAYERS.md M2).

A service's address comes from config.json (the first of its paths that is set), else its environment variable, else a
default. Defaults are for one box running everything (localhost); a spread-out install says where things are in
config.json. `missing(cfg)` lists what is still on a default and is needed by this install, and the server warns about
it at start, so a forgotten key shows up instead of quietly pointing at localhost. Optional services (OPTIONAL) are
not warned about unless the config has a section of its own for them; `missing(cfg, optional=True)` lists the rest
(the server logs those at INFO level).

    endpoints.url("coordinator", cfg)          -> "http://10.0.0.5:8001/v1" (from cluster.coordinator_url)
    endpoints.host("valkey", cfg), endpoints.port("valkey", cfg)
    endpoints.ssh_target(cfg)                  -> "user@engine-host"  (cluster.ssh_user @ the coordinator's host)
    endpoints.ssh_cmd(cfg, "systemctl ...")    -> ["ssh", "-o", "BatchMode=yes", ..., "--", "user@host", "systemctl ..."]
"""

import getpass
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from lodestone.system_profile import load_cfg

# name: (config paths, first set wins; environment variable; default). An empty default = not part of a basic install
# (callers treat "" as "not configured").
SERVICES: Dict[str, Tuple[Tuple[Tuple[str, ...], ...], str, str]] = {
    "coordinator": ((("cluster", "coordinator_url"),), "COORDINATOR_URL", "http://127.0.0.1:8001/v1"),
    "worker": ((("cluster", "worker_url"),), "WORKER_URL", "http://127.0.0.1:8002/v1"),
    "embedder": ((("cluster", "embedder_url"),), "EMBEDDER_URL", "http://127.0.0.1:8003/v1"),
    "vision": ((("cluster", "vision_url"),), "VISION_URL", "http://127.0.0.1:8004/v1"),
    "mcp": ((("cluster", "mcp_url"),), "CLUSTER_MCP_URL", "http://127.0.0.1:8765"),
    "qdrant": ((("cluster", "qdrant_url"), ("qdrant", "url")), "QDRANT_URL", "http://127.0.0.1:6333"),
    "valkey": ((("cluster", "valkey_url"), ("cluster", "valkey_host")), "VALKEY_URL", "redis://127.0.0.1:6379"),
    "homeassistant": ((("homeassistant", "url"), ("hass", "url")), "HASS_URL", "http://homeassistant.local:8123"),
    "couchdb": ((("couchdb", "url"),), "COUCHDB_URL", "http://127.0.0.1:5984"),
    "prometheus": ((("metrics", "prometheus_url"),), "PROMETHEUS_URL", "http://127.0.0.1:9090"),
    "frontier": ((("frontier", "url"),), "FRONTIER_URL", "http://127.0.0.1:8770"),
    "voice": ((("voice", "stt_url"),), "VOICE_URL", "http://127.0.0.1:8210"),
    "voice_legacy": ((("voice", "legacy_host"), ("voice", "host")), "VOICE_LEGACY_HOST", ""),
    "proxmox": ((("proxmox", "url"), ("proxmox", "host")), "PROXMOX_URL", ""),
    "immich": ((("immich", "url"),), "IMMICH_URL", ""),
    "freshrss": ((("freshrss", "url"),), "FRESHRSS_URL", ""),
    "kavita": ((("kavita", "url"),), "KAVITA_URL", ""),
    "gemini_web": ((("gemini_web", "endpoint"),), "GEMINI_WEB_URL", "http://127.0.0.1:8087"),
    "stonesage": ((("server", "public_url"), ("boost", "stonesage_url")), "STONESAGE_URL", "http://127.0.0.1:8888"),
    "blender": ((("blender", "url"),), "BLENDER_URL", ""),          # shelved (Citadel 3D)
    "assembly": ((("cluster", "assembly_url"),), "ASSEMBLY_HTTP", "http://127.0.0.1:8766"),   # shelved
}

# Services a basic install does without. Never warned about for sitting on their localhost default; a service in this
# set counts as used once config.json has a section of its own for it (voice, couchdb, qdrant, frontier, ...).
# Not in the set: the two core engines (coordinator, embedder) always, and homeassistant while the Hearth app is loaded.
OPTIONAL = frozenset({"worker", "vision", "mcp", "qdrant", "valkey", "couchdb", "prometheus", "frontier", "voice",
                      "voice_legacy", "proxmox", "immich", "freshrss", "kavita", "gemini_web", "stonesage", "blender",
                      "assembly", "homeassistant"})
_HOME_APPS = ("hearth", "home")
_SHARED_SECTIONS = ("cluster", "server", "boost")      # sections that exist in every install, whatever is used

# A bare host in config ("10.0.0.5", "nas.local:8123") borrows the scheme and port of the service's default.
_SCHEMES = {"valkey": "redis", "voice_legacy": "http", "proxmox": "https"}
_PORTS = {"valkey": 6379, "proxmox": 8006}


def _cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return load_cfg() if cfg is None else cfg


def _configured(name: str, cfg: Dict[str, Any]) -> str:
    paths, env, _ = SERVICES[name]
    for path in paths:
        node: Any = cfg
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, str) and node.strip():
            return node.strip()
    return os.environ.get(env, "").strip()


def _normalise(name: str, value: str) -> str:
    if not value or "://" in value:
        return value.rstrip("/")
    default = urlparse(SERVICES[name][2])
    scheme = _SCHEMES.get(name) or default.scheme or "http"
    port = default.port or _PORTS.get(name)
    has_port = ":" in value.split("/")[0]
    return f"{scheme}://{value.rstrip('/')}" + ("" if has_port or not port else f":{port}")


def url(name: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """The service's base URL ('' when it is optional and not configured). cfg: config.json; None = read it."""
    return _normalise(name, _configured(name, _cfg(cfg)) or SERVICES[name][2])


def host(name: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    return urlparse(url(name, cfg)).hostname or ""


def port(name: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    u = urlparse(url(name, cfg))
    return u.port or _PORTS.get(name) or {"http": 80, "https": 443}.get(u.scheme)


def _has_section(name: str, cfg: Dict[str, Any]) -> bool:
    """True when config.json has a section of its own for the service (not the shared 'cluster' one)."""
    for path in SERVICES[name][0]:
        if path[0] not in _SHARED_SECTIONS and isinstance(cfg.get(path[0]), dict) and cfg[path[0]]:
            return True
    return False


def needed(name: str, cfg: Dict[str, Any]) -> bool:
    """Does this install use the service? The core engines always; Home Assistant while the Hearth app is loaded;
    any other when config.json has a section for it."""
    if name not in OPTIONAL:
        return True
    if name == "homeassistant":
        apps = cfg.get("apps", ["hearth"])
        return any(a in _HOME_APPS for a in (apps if isinstance(apps, list) else [])) or _has_section(name, cfg)
    return _has_section(name, cfg)


def missing(cfg: Optional[Dict[str, Any]] = None, names: Sequence[str] = (), optional: bool = False) -> List[str]:
    """Services with a default that run on it because config.json and the environment say nothing (warned about at
    start). Only the ones this install needs; optional=True gives the others instead (for an INFO line)."""
    cfg = _cfg(cfg)
    return [n for n in (names or SERVICES)
            if SERVICES[n][2] and not _configured(n, cfg) and needed(n, cfg) is not optional]


# ------------------------------------------------------------------------------------------------ the brain's speed ----
# A coordinator on a CPU answers a few tokens a second and reads a long prompt for minutes: every wait on it is longer.
CPU_LLM_TIMEOUT_S = 600.0
GPU_LLM_TIMEOUT_S = 90.0


def _truthy(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip().lower() in ("1", "true", "yes", "on")
    return None


def coordinator_on_cpu(cfg: Optional[Dict[str, Any]] = None, profile: Optional[Dict[str, Any]] = None) -> bool:
    """Does the coordinator engine run on a CPU? A hint wins: config cluster.coordinator_cpu (true/false), else the
    environment COORDINATOR_CPU (the Docker stack sets it), else the live system profile when it was already read
    (the coordinator running with nothing resident in any GPU's memory), else False."""
    cfg = _cfg(cfg)
    hint = _truthy((cfg.get("cluster") or {}).get("coordinator_cpu"))
    if hint is None:
        hint = _truthy(os.environ.get("COORDINATOR_CPU"))
    if hint is not None:
        return hint
    eng = ((profile or {}).get("engines") or {}).get("coordinator") or {}
    return bool(eng.get("online") and eng.get("offloaded"))


# ------------------------------------------------------------------------------------------------------------ SSH ----
# The engines' host is reached over SSH for unit files, model files and probes: the coordinator's host, as the user in
# cluster.ssh_user. Host keys: "accept-new" records a host's key the first time and refuses a changed one afterwards;
# config ssh.host_key_checking can say "yes" (known_hosts must already have it) or, for a lab, "no".

def ssh_user(cfg: Optional[Dict[str, Any]] = None) -> str:
    return (_cfg(cfg).get("cluster") or {}).get("ssh_user") or os.environ.get("STONESAGE_SSH_USER") or getpass.getuser()


SSH_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:-]{0,252}$")      # a user or host name: never starts with '-' (an ssh option)
HOST_KEY_MODES = ("yes", "accept-new")                                 # "no" (trust anyone) is not offered, even if a config says it


def ssh_target(cfg: Optional[Dict[str, Any]] = None, host_name: str = "") -> str:
    cfg = _cfg(cfg)
    user, where = str(ssh_user(cfg)), str(host_name or host('coordinator', cfg))
    if not SSH_NAME.match(user) or not SSH_NAME.match(where):
        raise ValueError("ssh user or host in the config is not a plain name or address")
    return f"{user}@{where}"


def ssh_options(cfg: Optional[Dict[str, Any]] = None, connect_timeout: Optional[int] = None, batch: bool = True) -> List[str]:
    """batch=False for an interactive session (a terminal): it may ask for a password."""
    checking = (_cfg(cfg).get("ssh") or {}).get("host_key_checking") or "accept-new"
    if checking not in HOST_KEY_MODES:
        checking = "accept-new"
    opts = (["-o", "BatchMode=yes"] if batch else []) + ["-o", f"StrictHostKeyChecking={checking}"]
    return opts + (["-o", f"ConnectTimeout={int(connect_timeout)}"] if connect_timeout else [])


def check_ssh_target(target: str) -> str:
    """The target back when it is a plain `user@host` (or `host`); ValueError when either part could be read as an ssh option."""
    user, _, where = str(target).rpartition("@")
    if not SSH_NAME.match(where) or (user and not SSH_NAME.match(user)):
        raise ValueError("ssh target is not a plain user@host")
    return str(target)


def ssh_cmd(cfg: Optional[Dict[str, Any]], *remote: str, target: str = "", no_stdin: bool = False,
            connect_timeout: Optional[int] = None) -> List[str]:
    """argv for running `remote` on the engines' host (or `target`, "user@host")."""
    cfg = _cfg(cfg)
    if target:
        check_ssh_target(target)
    # `--` ends ssh's options, so nothing in the target can be read as one
    return ["ssh"] + (["-n"] if no_stdin else []) + ssh_options(cfg, connect_timeout) + ["--", target or ssh_target(cfg)] + list(remote)
