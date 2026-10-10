"""Where each service lives: one table, so no code names a LAN address (docs/LAYERS.md M2).

A service's address comes from config.json (the first of its paths that is set), else its environment variable, else a
default. Defaults are for one box running everything (localhost); a spread-out install says where things are in
config.json. `missing(cfg)` lists what is still on a default, and the server logs it at start, so a forgotten key shows
up instead of quietly pointing at localhost.

    endpoints.url("coordinator", cfg)          -> "http://10.0.0.5:8001/v1" (from cluster.coordinator_url)
    endpoints.host("valkey", cfg), endpoints.port("valkey", cfg)
    endpoints.ssh_target(cfg)                  -> "user@engine-host"  (cluster.ssh_user @ the coordinator's host)
    endpoints.ssh_cmd(cfg, "systemctl ...")    -> ["ssh", "-o", "BatchMode=yes", ..., "user@host", "systemctl ..."]
"""

import getpass
import os
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


def missing(cfg: Optional[Dict[str, Any]] = None, names: Sequence[str] = ()) -> List[str]:
    """Services with a default that run on it because config.json and the environment say nothing (logged at start)."""
    cfg = _cfg(cfg)
    return [n for n in (names or SERVICES) if SERVICES[n][2] and not _configured(n, cfg)]


# ------------------------------------------------------------------------------------------------------------ SSH ----
# The engines' host is reached over SSH for unit files, model files and probes: the coordinator's host, as the user in
# cluster.ssh_user. Host keys: "accept-new" records a host's key the first time and refuses a changed one afterwards;
# config ssh.host_key_checking can say "yes" (known_hosts must already have it) or, for a lab, "no".

def ssh_user(cfg: Optional[Dict[str, Any]] = None) -> str:
    return (_cfg(cfg).get("cluster") or {}).get("ssh_user") or os.environ.get("STONESAGE_SSH_USER") or getpass.getuser()


def ssh_target(cfg: Optional[Dict[str, Any]] = None, host_name: str = "") -> str:
    cfg = _cfg(cfg)
    return f"{ssh_user(cfg)}@{host_name or host('coordinator', cfg)}"


def ssh_options(cfg: Optional[Dict[str, Any]] = None, connect_timeout: Optional[int] = None, batch: bool = True) -> List[str]:
    """batch=False for an interactive session (a terminal): it may ask for a password."""
    checking = (_cfg(cfg).get("ssh") or {}).get("host_key_checking") or "accept-new"
    opts = (["-o", "BatchMode=yes"] if batch else []) + ["-o", f"StrictHostKeyChecking={checking}"]
    return opts + (["-o", f"ConnectTimeout={connect_timeout}"] if connect_timeout else [])


def ssh_cmd(cfg: Optional[Dict[str, Any]], *remote: str, target: str = "", no_stdin: bool = False,
            connect_timeout: Optional[int] = None) -> List[str]:
    """argv for running `remote` on the engines' host (or `target`, "user@host")."""
    cfg = _cfg(cfg)
    return ["ssh"] + (["-n"] if no_stdin else []) + ssh_options(cfg, connect_timeout) + [target or ssh_target(cfg)] + list(remote)
