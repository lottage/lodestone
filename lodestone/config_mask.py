"""
Keep secrets out of GET /api/config (it used to return every API key and token in config.json as-is), and keep a settings
form from sending a stored secret somewhere new.

mask()                  copy of the config with secret-looking string values replaced by "••••" + last 4 characters, and
                        the password of a URL such as redis://user:pass@host masked
drop_masked()           removes masked values from a POSTed update, so saving a form that echoes them back
                        never overwrites the real secret with its mask
destination_errors()    an update that changes WHERE something is sent (a url / host / base_url / tier) next to a stored
                        secret must carry that secret again in the same request; otherwise the stored key would silently
                        follow the new address (an attacker's server, or Boost treating it as local)
protected_keys()        top-level sections a settings POST may not write at all (the login key, feature flags, app list,
                        workspace roots): those are changed in config.json by the person who runs the server
"""

import re
from typing import Any, Dict, List

# What counts as a secret by its KEY: api keys, tokens, passwords, secrets, a bare `key` or `*_key`, session cookies (psid...),
# credentials, private material.
SECRET_KEY = re.compile(r"(api_?key|token|password|passwd|secret|psid|cookie|credential|private|(^|_)key$|^auth$|authorization)", re.I)
# Keys that say where something is sent (or how far it is trusted)
DESTINATION_KEY = re.compile(r"((^|_)(url|uri|host|hostname|endpoint|address|server)$|^tier$)", re.I)
PROTECTED_TOP_LEVEL = ("api", "features", "apps", "workspace_roots")
MASK = "••••"
_URL_USERINFO = re.compile(r"^([a-z][a-z0-9+.-]*://[^/\s:@]*:)([^/\s@]+)(@)", re.I)


def _masked_value(v: str) -> str:
    return MASK + (v[-4:] if len(v) >= 12 else "")


def _mask_url_password(v: str) -> str:
    return _URL_USERINFO.sub(lambda m: m.group(1) + MASK + m.group(3), v)


def mask(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (_masked_value(v) if isinstance(v, str) and v and SECRET_KEY.search(str(k)) else mask(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask(v) for v in obj]
    if isinstance(obj, str):
        return _mask_url_password(obj)
    return obj


def drop_masked(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: drop_masked(v) for k, v in obj.items() if not (isinstance(v, str) and v.startswith(MASK))}
    if isinstance(obj, list):
        return [drop_masked(v) for v in obj]
    return obj


def protected_keys(update: Any) -> List[str]:
    """The top-level keys of a POSTed update that may not be written through the settings API."""
    return [k for k in PROTECTED_TOP_LEVEL if isinstance(update, dict) and k in update]


def _has_secret(d: Dict[str, Any]) -> bool:
    return any(isinstance(v, str) and v.strip() and SECRET_KEY.search(str(k)) for k, v in d.items())


def destination_errors(old: Any, update: Any, path: str = "") -> List[str]:
    """Places in `update` (already passed through drop_masked) that change a destination while `old` holds a secret in the
    same section and the update does not give a new one. One message per place."""
    errors: List[str] = []
    if not isinstance(update, dict):
        return errors
    old = old if isinstance(old, dict) else {}
    for k, v in update.items():
        if isinstance(v, dict):
            errors += destination_errors(old.get(k), v, f"{path}{k}.")
    stored_secret = _has_secret(old)
    given_secret = _has_secret(update)
    for k, v in update.items():
        if isinstance(v, str) and DESTINATION_KEY.search(str(k)) and v.strip() != str(old.get(k) or "").strip():
            if stored_secret and not given_secret:
                errors.append(f"{path}{k} changed: enter the key/token for it again in the same request "
                              f"(a stored secret is never sent to a new address)")
    return errors
