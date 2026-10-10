"""
Keep secrets out of GET /api/config (it used to return every API key and token in config.json as-is).

mask()        copy of the config with secret-looking string values replaced by "••••" + last 4 characters
drop_masked() removes masked values from a POSTed update, so saving a form that echoes them back
              never overwrites the real secret with its mask
"""

import re
from typing import Any

SECRET_KEY = re.compile(r"(api_?key|token|password|passwd|secret)", re.I)
MASK = "••••"


def _masked_value(v: str) -> str:
    return MASK + (v[-4:] if len(v) >= 12 else "")


def mask(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (_masked_value(v) if isinstance(v, str) and v and SECRET_KEY.search(str(k)) else mask(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask(v) for v in obj]
    return obj


def drop_masked(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: drop_masked(v) for k, v in obj.items() if not (isinstance(v, str) and v.startswith(MASK))}
    if isinstance(obj, list):
        return [drop_masked(v) for v in obj]
    return obj
