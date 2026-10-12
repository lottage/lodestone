"""Which version of the layer below am I running on? (docs/LAYERS.md 7c, "M5: Backups, upgrades and the rest of the review")

Each layer package carries `__version__` (the release) and `API_VERSION` ("major.minor", what the layer above is written
against). The layer above names the API it was built for and checks it when it starts, so a mismatched install stops with
one clear sentence instead of an AttributeError from the middle of a request.

Compatible means: while the major number is 0, the minor number must be the same ("0.1" runs on "0.1", not on "0.2");
from 1 on, the major number must be the same and the layer below may be newer in its minor number.
Standard library only; imports nothing else of this project."""

import re
from typing import Tuple

_API = re.compile(r"^(\d+)\.(\d+)$")


class LayerMismatch(RuntimeError):
    """The layer below is not one this layer was written for. The message says what to install."""


def parse_api(version: str) -> Tuple[int, int]:
    m = _API.match(str(version or "").strip())
    if not m:
        raise ValueError(f"not an API version (want 'major.minor'): {version!r}")
    return int(m.group(1)), int(m.group(2))


def compatible(have: str, wanted: str) -> bool:
    """Can code written for API `wanted` run on a layer that offers API `have`?"""
    h_major, h_minor = parse_api(have)
    w_major, w_minor = parse_api(wanted)
    if h_major != w_major:
        return False
    if h_major == 0:
        return h_minor == w_minor
    return h_minor >= w_minor


def require(consumer: str, consumer_version: str, layer: str, have: str, wanted: str) -> None:
    """Raise LayerMismatch unless `layer` (offering API `have`) suits `consumer` (written for API `wanted`)."""
    try:
        ok = compatible(have, wanted)
    except ValueError as e:
        raise LayerMismatch(f"{consumer} {consumer_version}: cannot read the {layer} API version ({e})") from e
    if not ok:
        raise LayerMismatch(
            f"{consumer} {consumer_version} was written for {layer} API {wanted}, but the installed {layer} offers API {have}. "
            f"Install the {layer} release that matches {consumer} (see docs/UPGRADING.md); the two are upgraded together.")
