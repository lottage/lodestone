"""
Boost: free extra inference for Courage and the agents (plan 2026-09-24).

    providers.py  the free sources (Groq, Cloudflare, Gemini free, OpenCode Zen free, OpenRouter :free, edge nodes)
    quota.py      per-source free-tier accounting, 429 cooldowns, background share
    egress.py     what may leave the house (tiered: local / no_training / training)
    router.py     picks a source per call, falls back across sources, then to the local coordinator
    frontier.py   client for the Claude/Gemini CLI job runner on VM 102

The server calls `configure()` once; everything else uses `get_router()`.
"""

import threading
from typing import Any, Callable, Dict, Iterable, Optional, Union

from .router import BoostRouter, iter_sse

_router: Optional[BoostRouter] = None
_lock = threading.Lock()
_setup: Dict[str, Any] = {}


def configure(config_fn: Callable[[], Dict[str, Any]], data_dir: Optional[str],
              home_terms: Union[Iterable[str], Callable[[], Iterable[str]]] = (),
              on_call: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    """home_terms: fixed names, or a function read on every call (profiles and family change while running).
    on_call: receives one metadata-only record per call (the StoneSage trace); set on every router made here."""
    global _router
    with _lock:
        _setup.update(config_fn=config_fn, data_dir=data_dir,
                      home_terms=home_terms if callable(home_terms) else tuple(home_terms), on_call=on_call)
        _router = None


def get_router() -> Optional[BoostRouter]:
    """The shared router, or None before configure() (e.g. in the harness process)."""
    global _router
    with _lock:
        if _router is None and _setup:
            _router = BoostRouter(_setup["config_fn"], _setup["data_dir"], home_terms=_setup["home_terms"])
            _router.on_call = _setup.get("on_call")
        return _router


__all__ = ["BoostRouter", "configure", "get_router", "iter_sse"]
