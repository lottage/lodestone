"""
What may leave the house (the owner's "tiered" rule, 2026-09-24).

Content classes, most to least sensitive:
    secret   tokens, passwords, API keys: never leave local hardware
    home     presence, HA entities and states, LAN addresses, camera names, family/pet names
    code     source code, diffs, stack traces
    general  anything else

Callers declare a class; `classify()` can only raise it (a deterministic scan, not an LLM judge: small models
grade too leniently, see CLAUDE.md). `egress_allowed()` is the policy on top.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

CLASS_RANK = {"general": 0, "code": 1, "home": 2, "secret": 3}

_HA_DOMAINS = ("light|switch|camera|sensor|binary_sensor|climate|lock|cover|person|device_tracker|media_player|fan|"
               "alarm_control_panel|input_boolean|input_number|automation|script|scene|zone|vacuum|notify")
HOME_PATTERNS = [
    ("ha_entity", re.compile(rf"\b(?:{_HA_DOMAINS})\.[a-z0-9_]{{3,}}\b")),
    ("lan_ip", re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")),
]
SECRET_PATTERNS = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),   # HA long-lived tokens
    ("pve_token", re.compile(r"PVEAPIToken\s*=?\s*\S+", re.I)),
    ("api_key", re.compile(r"\b(?:sk|gsk|sk-or|sk-ant|csk|AIza)[-_][A-Za-z0-9_-]{16,}")),
    ("assignment", re.compile(r"(?i)(?:api[_-]?key|token|password|passwd|secret)\s*[:=]\s*[\"']?[^\s\"']{8,}")),
]


def _texts(messages: Iterable[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.extend(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        for tc in m.get("tool_calls") or []:
            out.append(str((tc.get("function") or {}).get("arguments", "")))
    return out


def has_images(messages: Iterable[Dict[str, Any]]) -> bool:
    for m in messages or []:
        if m.get("images"):
            return True
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "input_image") for p in c):
            return True
    return False


def scan(text: str, home_terms: Iterable[str] = ()) -> List[str]:
    """Kinds of sensitive content found in `text` ('secret:jwt', 'home:lan_ip', 'home:term:<name>', ...)."""
    hits: List[str] = []
    for kind, rx in SECRET_PATTERNS:
        if rx.search(text):
            hits.append(f"secret:{kind}")
    for kind, rx in HOME_PATTERNS:
        if rx.search(text):
            hits.append(f"home:{kind}")
    low = text.lower()
    for term in home_terms:
        t = term.lower().strip()
        if t and re.search(rf"\b{re.escape(t)}\b", low):
            hits.append(f"home:term:{t}")
    return hits


def classify(declared: str, messages: Iterable[Dict[str, Any]], home_terms: Iterable[str] = ()) -> Dict[str, Any]:
    """Effective class = the more sensitive of what the caller declared and what the scan finds."""
    cls = declared if declared in CLASS_RANK else "general"
    hits: List[str] = []
    for t in _texts(messages):
        hits.extend(scan(t, home_terms))
    for h in hits:
        found = h.split(":", 1)[0]
        if CLASS_RANK[found] > CLASS_RANK[cls]:
            cls = found
    return {"class": cls, "hits": sorted(set(hits)), "images": has_images(messages)}


def egress_allowed(tier: str, content_class: str, images: bool = False) -> bool:
    """May content of `content_class` (already raised by classify) go to a source of `tier`?

    tier: 'local' | 'no_training' | 'training'
    content_class: 'general' | 'code' | 'home' | 'secret'
    images: the request carries camera frames or other pictures
    """
    if tier not in ("local", "no_training", "training"):
        return False                                   # unknown tier: fail closed
    if content_class == "secret" or images:
        return tier == "local"                         # secrets and pictures never leave our hardware
    if content_class == "home":
        return tier in ("local", "no_training")        # home text only where it isn't kept or trained on
    return True                                        # general and code go anywhere


def refusal_reason(tier: str, verdict: Dict[str, Any]) -> Optional[str]:
    if egress_allowed(tier, verdict["class"], verdict.get("images", False)):
        return None
    what = "pictures" if verdict.get("images") else verdict["class"]
    hits = ", ".join(verdict.get("hits") or [])[:120]
    return f"{what} content may not go to a {tier.replace('_', '-')} source" + (f" ({hits})" if hits else "")
