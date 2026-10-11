"""
Free inference sources Boost can route to (researched 2026-09-24; limits change often, so every field can be
overridden in config.json `boost.providers.<id>`).

Tiers decide what content may go where (see egress.py):
    local        our own hardware (the coordinator on the engine host, edge nodes such as the workstation or the ROG Ally)
    no_training  the provider says prompts are not retained or used for training (Groq, Cloudflare Workers AI)
    training     free tier whose prompts may be used for training (Gemini free, OpenRouter :free, OpenCode Zen free)

Keys come from the environment (/etc/stonesage/secrets.env via systemd EnvironmentFile), then config.json.
"""

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TIERS = ("local", "no_training", "training")


@dataclass
class Provider:
    id: str
    label: str
    base_url: str
    tier: str
    key_env: Optional[str] = None           # None = no key needed (local engines)
    limits: Dict[str, int] = field(default_factory=dict)   # rpm, rpd, tpd (0/absent = unknown, only 429s count)
    tools_ok: bool = True
    models: List[str] = field(default_factory=list)       # preferred order; empty = use the live /models list
    fallback_models: List[str] = field(default_factory=list)  # only when the live list is empty or unreachable
    free_filter: str = ""                   # regex a model id must match to count as free ("" = all)
    list_models: bool = True                # provider serves an OpenAI-style GET /models
    extra_headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    api_key: str = ""                       # resolved at load time, never serialized to clients

    @property
    def has_key(self) -> bool:
        return self.key_env is None or bool(self.api_key)

    def headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def public(self) -> Dict[str, Any]:
        """Safe view for the UI: no key, only whether one is set."""
        return {"id": self.id, "label": self.label, "tier": self.tier, "enabled": self.enabled,
                "has_key": self.has_key, "key_env": self.key_env, "limits": dict(self.limits),
                "tools_ok": self.tools_ok, "models": list(self.models), "base_url": self.base_url}


DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "label": "Groq", "base_url": "https://api.groq.com/openai/v1", "tier": "no_training",
        "key_env": "GROQ_API_KEY", "limits": {"rpm": 30, "rpd": 1000, "tpd": 200000},
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    },
    "cloudflare": {
        "label": "Cloudflare Workers AI",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "tier": "no_training",
        "key_env": "CLOUDFLARE_API_TOKEN", "limits": {"rpm": 60, "rpd": 300},  # 10k neurons/day; 429s are the real limit
        "models": ["@cf/openai/gpt-oss-120b", "@cf/qwen/qwen3-30b-a3b-fp8", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"],
        "list_models": False,
    },
    "gemini": {
        "label": "Gemini API (free tier)", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "tier": "training", "key_env": "GEMINI_API_KEY", "limits": {"rpm": 10, "rpd": 250},
        "models": [], "free_filter": "flash", "fallback_models": ["gemini-2.5-flash"],
    },
    "zen": {
        "label": "OpenCode Zen (free models)", "base_url": "https://opencode.ai/zen/v1", "tier": "training",
        "key_env": "OPENCODE_API_KEY", "limits": {"rpm": 20},
        "models": ["big-pickle", "nemotron-3-ultra-free", "mimo-v2.6-flash-free"], "free_filter": "free|big-pickle",
    },
    "openrouter": {
        "label": "OpenRouter (:free models)", "base_url": "https://openrouter.ai/api/v1", "tier": "training",
        "key_env": "OPENROUTER_API_KEY", "limits": {"rpm": 20, "rpd": 50},
        "models": [], "free_filter": ":free",
        "extra_headers": {"HTTP-Referer": "http://localhost", "X-Title": "StoneSage"},   # never a LAN address: it leaves the house
    },
}

DEFAULT_PRIORITY = ["groq", "cloudflare", "gemini", "zen", "openrouter"]


def _key_for(pid: str, spec: Dict[str, Any], env: Dict[str, str]) -> str:
    k = spec.get("key_env")
    return ((env.get(k, "") if k else "") or spec.get("api_key", "")).strip()


def load_providers(cfg: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> Dict[str, Provider]:
    """Default table, overridden per field by config.json boost.providers, plus edge nodes from harness_instances."""
    env = dict(os.environ) if env is None else env
    overrides = ((cfg.get("boost") or {}).get("providers") or {})
    out: Dict[str, Provider] = {}
    for pid, base in DEFAULT_PROVIDERS.items():
        spec = copy.deepcopy(base)
        spec.update(overrides.get(pid) or {})
        base_url = spec["base_url"]
        if "{account_id}" in base_url:
            acct = (env.get("CLOUDFLARE_ACCOUNT_ID", "") or spec.get("account_id", "")).strip()
            base_url = base_url.replace("{account_id}", acct) if acct else ""
        p = Provider(id=pid, label=spec["label"], base_url=base_url, tier=spec["tier"], key_env=spec.get("key_env"),
                     limits=dict(spec.get("limits") or {}), tools_ok=spec.get("tools_ok", True),
                     models=list(spec.get("models") or []), fallback_models=list(spec.get("fallback_models") or []),
                     free_filter=spec.get("free_filter", ""),
                     list_models=spec.get("list_models", True), extra_headers=dict(spec.get("extra_headers") or {}),
                     enabled=spec.get("enabled", True), api_key=_key_for(pid, spec, env))
        if not base_url:
            p.enabled = False  # Cloudflare without an account id
        out[pid] = p
    # our own edge nodes (workstation, ROG Ally, ...) opt in with "boost": true in harness_instances
    for inst in cfg.get("harness_instances") or []:
        if not inst.get("boost") or not inst.get("url"):
            continue
        pid = "edge:" + str(inst.get("id") or inst.get("name") or inst["url"])
        url = inst["url"].rstrip("/")
        out[pid] = Provider(id=pid, label=inst.get("name") or pid, base_url=url if url.endswith("/v1") else url + "/v1",
                            tier="local", key_env=None, models=list(inst.get("boost_models") or []))
    return out
