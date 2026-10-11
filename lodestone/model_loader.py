"""
Model Loader: one place to choose a model for any engine on any node, size it against the real hardware,
review the exact change, and apply it with automatic rollback.

Targets
  engine:<role>      llama-server engines on the inference host (systemd units, one flag per line)
  lmstudio:<id>      LM Studio nodes from config.json harness_instances (REST API v1 load/unload)

Memory plan
  weights + KV cache (2 x layers x kv_heads x head_dim x ctx x slots x bytes/elem) + compute buffer,
  against the free VRAM of the chosen GPU (what the other engines on it use is subtracted).
  Geometry comes from the GGUF header (system_profile.get_models); when the planned model is the one
  already loaded, the live fdinfo measurement calibrates the estimate. LM Studio nodes have no header
  access, so their KV size is an approximation from the parameter count.

Apply (engines): preview returns a unified diff of the unit and a token (hash of the new unit); apply only
accepts that token, so what runs is exactly what was reviewed. The job backs up the unit, installs it,
restarts, waits for /health and checks /props, and restores the backup if anything fails.
"""

import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from lodestone import engine_options
from lodestone import system_profile
from lodestone import endpoints

GB = 1024 ** 3
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
SAFETY_GB = 0.35            # headroom left free on a GPU (driver, fragmentation)
LMSTUDIO_OS_RESERVE_GB = 4  # default when config.json hardware has no os_reserve_gb
LMSTUDIO_KV_MB_PER_B_PER_TOKEN = 0.012  # fp16 KV approximation (GQA models, ~0.16 MB/token for 14B)
VALUE_FLAGS = {  # flags that take a value in llama-server units (anything else is a switch)
    "--model", "-m", "--spec-draft-model", "-md", "--model-draft", "--host", "--port", "--device", "-ngl",
    "--n-gpu-layers", "-c", "--ctx-size", "-np", "--parallel", "-b", "--batch-size", "-ub", "--ubatch-size",
    "-t", "--threads", "-tb", "--threads-batch", "--flash-attn", "-fa", "-ctk", "--cache-type-k", "-ctv",
    "--cache-type-v", "--alias", "-a", "--mmproj", "--temp", "--top-p", "--top-k", "--min-p",
    "--repeat-penalty", "--presence-penalty", "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min",
    "--spec-draft-ngl", "--ctx-size-draft", "-cd", "--reasoning-format", "--chat-template", "--pooling",
    "--embd-normalize", "--rope-scaling", "--rope-freq-base", "--defrag-thold", "-C", "--prio", "--poll",
}
ALIASES = {"-m": "--model", "-md": "--spec-draft-model", "--model-draft": "--spec-draft-model",
           "--n-gpu-layers": "-ngl", "--ctx-size": "-c", "--parallel": "-np", "--cache-type-k": "-ctk",
           "--cache-type-v": "-ctv", "-fa": "--flash-attn", "--batch-size": "-b", "--ubatch-size": "-ub"}
SAMPLING_FLAGS = {"temp": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p",
                  "repeat_penalty": "--repeat-penalty", "presence_penalty": "--presence-penalty"}

LEASE = None  # engine_lease.EngineLease, set by server.py: no engine changes while a lease holds the GPUs
PROVENANCE = None  # provenance.Provenance, set by server.py: lineage in the library, hard block in preview (Phase 6)

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_apply_lock = threading.Lock()  # one engine change at a time


# ---------------------------------------------------------------- helpers ----
def _cfg() -> Dict[str, Any]:
    return system_profile.load_cfg()


def _ssh_target(cfg: Dict[str, Any]) -> str:
    cl = cfg.get("cluster", {})
    host = urlparse(system_profile._base(cl.get("coordinator_url", ""))).hostname
    return endpoints.ssh_target(cfg, host_name=host or "")


def _ssh(cfg: Dict[str, Any], cmd: str, stdin: Optional[str] = None, timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "--", _ssh_target(cfg), cmd],
                          input=stdin, capture_output=True, text=True, timeout=timeout)


def _http_json(url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 5) -> Tuple[Optional[int], Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "User-Agent": "StoneSage-ModelLoader"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw) if raw else {}
            except ValueError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw
    except Exception as e:
        return None, str(e)


def _num(v: Any) -> Optional[float]:
    try:
        return float(str(v).split("-")[0].rstrip("BbMm")) * (0.001 if str(v).split("-")[0][-1:] in "Mm" else 1)
    except (ValueError, IndexError):
        return None


def _int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


# ------------------------------------------------------------- unit edits ----
def parse_unit(text: str, value_flags: Optional[set] = None) -> Dict[str, Any]:
    """Split a unit into lines around ExecStart; ExecStart becomes an ordered [(flag, value|None)] list."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("ExecStart=")), -1)
    if start < 0:
        raise ValueError("unit has no ExecStart=")
    end, parts = start, [lines[start][len("ExecStart="):]]
    while parts[-1].rstrip().endswith("\\") and end + 1 < len(lines):
        parts[-1] = parts[-1].rstrip()[:-1]
        end += 1
        parts.append(lines[end])
    tokens = [t.replace("%%", "%").replace("$$", "$") for t in shlex.split(" ".join(parts))]    # the doubling unit_token adds
    binary, args, i = tokens[0], [], 1
    while i < len(tokens):
        tok = tokens[i]
        if (tok in VALUE_FLAGS or (value_flags and tok in value_flags)) and i + 1 < len(tokens):
            args.append([tok, tokens[i + 1]])
            i += 2
        else:
            args.append([tok, None])
            i += 1
    env = {}
    for l in lines:
        m = re.match(r'Environment="?([A-Za-z_][A-Za-z0-9_]*)=([^"]*)"?\s*$', l)
        if m:
            env[m.group(1)] = m.group(2)
    return {"lines": lines, "start": start, "end": end, "binary": binary, "args": args, "env": env}


def flag_get(unit: Dict[str, Any], flag: str) -> Optional[str]:
    names = {flag} | {k for k, v in ALIASES.items() if v == flag}
    return next((v for f, v in unit["args"] if f in names or ALIASES.get(f) == flag), None)


def flag_set(unit: Dict[str, Any], flag: str, value: Optional[str], present: bool = True) -> None:
    """Set/replace a flag (keeping its position), add it before --alias/--metrics, or remove it (present=False)."""
    names = {flag} | {k for k, v in ALIASES.items() if v == flag}
    idx = next((i for i, (f, _) in enumerate(unit["args"]) if f in names), None)
    if not present:
        if idx is not None:
            unit["args"].pop(idx)
        return
    if idx is not None:
        unit["args"][idx] = [unit["args"][idx][0], value]
        return
    anchor = next((i for i, (f, _) in enumerate(unit["args"]) if f in ("--alias", "-a", "--metrics")), len(unit["args"]))
    unit["args"].insert(anchor, [flag, value])


def set_option(unit: Dict[str, Any], opt: Dict[str, Any], value: Any) -> None:
    """Apply one UI value for an engine option. None = drop the flag (engine default); True/False = switch or
    on/off form of a tri-state flag; anything else = the flag's value. Keeps the spelling already in the unit."""
    names = set(opt["names"])
    idx = next((i for i, (f, _) in enumerate(unit["args"]) if f in names), None)
    existing = unit["args"][idx][0] if idx is not None else None
    if opt.get("required_on") and (value is None or value is False):
        return  # StoneSage depends on /metrics, /slots and /props
    if value is None or (value is False and opt.get("kind") == "switch"):
        unit["args"] = [a for a in unit["args"] if a[0] not in names]
        return
    if value is True or value is False:
        on = existing if existing and existing != opt.get("off_flag") and not existing.startswith(("--no-", "-no-")) else opt["flag"]
        name = on if value else opt.get("off_flag")
        if not name:
            return
        val = None
    else:
        name = existing if existing and existing != opt.get("off_flag") else opt["flag"]
        val = str(value)
    unit["args"] = [a for a in unit["args"] if a[0] not in names]
    anchor = idx if idx is not None and idx <= len(unit["args"]) else next(
        (i for i, (f, _) in enumerate(unit["args"]) if f in ("--alias", "-a", "--metrics")), len(unit["args"]))
    unit["args"].insert(anchor, [name, val])


def current_options(unit: Dict[str, Any], opts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """UI values for what a unit sets now: True/False for switches, strings for values."""
    by_name = {n: o for o in opts for n in o["names"]}
    out = {}
    for f, v in unit["args"]:
        o = by_name.get(f)
        if not o:
            continue
        if o["kind"] in ("switch", "tristate"):
            out[o["flag"]] = not (f == o.get("off_flag") or f.startswith(("--no-", "-no-")))
        else:
            out[o["flag"]] = v
    return out


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")      # a newline in a unit value would start a new directive
FLAG_NAME = re.compile(r"^-{1,2}[A-Za-z0-9][A-Za-z0-9_.-]*$")


def unit_token(value: str) -> str:
    """One value as a systemd ExecStart= word: no control characters at all (ValueError), `%` and `$` doubled so systemd does
    not expand them, then shell-quoted. parse_unit undoes the doubling, so a unit round-trips."""
    v = str(value)
    if CONTROL_CHARS.search(v):
        raise ValueError("a value for the engine unit contains a control character (newline, NUL...)")
    return shlex.quote(v.replace("%", "%%").replace("$", "$$"))


def render_unit(unit: Dict[str, Any], description: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> str:
    lines = list(unit["lines"])
    if CONTROL_CHARS.search(str(unit["binary"])) or any(not FLAG_NAME.match(str(f)) for f, _ in unit["args"]):
        raise ValueError("the engine unit has a flag name or program path that is not plain")
    body = [f"ExecStart={unit_token(unit['binary'])} \\"]
    for n, (f, v) in enumerate(unit["args"]):
        tail = "" if n == len(unit["args"]) - 1 else " \\"
        body.append(f"    {f}{' ' + unit_token(v) if v is not None else ''}{tail}")
    lines[unit["start"]:unit["end"] + 1] = body
    if description:
        if CONTROL_CHARS.search(description):
            raise ValueError("the description contains a control character")
        lines = [f"Description={description}" if l.startswith("Description=") else l for l in lines]
    for key, val in (env or {}).items():
        if CONTROL_CHARS.search(f"{key}{val}") or '"' in f"{key}{val}" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError("an Environment value for the engine unit is not plain")
        new = f'Environment="{key}={val}"'
        hit = [i for i, l in enumerate(lines) if re.match(rf'Environment="?{re.escape(key)}=', l)]
        if hit:
            lines[hit[0]] = new
        else:
            lines.insert(next(i for i, l in enumerate(lines) if l.startswith("ExecStart=")), new)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ state ----
def _lmstudio_nodes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for inst in cfg.get("harness_instances", []):
        url = (inst.get("url") or "").rstrip("/")
        if not (inst.get("engine") or url.endswith(":1234")):
            continue
        code, data = _http_json(f"{url}/api/v1/models", timeout=2.5)
        models = data.get("models", []) if code == 200 and isinstance(data, dict) else []
        loaded = [m for m in models if m.get("loaded_instances")]
        hw = inst.get("hardware") or {}
        ram = hw.get("ram_gb")
        out.append({
            "id": f"lmstudio:{inst['id']}", "kind": "lmstudio", "name": inst.get("name", inst["id"]), "url": url,
            "online": code == 200, "hardware": hw,
            "budget_gb": round(ram - hw.get("os_reserve_gb", LMSTUDIO_OS_RESERVE_GB), 1) if ram else None,
            "loaded": [{"key": m["key"], "name": m.get("display_name") or m["key"],
                        "instances": [{"id": i.get("id"), "context_length": (i.get("config") or {}).get("context_length")}
                                      for i in m.get("loaded_instances", [])]} for m in loaded],
            "model_count": len([m for m in models if m.get("type", "llm") == "llm"]),
        })
    return out


def get_state(fresh: bool = False) -> Dict[str, Any]:
    """Every target the loader can change, with its hardware and what it runs now."""
    cfg = _cfg()
    prof = system_profile.get_profile(cfg, fresh=fresh)
    engines = prof.get("engines") or {}
    host = prof.get("host") or {}
    gpus = prof.get("gpus") or []
    inference = {
        "id": "host:inference", "kind": "llama-host", "name": host.get("hostname") or "inference host",
        "cpu": host.get("cpu"), "ram_gb": round((host.get("ram_total_mb") or 0) / 1024, 1) or None,
        "gpus": [{"index": g["index"], "name": g["name"], "short": g["short"], "total_gb": g["vram_total_gb"],
                  "used_gb": g["vram_used_gb"], "engines": g.get("engines", [])} for g in gpus],
        "targets": [],
    }
    for role, e in engines.items():
        inference["targets"].append({
            "id": f"engine:{role}", "role": role, "purpose": e.get("purpose"), "online": e.get("online"),
            "port": e.get("port"), "unit": e.get("unit"), "model": e.get("model"), "model_file": e.get("model_file"),
            "label": e.get("label"), "quant": e.get("quant"), "params": e.get("params"),
            "ctx_per_slot": e.get("ctx_per_slot"), "slots": e.get("slots"), "kv_cache": e.get("kv_cache"),
            "draft_model": e.get("draft_model"), "draft_model_file": e.get("draft_model_file"), "mmproj": e.get("mmproj"),
            "flash_attn": e.get("flash_attn"),
            "gpu_index": (e.get("gpu") or {}).get("index"), "vram_gb": round((e.get("vram_mb") or 0) / 1024, 2),
            "is_courage": role == "coordinator",
            "editable": role in ("coordinator", "worker", "vision"),  # the embedder is paired with memory search
        })
    return {"updated_at": time.time(), "nodes": [inference] + _lmstudio_nodes(cfg), "probe_error": prof.get("probe_error")}


def get_library(node_id: str, fresh: bool = False) -> Dict[str, Any]:
    cfg = _cfg()
    if node_id == "host:inference":
        items = []
        for m in system_profile.get_models(cfg, fresh=fresh):
            if m.get("is_projector") or m.get("architecture") in ("bert", "nomic-bert"):
                kind = "projector" if m.get("is_projector") else "embedding"
            else:
                kind = "llm"
            kv = _int(m.get("kv_heads"))
            items.append({
                "key": m["path"], "file": m["file"], "name": m.get("name") or m["file"], "kind": kind,
                "arch": m.get("architecture"), "params": m.get("params"), "quant": m.get("quant"),
                "size_gb": round((m.get("size_bytes") or 0) / GB, 2), "trained_ctx": m.get("context_length"),
                "layers": _int(m.get("layers")), "kv_heads": kv, "embedding": _int(m.get("embedding")),
                "heads": _int(m.get("heads")), "experts": m.get("experts"), "is_loaded": m.get("is_loaded"),
                "geometry": "gguf" if (_int(m.get("layers")) and kv and _int(m.get("heads"))) else "partial",
            })
            prov = PROVENANCE.lookup(m["path"]) if PROVENANCE else None
            if prov:
                from lodestone.provenance import lineage
                items[-1].update(sha256=prov.get("sha256"), loadable_by=prov.get("loadable_by"),
                                 lineage=lineage(prov.get("gguf") or {}))
        return {"node": node_id, "models": items}
    inst_id = node_id.split(":", 1)[1]
    inst = next((i for i in cfg.get("harness_instances", []) if i.get("id") == inst_id), None)
    if not inst:
        return {"node": node_id, "models": [], "error": "unknown node"}
    code, data = _http_json(f"{inst['url'].rstrip('/')}/api/v1/models", timeout=4)
    if code != 200 or not isinstance(data, dict):
        return {"node": node_id, "models": [], "error": f"LM Studio unreachable ({data})"}
    items = []
    for m in data.get("models", []):
        items.append({
            "key": m["key"], "file": m["key"], "name": m.get("display_name") or m["key"],
            "kind": "llm" if m.get("type", "llm") == "llm" else m.get("type"), "arch": m.get("architecture"),
            "params": m.get("params_string"), "quant": (m.get("quantization") or {}).get("name"),
            "size_gb": round((m.get("size_bytes") or 0) / GB, 2), "trained_ctx": m.get("max_context_length"),
            "is_loaded": bool(m.get("loaded_instances")), "geometry": "approx",
            "vision": bool((m.get("capabilities") or {}).get("vision")),
            "tools": bool((m.get("capabilities") or {}).get("trained_for_tool_use")),
        })
    return {"node": node_id, "models": items}


# ------------------------------------------------------------------- plan ----
def _kv_gb(model: Dict[str, Any], ctx_total: int, kv_type: str) -> Tuple[float, str]:
    bpe = KV_BYTES.get((kv_type or "f16").lower(), 2.0)
    layers, kvh, heads, emb = model.get("layers"), model.get("kv_heads"), model.get("heads"), model.get("embedding")
    if layers and kvh and heads and emb:
        return 2 * layers * kvh * (emb // heads) * ctx_total * bpe / GB, "gguf"
    p = _num(model.get("params")) or model.get("size_gb", 4) * 1.6
    return p * LMSTUDIO_KV_MB_PER_B_PER_TOKEN * ctx_total * (bpe / 2.0) / 1024, "approx"


def _compute_gb(model: Dict[str, Any], ubatch: int) -> float:
    p = _num(model.get("params")) or 4
    return 0.25 + 0.02 * p + 0.0004 * ubatch


def plan(req: Dict[str, Any]) -> Dict[str, Any]:
    """Memory estimate and verdict for loading `model` on `target` with the given settings."""
    target = req.get("target", "")
    node = "host:inference" if target.startswith("engine:") else target
    lib = {m["key"]: m for m in get_library(node)["models"]}
    model = lib.get(req.get("model")) or {}
    if not model:
        return {"ok": False, "error": "model not found on that node"}
    ctx = max(512, int(req.get("ctx_per_slot") or 4096))
    slots = max(1, int(req.get("slots") or 1))
    kv_type = req.get("kv_cache") or "f16"
    ctx_total = ctx * slots
    kv_k, kv_v = req.get("kv_cache_k") or kv_type, req.get("kv_cache_v") or kv_type
    kv_gb, geometry = _kv_gb(model, ctx_total, kv_k)
    kv_gb = (kv_gb + _kv_gb(model, ctx_total, kv_v)[0]) / 2  # K and V are half the cache each
    weights_gb = model.get("size_gb") or 0
    compute_gb = _compute_gb(model, int(req.get("ubatch") or 512))
    draft_gb = 0.0
    if req.get("draft"):
        d = lib.get(req["draft"]) or {}
        draft_gb = (d.get("size_gb") or 0) + _kv_gb(d, ctx_total, "f16")[0] * 0.5 if d else 0.0
    opts = req.get("options") or {}
    if target.startswith("lmstudio:") and opts.get("parallel"):
        ctx_total = ctx  # LM Studio's parallel requests share one context
        kv_gb, geometry = _kv_gb(model, ctx_total, kv_type)
    if opts.get("--ubatch-size") or opts.get("physical_batch_size"):
        compute_gb = _compute_gb(model, int(opts.get("--ubatch-size") or opts.get("physical_batch_size")))
    kv_in_ram = opts.get("--kv-offload") is False or opts.get("offload_kv_cache_to_gpu") is False
    moe_cpu_layers = model.get("layers") if opts.get("--cpu-moe") else int(opts.get("--n-cpu-moe") or 0)
    ngl = req.get("ngl")
    offload = 1.0
    if ngl not in (None, "", "auto") and model.get("layers"):
        offload = max(0.0, min(1.0, int(ngl) / model["layers"]))
    notes = []
    if model.get("trained_ctx") and ctx > model["trained_ctx"]:
        notes.append(f"Context per slot is above the model's trained {model['trained_ctx']:,} tokens.")
    if geometry != "gguf":
        notes.append("KV size is approximate (no layer/head data for this file).")

    if target.startswith("engine:"):
        role = target.split(":", 1)[1]
        state = get_state()
        host = state["nodes"][0]
        tgt = next((t for t in host["targets"] if t["id"] == target), None)
        if not tgt:
            return {"ok": False, "error": "unknown engine"}
        gpu_index = req.get("gpu_index", tgt.get("gpu_index"))
        gpu = next((g for g in host["gpus"] if g["index"] == gpu_index), None)
        others = sum(t["vram_gb"] for t in host["targets"] if t["id"] != target and t.get("gpu_index") == gpu_index)
        # calibration: same model already loaded here -> use the measured footprint for weights+compute
        calibrated = False
        if tgt.get("model_file") == model.get("file") and tgt.get("vram_gb"):
            cur_kv, _ = _kv_gb(model, (tgt.get("ctx_per_slot") or ctx) * (tgt.get("slots") or 1), tgt.get("kv_cache") or "f16")
            base = tgt["vram_gb"] - cur_kv
            if base > 0:
                weights_gb, compute_gb, calibrated = base, 0.0, True
                cur_draft = ((system_profile.get_profile(_cfg()).get("engines") or {}).get(role) or {}).get("draft_model_file")
                if req.get("draft") and cur_draft and os.path.basename(req["draft"]) == cur_draft:
                    draft_gb = 0.0  # the measurement already includes the running draft model
                notes.append(f"Calibrated against the live measurement ({tgt['vram_gb']:.2f} GB in use now).")
        sel = req.get("gpu_indexes") or ([gpu_index] if gpu_index is not None else [])
        sel_gpus = [g for g in host["gpus"] if g["index"] in sel] or ([gpu] if gpu else [])
        others = sum(t["vram_gb"] for t in host["targets"] if t["id"] != target and t.get("gpu_index") in {g["index"] for g in sel_gpus})
        moe_share = 0.0
        if moe_cpu_layers and model.get("layers"):
            moe_share = 0.85 * min(1.0, moe_cpu_layers / model["layers"])  # expert tensors are most of an MoE model
            notes.append(f"{moe_cpu_layers} layers of expert weights stay in system RAM (about {weights_gb * moe_share:.1f} GB).")
        if kv_in_ram:
            notes.append(f"KV cache ({kv_gb:.2f} GB) is kept in system RAM: less VRAM, slower long prompts.")
        on_gpu = weights_gb * offload * (1 - moe_share) + (0 if kv_in_ram else kv_gb) + compute_gb + draft_gb
        capacity = sum(g["total_gb"] for g in sel_gpus) - others - SAFETY_GB * max(1, len(sel_gpus))
        spill = weights_gb * (1 - offload) + weights_gb * offload * moe_share + (kv_gb if kv_in_ram else 0)
        gpu = sel_gpus[0] if len(sel_gpus) == 1 else gpu
        if len(sel_gpus) > 1:
            notes.append("Split across " + " + ".join(g["short"] for g in sel_gpus) + " (set --split-mode/--tensor-split below).")
        if role == "coordinator":
            notes.append("Courage runs on this engine: it is unavailable while the engine restarts.")
        names = " + ".join(g["short"] for g in sel_gpus) or "no GPU"
        budget_label = f"{names} ({sum(g['total_gb'] for g in sel_gpus)} GB, {others:.1f} GB used by other engines)"
    else:
        state_nodes = {n["id"]: n for n in get_state()["nodes"]}
        n = state_nodes.get(target) or {}
        capacity = n.get("budget_gb") or 0
        on_gpu, spill, others, calibrated = weights_gb + (0 if kv_in_ram else kv_gb) + compute_gb, 0.0, 0.0, False
        hw = n.get("hardware") or {}
        budget_label = f"{hw.get('device') or n.get('name')} ({hw.get('ram_gb')} GB unified, {hw.get('os_reserve_gb', LMSTUDIO_OS_RESERVE_GB)} GB kept for the OS)"
        if n.get("loaded"):
            notes.append("The model loaded there now will be unloaded first.")

    fits = on_gpu <= capacity
    tight = fits and on_gpu > capacity * 0.92
    # largest context per slot that still fits, in 1024 steps, capped by the trained context
    per_tok = kv_gb / ctx_total if ctx_total else 0
    room = capacity - (on_gpu - kv_gb)
    max_ctx = int(room / per_tok / slots // 1024 * 1024) if per_tok > 0 and room > 0 else 0
    if model.get("trained_ctx"):
        max_ctx = min(max_ctx, model["trained_ctx"])
    return {
        "ok": True, "model": model, "target": target,
        "memory": {"weights_gb": round(weights_gb * offload, 2), "kv_gb": round(kv_gb, 2), "compute_gb": round(compute_gb, 2),
                   "draft_gb": round(draft_gb, 2), "total_gb": round(on_gpu, 2), "capacity_gb": round(capacity, 2),
                   "others_gb": round(others, 2), "spill_to_ram_gb": round(spill, 2)},
        "verdict": "fits" if fits and not tight else ("tight" if tight else "too_big"),
        "budget_label": budget_label, "geometry": geometry, "calibrated": calibrated,
        "max_ctx_per_slot": max_ctx, "notes": notes,
    }


# ------------------------------------------------------ preview and apply ----
_llama_schema: Dict[str, Any] = {"version": None, "options": []}


def llama_schema(cfg: Dict[str, Any], fresh: bool = False) -> List[Dict[str, Any]]:
    """Options of the installed llama-server build, parsed from its --help (re-read when the build changes)."""
    ver = _ssh(cfg, "/usr/local/bin/llama-server --version 2>&1 | head -2", timeout=20).stdout.strip()
    if fresh or ver != _llama_schema["version"] or not _llama_schema["options"]:
        help_text = _ssh(cfg, "/usr/local/bin/llama-server --help 2>&1", timeout=30).stdout
        _llama_schema.update(version=ver, options=engine_options.parse_llama_help(help_text))
    return _llama_schema["options"]


def _value_flags(opts: List[Dict[str, Any]]) -> set:
    return {n for o in opts if o.get("arg") for n in o["names"]}


def get_options(target: str, model_key: str = "", gpu_indexes: Optional[List[int]] = None) -> Dict[str, Any]:
    """Every option the target engine accepts, narrowed to the chosen model and hardware, with current values."""
    cfg = _cfg()
    node = "host:inference" if target.startswith("engine:") else target
    model = next((m for m in get_library(node)["models"] if m["key"] == model_key), {}) if model_key else {}
    prof = system_profile.get_profile(cfg)
    if target.startswith("engine:"):
        role = target.split(":", 1)[1]
        opts = llama_schema(cfg)
        _, text = _engine_unit(cfg, role)
        unit = parse_unit(text, _value_flags(opts))
        visible = unit["env"].get("GGML_VK_VISIBLE_DEVICES", "")
        n_gpus = len(gpu_indexes) if gpu_indexes else (len([x for x in visible.split(",") if x.strip()]) or len(prof.get("gpus") or []))
        hw = {"cpu_threads": (prof.get("host") or {}).get("cpu_threads"), "gpu_count": n_gpus}
        refined = engine_options.refine(opts, model, hw, "llama", role)
        return {"engine": "llama-server", "build": _llama_schema["version"], "options": refined,
                "current": current_options(unit, opts), "visible_devices": visible}
    inst = next((i for i in cfg.get("harness_instances", []) if f"lmstudio:{i.get('id')}" == target), None)
    if not inst:
        return {"engine": "lmstudio", "options": [], "current": {}, "error": "unknown node"}
    opts = engine_options.probe_lmstudio(inst["url"].rstrip("/"))
    refined = engine_options.refine(opts, model, inst.get("hardware") or {}, "lmstudio")
    current = {}
    code, data = _http_json(f"{inst['url'].rstrip('/')}/api/v1/models", timeout=4)
    for m in (data.get("models", []) if isinstance(data, dict) else []):
        for li in m.get("loaded_instances", []):
            cfg_li = li.get("config") or {}
            current = {k: v for k, v in cfg_li.items() if k in {o["flag"] for o in opts}}
    return {"engine": "lmstudio", "options": refined, "current": current}


def _engine_unit(cfg: Dict[str, Any], role: str) -> Tuple[str, str]:
    eng = (system_profile.get_profile(cfg).get("engines") or {}).get(role) or {}
    unit = (eng.get("unit") or f"llama-{role}") + ".service"
    res = _ssh(cfg, f"cat /etc/systemd/system/{shlex.quote(unit)}")
    if res.returncode != 0:
        raise RuntimeError(f"cannot read {unit}: {res.stderr.strip()[:200]}")
    return unit, res.stdout


def preview(req: Dict[str, Any]) -> Dict[str, Any]:
    target = req.get("target", "")
    if target.startswith("lmstudio:"):
        body = {"model": req.get("model"), "context_length": int(req.get("ctx_per_slot") or 4096), "echo_load_config": True}
        inst_url = next((i["url"].rstrip("/") for i in _cfg().get("harness_instances", []) if f"lmstudio:{i.get('id')}" == target), "")
        kinds = {o["flag"]: o["kind"] for o in engine_options.probe_lmstudio(inst_url)} if inst_url else {}
        for key, val in (req.get("options") or {}).items():
            if key not in kinds or val in (None, "") or key == "context_length":
                continue
            body[key] = bool(val) if kinds[key] in ("switch", "tristate") else (int(val) if kinds[key] in ("int", "enum") else val)
        token = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        return {"ok": True, "kind": "lmstudio", "request": body, "token": token,
                "summary": f"Unload what is loaded, then POST /api/v1/models/load on {target.split(':', 1)[1]}"}

    role = target.split(":", 1)[1]
    cfg = _cfg()
    unit_name, text = _engine_unit(cfg, role)
    schema = llama_schema(cfg)
    u = parse_unit(text, _value_flags(schema))
    lib = {m["key"]: m for m in get_library("host:inference")["models"]}
    model = lib.get(req.get("model"))
    if not model:
        return {"ok": False, "error": "model not found on the inference host"}
    if PROVENANCE:   # hard block, no override: a model this unit's llama-server build cannot load (e.g. needs a fork)
        for path in [model["key"]] + ([req["draft"]] if req.get("draft") else []):
            why = PROVENANCE.check(path, text)
            if why:
                return {"ok": False, "blocked": True, "error": why}
    ctx, slots = int(req.get("ctx_per_slot") or 4096), int(req.get("slots") or 1)
    flag_set(u, "--model", model["key"])
    flag_set(u, "-c", str(ctx * slots))
    flag_set(u, "-np", str(slots))
    if req.get("kv_cache") or req.get("kv_cache_k"):
        flag_set(u, "-ctk", req.get("kv_cache_k") or req["kv_cache"])
    if req.get("kv_cache") or req.get("kv_cache_v"):
        flag_set(u, "-ctv", req.get("kv_cache_v") or req["kv_cache"])
    if req.get("ngl") not in (None, ""):
        flag_set(u, "-ngl", str(req["ngl"]))
    if req.get("flash_attn"):
        flag_set(u, "--flash-attn", req["flash_attn"])
    draft = req.get("draft")
    if draft:
        flag_set(u, "--spec-draft-model", draft)
    elif "draft" in req:  # explicitly cleared: drop the draft and its tuning flags
        for f in ("--spec-draft-model", "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min", "--spec-draft-ngl"):
            flag_set(u, f, None, present=False)
    if model.get("kind") == "llm" and flag_get(u, "--mmproj") and role != "vision":
        flag_set(u, "--mmproj", None, present=False)
    for key, flag in SAMPLING_FLAGS.items():
        val = (req.get("sampling") or {}).get(key)
        if val not in (None, ""):
            flag_set(u, flag, str(val))
    by_flag = {o["flag"]: o for o in schema}
    for flag, val in (req.get("options") or {}).items():
        opt = by_flag.get(flag)
        if opt and not opt.get("managed"):
            set_option(u, opt, None if val == "" else val)
    env = {}
    gpus_sel = req.get("gpu_indexes") or ([req["gpu_index"]] if req.get("gpu_index") is not None else None)
    if gpus_sel:
        env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in gpus_sel)
    prof = system_profile.get_profile(cfg)
    gpu = next((g for g in prof.get("gpus") or [] if g["index"] == int(req.get("gpu_index", -1) if req.get("gpu_index") is not None else -1)), None)
    gpu = gpu or next((g for g in prof.get("gpus") or [] if g["index"] == ((prof.get("engines") or {}).get(role, {}).get("gpu") or {}).get("index")), None)
    draft_file = flag_get(u, "--spec-draft-model")
    draft_name = (lib.get(draft_file) or {}).get("name") if draft_file else None
    desc = (f"llama-server {role}: {model['name']}" + (f" {model['quant']}" if model.get("quant") else "")
            + (f" + {draft_name} draft" if draft_name else "") + (f" on {gpu['name']}" if gpu else ""))
    new_text = render_unit(u, description=desc, env=env)
    diff = "".join(difflib.unified_diff(text.splitlines(True), new_text.splitlines(True),
                                        f"a/{unit_name}", f"b/{unit_name}"))
    token = hashlib.sha256(new_text.encode()).hexdigest()[:16]
    return {"ok": True, "kind": "unit", "unit": unit_name, "diff": diff or "(no change)", "changed": bool(diff),
            "token": token, "new_unit": new_text, "port": prof.get("engines", {}).get(role, {}).get("port"),
            "summary": f"Rewrite {unit_name}, restart it, wait for /health, roll back on failure"}


def _job(job_id: str, **upd) -> None:
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {"id": job_id, "steps": [], "created": time.time()})
        if "step" in upd:
            job["steps"].append({"t": round(time.time() - job["created"], 1), "text": upd.pop("step")})
        job.update(upd)


def get_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        return dict(_jobs.get(job_id) or {"error": "unknown job"})


def apply(req: Dict[str, Any]) -> Dict[str, Any]:
    if LEASE is not None and LEASE.lease and str(req.get("target", "")).startswith("engine:"):
        return {"ok": False, "error": f"An engine lease ({LEASE.lease['layout']}) holds the GPUs; release it first."}
    pv = preview(req)
    if not pv.get("ok"):
        return pv
    if req.get("token") != pv["token"]:
        return {"ok": False, "error": "The change differs from what was reviewed (the engine changed meanwhile). Review again."}
    if pv["kind"] == "unit" and not pv["changed"]:
        return {"ok": False, "error": "Nothing to change."}
    if not _apply_lock.acquire(blocking=False):
        return {"ok": False, "error": "Another engine change is still running."}
    job_id = uuid.uuid4().hex[:10]
    _job(job_id, status="running", target=req.get("target"), step="Queued")
    worker = _apply_unit if pv["kind"] == "unit" else _apply_lmstudio
    threading.Thread(target=_run_job, args=(worker, job_id, req, pv), daemon=True).start()
    return {"ok": True, "job": job_id}


def _run_job(worker, job_id, req, pv):
    try:
        worker(job_id, req, pv)
    except Exception as e:
        _job(job_id, status="failed", step=f"Error: {type(e).__name__}: {e}")
    finally:
        _apply_lock.release()
        system_profile.get_profile(_cfg(), fresh=True)
        system_profile.get_models(_cfg(), fresh=True)


def _wait_healthy(base: str, expect_file: Optional[str], job_id: str, timeout: float = 300) -> bool:
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        code, _ = _http_json(f"{base}/health", timeout=3)
        state = "ok" if code == 200 else ("loading" if code == 503 else "down")
        if state != last:
            _job(job_id, step={"ok": "Engine answers /health", "loading": "Loading weights...", "down": "Waiting for the engine to start..."}[state])
            last = state
        if code == 200:
            pc, props = _http_json(f"{base}/props", timeout=3)
            loaded = os.path.basename((props or {}).get("model_path", "")) if isinstance(props, dict) else ""
            if expect_file and loaded and loaded != expect_file:
                _job(job_id, step=f"Engine is up but serves {loaded}, expected {expect_file}")
                return False
            return True
        time.sleep(2)
    return False


def _apply_unit(job_id: str, req: Dict[str, Any], pv: Dict[str, Any]) -> None:
    cfg = _cfg()
    unit = pv["unit"]
    path = f"/etc/systemd/system/{unit}"
    backup = f"{path}.bak-loader-{time.strftime('%Y%m%d-%H%M%S')}"
    role = req["target"].split(":", 1)[1]
    base = system_profile._base(cfg.get("cluster", {}).get(f"{role}_url", ""))
    _job(job_id, step=f"Backing up {unit}")
    r = _ssh(cfg, f"sudo cp {shlex.quote(path)} {shlex.quote(backup)}")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    _job(job_id, step="Installing the reviewed unit", backup=backup)
    r = _ssh(cfg, f"sudo tee {shlex.quote(path)} >/dev/null && sudo systemctl daemon-reload", stdin=pv["new_unit"])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    _job(job_id, step=f"Restarting {unit}")
    _ssh(cfg, f"sudo systemctl restart {shlex.quote(unit)}", timeout=120)
    expect = os.path.basename(req.get("model", ""))
    if _wait_healthy(base, expect, job_id):
        _job(job_id, status="done", step="Done: engine healthy with the new model")
        return
    _job(job_id, step="Health check failed; restoring the previous unit")
    _ssh(cfg, f"sudo cp {shlex.quote(backup)} {shlex.quote(path)} && sudo systemctl daemon-reload && sudo systemctl restart {shlex.quote(unit)}", timeout=120)
    ok = _wait_healthy(base, None, job_id, timeout=240)
    logs = _ssh(cfg, f"journalctl -u {shlex.quote(unit)} -n 30 --no-pager", timeout=20).stdout[-3000:]
    _job(job_id, status="rolled_back" if ok else "failed", log=logs,
         step="Rolled back: previous engine is running again" if ok else "Rollback did not come up healthy; check the log")


def _apply_lmstudio(job_id: str, req: Dict[str, Any], pv: Dict[str, Any]) -> None:
    cfg = _cfg()
    inst_id = req["target"].split(":", 1)[1]
    inst = next(i for i in cfg.get("harness_instances", []) if i.get("id") == inst_id)
    url = inst["url"].rstrip("/")
    code, data = _http_json(f"{url}/api/v1/models", timeout=5)
    for m in (data.get("models", []) if isinstance(data, dict) else []):
        for li in m.get("loaded_instances", []):
            _job(job_id, step=f"Unloading {m.get('display_name') or m['key']}")
            _http_json(f"{url}/api/v1/models/unload", {"instance_id": li.get("id")}, timeout=30)
    _job(job_id, step=f"Loading {req.get('model')} (this can take a minute)")
    code, data = _http_json(f"{url}/api/v1/models/load", pv["request"], timeout=300)
    if code in (200, 201):
        _job(job_id, status="done", step="Done: model loaded", result=data)
    else:
        msg = data.get("error", data) if isinstance(data, dict) else data
        _job(job_id, status="failed", step=f"LM Studio refused the load: {str(msg)[:300]}")
