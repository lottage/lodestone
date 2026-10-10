"""
Every option an engine accepts, as a UI schema, discovered from the engine itself.

llama-server: `llama-server --help` of the installed build is parsed into options (flags, aliases, argument kind,
choices, numeric range, default, env var, section). Removed/deprecated flags are dropped. A new llama.cpp build
brings its new flags with it; nothing here lists flags by hand except the few the loader manages itself.

LM Studio: its REST API v1 validates load requests strictly (unknown keys are rejected, known keys report their
type), so the accepted fields are probed with a request for a model that does not exist (nothing gets loaded).

refine() then narrows each option to what makes sense for the chosen model and hardware
(threads <= CPU threads, GPU layers <= model layers, MoE options only for MoE models, ...).
"""

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Flags the loader sets from its own controls (model, context, slots, GPU choice...) or that StoneSage depends on.
MANAGED = {"--model", "--ctx-size", "--parallel", "--n-gpu-layers", "--cache-type-k", "--cache-type-v", "--flash-attn",
           "--spec-draft-model", "--device", "--mmproj"}
REQUIRED_ON = {"--metrics", "--slots", "--props"}  # StoneSage reads /metrics, /slots and /props
HIDDEN = {"--help", "--version", "--cache-list", "--completion-bash", "--list-devices", "--host", "--port", "--alias",
          "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file", "--path", "--api-prefix", "--model-url",
          "--hf-repo", "--hf-file", "--hf-token", "--hf-repo-v", "--hf-file-v", "--docker-repo", "--offline",
          "--spec-draft-hf", "--usage", "--log-file", "--webui-config-file", "--models-dir", "--models-preset",
          "--mmproj-url", "--lora-init-without-apply"}

NAME_COL = 40


def _canonical(names: List[str]) -> str:
    longs = [n for n in names if n.startswith("--") and not n.startswith("--no-")]
    return longs[-1] if longs else names[0]


def parse_llama_help(text: str) -> List[Dict[str, Any]]:
    options, section, cur = [], "common", None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^----- (.+?) params -----$", line)
        if m:
            section = m.group(1)
            continue
        if line.startswith("-"):
            # names end at column 40 when a description follows on the same line; longer name lists take the
            # whole line and the description starts on the next one
            if len(line) > NAME_COL and line[NAME_COL - 2:NAME_COL] == "  ":
                head, desc = line[:NAME_COL].rstrip(), line[NAME_COL:].strip()
            else:
                head, desc = line.strip(), ""
            cur = {"head": head, "desc": [desc] if desc else [], "section": section}
            options.append(cur)
        elif cur is not None and line.strip():
            cur["desc"].append(line.strip())
    out = []
    for o in options:
        pieces = [p.strip() for p in re.split(r",\s+(?=-)", o["head"])]
        names, arg = [], None
        for p in pieces:
            bits = p.split(None, 1)
            names.append(bits[0])
            if len(bits) > 1:
                arg = bits[1]
        desc = " ".join(o["desc"])
        listed = [m.group(1) for l in o["desc"] for m in [re.match(r"^- ([\w.-]+):", l)] if m]
        if re.search(r"has been removed|DEPRECATED", desc):
            continue
        env = re.search(r"\(env: ([A-Z0-9_]+)\)", desc)
        desc_clean = re.sub(r"\s*\(env: [A-Z0-9_]+\)", "", desc)
        desc_clean = re.sub(r"\[\(more\s+info\)\]\([^)]*\)", "", desc_clean).strip()
        default = re.search(r"\(default:\s*([^)]*)\)", desc_clean)
        dval = re.match(r"\s*'?(-?[\w.]+)", default.group(1)) if default else None
        opt = {"flag": _canonical(names), "names": names, "arg": arg, "section": o["section"],
               "desc": desc_clean, "env": env.group(1) if env else None,
               "default": default.group(1).strip().strip("'\"") if default else None,
               "default_value": dval.group(1) if dval else None}
        negs = [n for n in names if n.startswith("--no-") or n.startswith("-no-") or (n.startswith("-n") and n[2:] and "-" + n[2:] in names)]
        if arg is None:
            opt["kind"] = "tristate" if negs else "switch"
            opt["off_flag"] = next((n for n in names if n.startswith("--no-")), negs[0] if negs else None)
        else:
            a = arg.strip()
            choices = None
            if re.fullmatch(r"\{[^}]+\}", a):
                choices = a[1:-1].split(",")
            elif re.fullmatch(r"\[[^\]]+\]", a):
                choices = a[1:-1].split("|")
            elif re.fullmatch(r"<[\w.]+(\|[\w.]+)+>", a):
                choices = a[1:-1].split("|")
            elif re.fullmatch(r"[a-z0-9-]+(,[a-z0-9-]+){2,}", a):
                choices = a.split(",")
            allowed = re.search(r"allowed values:\s*([\w\s,]+)", desc_clean)
            if not choices and allowed:
                choices = [c.strip() for c in allowed.group(1).split(",") if c.strip()]
            if not choices and len(listed) >= 2:  # "one of:" followed by "- name: meaning" lines
                choices = listed
            rng = re.fullmatch(r"<(-?\d+)\.\.\.(-?\d+)>", a)
            if choices:
                opt.update(kind="enum", choices=choices)
            elif rng:
                opt.update(kind="int", min=int(rng.group(1)), max=int(rng.group(2)))
            elif re.fullmatch(r"[A-Z]+\d?", a) and (a in ("N", "P", "L", "INDEX", "SEED") or re.match(r"^-?\d", opt["default"] or "")):
                d = opt["default"] or ""
                opt["kind"] = "float" if re.match(r"^-?\d+\.\d+", d) else "int"
            elif a in ("FNAME", "PATH") or "FNAME" in a:
                opt["kind"] = "path"
            else:
                opt["kind"] = "text"
            # enumerated levels written in the description: "low(-1), normal(0), medium(1)"
            lv = re.findall(r"(\w+)\((-?\d+)\)", desc_clean)
            if opt["kind"] == "int" and len(lv) >= 3:
                opt.update(kind="enum", choices=[v for _, v in lv], choice_labels=[f"{n} ({v})" for n, v in lv])
        opt["managed"] = opt["flag"] in MANAGED or any(n in MANAGED for n in names)
        opt["required_on"] = opt["flag"] in REQUIRED_ON
        if opt["flag"] in HIDDEN or any(n in HIDDEN for n in names):
            continue
        out.append(opt)
    return out


# --------------------------------------------------------------- LM Studio ----
LMSTUDIO_FIELDS = {  # meaning of each field LM Studio's load API accepts (the probe decides which exist)
    "context_length": ("Context length", "Tokens the model can attend to; KV cache memory grows with it."),
    "parallel": ("Parallel requests", "Concurrent predictions sharing the context (continuous batching)."),
    "eval_batch_size": ("Evaluation batch size", "Tokens processed per step while reading the prompt (logical batch)."),
    "physical_batch_size": ("Physical batch size", "Tokens per GPU pass (ubatch); higher is faster, uses more compute memory."),
    "num_experts": ("Active experts", "Experts used per token for Mixture-of-Experts models."),
    "flash_attention": ("Flash attention", "Faster attention with less memory; required for quantized KV cache."),
    "offload_kv_cache_to_gpu": ("KV cache on GPU", "Keep the KV cache in GPU memory (off keeps it in system RAM)."),
    "echo_load_config": ("Echo load config", "Return the effective load settings in the response."),
}
_lms_cache: Dict[str, List[Dict[str, Any]]] = {}


def probe_lmstudio(url: str, fresh: bool = False) -> List[Dict[str, Any]]:
    if url in _lms_cache and not fresh:
        return _lms_cache[url]
    out = []
    for key, (label, desc) in LMSTUDIO_FIELDS.items():
        req = urllib.request.Request(f"{url}/api/v1/models/load", json.dumps({"model": "__stonesage_probe__", key: {"x": 1}}).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                continue
        except urllib.error.HTTPError as e:
            err = (json.loads(e.read().decode() or "{}").get("error") or {})
        except Exception:
            continue
        if err.get("code") == "unrecognized_keys":
            continue
        typ = "boolean" if "boolean" in err.get("message", "") else "number" if "number" in err.get("message", "") else "string"
        out.append({"flag": key, "names": [key], "label": label, "desc": desc, "section": "load",
                    # booleans are tri-state: leaving them unset keeps LM Studio's own default
                    "kind": {"boolean": "tristate", "number": "int"}.get(typ, "text"), "managed": key == "context_length"})
    _lms_cache[url] = out
    return out


# ------------------------------------------------------------------ refine ----
def refine(opts: List[Dict[str, Any]], model: Dict[str, Any], hw: Dict[str, Any], engine: str, role: str = "") -> List[Dict[str, Any]]:
    """Copy options with limits narrowed to the model and hardware; options that cannot apply are marked hidden."""
    threads = hw.get("cpu_threads") or 16
    layers = model.get("layers") or 0
    trained = model.get("trained_ctx") or 0
    is_moe = bool(model.get("experts")) or bool(re.search(r"moe|A\d+(\.\d+)?B", f"{model.get('arch')} {model.get('params')}", re.I))
    out = []
    for o in opts:
        o = dict(o)
        f = o["flag"]
        why = None
        if engine == "llama":
            if re.search(r"threads", f) and o["kind"] == "int":
                o.update(min=-1, max=threads, hint=f"{threads} CPU threads on this host")
            elif f in ("--n-gpu-layers", "--spec-draft-ngl") or "gpu-layers" in f:
                o.update(min=0, max=(layers + 1) if layers else 999)
            elif "cpu-moe" in f or "n-cpu-moe" in f:
                if not is_moe:
                    why = "only for Mixture-of-Experts models"
                elif o["kind"] == "int":
                    o.update(min=0, max=layers or 99)
            elif f in ("--batch-size", "--ubatch-size"):
                o.update(kind="enum", choices=[str(2 ** i) for i in range(5, 14)])
            elif f in ("--keep", "--cache-reuse") and trained:
                o.update(min=-1 if f == "--keep" else 0, max=trained)
            elif f.startswith("--yarn") or f.startswith("--rope"):
                o["hint"] = f"trained context {trained:,}" if trained else None
            elif f.startswith("--mmproj") and role != "vision":
                why = "only for the vision engine"
            elif f in ("--embedding", "--pooling", "--reranking", "--embd-normalize") and model.get("kind") != "embedding":
                why = "only for embedding models"
            elif f in ("--split-mode", "--tensor-split", "--main-gpu") and (hw.get("gpu_count") or 1) < 2:
                why = "needs more than one GPU visible to this engine"
            elif o["section"] == "speculative" and f not in ("--spec-type", "--spec-default"):
                o["needs_spec"] = True
            if o.get("kind") == "int" and "min" not in o:
                d = o.get("default")
                if d and re.fullmatch(r"-?\d+", d.split(",")[0].strip()):
                    o.setdefault("min", -1 if d.startswith("-") else 0)
        else:  # LM Studio
            if f == "context_length":
                o.update(min=512, max=trained or 131072, step=512)
            elif f == "parallel":
                o.update(min=1, max=8)
            elif f in ("eval_batch_size", "physical_batch_size"):
                o.update(kind="enum", choices=[str(2 ** i) for i in range(5, 14)])
            elif f == "num_experts":
                if not is_moe:
                    why = "only for Mixture-of-Experts models"
                else:
                    o.update(min=1, max=model.get("experts") or 16)
        if why:
            o.update(hidden=True, hidden_reason=why)
        out.append(o)
    return out
