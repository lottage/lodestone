"""Install Lodestone on a fresh machine (docs/LAYERS.md M4): engines plus the gateway, from a bare Ubuntu 24.04.

    python -m lodestone.installer                       # review: detect the machine, print the plan, change nothing
    sudo python -m lodestone.installer --apply          # run it, step by step, stopping at the first failure
    options: --profile auto|<tier>   --prefix /opt/lodestone   --config PATH   --vision   --json
    python -m lodestone.installer --list-profiles       # the GPU ladder: which card gets which models

    python -m lodestone.installer --docker [--gpu auto|vulkan|cuda|cpu] [--out DIR] [--build|--no-build]
                                                         # the same plan as a Docker Compose stack

Two ways in. Docker (below, "Docker") is for users on any OS that runs Docker; this file's native steps (the numbered list)
are the bare-metal path: Ubuntu 24.04, systemd units, llama.cpp built from source.

What it does, in order (each step says why when printed):
    1. checks   Linux, root, Ubuntu/Debian, free disk, RAM
    2. packages apt: a compiler, cmake, git and (GPU profiles) the Vulkan headers, drivers and shader compiler
    3. user     a system user `lodestone` (member of render and video, which GPU access needs)
    4. llama.cpp   cloned at ONE pinned commit (LLAMA_CPP_REF) and built with cmake: Vulkan for the GPU profiles,
                   plain CPU otherwise; the binary lands in PREFIX/bin/llama-server
    5. models   the profile's GGUF files from Hugging Face through lodestone.model_downloader (atomic, resumable,
                SHA-256 recorded in PREFIX/models/SHA256SUMS)
    6. units    one systemd unit per engine (coordinator, worker, embedder, optional vision) and one for the gateway
    7. config   endpoint keys (cluster.*_url, gateway.*) merged into the config file: nothing is dropped, a key that is
                already set is kept, the old file is copied to <file>.bak-<time> first
    8. start    systemctl enable --now, then wait for every /health

Hardware: GPUs are driven through Vulkan (AMD, Intel and NVIDIA all have a Vulkan driver); no GPU, or one under 2 GB, means
the CPU profile. The profile is a rung of the GPU ladder (PROFILES below, `--list-profiles`): the largest discrete GPU's VRAM
picks the rung (`auto`), or --profile forces one, also on a card the probes cannot size. Every engine listens on 127.0.0.1
only; the gateway is the one address to expose (and it should get gateway.api_key first).

Design: detect() turns text (os-release, /proc/meminfo, `vulkaninfo --summary`, sysfs, nvidia-smi) into one dict through small
pure parsers; plan() turns that dict into an ordered list of steps (data, nothing executed); apply() runs them, or only
prints them (the default). A dry run still probes the machine, read-only (vulkaninfo, nvidia-smi, llama-server --version).
Standard library only. NOT yet tried on real hardware: see docs/LAYERS.md (M4).

Rules baked into every unit (learned the hard way on the project's own engines):
  --device VulkanN (never an integer) with GGML_VK_VISIBLE_DEVICES pinning the card, so VulkanN is "the Nth visible";
  --flash-attn always with a value; --jinja on the chat engines (native tool calls); --metrics on (the gateway and
  the metrics readers use it); an ExecStartPre that waits until Vulkan sees every GPU, because an engine that starts
  before the driver is ready silently runs on the CPU.
"""

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

UNKNOWN = "unknown"
DEFAULT_PREFIX = "/opt/lodestone"
SERVICE_USER = "lodestone"
ENGINE_HOST = "127.0.0.1"
GATEWAY_PORT = 8000

# llama.cpp: ONE pinned commit (full SHA), the build the project's own engines were verified on (2026-09-04, mainline).
# Newer is not better here: a flag that moves or disappears breaks every unit below. Change it deliberately: build it, run
# the engines, then edit this line. The build is skipped when PREFIX/bin/llama-server.ref already holds this value.
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_REF = "8b4b3558f1459c13e4aa38d5c94d306a00dc6acd"

# --------------------------------------------------------------------------------------------------- the tables ----
# Every model the installer can fetch: key -> Hugging Face repository + file name + size and SHA-256 as the repository's own
# file list gives them (api/models/<repo>?blobs=true; re-checked 2026-10-12, all 10 files exist and match). size_bytes lets a
# re-run skip a finished download without the network and feeds the free-disk check; sha256 is what `sha256sum -c` should
# print for the file (model_downloader records the digest of what it fetched in SHA256SUMS; this table is the reference).
MODELS: Dict[str, Dict[str, Any]] = {
    "qwen3-32b-q4_k_m": {"repo": "Qwen/Qwen3-32B-GGUF", "file": "Qwen3-32B-Q4_K_M.gguf", "size_bytes": 19762149024,
                         "sha256": "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"},
    "qwen3-14b-q4_k_m": {"repo": "Qwen/Qwen3-14B-GGUF", "file": "Qwen3-14B-Q4_K_M.gguf", "size_bytes": 9001752960,
                         "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"},
    "qwen3-8b-q4_k_m": {"repo": "Qwen/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q4_K_M.gguf", "size_bytes": 5027783488,
                        "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"},
    "qwen3-4b-q4_k_m": {"repo": "Qwen/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q4_K_M.gguf", "size_bytes": 2497280256,
                        "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"},
    "qwen3-0.6b-q8_0": {"repo": "Qwen/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q8_0.gguf", "size_bytes": 639446688,
                        "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"},
    "granite-4.0-h-tiny-q4_k_m": {"repo": "ibm-granite/granite-4.0-h-tiny-GGUF", "file": "granite-4.0-h-tiny-Q4_K_M.gguf",
                                  "size_bytes": 4230976352,
                                  "sha256": "5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd"},
    "qwen2.5-coder-3b-q5_k_m": {"repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "file": "qwen2.5-coder-3b-instruct-q5_k_m.gguf",
                                "size_bytes": 2438740416,
                                "sha256": "eb863f2a1a9b67e33bbf2dad98ea09c03b71c8052aeb4835171cf6f7a7a12db4"},
    "bge-large-en-v1.5-f16": {"repo": "CompendiumLabs/bge-large-en-v1.5-gguf", "file": "bge-large-en-v1.5-f16.gguf",
                              "size_bytes": 669603712,
                              "sha256": "3379a0e9cea28fc6d7136df8ea7a88ef99ccce5963b9a6f7af9609997be762e3"},
    "qwen2.5-vl-7b-q4_k_m": {"repo": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", "file": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                             "size_bytes": 4683072032,
                             "sha256": "9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392"},
    "qwen2.5-vl-7b-mmproj-f16": {"repo": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
                                 "file": "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "size_bytes": 1354162912,
                                 "sha256": "c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde"},
}

# ---- the GPU ladder ----
# One rung per common card size; the largest discrete GPU's VRAM picks the rung (the biggest `min_vram_mb` it reaches).
# HEADROOM, the arithmetic: a role's `vram_mb` is the room it needs (weights + KV cache + compute buffers); place_roles plans
# only VRAM_MARGIN (95 %) of a card, the rest being the driver's context and fragmentation. A rung's `min_vram_mb` is the
# marketed size minus ~4 % (cards report a little under their name: a "12 GB" card reads 11.9 to 12.3 GB, a "4 GB" one 4.0)
# and the rung's coordinator must fit in min x 0.95 (a test checks every rung). KV cache per token = layers x 2 x KV heads x
# head_dim x bytes per element (q8_0 1.06, q4_0 0.56); the numbers are from each model's config.json: Qwen3-4B 36 layers,
# 8 KV heads, 128 head_dim; Qwen3-14B 40/8/128; Qwen3-32B 64/8/128; Granite 4.0 h-tiny 40 layers of which only 4 are attention
# (its cache is tiny, the rest is mamba state). Role keys: `ngl` = layers on the GPU (default 99 = all; fewer = partial
# offload, the rest runs on the CPU), `draft` = a small model for speculative decoding, `fallback` = what happens when no GPU
# has the room left: "cpu" runs the role on the CPU, "omit" leaves it out, None = the install stops. kind: chat | embedding |
# vision. A figure marked "measured" was measured on the project's own engines; the others are arithmetic and say so.
_EMBEDDER: Dict[str, Any] = {"kind": "embedding", "model": "bge-large-en-v1.5-f16", "port": 8003, "ctx": 512,
                             "vram_mb": 600, "fallback": "cpu"}
_WORKER: Dict[str, Any] = {"kind": "chat", "model": "qwen2.5-coder-3b-q5_k_m", "port": 8002, "ctx": 8192, "parallel": 2,
                           "kv": "q8_0", "batch": (2048, 512), "vram_mb": 2350, "fallback": "omit"}
_COORD_14B: Dict[str, Any] = {   # measured on an RX 6750 XT: 8.96 GB with the draft, ~41 tok/s, Computer's tool eval 45/46
    "kind": "chat", "model": "qwen3-14b-q4_k_m", "draft": "qwen3-0.6b-q8_0", "port": 8001, "ctx": 12288, "parallel": 2,
    "kv": "q4_0", "batch": (2048, 1024), "vram_mb": 8960, "fallback": None}
_GRANITE: Dict[str, Any] = {     # measured on an RX 6600: 4.2 GB, 75 tok/s; tool eval 44/46 single, 41/46 mid, 0 wrong actions
    "kind": "chat", "model": "granite-4.0-h-tiny-q4_k_m", "port": 8001, "ctx": 8192, "parallel": 1, "kv": "q8_0",
    "batch": (2048, 512), "vram_mb": 4300, "fallback": None}

PROFILES: Dict[str, Dict[str, Any]] = {
    "gpu-24gb": {
        "min_vram_mb": 23000, "examples": "RTX 3090 / 4090, RX 7900 XTX, RTX A5000",
        "speed": "not measured",
        "note": "Qwen3-32B Q4_K_M (18.4 GiB of weights + 0.6 GiB KV at 8192 tokens, q4_0, one slot + ~0.6 GiB buffers = "
                "~20 GB of the 22 GB the margin allows) plus the embedder. Chosen over 14B at Q8_0 + vision because the extra "
                "VRAM buys the most as model quality (the 14B is already the proven size; 32B is the next Qwen3 step); "
                "vision stays one flag away on a second card or via --profile gpu-16gb. NOT measured: the 32B has not been "
                "run against Computer's tool eval.",
        "roles": {
            "coordinator": {"kind": "chat", "model": "qwen3-32b-q4_k_m", "port": 8001, "ctx": 8192, "parallel": 1,
                            "kv": "q4_0", "batch": (2048, 512), "vram_mb": 20200, "fallback": None},
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-16gb": {
        "min_vram_mb": 15500, "examples": "RTX 4060 Ti 16 GB, 4070 Ti Super, 4080, RTX A4000, RX 6800 / 6900 / 7800 XT",
        "speed": "~41 tok/s for the 14B on an RX 6750 XT (same engine as gpu-12gb)",
        "note": "The 12 GB stack's 14B (+ draft, 9.0 GB) with the vision model co-resident (Qwen2.5-VL-7B Q4_K_M + mmproj f16, "
                "6.4 GB measured): 8960 + 6500 = 15.5 GB, so the embedder runs on the CPU (a second card takes it) and "
                "there is no worker. Vision rather than a longer context: 12288 tokens x 2 slots already "
                "holds Computer's prompt with room, while the cameras are what a house needs next.",
        "roles": {
            "coordinator": dict(_COORD_14B),
            "vision": {"kind": "vision", "model": "qwen2.5-vl-7b-q4_k_m", "mmproj": "qwen2.5-vl-7b-mmproj-f16", "port": 8004,
                       "ctx": 4096, "parallel": 1, "vram_mb": 6500, "fallback": "omit"},
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-12gb": {       # the old "gpu-large"
        "min_vram_mb": 11500, "examples": "RTX 3060 12 GB, RTX 4070 / 4070 Ti, RX 6700 XT / 6750 XT, Arc B580",
        "speed": "~41 tok/s (RX 6750 XT, measured); Computer's tool eval 45/46",
        "note": "Qwen3-14B Q4_K_M with the Qwen3-0.6B draft (speculative decoding), KV q4_0, two slots of 6144 tokens: 9.0 GB "
                "measured. The 3B code worker takes the next 2.3 GB; on a single 12 GB card that leaves 0.3 GB, so the embedder "
                "(0.6 GB) runs on the CPU there (a second card takes it); a card under 11.9 GB loses the worker instead.",
        "roles": {
            "coordinator": dict(_COORD_14B),
            "worker": dict(_WORKER),
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-8gb": {        # the old "gpu-small"
        "min_vram_mb": 7500, "examples": "RTX 3060 Ti / 3070 / 4060, RX 6600 / 7600, Arc A750",
        "speed": "not measured",
        "note": "One 8B chat model (Qwen3-8B Q4_K_M, 5.0 GB, KV q8_0, 8192 tokens, one slot: ~6.3 GB) and the embedder; "
                "nothing else competes for the card. Granite 4.0 h-tiny (next rung down) is smaller and was the best small "
                "model in Computer's eval; the 8B stays here because the headroom lets it be the stronger reader.",
        "roles": {
            "coordinator": {"kind": "chat", "model": "qwen3-8b-q4_k_m", "port": 8001, "ctx": 8192, "parallel": 1,
                            "kv": "q8_0", "batch": (2048, 512), "vram_mb": 6300, "fallback": None},
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-6gb": {
        "min_vram_mb": 5500, "examples": "GTX 1060 6 GB, RTX 2060, RTX 3060 laptop 6 GB, RX 5600 XT",
        "speed": "75 tok/s (RX 6600, measured)",
        "note": "Granite 4.0 h-tiny Q4_K_M fully offloaded (4.2 GB measured; a mixture-of-experts model with ~1B active "
                "parameters) and the embedder: 4300 + 600 = 4.9 GB of the 5.2 GB the margin allows. The best small model "
                "tested: Computer's tool eval 44/46 single, 41/46 mid-conversation, 0 wrong actions.",
        "roles": {
            "coordinator": dict(_GRANITE),
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-4gb": {
        "min_vram_mb": 3500, "examples": "GTX 1650, RTX 3050 4 GB laptop, Quadro M2000 / P1000 / T1000, RX 6400 / 6500 XT",
        "speed": "13.1 tok/s through the gateway (Quadro M2000, measured 2026-10-11)",
        "note": "Qwen3-4B Q4_K_M fully offloaded: 2.3 GiB of weights + 0.45 GiB KV (6144 tokens, q8_0, one slot) + ~0.3 GiB "
                "buffers (batch 1024 / 256 keeps the logits buffer small) = ~3.1 GiB, planned as 3300 MB of the 3325 MB the "
                "margin allows at the rung's minimum. On a 4 GB card the embedder does not fit beside it and runs on the CPU. Granite h-tiny does not fit whole "
                "(4.2 GB), but a partial offload of it is the candidate to try next (not measured).",
        "roles": {
            "coordinator": {"kind": "chat", "model": "qwen3-4b-q4_k_m", "port": 8001, "ctx": 6144, "parallel": 1,
                            "kv": "q8_0", "batch": (1024, 256), "vram_mb": 3300, "fallback": None},
            "embedder": dict(_EMBEDDER),
        },
    },
    "gpu-2gb": {
        "min_vram_mb": 1900, "examples": "GT 1030, GTX 1050 2 GB, GTX 1060 3 GB, MX150 / MX250 laptop GPUs",
        "speed": "not measured (partial offload: the CPU's speed, a little better)",
        "note": "Granite 4.0 h-tiny Q4_K_M with 12 of its 40 layers on the card (~1.3 GB of weights + buffers = 1.6 GB of the "
                "1.8 GB the margin allows at the rung's minimum); the other 28 layers and the embedder run on the CPU. Chosen "
                "over Qwen3-4B (partly offloaded) and Qwen3-1.7B (whole) for tool-calling quality: Granite scored 44/46 on "
                "Computer's tool eval on a CPU against 42/46 for Qwen3-4B, with no answers invented from nothing, and a "
                "partly offloaded model is still faster than none. NOT measured on a 2 GB card.",
        "roles": {
            "coordinator": dict(_GRANITE, ngl=12, vram_mb=1600),
            "embedder": dict(_EMBEDDER, vram_mb=0, cpu_only=True),
        },
    },
    "cpu": {            # no usable GPU: a small mixture-of-experts model that decodes well on a CPU
        "min_vram_mb": 0, "examples": "no GPU, integrated graphics, a card under 2 GB",
        "speed": "20-40 tok/s on a modern desktop CPU (Granite: 35 on an i7-12700K, measured); a few on an old server CPU",
        "note": "Granite 4.0 h-tiny Q4_K_M and the embedder, both on the CPU.",
        "roles": {
            "coordinator": {"kind": "chat", "model": "granite-4.0-h-tiny-q4_k_m", "port": 8001, "ctx": 8192,
                            "parallel": 1, "kv": "q8_0", "batch": (2048, 512), "vram_mb": 0, "fallback": None,
                            "cpu_only": True},
            "embedder": {"kind": "embedding", "model": "bge-large-en-v1.5-f16", "port": 8003, "ctx": 512,
                         "vram_mb": 0, "fallback": None, "cpu_only": True},
        },
    },
}
# Old names, so earlier commands and the profile recorded in a config still work.
PROFILE_ALIASES = {"gpu-large": "gpu-12gb", "gpu-small": "gpu-8gb"}
# Optional (--vision): needs a GPU of its own with this much room left, else it is left out with a note.
VISION_ROLE: Dict[str, Any] = {"kind": "vision", "model": "qwen2.5-vl-7b-q4_k_m", "mmproj": "qwen2.5-vl-7b-mmproj-f16",
                               "port": 8004, "ctx": 4096, "parallel": 1, "vram_mb": 6500, "fallback": "omit"}
ROLE_ORDER = ("coordinator", "vision", "worker", "embedder")     # placement order: biggest first
VRAM_MARGIN = 0.95          # never plan to fill a card
BUILD_SPACE_MB = 3000       # llama.cpp source + build tree
VULKAN_PACKAGES = ["libvulkan-dev", "vulkan-tools", "mesa-vulkan-drivers", "libdrm-dev", "glslc", "glslang-tools",
                   "spirv-tools", "spirv-headers", "libspirv-tools-dev"]
BASE_PACKAGES = ["build-essential", "cmake", "git", "curl", "ca-certificates", "python3", "pkg-config"]
SPIRV_CMAKE_DIRS = ["/usr/local/share/cmake/SPIRV-Headers", "/usr/share/cmake/SPIRV-Headers"]
VENDORS = {0x1002: "amd", 0x10DE: "nvidia", 0x8086: "intel"}


class InstallerError(Exception):
    """Something the installer cannot (or must not) go on with; the message says what to do about it."""


# ------------------------------------------------------------------------------------------------------ parsers ----
def parse_os_release(text: Optional[str]) -> Dict[str, str]:
    out = {"id": UNKNOWN, "version_id": UNKNOWN, "pretty_name": UNKNOWN}
    for line in (text or "").splitlines():
        key, sep, val = line.partition("=")
        key = key.strip().lower()
        if sep and key in out:
            out[key] = val.strip().strip('"').strip("'") or UNKNOWN
    out["id"] = out["id"].lower()
    return out


def parse_meminfo(text: Optional[str]) -> Optional[int]:
    """MemTotal in MB, or None."""
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text or "", re.M)
    return int(m.group(1)) // 1024 if m else None


_GPU_HEAD = re.compile(r"^\s*GPU(?:\s+id\s*:)?\s*(\d+)\s*(?:\((.*)\))?\s*:\s*$")


def parse_vulkaninfo_summary(text: Optional[str]) -> List[Dict[str, Any]]:
    """Every Vulkan device `vulkaninfo --summary` lists, in its order (that order is what GGML_VK_VISIBLE_DEVICES indexes):
    {index, name, vendor_id, device_id, type (discrete | integrated | virtual | cpu | other), driver}."""
    devices: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for line in (text or "").splitlines():
        if line.strip().startswith("Device Groups"):
            break
        head = _GPU_HEAD.match(line)
        if head:
            cur = {"index": int(head.group(1)), "name": (head.group(2) or "").strip(), "vendor_id": None,
                   "device_id": None, "type": "other", "driver": ""}
            devices.append(cur)
            continue
        key, sep, val = line.partition("=")
        if cur is None or not sep:
            continue
        key, val = key.strip(), val.strip()
        if key in ("vendorID", "deviceID"):
            try:
                cur["vendor_id" if key == "vendorID" else "device_id"] = int(val, 16)
            except ValueError:
                pass
        elif key == "deviceType":
            m = re.search(r"PHYSICAL_DEVICE_TYPE_([A-Z]+)", val)
            cur["type"] = (m.group(1).lower() if m else "other")
        elif key == "deviceName" and val:
            cur["name"] = val
        elif key == "driverName":
            cur["driver"] = val
    return devices


def parse_nvidia_smi(text: Optional[str]) -> List[Dict[str, Any]]:
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` -> [{name, vram_mb}]."""
    out = []
    for line in (text or "").splitlines():
        name, sep, mem = line.rpartition(",")
        if sep and mem.strip().isdigit() and name.strip():
            out.append({"name": name.strip(), "vram_mb": int(mem.strip())})
    return out


def parse_llama_version(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """`llama-server --version` -> {build, commit} or None."""
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-fA-F]+)\)", text or "")
    return {"build": int(m.group(1)), "commit": m.group(2).lower()} if m else None


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _run_text(argv: List[str], timeout: float = 15) -> Optional[str]:
    """Output (stdout then stderr) of a read-only probe, or None when it is missing, fails to start or hangs."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return (done.stdout or "") + (done.stderr or "")


def read_drm(root: str = "/sys/class/drm", read: Callable[[str], Optional[str]] = _read_text,
             listdir: Callable[[str], List[str]] = os.listdir) -> List[Dict[str, Any]]:
    """amdgpu (and any driver that publishes it) tells its VRAM in sysfs: [{vendor_id, device_id, vram_mb}] per card."""
    out = []
    try:
        names = sorted(n for n in listdir(root) if re.fullmatch(r"card\d+", n))
    except OSError:
        return out
    for n in names:
        base = os.path.join(root, n, "device")
        vendor, device, vram = (read(os.path.join(base, f)) for f in ("vendor", "device", "mem_info_vram_total"))
        try:
            out.append({"vendor_id": int((vendor or "").strip(), 16), "device_id": int((device or "").strip(), 16),
                        "vram_mb": int((vram or "").strip()) // (1024 * 1024)})
        except ValueError:
            continue
    return out


def _merge(vulkan: List[Dict[str, Any]], drm: List[Dict[str, Any]], nvidia: List[Dict[str, Any]]
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(gpus, sysfs cards no Vulkan device matched, nvidia-smi cards no Vulkan device matched). See merge_gpus."""
    drm_left, smi_left = list(drm), list(nvidia)
    gpus = []
    for dev in vulkan:
        if dev["type"] == "cpu":
            continue
        vram, source = None, UNKNOWN
        hit = next((d for d in drm_left if d["vendor_id"] == dev["vendor_id"] and d["device_id"] == dev["device_id"]), None)
        if hit is not None:
            drm_left.remove(hit)
            vram, source = hit["vram_mb"], "sysfs"
        elif dev["vendor_id"] == 0x10DE and smi_left:
            vram, source = smi_left.pop(0)["vram_mb"], "nvidia-smi"
        gpus.append({"index": dev["index"], "name": dev["name"] or UNKNOWN, "vendor": VENDORS.get(dev["vendor_id"], UNKNOWN),
                     "type": dev["type"], "driver": dev["driver"], "vram_mb": vram, "vram_source": source,
                     "integrated": dev["type"] == "integrated"})
    return gpus, drm_left, smi_left


def merge_gpus(vulkan: List[Dict[str, Any]], drm: List[Dict[str, Any]], nvidia: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Real GPUs (the software renderer is dropped) with their VRAM, from sysfs (matched on vendor and device id, so two
    identical cards pair up in order) or, for NVIDIA's proprietary driver, from nvidia-smi (in order). vram_mb None = unknown."""
    return _merge(vulkan, drm, nvidia)[0]


VENDOR_NAMES = {"amd": "AMD", "nvidia": "NVIDIA", "intel": "Intel"}


def build_cards(gpus: List[Dict[str, Any]], drm_left: List[Dict[str, Any]], smi_left: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every card any probe found, once: {name, vendor, vram_mb (None = unknown), source (where the size came from: nvidia-smi |
    sysfs | vulkan = listed by Vulkan with no size), integrated, vulkan (True = Vulkan lists it, so the native install can use
    it), index (its Vulkan index)}. The Vulkan devices come first, then the cards only nvidia-smi or sysfs know (a host
    without vulkaninfo, or an NVIDIA driver without its Vulkan part): they are named, never lost."""
    cards = [{"name": g["name"], "vendor": g["vendor"], "vram_mb": g["vram_mb"],
              "source": g["vram_source"] if g["vram_mb"] else "vulkan", "integrated": g["integrated"], "vulkan": True,
              "index": g["index"]} for g in gpus]
    cards += [{"name": s["name"], "vendor": "nvidia", "vram_mb": s["vram_mb"], "source": "nvidia-smi", "integrated": False,
               "vulkan": False, "index": None} for s in smi_left]
    for x in drm_left:
        vendor = VENDORS.get(x["vendor_id"], UNKNOWN)
        cards.append({"name": "%s GPU (device 0x%04x)" % (VENDOR_NAMES.get(vendor, "Unknown"), x["device_id"]), "vendor": vendor,
                      "vram_mb": x["vram_mb"], "source": "sysfs", "integrated": False, "vulkan": False, "index": None})
    return cards


# --------------------------------------------------------------------------------------------------- detection ----
def gather(prefix: str = DEFAULT_PREFIX, config_path: str = "", *, read: Callable[[str], Optional[str]] = _read_text,
           run: Callable[[List[str]], Optional[str]] = _run_text, which: Callable[[str], Optional[str]] = shutil.which,
           drm: Callable[[], List[Dict[str, Any]]] = read_drm) -> Dict[str, Any]:
    """The raw facts of this machine, as text. Read-only. Everything a probe cannot get is None."""
    llama = next((p for p in (os.path.join(prefix, "bin", "llama-server"), which("llama-server")) if p and os.path.isfile(p)), None)
    config: Any = None
    if config_path:
        text = read(config_path)
        if text is not None:
            try:
                config = json.loads(text.lstrip("﻿"))
            except ValueError:
                config = "invalid"
    free = None
    probe = prefix
    while probe and not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    try:
        free = shutil.disk_usage(probe or "/").free // (1024 * 1024)
    except OSError:
        pass
    return {
        "system": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "os_release": read("/etc/os-release"), "meminfo": read("/proc/meminfo"),
        "vulkaninfo": run(["vulkaninfo", "--summary"]) if which("vulkaninfo") else None,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if which("nvidia-smi") else None,
        "drm": drm(), "llama_server_path": llama,
        "llama_server_version": run([llama, "--version"]) if llama else None,
        "disk_free_mb": free, "config": config,
    }


def detect(sources: Optional[Dict[str, Any]] = None, prefix: str = DEFAULT_PREFIX, config_path: str = "") -> Dict[str, Any]:
    """The machine as one dict. `sources` is what gather() returns (tests pass fixture text); None = look at this machine.
    A fact that cannot be found is "unknown" (words) or None (numbers), never an exception."""
    s = sources if sources is not None else gather(prefix, config_path)
    osr = parse_os_release(s.get("os_release"))
    vulkan_text = s.get("vulkaninfo")
    gpus, drm_left, smi_left = _merge(parse_vulkaninfo_summary(vulkan_text), s.get("drm") or [],
                                      parse_nvidia_smi(s.get("nvidia_smi")))
    version = parse_llama_version(s.get("llama_server_version"))
    system = s.get("system") or UNKNOWN
    return {
        "system": system, "is_linux": system == "Linux", "machine": s.get("machine") or UNKNOWN,
        "os": {"id": osr["id"], "version_id": osr["version_id"], "pretty": osr["pretty_name"]},
        "cpu_count": s.get("cpu_count"), "ram_mb": parse_meminfo(s.get("meminfo")),
        "euid": s.get("euid"), "is_root": (s.get("euid") == 0) if s.get("euid") is not None else None,
        "vulkan": UNKNOWN if vulkan_text is None else ("ok" if gpus else "no-gpu"),
        "gpus": gpus, "cards": build_cards(gpus, drm_left, smi_left), "nvidia": parse_nvidia_smi(s.get("nvidia_smi")),
        "llama_server": ({"path": s["llama_server_path"], **(version or {"build": None, "commit": None})}
                         if s.get("llama_server_path") else None),
        "disk_free_mb": s.get("disk_free_mb"), "config": s.get("config"),
    }


def gb(vram_mb: Optional[int]) -> str:
    return ("%.1f GB" % (vram_mb / 1024)) if vram_mb else "VRAM unknown"


def cards_of(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """d["cards"] (detect() makes it); for a hand-made dict that has only "gpus", the same cards built from those."""
    if d.get("cards") is not None:
        return d["cards"]
    return [{"name": g["name"], "vendor": g["vendor"], "vram_mb": g["vram_mb"],
             "source": g.get("vram_source") if g["vram_mb"] else "vulkan", "integrated": g["integrated"], "vulkan": True,
             "index": g["index"]} for g in d["gpus"]]


def describe(d: Dict[str, Any], native: bool = True) -> List[str]:
    """The machine report. native=False (the Docker path) leaves out the lines about the bare-metal install (Vulkan
    packages, a llama-server binary on this host): the Docker engines use their own images."""
    cards = cards_of(d)
    gpu = ["  %s, %s (%s%s)%s" % (c["name"], gb(c["vram_mb"]), c["source"], "" if c["vulkan"] else "; Vulkan does not list it",
                                  " (integrated)" if c["integrated"] else "") for c in cards]
    ls = d["llama_server"]
    vulkan = {"ok": "ok", "no-gpu": "installed, lists no GPU", UNKNOWN: "not installed (vulkaninfo missing)"}[d["vulkan"]]
    lines = ["machine:  %s %s, %s, %s" % (d["system"], d["machine"], d["os"]["pretty"],
                                           ("root" if d["is_root"] else "not root") if d["is_root"] is not None else "root unknown"),
             "cpu/ram:  %s threads, %s" % (d["cpu_count"] or UNKNOWN, ("%.1f GB" % (d["ram_mb"] / 1024)) if d["ram_mb"] else UNKNOWN)]
    if native:
        lines.append("vulkan:   %s" % vulkan)
    lines.append("gpus:     %s" % (("%d found (name, VRAM, where the size came from)" % len(cards)) if cards else
                                   "none found (looked at nvidia-smi, sysfs and vulkaninfo)"))
    lines += gpu
    if native:
        lines.append("llama-server: %s" % (("%s (build %s)" % (ls["path"], ls["build"] or UNKNOWN)) if ls else "not installed"))
    return lines


# ---------------------------------------------------------------------------------------------- profile + layout ----
def usable_gpus(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discrete GPUs with a known VRAM size: the ones a profile can be chosen from. (An integrated GPU shares system
    RAM, and a card whose VRAM cannot be read cannot be sized; --profile still lets a person force one.)"""
    return [g for g in d["gpus"] if g["vram_mb"] and not g["integrated"]]


def canonical_profile(name: str) -> str:
    """A profile name as PROFILES spells it (the old gpu-large / gpu-small map to gpu-12gb / gpu-8gb); InstallerError if unknown."""
    name = PROFILE_ALIASES.get(name, name)
    if name not in PROFILES:
        raise InstallerError("unknown profile %r (choose from auto, %s; see --list-profiles)" % (name, ", ".join(PROFILES)))
    return name


def profile_choices() -> List[str]:
    """What --profile accepts."""
    return ["auto"] + list(PROFILES) + list(PROFILE_ALIASES)


def tier_for(vram_mb: Optional[int]) -> str:
    """The rung of the ladder a card of this size reaches (the highest rung whose minimum it meets); cpu under the lowest."""
    for name, p in PROFILES.items():                      # largest first
        if name != "cpu" and vram_mb and vram_mb >= p["min_vram_mb"]:
            return name
    return "cpu"


MIN_GPU_MB = PROFILES["gpu-2gb"]["min_vram_mb"]


def explain_profile(cards: List[Dict[str, Any]], requested: str = "auto") -> Tuple[str, List[str]]:
    """(profile, notes). auto: the largest usable card sets the rung. A card is usable when it is discrete, its VRAM is
    known and (unless the row says vulkan=False) Vulkan lists it. Every card gets a note that names it with its size and says
    what became of it: "found Quadro M2000, 4.0 GB -> gpu-4gb", "found GT 710, 1.0 GB: below the 2 GB minimum, using the
    CPU". `requested` other than auto is returned as given (canonical name), whatever the cards are."""
    if requested != "auto":
        return canonical_profile(requested), []
    usable = [c for c in cards if c.get("vulkan", True) and not c.get("integrated") and c.get("vram_mb")]
    best = max(usable, key=lambda c: c["vram_mb"], default=None)
    profile = tier_for(best["vram_mb"]) if best else "cpu"
    if not cards:
        return profile, ["no GPU found (looked at nvidia-smi, sysfs and vulkaninfo): using the CPU"]
    notes: List[str] = []
    for c in cards:
        label = "%s, %s" % (c["name"], gb(c["vram_mb"])) if c.get("vram_mb") else c["name"]
        if c.get("integrated"):
            notes.append("found %s (integrated graphics share system RAM): not used" % label)
        elif not c.get("vram_mb"):
            notes.append("found %s: VRAM size unknown, not used (if you know the card, name its rung: --profile gpu-8gb, "
                         "for example; see --list-profiles)" % c["name"])
        elif not c.get("vulkan", True):
            notes.append("found %s (%s): Vulkan does not list it, so the native install cannot use it (install the vendor's "
                         "Vulkan driver, or use --docker, which can use an NVIDIA card through CUDA)%s"
                         % (label, c.get("source") or UNKNOWN, ", using the CPU" if profile == "cpu" else ""))
        elif c["vram_mb"] < MIN_GPU_MB:
            notes.append("found %s: below the %d GB minimum, %s" % (label, round(MIN_GPU_MB / 1000),
                                                                   "using the CPU" if profile == "cpu" else "not used"))
        elif c is best:
            notes.append("found %s -> %s" % (label, profile))
        else:
            notes.append("found %s (second card: takes the jobs that do not fit next to the main model)" % label)
    return profile, notes


def choose_profile(d: Dict[str, Any], requested: str = "auto") -> str:
    return explain_profile(cards_of(d), requested)[0]


def profile_notes(d: Dict[str, Any], requested: str = "auto") -> List[str]:
    return explain_profile(cards_of(d), requested)[1]


def ladder_rows() -> List[Dict[str, Any]]:
    """The ladder for --list-profiles, largest first: {name, min_vram_mb, max_vram_mb (None = no upper bound), examples, speed,
    models (one line per role), note}."""
    names = list(PROFILES)
    rows = []
    for i, name in enumerate(names):
        p = PROFILES[name]
        upper = PROFILES[names[i - 1]]["min_vram_mb"] if i else None
        roles = []
        for rname, r in p["roles"].items():
            if r.get("cpu_only") or r["vram_mb"] == 0:
                where = "CPU"
            elif r.get("ngl"):
                where = "%d layers on the GPU, the rest on the CPU" % r["ngl"]
            else:
                where = "GPU"
            keys = [r["model"]] + ([r["mmproj"]] if r.get("mmproj") else []) + ([r["draft"]] if r.get("draft") else [])
            m = " + ".join(MODELS[k]["file"] + (" (draft)" if k == r.get("draft") else "") for k in keys)
            roles.append("%s: %s, %.1f GB [%s]" % (rname, m, sum(MODELS[k]["size_bytes"] for k in keys) / 1e9, where))
        rows.append({"name": name, "min_vram_mb": p["min_vram_mb"], "max_vram_mb": upper, "examples": p["examples"],
                     "speed": p["speed"], "models": roles, "note": p["note"]})
    return rows


def format_ladder(out: Callable[[str], None] = print) -> None:
    out("GPU profiles: the largest GPU's VRAM picks the rung (--profile auto); --profile <name> forces one.")
    out("Old names still work: %s." % ", ".join("%s = %s" % kv for kv in PROFILE_ALIASES.items()))
    for r in ladder_rows():
        lo, hi = (round(v / 500.0) / 2.0 if v else None for v in (r["min_vram_mb"], r["max_vram_mb"]))    # 1900 -> "2", 11500 -> "11.5"
        if r["name"] == "cpu":
            rng = "no GPU, or one under 2 GB"
        elif hi is None:
            rng = "%g GB and up" % lo
        else:
            rng = "%g to %g GB" % (lo, hi)
        out("")
        out("%s  (%s)" % (r["name"], rng))
        out("  cards:  %s" % r["examples"])
        for m in r["models"]:
            out("  %s" % m)
        out("  speed:  %s" % r["speed"])
        out("  why:    %s" % r["note"])


def place_roles(profile: str, d: Dict[str, Any], vision: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(roles, notes): each role = its table entry plus name, gpu (the Vulkan index it is pinned to, None = CPU) and
    model files. Greedy: the coordinator takes the largest GPU, each later role the GPU with the most room left
    (VRAM_MARGIN of each card is all that may be planned), else its fallback (CPU, or left out)."""
    profile = canonical_profile(profile)
    spec = {name: dict(r) for name, r in PROFILES[profile]["roles"].items()}
    notes: List[str] = []
    if vision and "vision" not in spec:
        if profile == "cpu":
            notes.append("vision skipped: it needs a GPU")
        else:
            spec["vision"] = dict(VISION_ROLE)
    gpus = usable_gpus(d)
    if not gpus and profile != "cpu":      # forced profile on a card whose size is unknown: trust the person, assume the tier
        gpus = [dict(g, vram_mb=PROFILES[profile]["min_vram_mb"]) for g in d["gpus"] if g["type"] != "cpu"]
        if not gpus:
            raise InstallerError("profile %s needs a Vulkan GPU and none was found (is vulkan-tools installed and the "
                                 "driver loaded? use --profile cpu to run without one)" % profile)
    free = {g["index"]: g["vram_mb"] * VRAM_MARGIN for g in gpus}
    roles = []
    for name in ROLE_ORDER:
        if name not in spec:
            continue
        r = dict(spec[name], name=name, gpu=None)
        if not r.get("cpu_only"):
            fits = [i for i in free if free[i] >= r["vram_mb"]]
            if fits:
                r["gpu"] = max(fits, key=lambda i: free[i])
                free[r["gpu"]] -= r["vram_mb"]
            elif r["fallback"] is None:
                big = max((g["vram_mb"] for g in gpus), default=0)
                raise InstallerError("the %s needs about %.1f GB of VRAM and the largest GPU has %.1f GB: use --profile %s"
                                     % (name, r["vram_mb"] / 1024, big / 1024, tier_for(big)))
            elif r["fallback"] == "omit":
                notes.append("%s left out: no GPU has %.1f GB free" % (name, r["vram_mb"] / 1024))
                continue
            else:
                notes.append("%s runs on the CPU: no GPU has %.1f GB free" % (name, r["vram_mb"] / 1024))
        roles.append(r)
    for r in roles:
        r["files"] = [MODELS[r["model"]]] + ([MODELS[r["mmproj"]]] if r.get("mmproj") else []) + (
            [MODELS[r["draft"]]] if r.get("draft") else [])
    return roles, notes


# -------------------------------------------------------------------------------------------------- unit files ----
def cpu_threads(d: Dict[str, Any]) -> int:
    return max(1, min(d.get("cpu_count") or 4, 8))      # past ~8 threads decoding gets slower, not faster


CPU_EXPECTATION = [
    "CPU profile: no GPU Lodestone can use was found, so the language model runs on the processor.",
    "  Expect roughly 20-40 tokens/s for the small model on a modern desktop CPU, and only a few tokens/s on an older server",
    "  CPU: a short answer takes seconds to a minute, and Computer's first reply of a conversation (it reads a long prompt) can",
    "  take minutes there. Waits for the model are lengthened to match (600 s). A GPU is strongly recommended, above all for",
    "  Computer's voice use (it cannot feel like a conversation at CPU speed).",
]


PARTIAL_EXPECTATION = [
    "Partial-offload profile: the card is too small to hold the language model, so only some of its layers run on the GPU and",
    "  the rest on the processor. Expect about the speed of the CPU profile, a little better; a first reply that reads a long",
    "  prompt can take a minute or more. Waits for the model are lengthened to match. A card of 4 GB or more runs a model whole.",
]


def coordinator_on_cpu(roles: List[Dict[str, Any]]) -> bool:
    """True when the language model is slow enough to need longer waits: it runs on the CPU, or only partly on a GPU."""
    return any(r["name"] == "coordinator" and (r["gpu"] is None or r.get("ngl", 99) < 99) for r in roles)


def expectation_lines(roles: List[Dict[str, Any]]) -> List[str]:
    """What to tell the person when the language model is not whole on a GPU ([] when it is)."""
    for r in roles:
        if r["name"] == "coordinator":
            if r["gpu"] is None:
                return CPU_EXPECTATION
            if r.get("ngl", 99) < 99:
                return PARTIAL_EXPECTATION
    return []


def engine_argv(role: Dict[str, Any], prefix: str, threads: int, *, backend: str = "vulkan", host: str = ENGINE_HOST,
                binary: bool = True, models_dir: Optional[str] = None) -> List[str]:
    """llama-server's command line for one role (the rules are in the module docstring). Defaults give the systemd unit's
    line; the Docker stack passes binary=False (the image's entrypoint is llama-server), host="0.0.0.0" (reachable by the
    other containers only: nothing is published), models_dir="/models" and backend "cuda" for NVIDIA ("CUDA0")."""
    mdir = models_dir or prefix + "/models"
    a = [prefix + "/bin/llama-server"] if binary else []
    a += ["--model", "%s/%s" % (mdir, role["files"][0]["file"])]
    if role.get("mmproj"):
        a += ["--mmproj", "%s/%s" % (mdir, MODELS[role["mmproj"]]["file"])]
    if role.get("draft"):       # speculative decoding: the flags of the live coordinator unit (a small model drafts, the big one checks)
        a += ["--spec-draft-model", "%s/%s" % (mdir, MODELS[role["draft"]]["file"]), "--spec-draft-n-max", "12",
              "--spec-draft-n-min", "2", "--spec-draft-p-min", "0.65", "--spec-draft-ngl", "99"]
    a += ["--host", host, "--port", str(role["port"])]
    if role["gpu"] is None:
        a += ["--device", "none", "-t", str(2 if role["kind"] == "embedding" else threads)]
    else:
        ngl = role.get("ngl", 99)
        a += ["--device", "CUDA0" if backend == "cuda" else "Vulkan0", "-ngl", str(ngl)]
        if ngl < 99:            # partial offload: the layers left on the CPU need threads
            a += ["-t", str(threads)]
    if role["kind"] == "embedding":
        a += ["--embedding", "-c", str(role["ctx"]), "-b", str(role["ctx"]), "-ub", str(role["ctx"])]
    else:
        a += ["-c", str(role["ctx"]), "-np", str(role["parallel"])]
        if role["kind"] == "chat":
            a += ["-b", str(role["batch"][0]), "-ub", str(role["batch"][1]), "--flash-attn", "on",
                  "-ctk", role["kv"], "-ctv", role["kv"], "--jinja"]
    a += ["--alias", role["name"], "--metrics"]
    if role["kind"] != "embedding":
        a += ["--slots", "--props"]
    return a


def unit_name(role: str) -> str:
    return "lodestone-%s.service" % role


def _wrap(argv: List[str]) -> str:
    """argv as a systemd ExecStart: one flag with its value per line, continued with a backslash."""
    lines: List[List[str]] = [[argv[0]]]
    for tok in argv[1:]:
        if tok.startswith("-"):
            lines.append([tok])
        else:
            lines[-1].append(tok)
    return " \\\n    ".join(" ".join(x) for x in lines)


def render_engine_unit(role: Dict[str, Any], prefix: str, threads: int, gpu_count: int) -> str:
    model = role["files"][0]["file"]
    env = ""
    pre = ""
    if role["gpu"] is not None:
        env = 'Environment="GGML_VK_VISIBLE_DEVICES=%d"\n' % role["gpu"]
        pre = "ExecStartPre=%s/bin/wait-for-gpus.sh %d\n" % (prefix, gpu_count)
    where = "CPU" if role["gpu"] is None else "Vulkan GPU %d" % role["gpu"]
    return (
        "[Unit]\n"
        "Description=Lodestone %s (%s, %s)\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "User=%s\n"
        "Group=%s\n"
        "WorkingDirectory=%s/models\n"
        "%s"
        "TimeoutStartSec=150\n"
        "%s"
        "ExecStart=%s\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "NoNewPrivileges=true\n"
        "ProtectHome=true\n"
        "LimitNOFILE=65535\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    ) % (role["name"], model, where, SERVICE_USER, SERVICE_USER, prefix, env, pre,
         _wrap(engine_argv(role, prefix, threads)))


def render_gateway_unit(prefix: str, config_path: str, engine_units: List[str]) -> str:
    return (
        "[Unit]\n"
        "Description=Lodestone gateway (one OpenAI-compatible address in front of the engines)\n"
        "After=network.target %s\n"
        "Wants=%s\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "User=%s\n"
        "Group=%s\n"
        "WorkingDirectory=%s\n"
        'Environment="PYTHONPATH=%s/app"\n'
        'Environment="STONESAGE_CONFIG_FILE=%s"\n'
        "ExecStart=/usr/bin/python3 -m lodestone.gateway\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "NoNewPrivileges=true\n"
        "ProtectHome=true\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    ) % (" ".join(engine_units), " ".join(engine_units), SERVICE_USER, SERVICE_USER, prefix, prefix, config_path)


WAIT_FOR_GPUS = """#!/bin/sh
# ExecStartPre of Lodestone's GPU engines. At boot an engine can start before the GPU driver has brought every card up;
# llama-server then finds fewer Vulkan devices than it was pinned to and silently runs on the CPU (a 14B model at
# under 1 token/s, /health still 200). Waits until `vulkaninfo` lists at least $1 GPUs (default 1), up to 90 s.
want=${1:-1}
i=0
have=0
while [ "$i" -lt 90 ]; do
  have=$(vulkaninfo --summary 2>/dev/null | grep -c 'deviceType *= *PHYSICAL_DEVICE_TYPE_\\(DISCRETE\\|INTEGRATED\\|VIRTUAL\\)_GPU')
  [ "$have" -ge "$want" ] && exit 0
  i=$((i + 1))
  sleep 1
done
echo "wait-for-gpus: Vulkan lists $have of $want GPUs after 90 s" >&2
exit 1
"""


# ------------------------------------------------------------------------------------------------------ config ----
def merge_config(existing: Dict[str, Any], updates: Dict[str, Any],
                 overwrite: Tuple[str, ...] = ()) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """existing + updates, as (merged, added, kept). Nothing in `existing` is ever dropped. A key that is already set
    (not null, not "") is kept, not replaced, unless its dotted path is in `overwrite`; `kept` lists the dotted paths of
    wanted values that were left as they were because the person had set something else."""
    added: List[str] = []
    kept: List[str] = []

    def walk(base: Dict[str, Any], new: Dict[str, Any], path: str) -> Dict[str, Any]:
        out = dict(base)
        for key, val in new.items():
            here = path + key
            cur = out.get(key)
            if isinstance(val, dict) and isinstance(cur, dict) and here not in overwrite:
                out[key] = walk(cur, val, here + ".")
            elif cur in (None, "") or here in overwrite:
                if cur != val:
                    added.append(here)
                out[key] = val
            elif cur != val:
                kept.append(here)
        return out

    return walk(existing, updates, ""), added, kept


def render_config(merged: Dict[str, Any]) -> str:
    return json.dumps(merged, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------------------------------------- plan ----
def _check(cid: str, ok: bool, message: str, fatal: bool = True) -> Dict[str, Any]:
    return {"id": cid, "ok": bool(ok), "fatal": fatal, "message": message}


def _safe_path(label: str, value: str) -> str:
    if not re.fullmatch(r"/[A-Za-z0-9_./-]*", value or "") or ".." in value.split("/"):
        raise InstallerError("%s must be an absolute path of letters, digits and _ . / - only (got %r)" % (label, value))
    if value.startswith(("/home/", "/root/")) or value in ("/home", "/root"):
        raise InstallerError("%s must not be under /home or /root: the services run with ProtectHome and could not read it"
                             % label)
    return value.rstrip("/") or "/"


def preflight_checks(d: Dict[str, Any], roles: List[Dict[str, Any]], notes: List[str], prefix: str) -> List[Dict[str, Any]]:
    need_mb = BUILD_SPACE_MB + sum(f["size_bytes"] for r in roles for f in r["files"]) // (1024 * 1024)
    osd = d["os"]
    checks = [
        _check("linux", d["is_linux"], "runs on Linux (this is %s)" % d["system"]),
        _check("root", d["is_root"] is True, "run as root (sudo): it installs packages and systemd units"
               + ("" if d["is_root"] is not None else " (cannot tell here)")),
    ]
    if osd["id"] == "ubuntu" and osd["version_id"] == "24.04":
        checks.append(_check("os", True, "Ubuntu 24.04, the tested target"))
    elif osd["id"] in ("ubuntu", "debian"):
        checks.append(_check("os", False, "%s is not the tested Ubuntu 24.04: package names may differ" % osd["pretty"], False))
    elif osd["id"] == UNKNOWN:
        checks.append(_check("os", False, "operating system not recognised (no /etc/os-release); apt is required", False))
    else:
        checks.append(_check("os", False, "%s: only apt-based systems (Ubuntu 24.04) are supported" % osd["pretty"]))
    if d["machine"] not in ("x86_64", "aarch64", "amd64", "arm64"):
        checks.append(_check("arch", False, "CPU architecture %s is untested" % d["machine"], False))
    if d["ram_mb"] is not None and d["ram_mb"] < 8000:
        checks.append(_check("ram", False, "%.1f GB of RAM: building llama.cpp and loading a model may run out" % (d["ram_mb"] / 1024), False))
    if d["disk_free_mb"] is None:
        checks.append(_check("disk", False, "free disk space under %s unknown; about %.0f GB are needed" % (prefix, need_mb / 1024), False))
    else:
        checks.append(_check("disk", d["disk_free_mb"] >= need_mb, "%.0f GB free under %s, about %.0f GB needed"
                             % (d["disk_free_mb"] / 1024, prefix, need_mb / 1024)))
    checks += [_check("note", False, n, False) for n in notes]
    return checks


def plan(d: Dict[str, Any], choices: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The whole install as an ordered list of steps; nothing is run or written here.
    A step: {"id", "why"} plus any of "commands" (argv lists, run in order), "files" (path -> content, written before the
    commands) and "modes" (path -> chmod), "checks" (preflight), "skip_if" (the step is done already when ALL of its
    conditions hold: exists, any_exists, file_contains [path, text], file_size [path, bytes]), "env" (added to the
    commands' environment), "secret" (print a summary, not the content of its files), "summary"."""
    c = choices or {}
    prefix = _safe_path("--prefix", c.get("prefix") or DEFAULT_PREFIX)
    config_path = _safe_path("--config", c.get("config") or prefix + "/config.json")
    pkg_dir = c.get("package_dir") or os.path.dirname(os.path.abspath(__file__))
    if d.get("config") == "invalid" or (d.get("config") is not None and not isinstance(d["config"], dict)):
        raise InstallerError("%s exists but is not a JSON object: fix or move it, the installer will not overwrite it" % config_path)
    profile = choose_profile(d, c.get("profile") or "auto")
    roles, notes = place_roles(profile, d, bool(c.get("vision")))
    threads = cpu_threads(d)
    gpu_build = profile != "cpu"
    gpu_count = len([g for g in d["gpus"] if g["type"] != "cpu"])
    src, app, models = prefix + "/src/llama.cpp", prefix + "/app", prefix + "/models"
    binary, marker = prefix + "/bin/llama-server", prefix + "/bin/llama-server.ref"
    steps: List[Dict[str, Any]] = []

    steps.append({"id": "preflight", "why": "refuse early, before anything is changed, on a machine this cannot work on",
                  "checks": preflight_checks(d, roles, notes, prefix)})
    packages = BASE_PACKAGES + (VULKAN_PACKAGES if gpu_build else [])
    steps.append({"id": "apt-packages", "why": "compiler, cmake and git to build llama.cpp" + (
        "; Vulkan headers, drivers and the shader compiler for the GPU build" if gpu_build else " (CPU build, no Vulkan)"),
        "env": {"DEBIAN_FRONTEND": "noninteractive"},
        "commands": [["apt-get", "update"], ["apt-get", "install", "-y"] + packages]})
    if gpu_build:
        steps.append({"id": "spirv-headers", "why": "llama.cpp's Vulkan build wants SPIRV-Headers' CMake config; some releases "
                      "ship the headers without it, so build it from Khronos' repository when missing",
                      "skip_if": {"any_exists": SPIRV_CMAKE_DIRS},
                      "commands": [["git", "clone", "--depth", "1", "https://github.com/KhronosGroup/SPIRV-Headers.git", "/tmp/SPIRV-Headers"],
                                   ["cmake", "-S", "/tmp/SPIRV-Headers", "-B", "/tmp/SPIRV-Headers/build"],
                                   ["cmake", "--install", "/tmp/SPIRV-Headers/build"],
                                   ["rm", "-rf", "/tmp/SPIRV-Headers"]]})
    steps.append({"id": "service-user", "why": "the engines run as an unprivileged system user, which needs the render and "
                  "video groups to open the GPU", "commands": [
                      ["sh", "-c", "id -u %s >/dev/null 2>&1 || useradd --system --home-dir %s --no-create-home "
                       "--shell /usr/sbin/nologin %s" % (SERVICE_USER, prefix, SERVICE_USER)],
                      ["sh", "-c", "for g in render video; do getent group $g >/dev/null && usermod -aG $g %s; done; true" % SERVICE_USER],
                      ["install", "-d", "-m", "755", prefix, prefix + "/bin", prefix + "/src", models],
                  ]})
    steps.append({"id": "llama-cpp-source", "why": "fetch llama.cpp at the pinned commit %s (not 'latest': a moved flag "
                  "would break the units below)" % LLAMA_CPP_REF[:12],
                  "skip_if": {"file_contains": [marker, LLAMA_CPP_REF], "exists": binary},
                  "commands": [["sh", "-c", "test -d %s/.git || git clone %s %s" % (src, LLAMA_CPP_URL, src)],
                               ["git", "-C", src, "checkout", "--detach", LLAMA_CPP_REF]]})
    cmake = ["cmake", "-S", src, "-B", src + "/build", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
             "-DLLAMA_CURL=OFF"] + (["-DGGML_VULKAN=ON"] if gpu_build else [])
    steps.append({"id": "llama-cpp-build", "why": "build llama-server (%s) and install it as %s" % (
        "Vulkan backend" if gpu_build else "CPU only", binary),
        "skip_if": {"file_contains": [marker, LLAMA_CPP_REF], "exists": binary},
        "commands": [cmake,
                     ["cmake", "--build", src + "/build", "--config", "Release", "-j", str(max(1, d.get("cpu_count") or 2)),
                      "--target", "llama-server"],
                     ["install", "-m", "755", src + "/build/bin/llama-server", binary],
                     ["sh", "-c", "echo %s > %s" % (LLAMA_CPP_REF, marker)]]})
    steps.append({"id": "lodestone-code", "why": "copy this package to %s so the service user can run the gateway and "
                  "the downloader whatever directory the installer was started from" % app,
                  "commands": [["rm", "-rf", app + "/lodestone"], ["mkdir", "-p", app], ["cp", "-r", pkg_dir, app + "/lodestone"],
                               ["chmod", "-R", "a+rX", app]]})
    seen = set()
    for r in roles:
        for f in r["files"]:
            if f["file"] in seen:
                continue
            seen.add(f["file"])
            steps.append({"id": "model-" + f["file"], "why": "%s for the %s (%.1f GB, from huggingface.co/%s); SHA-256 goes to %s/SHA256SUMS"
                          % (f["file"], r["name"], f["size_bytes"] / 1e9, f["repo"], models),
                          "skip_if": {"file_size": [models + "/" + f["file"], f["size_bytes"]]}, "env": {"PYTHONPATH": app},
                          "commands": [["python3", "-m", "lodestone.model_downloader", "--repo", f["repo"], "--file", f["file"],
                                        "--dest", models]]})
    files: Dict[str, str] = {}
    modes: Dict[str, int] = {}
    if gpu_build:
        files[prefix + "/bin/wait-for-gpus.sh"] = WAIT_FOR_GPUS
        modes[prefix + "/bin/wait-for-gpus.sh"] = 0o755
    for r in roles:
        files["/etc/systemd/system/" + unit_name(r["name"])] = render_engine_unit(r, prefix, threads, gpu_count)
    engine_units = [unit_name(r["name"]) for r in roles]
    files["/etc/systemd/system/lodestone-gateway.service"] = render_gateway_unit(prefix, config_path, engine_units)
    steps.append({"id": "systemd-units", "why": "one unit per engine (pinned GPU, 127.0.0.1 only, restarts on failure) and one "
                  "for the gateway", "files": files, "modes": modes})

    cluster_updates: Dict[str, Any] = {"%s_url" % r["name"]: "http://%s:%d/v1" % (ENGINE_HOST, r["port"]) for r in roles}
    if coordinator_on_cpu(roles):
        cluster_updates["coordinator_cpu"] = True        # lengthens the waits for the model's answers (endpoints.py)
    updates: Dict[str, Any] = {"cluster": cluster_updates,
                               "gateway": {"host": ENGINE_HOST, "port": GATEWAY_PORT},
                               "lodestone": {"install": {"profile": profile, "prefix": prefix, "llama_cpp_ref": LLAMA_CPP_REF}}}
    merged, added, kept = merge_config(d.get("config") or {}, updates, overwrite=("lodestone.install",))
    steps.append({"id": "config", "why": "tell the gateway and everything above it where the engines are (%s); existing keys are "
                  "kept, the old file is backed up first" % config_path,
                  "files": {config_path: render_config(merged)}, "modes": {config_path: 0o640}, "secret": True,
                  "summary": {"added": added, "kept": kept}, "commands": [["chgrp", SERVICE_USER, config_path]]})
    gateway_health = "http://%s:%d/health" % (ENGINE_HOST, GATEWAY_PORT)
    steps.append({"id": "start", "why": "start the engines, then the gateway, and wait until each answers /health (a large "
                  "model takes a minute or two to load)", "commands": [["systemctl", "daemon-reload"],
                  ["systemctl", "enable", "--now"] + engine_units, ["systemctl", "enable", "--now", "lodestone-gateway.service"]] + [
                      ["sh", "-c", "for i in $(seq 1 90); do curl -fsS http://%s:%d/health >/dev/null && exit 0; sleep 2; done; "
                       "echo \"%s did not answer /health in 3 minutes: journalctl -u %s\" >&2; exit 1"
                       % (ENGINE_HOST, r["port"], r["name"], unit_name(r["name"]))] for r in roles] + [
                      ["sh", "-c", "for i in $(seq 1 30); do curl -fsS %s >/dev/null && exit 0; sleep 1; done; exit 1" % gateway_health]]})
    return steps


# ---------------------------------------------------------------------------------------------------------- apply ----
def _skip(cond: Dict[str, Any], read: Callable[[str], Optional[str]] = _read_text) -> bool:
    """True when every condition of a step's skip_if holds (so the step was done on an earlier run)."""
    for kind, arg in cond.items():
        if kind == "exists":
            ok = os.path.exists(arg)
        elif kind == "any_exists":
            ok = any(os.path.exists(p) for p in arg)
        elif kind == "file_contains":
            ok = arg[1] in (read(arg[0]) or "")
        elif kind == "file_size":
            try:
                ok = os.path.getsize(arg[0]) == arg[1]
            except OSError:
                ok = False
        else:
            ok = False
        if not ok:
            return False
    return bool(cond)


def _run_cmd(argv: List[str], env: Optional[Dict[str, str]] = None) -> int:
    try:
        return subprocess.run(argv, env=env).returncode
    except OSError as e:
        print("  cannot start %s: %s" % (argv[0], e), file=sys.stderr)
        return 127


def write_file(path: str, content: str, mode: Optional[int], stamp: str) -> str:
    """Write content to path (LF line endings, atomic). An existing different file is copied to <path>.bak-<stamp> first;
    an identical one is left alone. Returns "unchanged" | "written" | "backed up and written"."""
    old = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            old = f.read()
        if old == content:
            return "unchanged"
        shutil.copy2(path, "%s.bak-%s" % (path, stamp))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)
    return "backed up and written" if old is not None else "written"


def show(steps: List[Dict[str, Any]], out: Callable[[str], None] = print) -> None:
    """The dry run: every step's why and what it would run or write. Executes nothing."""
    for n, s in enumerate(steps, 1):
        out("\n[%d/%d] %s: %s" % (n, len(steps), s["id"], s["why"]))
        for c in s.get("checks", []):
            out("    %s %s" % ("ok  " if c["ok"] else ("FAIL" if c["fatal"] else "warn"), c["message"]))
        if s.get("skip_if"):
            out("    (skipped when already done: %s)" % json.dumps(s["skip_if"]))
        for argv in s.get("commands", []):
            out("    $ " + shlex.join(argv))
        for path, content in s.get("files", {}).items():
            if s.get("secret"):
                sm = s.get("summary", {})
                out("    write %s (existing keys kept; added: %s; left as set: %s)"
                    % (path, ", ".join(sm.get("added", [])) or "nothing", ", ".join(sm.get("kept", [])) or "nothing"))
            else:
                out("    write %s" % path)
                for line in content.splitlines():
                    out("    | " + line)


def apply(steps: List[Dict[str, Any]], dry_run: bool = True, *, system: Optional[str] = None, euid: Optional[int] = None,
          run: Callable[..., int] = _run_cmd, out: Callable[[str], None] = print,
          now: Callable[[], str] = lambda: time.strftime("%Y%m%d-%H%M%S")) -> List[Tuple[str, str]]:
    """Dry run (the default): print the steps. dry_run=False: execute them, in order, and stop at the first failure with
    an InstallerError that names the step and the command. Refuses to execute off Linux or without root. Returns
    [(step id, "done" | "skipped")]."""
    if dry_run:
        show(steps, out)
        return []
    system = system if system is not None else platform.system()
    euid = euid if euid is not None else (os.geteuid() if hasattr(os, "geteuid") else None)
    if system != "Linux":
        raise InstallerError("--apply only runs on Linux (this is %s): the installer targets Ubuntu 24.04. Without --apply it "
                             "prints the plan." % system)
    if euid != 0:
        raise InstallerError("--apply must run as root (try: sudo python3 -m lodestone.installer --apply)")
    results: List[Tuple[str, str]] = []
    stamp = now()
    for n, s in enumerate(steps, 1):
        out("[%d/%d] %s: %s" % (n, len(steps), s["id"], s["why"]))
        for c in s.get("checks", []):
            if not c["ok"]:
                out("    %s %s" % ("FAIL" if c["fatal"] else "warn", c["message"]))
        bad = [c for c in s.get("checks", []) if c["fatal"] and not c["ok"]]
        if bad:
            raise InstallerError("step %s: %s" % (s["id"], "; ".join(c["message"] for c in bad)))
        if s.get("skip_if") and _skip(s["skip_if"]):
            out("    already done, skipped")
            results.append((s["id"], "skipped"))
            continue
        for path, content in s.get("files", {}).items():
            try:
                out("    %s: %s" % (path, write_file(path, content, s.get("modes", {}).get(path), stamp)))
            except OSError as e:
                raise InstallerError("step %s: cannot write %s: %s" % (s["id"], path, e))
        env = dict(os.environ, **s["env"]) if s.get("env") else None
        for argv in s.get("commands", []):
            out("    $ " + shlex.join(argv))
            code = run(argv, env=env)
            if code != 0:
                raise InstallerError("step %s failed: `%s` exited with %s. Fix that and run the installer again: finished "
                                     "steps are skipped." % (s["id"], shlex.join(argv), code))
        results.append((s["id"], "done"))
    out("\nDone. The gateway answers on http://%s:%d/v1 (set gateway.api_key in the config before exposing it)."
        % (ENGINE_HOST, GATEWAY_PORT))
    return results


# ----------------------------------------------------------------------------------------------------- docker ----
# The same plan as a Docker Compose stack (`--docker`): for people on any OS that runs Docker, instead of the bare-metal
# steps above. Engines run in llama.cpp's own published server images; the models come from one named volume that a
# one-shot service fills with model_downloader; the gateway (and, unless --no-stonesage, StoneSage with Hearth) point at the
# engines by service name. Nothing is published except the gateway and the cockpit, both on 127.0.0.1 unless .env says
# otherwise. Support: CPU runs anywhere (Linux, Windows, macOS incl. Apple Silicon: the images are multi-arch); NVIDIA via the
# cuda image on Linux and on Windows Docker Desktop (WSL2); AMD and Intel GPUs via /dev/dri (Vulkan) on Linux only.

# llama.cpp's published images, ONE build for all three variants. LLAMA_CPP_REF (the commit the native units are verified on,
# 2026-09-04) has no image of its own (images exist for some builds only); b10818 is the nearest build that has all of
# server-vulkan, server-cuda and server (multi-arch: amd64 + arm64): it is that commit plus six commits that touch only OpenCL,
# Metal, SYCL and the web UI. To change it: edit LLAMA_CPP_IMAGE_BUILD, or set LLAMA_CPP_IMAGE in the generated .env
# (any ghcr.io/ggml-org/llama.cpp tag from https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp), after checking the
# flags in the module docstring still exist in that build.
LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp"
LLAMA_CPP_IMAGE_BUILD = "b10818"
LLAMA_CPP_IMAGE_VARIANT = {"vulkan": "server-vulkan", "cuda": "server-cuda", "cpu": "server"}
# The release this copy of the installer belongs to. The public-repo builder stamps it (`build_public_repos.py --tag`): it is
# the git tag of all three repositories, and the compose file names images by it (never `latest`) and builds from it. The
# source tree carries the number of the next release; a tree built without --tag carries the placeholder, which a release
# refuses (release_public_repos.py). Image versions drop the leading v: tag v0.1.0-rc.2 = image 0.1.0-rc.2.
RELEASE_TAG = "v0.1.0-rc.4"
UNSET_TAG = "UNSET-TAG"
GITHUB_ORG = "https://github.com/lottage"
IMAGE_REGISTRY = "ghcr.io/lottage"
IMAGE_REPOS = ("lodestone", "stonesage", "hearth")        # the three layers; each repository's Dockerfile builds its image
GPU_KINDS = ("vulkan", "cuda", "cpu")
MODELS_MOUNT = "/models"
GATEWAY_CONTAINER_PORT = 8000
STONESAGE_CONTAINER_PORT = 8888
# the environment variables lodestone.endpoints reads (a test checks they agree; this file imports nothing from the package)
ENGINE_ROLE_ENV = {"coordinator": "COORDINATOR_URL", "worker": "WORKER_URL", "embedder": "EMBEDDER_URL", "vision": "VISION_URL"}


def engine_image(kind: str) -> str:
    return "%s:%s-%s" % (LLAMA_CPP_IMAGE, LLAMA_CPP_IMAGE_VARIANT[kind], LLAMA_CPP_IMAGE_BUILD)


def image_version(tag: Optional[str] = None) -> str:
    tag = tag or RELEASE_TAG
    return tag[1:] if tag.startswith("v") else tag


def layer_image(repo: str, tag: Optional[str] = None) -> str:
    """ghcr.io/lottage/<repo>:<version> (the image the layer's Dockerfile builds): lodestone = the gateway and model
    downloader, stonesage = the platform, hearth = StoneSage + the Hearth app (what the cockpit service runs)."""
    return "%s/%s:%s" % (IMAGE_REGISTRY, repo, image_version(tag))


def build_context(repo: str, tag: Optional[str] = None) -> str:
    """A git URL docker compose builds from without any clone: the repository at the release tag."""
    return "%s/%s.git#%s" % (GITHUB_ORG, repo, tag or RELEASE_TAG)


def lodestone_pip(tag: Optional[str] = None) -> str:
    return "lodestone @ git+%s/lodestone@%s" % (GITHUB_ORG, tag or RELEASE_TAG)


def default_build(tag: Optional[str] = None) -> bool:
    """Build the images locally unless this version has published ones. Images are published with a final release (v0.1.0),
    never for a pre-release (v0.1.0-rc.2) or an unstamped tree; --build / --no-build overrides."""
    tag = tag or RELEASE_TAG
    return tag == UNSET_TAG or "-" in tag


def _gpu_row(index: int, name: str, vendor: str, vram_mb: Optional[int], source: str) -> Dict[str, Any]:
    return {"index": index, "name": name, "vendor": vendor, "type": "discrete", "driver": "", "vram_mb": vram_mb,
            "vram_source": source, "integrated": False}


def docker_gpus(d: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    """The GPUs a container of this kind could use, as detect() rows. cuda: NVIDIA cards, numbered as the container sees
    them (0..n-1; from Vulkan when it lists them, else from nvidia-smi alone, which is all a Windows host has). vulkan:
    every real Vulkan device, numbered as `vulkaninfo` lists them."""
    real = [g for g in d["gpus"] if g["type"] != "cpu"]
    if kind == "cuda":
        nv = [g for g in real if g["vendor"] == "nvidia"]
        if nv:
            return [dict(g, index=i) for i, g in enumerate(nv)]
        return [_gpu_row(i, s["name"], "nvidia", s["vram_mb"], "nvidia-smi") for i, s in enumerate(d.get("nvidia") or [])]
    return real if kind == "vulkan" else []


def choose_docker_gpu(d: Dict[str, Any], requested: str = "auto") -> Tuple[str, List[str]]:
    """(kind, notes): vulkan | cuda | cpu. auto: NVIDIA only -> cuda; an AMD or Intel discrete GPU on Linux -> vulkan;
    anything else -> cpu, which runs everywhere and is the safe default when no GPU is visible."""
    if requested != "auto":
        if requested not in GPU_KINDS:
            raise InstallerError("unknown --gpu %r (choose from auto, %s)" % (requested, ", ".join(GPU_KINDS)))
        return requested, []
    notes: List[str] = []
    nvidia = docker_gpus(d, "cuda")
    others = [g for g in d["gpus"] if g["type"] != "cpu" and g["vendor"] != "nvidia" and not g["integrated"]]
    if others and d.get("system") == "Linux":
        return "vulkan", notes
    if others:
        notes.append("an AMD or Intel GPU was found, but Docker passes those through only on Linux (/dev/dri): using the CPU")
    if nvidia and not others:
        return "cuda", notes
    if nvidia:
        notes.append("an NVIDIA GPU was found next to the Vulkan one; --gpu cuda uses it")
    return "cpu", notes


def docker_stack(d: Dict[str, Any], choices: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The plan for a Compose stack: {"profile", "gpu" (kind), "roles", "notes", "threads", "stonesage", "build", "tag"}. Reuses
    PROFILES, place_roles and engine_argv; a GPU request that finds no GPU big enough (or none at all) becomes the CPU profile,
    unless --profile names a GPU profile on purpose (then one card of that tier is assumed, as the native installer does).
    build: write `build:` sections (images made from the GitHub tag) instead of expecting published images; None = per the tag."""
    c = choices or {}
    requested_kind = c.get("gpu") or "auto"
    requested_profile = c.get("profile") or "auto"
    kind, notes = choose_docker_gpu(d, requested_kind)
    if requested_profile != "auto":
        requested_profile = canonical_profile(requested_profile)
    if kind == "cpu" and requested_profile not in ("auto", "cpu"):
        raise InstallerError("profile %s needs a GPU and --gpu is cpu (or no GPU Docker can use was found): use --profile cpu, "
                             "or --gpu vulkan / --gpu cuda" % requested_profile)
    gpus = docker_gpus(d, kind)
    if kind != "cpu" and not gpus and requested_profile in PROFILES and requested_profile != "cpu":
        gpus = [_gpu_row(0, "GPU 0 (assumed)", "nvidia" if kind == "cuda" else UNKNOWN, None, UNKNOWN)]
    view = dict(d, gpus=gpus, cards=[dict(g, source=g.get("vram_source") if g.get("vram_mb") else "vulkan", vulkan=True)
                                     for g in gpus])
    if kind == "cpu":
        profile, why = "cpu", []
    else:
        profile, why = explain_profile(view["cards"], requested_profile)
    if profile == "cpu" and kind != "cpu":
        notes += why or ["no %s GPU with enough memory was found: running on the CPU (pass --profile gpu-8gb, or the rung that "
                         "matches your card, to assume a GPU you know is there)" % kind]
        kind = "cpu"
    else:
        notes += why
    roles, role_notes = place_roles(profile, view, bool(c.get("vision")))
    build = c.get("build")
    tag = c.get("tag") or RELEASE_TAG
    return {"profile": profile, "gpu": kind, "roles": roles, "notes": notes + role_notes, "threads": cpu_threads(d),
            "stonesage": c.get("stonesage", True), "tag": tag, "build": default_build(tag) if build is None else bool(build)}


def _py_http_check(port: int, path: str) -> List[str]:
    return ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%d%s', timeout=3)" % (port, path)]


def compose_services(stack: Dict[str, Any]) -> Dict[str, Any]:
    """The `services:` mapping, in start order (models, engines, gateway, stonesage)."""
    roles, kind = stack["roles"], stack["gpu"]
    svc: Dict[str, Any] = {}
    fetch, seen = [], set()
    for r in roles:
        for f in r["files"]:
            if f["file"] not in seen:
                seen.add(f["file"])
                fetch.append("python -m lodestone.model_downloader --repo %s --file %s --dest %s"
                             % (shlex.quote(f["repo"]), shlex.quote(f["file"]), MODELS_MOUNT))
    verify = 'cd %s && ( [ "$$LODESTONE_VERIFY_MODELS" != 1 ] || sha256sum -c SHA256SUMS )' % MODELS_MOUNT
    tag, build = stack.get("tag") or RELEASE_TAG, bool(stack.get("build"))
    # build: the images are made from the repositories at the release tag (a git URL: no clone needed). `models` and `gateway`
    # run the same image, but only `gateway` builds it: two services with a build section for one image name fail in a
    # parallel build ("image ... already exists"). `models` gets the image name and pull_policy: never, so it uses the
    # image the gateway build made (compose builds every image before it starts any container, so no depends_on is needed:
    # a depends_on on the gateway would be a cycle, since the gateway waits for the engines, which wait for models).
    gateway_build = {"build": {"context": build_context("lodestone", tag)}} if build else {}
    models_image = {"pull_policy": "never"} if build else {}
    svc["models"] = {
        "image": "${LODESTONE_IMAGE:-%s}" % layer_image("lodestone", tag), **models_image, "restart": "no",
        "command": ["sh", "-c", " && ".join(fetch + [verify])],
        "environment": {"LODESTONE_VERIFY_MODELS": "${LODESTONE_VERIFY_MODELS:-0}"},
        "volumes": ["models:%s" % MODELS_MOUNT]}
    for r in roles:
        s: Dict[str, Any] = {
            "image": "${LLAMA_CPP_IMAGE:-%s}" % engine_image(kind), "restart": "unless-stopped",
            "command": engine_argv(r, "", stack["threads"], backend=kind, host="0.0.0.0", binary=False, models_dir=MODELS_MOUNT),
            "volumes": ["models:%s:ro" % MODELS_MOUNT]}
        if r["gpu"] is not None and kind == "vulkan":
            s["environment"] = {"GGML_VK_VISIBLE_DEVICES": str(r["gpu"])}
            s["devices"] = ["/dev/dri:/dev/dri"]
            s["group_add"] = ["${VIDEO_GID:-44}", "${RENDER_GID:-992}"]
        elif r["gpu"] is not None and kind == "cuda":
            s["deploy"] = {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "device_ids": [str(r["gpu"])], "capabilities": ["gpu"]}]}}}
        s["healthcheck"] = {"test": ["CMD", "curl", "-f", "http://localhost:%d/health" % r["port"]], "interval": "15s",
                            "timeout": "5s", "retries": 20, "start_period": "120s"}
        s["depends_on"] = {"models": {"condition": "service_completed_successfully"}}
        svc[r["name"]] = s
    urls = {ENGINE_ROLE_ENV[r["name"]]: "http://%s:%d/v1" % (r["name"], r["port"]) for r in roles}
    cpu_brain = {"COORDINATOR_CPU": "1"} if coordinator_on_cpu(roles) else {}     # longer waits for its answers
    healthy = {r["name"]: {"condition": "service_healthy"} for r in roles}
    svc["gateway"] = {
        "image": "${LODESTONE_IMAGE:-%s}" % layer_image("lodestone", tag), **gateway_build, "restart": "unless-stopped",
        "command": ["python", "-m", "lodestone.gateway", "--host", "0.0.0.0", "--port", str(GATEWAY_CONTAINER_PORT)],
        "environment": dict(urls, LODESTONE_API_KEY="${LODESTONE_API_KEY:?set LODESTONE_API_KEY in .env}", **cpu_brain),
        "ports": ["${LODESTONE_BIND:-127.0.0.1}:${GATEWAY_PORT:-%d}:%d" % (GATEWAY_CONTAINER_PORT, GATEWAY_CONTAINER_PORT)],
        "healthcheck": {"test": _py_http_check(GATEWAY_CONTAINER_PORT, "/health"), "interval": "30s", "timeout": "5s",
                        "retries": 5, "start_period": "20s"},
        "depends_on": healthy}
    if stack["stonesage"]:
        hearth_build: Dict[str, Any] = {}
        if build:
            # Hearth's Dockerfile is FROM the StoneSage image, which exists nowhere yet: `stonesage-image` builds it (a one-shot
            # that only has to exist) and the Hearth build takes it as a named context (`service:`), which also makes Compose
            # build it first. Build order is therefore lodestone, stonesage, hearth, from `docker compose up -d --build` alone.
            svc["stonesage-image"] = {
                "image": layer_image("stonesage", tag),
                "build": {"context": build_context("stonesage", tag), "args": {"LODESTONE_PIP": lodestone_pip(tag)}},
                "entrypoint": ["true"], "restart": "no"}
            hearth_build = {"build": {"context": build_context("hearth", tag), "args": {"STONESAGE_IMAGE": "stonesage-image"},
                                      "additional_contexts": {"stonesage-image": "service:stonesage-image"}}}
        svc["stonesage"] = {
            "image": "${STONESAGE_IMAGE:-%s}" % layer_image("hearth", tag), **hearth_build, "restart": "unless-stopped",
            "environment": dict(urls, STONESAGE_APPS="hearth", HASS_URL="${HASS_URL:-http://homeassistant.local:8123}",
                                HASS_TOKEN="${HASS_TOKEN:-}", TZ="${TZ:-UTC}",
                                # the cockpit never runs open: compose stops here when .env has no key
                                STONESAGE_API_KEY="${STONESAGE_API_KEY:?set STONESAGE_API_KEY in .env}",
                                STONESAGE_ALLOWED_HOSTS="${STONESAGE_ALLOWED_HOSTS:-}", **cpu_brain),
            "ports": ["${STONESAGE_BIND:-127.0.0.1}:${STONESAGE_PORT:-%d}:%d" % (STONESAGE_CONTAINER_PORT, STONESAGE_CONTAINER_PORT)],
            "volumes": ["stonesage-config:/config", "stonesage-data:/data"],
            # /api/health is the one public route (ok + version, nothing else), so the check needs no key
            "healthcheck": {"test": _py_http_check(STONESAGE_CONTAINER_PORT, "/api/health"), "interval": "30s",
                            "timeout": "5s", "retries": 5, "start_period": "60s"},
            "depends_on": dict({n: v for n, v in healthy.items() if n in ("coordinator", "embedder")},
                               **({"stonesage-image": {"condition": "service_completed_successfully"}} if build else {}))}
    return svc


def compose_dict(stack: Dict[str, Any]) -> Dict[str, Any]:
    vols: Dict[str, Any] = {"models": {}}
    if stack["stonesage"]:
        vols.update({"stonesage-config": {}, "stonesage-data": {}})
    return {"name": "lodestone", "services": compose_services(stack), "volumes": vols}


# A small YAML writer for exactly the shapes compose_dict makes (mappings, lists, strings, ints, booleans): every string is
# JSON-quoted, which YAML reads as a double-quoted scalar, so `${VAR:-x}` and shell commands need no escaping rules.
def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return json.dumps(str(v), ensure_ascii=False)


def _yaml_lines(node: Any, indent: int) -> List[str]:
    pad = "  " * indent
    out: List[str] = []
    if isinstance(node, dict):
        for key, val in node.items():
            if isinstance(val, dict) and val:
                out += [pad + "%s:" % key] + _yaml_lines(val, indent + 1)
            elif isinstance(val, dict):
                out.append(pad + "%s: {}" % key)
            elif isinstance(val, list) and (len(val) > 4 or any(isinstance(x, (dict, list)) for x in val)):
                out += [pad + "%s:" % key] + _yaml_lines(val, indent + 1)
            elif isinstance(val, list):
                out.append(pad + "%s: [%s]" % (key, ", ".join(_yaml_scalar(x) for x in val)))
            else:
                out.append(pad + "%s: %s" % (key, _yaml_scalar(val)))
    else:                                               # a list in block style
        for item in node:
            if isinstance(item, dict):
                sub = _yaml_lines(item, indent + 1)
                out += [pad + "- " + sub[0][len(pad) + 2:]] + sub[1:]
            else:
                out.append(pad + "- " + _yaml_scalar(item))
    return out


def render_compose(stack: Dict[str, Any]) -> str:
    comp = compose_dict(stack)
    kind = stack["gpu"]
    why = {
        "models": "one-shot: downloads the models into the shared volume (resumable; SHA-256 in SHA256SUMS), then exits",
        "stonesage-image": "one-shot, build only: makes the StoneSage image that the Hearth image below is built FROM, then exits",
        "gateway": "the one address to use: an OpenAI-compatible API in front of the engines (http://localhost:8000/v1)",
        "stonesage": "StoneSage with the Hearth app: the cockpit on http://localhost:8888; fill HASS_TOKEN in .env first",
    }
    for r in stack["roles"]:
        where = ("the CPU" if r["gpu"] is None else "GPU %d" % r["gpu"])
        why[r["name"]] = "%s engine (%s) on %s; reachable by the other services only" % (r["name"], r["files"][0]["file"], where)
    head = [
        "# Generated by `python -m lodestone.installer --docker` (profile %s, %s). Edit freely; re-generating writes a backup." % (
            stack["profile"], {"vulkan": "AMD/Intel GPUs through /dev/dri (Linux)", "cuda": "NVIDIA GPUs (nvidia-container-toolkit)",
                               "cpu": "CPU only"}[kind]),
        "# Engines use llama.cpp's published images (%s); the tag is LLAMA_CPP_IMAGE in .env." % engine_image(kind),
        ("# Start:  docker compose up -d --build     The images are built from GitHub (%s), then the models are downloaded "
         "(several GB): docker compose logs -f models" % stack["tag"]) if stack.get("build") else
        "# Start:  docker compose up -d     Models are downloaded on the first start (several GB): docker compose logs -f models",
        "# Secrets: none in this file; the API key and the Home Assistant token go in .env.",
    ]
    lines = head + ["name: %s" % _yaml_scalar(comp["name"]), "services:"]
    for name, body in comp["services"].items():
        lines += ["  # " + why[name], "  %s:" % name] + _yaml_lines(body, 2) + [""]
    lines += ["volumes:"] + _yaml_lines(comp["volumes"], 1)
    return "\n".join(lines) + "\n"


def render_env(stack: Dict[str, Any], keys: Optional[Dict[str, str]] = None) -> str:
    """`keys`: the API keys to write (the installer generates them); without them the key lines are empty, which makes
    `docker compose up` stop and say so (the public example file is rendered that way)."""
    kind = stack["gpu"]
    keys = keys or {}
    lines = [
        "# Settings for docker-compose.yml. The API keys below were generated by the installer; the Home Assistant token is yours to fill.",
        "",
        "# llama.cpp image for the engines (see the note on LLAMA_CPP_IMAGE_BUILD in lodestone/installer.py before changing it)",
        "LLAMA_CPP_IMAGE=%s" % engine_image(kind),
        "# The Lodestone and Hearth images (release %s): %s." % (
            stack.get("tag") or RELEASE_TAG,
            "built from GitHub by `docker compose up -d --build`, named by version, never latest" if stack.get("build")
            else "pulled by version, never latest"),
        "LODESTONE_IMAGE=%s" % layer_image("lodestone", stack.get("tag")),
    ]
    if stack["stonesage"]:
        lines.append("STONESAGE_IMAGE=%s" % layer_image("hearth", stack.get("tag")))
    lines += [
        "",
        "# The gateway. Bound to this computer only; set LODESTONE_BIND=0.0.0.0 to offer it on your network AND set a key first.",
        "LODESTONE_BIND=127.0.0.1",
        "GATEWAY_PORT=%d" % GATEWAY_CONTAINER_PORT,
        "LODESTONE_API_KEY=%s" % keys.get("LODESTONE_API_KEY", ""),
        "",
        "# Set to 1 to re-check every model file against its recorded SHA-256 each time the models service runs (slow for big files)",
        "LODESTONE_VERIFY_MODELS=0",
    ]
    if stack["stonesage"]:
        lines += [
            "",
            "# StoneSage and Hearth: the cockpit, and Home Assistant (Profile > Security > Long-lived access tokens)",
            "STONESAGE_BIND=127.0.0.1",
            "STONESAGE_PORT=%d" % STONESAGE_CONTAINER_PORT,
            "# The cockpit asks for this key (login page); scripts send it as `Authorization: Bearer <key>`. Keep this file private.",
            "STONESAGE_API_KEY=%s" % keys.get("STONESAGE_API_KEY", ""),
            "# Host names (not addresses) you reach the cockpit by, comma separated, e.g. mybox.local; addresses and localhost always work.",
            "STONESAGE_ALLOWED_HOSTS=",
            "HASS_URL=http://homeassistant.local:8123",
            "HASS_TOKEN=",
            "TZ=UTC",
        ]
    if kind == "vulkan":
        lines += [
            "",
            "# Group ids of /dev/dri on this computer (`getent group video render`); only needed if the container user is not root",
            "VIDEO_GID=44",
            "RENDER_GID=992",
        ]
    return "\n".join(lines) + "\n"


def docker_files(d: Dict[str, Any], choices: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """({"docker-compose.yml": text, ".env": text}, stack)."""
    stack = docker_stack(d, choices)
    return {"docker-compose.yml": render_compose(stack), ".env": render_env(stack, (choices or {}).get("keys"))}, stack


def write_docker_files(files: Dict[str, str], out_dir: str, stamp: str, out: Callable[[str], None] = print) -> None:
    """The compose file is replaced (the old one is kept as .bak-<time>). An existing .env is never touched: it may hold the
    keys you filled in; the new defaults go to .env.new next to it."""
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        path = os.path.join(out_dir, name)
        if name == ".env" and os.path.exists(path):
            with open(path, encoding="utf-8-sig", newline="") as f:
                same = f.read() == text
            if same:
                out("  %s: unchanged" % path)
                continue
            path += ".new"
            out("  kept your existing .env; new defaults are in %s" % path)
        out("  %s: %s" % (path, write_file(path, text, 0o600 if name == ".env" else None, stamp)))


def api_keys(out_dir: str) -> Dict[str, str]:
    """The keys for .env: the ones an existing .env already holds (re-running never changes your keys), else new random ones."""
    import secrets
    have: Dict[str, str] = {}
    try:
        with open(os.path.join(out_dir, ".env"), encoding="utf-8-sig") as f:
            for line in f:
                name, _, val = line.strip().partition("=")
                if name in ("STONESAGE_API_KEY", "LODESTONE_API_KEY") and val.strip():
                    have[name] = val.strip()
    except OSError:
        pass
    return {n: have.get(n) or secrets.token_urlsafe(32) for n in ("STONESAGE_API_KEY", "LODESTONE_API_KEY")}


def docker_main(args: argparse.Namespace, detected: Optional[Dict[str, Any]], out: Callable[[str], None]) -> int:
    d = detected if detected is not None else detect(None, DEFAULT_PREFIX, "")
    keys = api_keys(args.out)
    files, stack = docker_files(d, {"profile": args.profile, "gpu": args.gpu, "vision": args.vision,
                                    "stonesage": not args.no_stonesage, "keys": keys, "build": args.build})
    for line in describe(d, native=False):
        out(line)
    out("docker:    profile %s, %s engine images, release %s (%s)" % (
        stack["profile"], stack["gpu"], stack["tag"], "built from GitHub on first start" if stack["build"] else "pulled from ghcr.io"))
    for r in stack["roles"]:
        out("  %-12s %s on %s" % (r["name"], r["files"][0]["file"], "CPU" if r["gpu"] is None else "GPU %d" % r["gpu"]))
    for n in stack["notes"]:
        out("  note: " + n)
    for line in expectation_lines(stack["roles"]):
        out(line)
    if args.json:
        out(json.dumps(compose_dict(stack), indent=2))
        return 0
    write_docker_files(files, args.out, time.strftime("%Y%m%d-%H%M%S"), out)
    env_path = os.path.join(args.out, ".env")
    out("API keys: generated and written to %s (owner-only). %s" % (env_path, (
        "The cockpit's login asks for STONESAGE_API_KEY; the gateway wants LODESTONE_API_KEY as a bearer."
        if stack["stonesage"] else "The gateway wants LODESTONE_API_KEY as a bearer.")))
    out("Next: %srun `cd %s && docker compose up -d%s`" % (
        "fill HASS_TOKEN in .env, then " if stack["stonesage"] else "", shlex.quote(os.path.abspath(args.out)),
        " --build" if stack["build"] else ""))
    return 0


# ------------------------------------------------------------------------------------------------------------ cli ----
def main(argv: Optional[List[str]] = None, *, detected: Optional[Dict[str, Any]] = None,
         out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="python -m lodestone.installer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="really install (Linux, root); without it the plan is only printed")
    ap.add_argument("--profile", default="auto", choices=profile_choices(),
                    help="auto (default): the rung of the GPU ladder the largest GPU's VRAM reaches; or force a rung (--list-profiles)")
    ap.add_argument("--list-profiles", action="store_true", help="print the GPU ladder (rungs, cards, models, speed) and exit")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="install directory (default %s)" % DEFAULT_PREFIX)
    ap.add_argument("--config", help="config file to merge endpoint keys into (default PREFIX/config.json)")
    ap.add_argument("--vision", action="store_true", help="also install the vision engine (needs a GPU with ~6.5 GB to spare)")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON (nothing is executed)")
    ap.add_argument("--docker", action="store_true", help="render a Docker Compose stack (docker-compose.yml + .env) instead")
    ap.add_argument("--gpu", default="auto", choices=["auto"] + list(GPU_KINDS),
                    help="with --docker: vulkan (AMD/Intel, Linux), cuda (NVIDIA), cpu; default: from what is found, else cpu")
    ap.add_argument("--out", default=".", help="with --docker: directory for docker-compose.yml and .env (default: here)")
    ap.add_argument("--no-stonesage", action="store_true", help="with --docker: engines, models and gateway only")
    ap.add_argument("--build", action=argparse.BooleanOptionalAction, default=None,
                    help="with --docker: write build sections (images made from GitHub at the release tag, no clone) instead of "
                         "pulling published images; default: on for a pre-release or unstamped tree, off for a final release")
    args = ap.parse_args(argv)
    if args.list_profiles:
        if args.json:
            out(json.dumps(ladder_rows(), indent=2))
        else:
            format_ladder(out)
        return 0
    if args.docker and args.apply:
        print("installer: --docker writes files for `docker compose up`; --apply is the bare-metal install", file=sys.stderr)
        return 2
    if args.docker:
        try:
            return docker_main(args, detected, out)
        except InstallerError as e:
            print("installer: %s" % e, file=sys.stderr)
            return 1
    try:
        prefix = _safe_path("--prefix", args.prefix)
        config_path = _safe_path("--config", args.config or prefix + "/config.json")
        d = detected if detected is not None else detect(None, prefix, config_path)
        steps = plan(d, {"profile": args.profile, "prefix": prefix, "config": config_path, "vision": args.vision})
        if args.json:
            out(json.dumps(steps, indent=2))
            return 0
        for line in describe(d):
            out(line)
        profile, notes = explain_profile(cards_of(d), args.profile)
        roles, role_notes = place_roles(profile, d, args.vision)
        for n in notes + role_notes:
            out("  note: " + n)
        out("profile:   %s" % profile)
        for r in roles:
            out("  %-12s %s on %s, port %d" % (r["name"], r["files"][0]["file"],
                                               "CPU" if r["gpu"] is None else "GPU %d" % r["gpu"], r["port"]))
        for line in expectation_lines(roles):
            out(line)
        out("DRY RUN: nothing is changed. Re-run with --apply (as root, on Linux) to install." if not args.apply else "APPLYING:")
        apply(steps, dry_run=not args.apply, out=out)
    except InstallerError as e:
        print("installer: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
