"""Offline tests for the Docker renderer of the Lodestone installer (`python -m lodestone.installer --docker`): for each profile
and GPU kind, the Compose file and .env it writes. Nothing here needs Docker or a network: the files are parsed back with a
small reader for exactly the YAML subset the renderer emits (PyYAML too, when it happens to be installed), and checked
against the rules the native units follow (--device VulkanN never an integer, --flash-attn with a value, --jinja on chat,
--metrics), the GPU hand-off of each kind, the shared read-only model volume, the start order, service-name URLs, and the
absence of secrets and LAN addresses. When `docker` is installed, `docker compose config` also validates every file."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import installer as inst  # noqa: E402

UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'


def _gpu(n, vendor, device, kind, name):
    return ("GPU%d:\n\tvendorID           = 0x%04x\n\tdeviceID           = 0x%04x\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_%s\n"
            "\tdeviceName         = %s\n\tdriverName         = radv\n\n" % (n, vendor, device, kind, name))


def _vulkaninfo(*blocks):
    return "Devices:\n========\n" + "".join(blocks) + "Device Groups:\n=============\n"


RX6750 = _gpu(0, 0x1002, 0x73DF, "DISCRETE_GPU", "AMD Radeon RX 6750 XT")
RX6600 = _gpu(1, 0x1002, 0x73FF, "DISCRETE_GPU", "AMD Radeon RX 6600")
RTX = _gpu(0, 0x10DE, 0x2504, "DISCRETE_GPU", "NVIDIA GeForce RTX 3060")
IGPU = _gpu(0, 0x8086, 0x4680, "INTEGRATED_GPU", "Intel UHD Graphics 770")
LLVMPIPE = _gpu(2, 0x10005, 0, "CPU", "llvmpipe")
DRM2 = [{"vendor_id": 0x1002, "device_id": 0x73DF, "vram_mb": 12272}, {"vendor_id": 0x1002, "device_id": 0x73FF, "vram_mb": 8176}]
DRM1 = [{"vendor_id": 0x1002, "device_id": 0x73FF, "vram_mb": 8176}]


def machine(kind, system="Linux"):
    s = {"system": system, "machine": "x86_64", "cpu_count": 16, "euid": 0, "os_release": UBUNTU, "meminfo": None,
         "vulkaninfo": None, "nvidia_smi": None, "drm": [], "llama_server_path": None, "llama_server_version": None,
         "disk_free_mb": 500000, "config": None}
    if kind == "amd2":
        s.update(vulkaninfo=_vulkaninfo(RX6750, RX6600, LLVMPIPE), drm=DRM2)
    elif kind == "amd1":
        s.update(vulkaninfo=_vulkaninfo(RX6600), drm=DRM1)
    elif kind == "nvidia":                       # Vulkan lists the card, nvidia-smi gives its memory
        s.update(vulkaninfo=_vulkaninfo(RTX), nvidia_smi="NVIDIA GeForce RTX 3060, 12288\n")
    elif kind == "nvidia-smi-only":              # a Windows host: no vulkaninfo at all
        s.update(nvidia_smi="NVIDIA GeForce RTX 3060, 12288\n")
    elif kind == "igpu":
        s.update(vulkaninfo=_vulkaninfo(IGPU, LLVMPIPE))
    return inst.detect(s)


# ------------------------------------------------------------------------------------------------ YAML subset reader ----
def load_yaml(text):
    """The subset render_compose writes: block mappings and sequences, JSON-quoted strings, ints, booleans, [] flow lists of
    scalars, {} and `# comments`. Anything else raises, so a change in the writer cannot slip through unread."""
    rows = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rows.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    pos = [0]

    def scalar(tok):
        if tok == "{}":
            return {}
        if tok in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[tok]
        return json.loads(tok)               # "string", 12, ["a", "b"]

    def split_key(line):
        m = re.match(r'^([A-Za-z0-9_.-]+):(?: (.*))?$', line)
        if not m:
            raise ValueError("cannot read: " + line)
        return m.group(1), m.group(2)

    def block(indent):
        if rows[pos[0]][1].startswith("- "):
            return seq(indent)
        return mapping(indent)

    def mapping(indent, first=None):
        out = {}
        while pos[0] < len(rows) or first:
            if first:
                line, ind = first, indent
                first = None
            else:
                ind, line = rows[pos[0]]
                if ind < indent:
                    break
                if ind > indent:
                    raise ValueError("bad indent: " + line)
                pos[0] += 1
            key, rest = split_key(line)
            if rest is None:
                out[key] = block(rows[pos[0]][0]) if pos[0] < len(rows) and rows[pos[0]][0] > ind or (
                    pos[0] < len(rows) and rows[pos[0]][1].startswith("- ") and rows[pos[0]][0] >= ind) else None
            else:
                out[key] = scalar(rest)
        return out

    def seq(indent):
        out = []
        while pos[0] < len(rows):
            ind, line = rows[pos[0]]
            if ind != indent or not line.startswith("- "):
                break
            pos[0] += 1
            body = line[2:]
            if re.match(r'^[A-Za-z0-9_.-]+:(?: |$)', body):          # "- key: value" starts a mapping
                out.append(mapping(indent + 2, first=body))
            else:
                out.append(scalar(body))
        return out

    return mapping(0)


def render(machine_kind, **choices):
    d = machine(machine_kind, choices.pop("system", "Linux"))
    files, stack = inst.docker_files(d, choices)
    return files, stack, load_yaml(files["docker-compose.yml"])


def slurp(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def flag(args, name):
    return args[args.index(name) + 1]


class ReaderTests(unittest.TestCase):
    def test_reads_what_the_writer_writes(self):
        stack = inst.docker_stack(machine("amd2"), {})
        text = inst.render_compose(stack)
        self.assertEqual(load_yaml(text), json.loads(json.dumps(inst.compose_dict(stack))))

    def test_agrees_with_a_real_yaml_parser_when_there_is_one(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        for kind in ("amd2", "nvidia", "igpu"):
            stack = inst.docker_stack(machine(kind), {"gpu": "auto"})
            text = inst.render_compose(stack)
            self.assertEqual(yaml.safe_load(text), load_yaml(text), kind)

    def test_refuses_what_it_cannot_read(self):
        with self.assertRaises(ValueError):
            load_yaml("a:\n  - b: 1\n  *oops\n")


# ----------------------------------------------------------------------------------------------- the rules per file ----
CASES = [   # (machine, choices, expected profile, expected kind)
    ("amd2", {"gpu": "vulkan"}, "gpu-large", "vulkan"),
    ("amd1", {"gpu": "vulkan"}, "gpu-small", "vulkan"),
    ("amd2", {"gpu": "auto"}, "gpu-large", "vulkan"),
    ("nvidia", {"gpu": "cuda"}, "gpu-large", "cuda"),
    ("nvidia", {"gpu": "auto"}, "gpu-large", "cuda"),
    ("nvidia-smi-only", {"gpu": "auto", "system": "Windows"}, "gpu-large", "cuda"),
    ("nvidia", {"gpu": "cuda", "profile": "gpu-small"}, "gpu-small", "cuda"),
    ("amd2", {"gpu": "vulkan", "profile": "gpu-small"}, "gpu-small", "vulkan"),
    ("amd2", {"gpu": "cpu"}, "cpu", "cpu"),
    ("none", {"gpu": "auto"}, "cpu", "cpu"),
    ("none", {"gpu": "cpu", "profile": "cpu"}, "cpu", "cpu"),
    ("amd2", {"gpu": "vulkan", "vision": True}, "gpu-large", "vulkan"),
]
SECRET_SHAPES = [re.compile(p) for p in (r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", r"\b[0-9a-f]{32,}\b", r"gh[pousr]_[A-Za-z0-9]{20,}",
                                         r"-----BEGIN", r"\bsk-[A-Za-z0-9]{16,}")]


class ComposeRulesTests(unittest.TestCase):
    def check(self, mkind, choices, profile, kind):
        label = "%s %s" % (mkind, choices)
        files, stack, comp = render(mkind, **dict(choices))
        compose, env = files["docker-compose.yml"], files[".env"]
        self.assertEqual((stack["profile"], stack["gpu"]), (profile, kind), label)
        svcs = comp["services"]
        roles = {r["name"]: r for r in stack["roles"]}
        self.assertEqual(list(svcs), ["models"] + list(roles) + ["gateway", "stonesage"], label)
        self.assertIn("coordinator", roles, label)
        self.assertEqual(sorted(comp["volumes"]), ["models", "stonesage-config", "stonesage-data"], label)

        # the model downloader: one-shot, writes the volume, has every file of the plan
        models = svcs["models"]
        self.assertEqual(models["restart"], "no", label)
        self.assertIn("models:/models", models["volumes"], label)
        for r in roles.values():
            for f in r["files"]:
                self.assertIn("--repo %s --file %s --dest /models" % (f["repo"], f["file"]), models["command"][2], label)
        self.assertIn("lodestone.model_downloader", models["command"][2])
        self.assertIn("sha256sum -c SHA256SUMS", models["command"][2])
        self.assertEqual(models["environment"]["LODESTONE_VERIFY_MODELS"], "${LODESTONE_VERIFY_MODELS:-0}")

        image = "ghcr.io/ggml-org/llama.cpp:%s-%s" % ({"vulkan": "server-vulkan", "cuda": "server-cuda", "cpu": "server"}[kind],
                                                         inst.LLAMA_CPP_IMAGE_BUILD)
        for name, r in roles.items():
            s = svcs[name]
            args = s["command"]
            self.assertTrue(re.fullmatch(r"\$\{LLAMA_CPP_IMAGE:-%s\}" % re.escape(image), s["image"]), (label, s["image"]))
            self.assertRegex(s["image"], r"-b\d+\}$", "a concrete build, never a moving tag")
            self.assertEqual(s["restart"], "unless-stopped", label)
            # llama-server rules, as in the systemd units
            self.assertNotIn("llama-server", " ".join(args), "the image's entrypoint is llama-server")
            dev = flag(args, "--device")
            if r["gpu"] is None:
                self.assertEqual(dev, "none", (label, name))
                self.assertIn("-t", args)
                self.assertNotIn("-ngl", args)
            else:
                self.assertEqual(dev, "CUDA0" if kind == "cuda" else "Vulkan0", (label, name))
                self.assertRegex(dev, r"^(Vulkan|CUDA)\d$")
                self.assertEqual(flag(args, "-ngl"), "99")
            self.assertNotRegex(" ".join(args), r"--device \d", "never an integer")
            self.assertEqual(flag(args, "--host"), "0.0.0.0")
            self.assertEqual(flag(args, "--port"), str(r["port"]))
            self.assertEqual(flag(args, "--alias"), name)
            self.assertIn("--metrics", args)
            self.assertEqual(flag(args, "--model"), "/models/" + r["files"][0]["file"])
            if r["kind"] == "chat":
                self.assertIn(flag(args, "--flash-attn"), ("on", "off", "auto"), "always a value")
                self.assertIn("--jinja", args)
                self.assertEqual(flag(args, "-ctk"), r["kv"])
                self.assertEqual(flag(args, "-ctv"), r["kv"])
                self.assertEqual(flag(args, "-np"), str(r["parallel"]))
            else:
                self.assertNotIn("--jinja", args)
                self.assertNotIn("--flash-attn", args)
            if r["kind"] == "embedding":
                self.assertIn("--embedding", args)
            if r["kind"] == "vision":
                self.assertEqual(flag(args, "--mmproj"), "/models/" + r["files"][1]["file"])
            self.assertEqual(flag(args, "-c"), str(r["ctx"]))
            # models: shared named volume, read-only on engines, and the engines wait for the download to finish
            self.assertEqual(s["volumes"], ["models:/models:ro"], label)
            self.assertEqual(s["depends_on"], {"models": {"condition": "service_completed_successfully"}}, label)
            # health check on the engine's own port
            self.assertEqual(s["healthcheck"]["test"][:3], ["CMD", "curl", "-f"], label)
            self.assertEqual(s["healthcheck"]["test"][3], "http://localhost:%d/health" % r["port"], label)
            self.assertNotIn("ports", s, "engines are not published; the gateway is the one address")
            # GPU hand-off per kind
            has_dev = "devices" in s
            has_nv = "deploy" in s
            has_vk = "GGML_VK_VISIBLE_DEVICES" in s.get("environment", {})
            if r["gpu"] is None or kind == "cpu":
                self.assertFalse(has_dev or has_nv or has_vk or "group_add" in s, (label, name))
            elif kind == "vulkan":
                self.assertEqual(s["devices"], ["/dev/dri:/dev/dri"], label)
                self.assertEqual(s["environment"]["GGML_VK_VISIBLE_DEVICES"], str(r["gpu"]), (label, name))
                self.assertFalse(has_nv, label)
                self.assertEqual(len(s["group_add"]), 2)
            else:
                dev = s["deploy"]["resources"]["reservations"]["devices"]
                self.assertEqual(dev, [{"driver": "nvidia", "device_ids": [str(r["gpu"])], "capabilities": ["gpu"]}], label)
                self.assertFalse(has_dev or has_vk, label)

        # gateway and stonesage reach engines by service name
        for want in ("gateway", "stonesage"):
            senv = svcs[want]["environment"]
            for name, r in roles.items():
                var = {"coordinator": "COORDINATOR_URL", "worker": "WORKER_URL", "embedder": "EMBEDDER_URL", "vision": "VISION_URL"}[name]
                self.assertEqual(senv[var], "http://%s:%d/v1" % (name, r["port"]), (label, want))
            for var in ("COORDINATOR_URL", "WORKER_URL", "EMBEDDER_URL", "VISION_URL"):
                if var.split("_")[0].lower() not in roles:
                    self.assertNotIn(var, senv, (label, want, "an absent role is not configured"))
        gw = svcs["gateway"]
        self.assertEqual(flag(gw["command"], "--host"), "0.0.0.0")
        self.assertEqual(gw["environment"]["LODESTONE_API_KEY"], "${LODESTONE_API_KEY:-}")
        self.assertEqual(gw["ports"], ["${LODESTONE_BIND:-127.0.0.1}:${GATEWAY_PORT:-8000}:8000"], "loopback unless .env says otherwise")
        self.assertEqual(set(gw["depends_on"]), set(roles))
        self.assertTrue(all(v == {"condition": "service_healthy"} for v in gw["depends_on"].values()))
        ss = svcs["stonesage"]
        self.assertEqual(ss["environment"]["STONESAGE_APPS"], "hearth")
        self.assertEqual(ss["environment"]["HASS_TOKEN"], "${HASS_TOKEN:-}")
        self.assertEqual(ss["ports"], ["${STONESAGE_BIND:-127.0.0.1}:${STONESAGE_PORT:-8888}:8888"])
        self.assertEqual(ss["volumes"], ["stonesage-config:/config", "stonesage-data:/data"])
        self.assertEqual(set(ss["depends_on"]), {"coordinator", "embedder"} & set(roles) | {"coordinator"})
        for s in svcs.values():
            if "healthcheck" in s:
                self.assertIn("test", s["healthcheck"])

        # no secrets, no LAN, every variable the compose file uses has a default
        for name, text in (("compose", compose), (".env", env)):
            self.assertNotRegex(text, r"\b(?:10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)\b",
                                (label, name, "no LAN addresses"))
            for pat in SECRET_SHAPES:
                self.assertIsNone(pat.search(text), (label, name, pat.pattern))
        self.assertNotIn("192.168.", compose + env)
        for var in re.findall(r"\$\{([A-Z_]+)(:-|\})", compose):
            self.assertEqual(var[1], ":-", "%s has no default" % var[0])
        for line in env.splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if re.search(r"TOKEN|KEY|SECRET|PASSWORD", k):
                    self.assertEqual(v, "", "%s must be a placeholder to fill" % k)
        keys = dict(l.split("=", 1) for l in env.splitlines() if "=" in l and not l.startswith("#"))
        self.assertEqual(keys["LLAMA_CPP_IMAGE"], image)
        for var in set(v for v, _ in re.findall(r"\$\{([A-Z_]+)(:-|\})", compose)):
            if var not in ("VIDEO_GID", "RENDER_GID") or kind == "vulkan":
                self.assertIn(var, keys, "%s is used by docker-compose.yml but not set in .env" % var)
        self.assertEqual(sorted(k for k in keys if k.endswith("_GID")), ["RENDER_GID", "VIDEO_GID"] if kind == "vulkan" else [])
        self.assertTrue(compose.endswith("\n") and "\r" not in compose and "\r" not in env)

    def test_every_profile_and_gpu_kind(self):
        for mkind, choices, profile, kind in CASES:
            with self.subTest(machine=mkind, choices=choices):
                self.check(mkind, choices, profile, kind)

    def test_all_three_profiles_and_all_three_kinds_are_covered(self):
        self.assertEqual({c[2] for c in CASES}, {"gpu-large", "gpu-small", "cpu"})
        self.assertEqual({c[3] for c in CASES}, {"vulkan", "cuda", "cpu"})

    def test_two_cards_get_their_own_pin(self):
        _, stack, comp = render("amd2", gpu="vulkan")
        pins = {n: comp["services"][n]["environment"]["GGML_VK_VISIBLE_DEVICES"] for n in ("coordinator", "worker", "embedder")
                if "environment" in comp["services"][n]}
        self.assertEqual(pins["coordinator"], "0")
        self.assertNotEqual(pins["coordinator"], pins["worker"], "the worker goes where the coordinator is not")

    def test_vision_engine_carries_its_projector(self):
        _, stack, comp = render("amd2", gpu="vulkan", vision=True)
        self.assertIn("vision", comp["services"])
        self.assertIn("--mmproj", comp["services"]["vision"]["command"])
        self.assertEqual(comp["services"]["gateway"]["environment"]["VISION_URL"], "http://vision:8004/v1")

    def test_without_stonesage(self):
        files, stack, comp = render("amd2", gpu="vulkan", stonesage=False)
        self.assertNotIn("stonesage", comp["services"])
        self.assertEqual(list(comp["volumes"]), ["models"])
        self.assertNotIn("HASS_TOKEN", files[".env"])
        self.assertNotIn("STONESAGE_IMAGE", files[".env"])

    def test_docker_args_are_the_native_units_args(self):
        """One arg builder: the compose command equals the systemd unit's, apart from what a container must change."""
        d = machine("amd2")
        stack = inst.docker_stack(d, {"gpu": "vulkan", "vision": True})
        for r in stack["roles"]:
            native = inst.engine_argv(r, "/opt/lodestone", stack["threads"])[1:]
            docker = inst.engine_argv(r, "", stack["threads"], host="0.0.0.0", binary=False, models_dir="/models")
            fixed = [a.replace("/models/", "/opt/lodestone/models/") if a.startswith("/models/") else a for a in docker]
            fixed[fixed.index("--host") + 1] = inst.ENGINE_HOST
            self.assertEqual(fixed, native, r["name"])

    def test_role_env_names_are_the_ones_endpoints_reads(self):
        from lodestone import endpoints
        self.assertEqual(inst.ENGINE_ROLE_ENV, {r: endpoints.SERVICES[r][1] for r in inst.ENGINE_ROLE_ENV})

    def test_native_argv_is_unchanged_by_the_refactor(self):
        r = dict(inst.PROFILES["gpu-large"]["roles"]["coordinator"], name="coordinator", gpu=0)
        r["files"] = [inst.MODELS[r["model"]]]
        argv = inst.engine_argv(r, "/opt/lodestone", 8)
        self.assertEqual(argv[:6], ["/opt/lodestone/bin/llama-server", "--model", "/opt/lodestone/models/Qwen3-14B-Q4_K_M.gguf",
                                    "--host", "127.0.0.1", "--port"])
        self.assertEqual(flag(argv, "--device"), "Vulkan0")


# --------------------------------------------------------------------------------------------- choosing what to run ----
class ChoiceTests(unittest.TestCase):
    def test_no_gpu_is_cpu_everywhere(self):
        for system in ("Linux", "Darwin", "Windows"):
            _, stack, _ = render("none", gpu="auto", system=system)
            self.assertEqual((stack["profile"], stack["gpu"]), ("cpu", "cpu"), system)

    def test_a_requested_gpu_that_is_not_there_falls_back_to_cpu_with_a_note(self):
        for want in ("vulkan", "cuda"):
            _, stack, _ = render("none", gpu=want)
            self.assertEqual((stack["profile"], stack["gpu"]), ("cpu", "cpu"))
            self.assertTrue(any("running on the CPU" in n for n in stack["notes"]), stack["notes"])

    def test_forcing_a_gpu_profile_assumes_one_card_of_that_tier(self):
        _, stack, _ = render("none", gpu="vulkan", profile="gpu-small")
        self.assertEqual((stack["profile"], stack["gpu"]), ("gpu-small", "vulkan"))
        self.assertEqual([r["name"] for r in stack["roles"]], ["coordinator", "embedder"])
        self.assertEqual(stack["roles"][0]["gpu"], 0)

    def test_integrated_graphics_alone_is_cpu(self):
        _, stack, _ = render("igpu", gpu="auto")
        self.assertEqual((stack["profile"], stack["gpu"]), ("cpu", "cpu"))

    def test_amd_on_windows_or_mac_is_cpu_and_says_why(self):
        for system in ("Windows", "Darwin"):
            _, stack, _ = render("amd2", gpu="auto", system=system)
            self.assertEqual(stack["gpu"], "cpu", system)
            self.assertTrue(any("only on Linux" in n for n in stack["notes"]), stack["notes"])

    def test_nvidia_smi_alone_is_enough_for_cuda(self):
        _, stack, comp = render("nvidia-smi-only", gpu="auto", system="Windows")
        self.assertEqual(stack["gpu"], "cuda")
        self.assertIn("deploy", comp["services"]["coordinator"])

    def test_a_gpu_profile_on_a_cpu_stack_is_refused(self):
        with self.assertRaises(inst.InstallerError):
            inst.docker_stack(machine("amd2"), {"gpu": "cpu", "profile": "gpu-large"})
        with self.assertRaises(inst.InstallerError):
            inst.docker_stack(machine("none"), {"gpu": "auto", "profile": "gpu-large"})

    def test_unknown_names_are_refused(self):
        with self.assertRaises(inst.InstallerError):
            inst.docker_stack(machine("none"), {"gpu": "metal"})
        with self.assertRaises(inst.InstallerError):
            inst.docker_stack(machine("none"), {"profile": "huge"})

    def test_vision_is_left_out_on_cpu(self):
        _, stack, comp = render("none", gpu="auto", vision=True)
        self.assertNotIn("vision", comp["services"])
        self.assertTrue(any("vision skipped" in n for n in stack["notes"]))

    def test_cpu_is_the_default_profile_image(self):
        _, stack, comp = render("none")
        self.assertIn("server-b", comp["services"]["coordinator"]["image"])
        self.assertNotIn("vulkan", comp["services"]["coordinator"]["image"])


# ---------------------------------------------------------------------------------------------------- writing files ----
class WriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="docker-render-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def run_main(self, *argv, detected=None):
        lines = []
        code = inst.main(list(argv), detected=detected or machine("none"), out=lines.append)
        return code, lines

    def test_writes_compose_and_env(self):
        code, lines = self.run_main("--docker", "--out", self.tmp)
        self.assertEqual(code, 0, lines)
        self.assertEqual(sorted(os.listdir(self.tmp)), [".env", "docker-compose.yml"])
        load_yaml(slurp(os.path.join(self.tmp, "docker-compose.yml")))
        self.assertTrue(any("docker compose up -d" in l for l in lines))

    def test_rerun_backs_up_the_compose_file_and_keeps_your_env(self):
        self.run_main("--docker", "--out", self.tmp)
        env = os.path.join(self.tmp, ".env")
        with open(env, "a", encoding="utf-8") as f:
            f.write("MY_NOTE=1\n")
        with open(os.path.join(self.tmp, "docker-compose.yml"), "a", encoding="utf-8") as f:
            f.write("# hand edit\n")
        code, lines = self.run_main("--docker", "--out", self.tmp)
        self.assertEqual(code, 0)
        names = sorted(os.listdir(self.tmp))
        self.assertIn(".env.new", names)
        self.assertTrue(any(n.startswith("docker-compose.yml.bak-") for n in names), names)
        self.assertIn("MY_NOTE=1", slurp(env), "an existing .env is never overwritten")
        self.assertNotIn("MY_NOTE", slurp(os.path.join(self.tmp, ".env.new")))
        self.assertNotIn("# hand edit", slurp(os.path.join(self.tmp, "docker-compose.yml")))

    def test_identical_rerun_changes_nothing(self):
        self.run_main("--docker", "--out", self.tmp)
        self.run_main("--docker", "--out", self.tmp)
        self.assertEqual(sorted(os.listdir(self.tmp)), [".env", "docker-compose.yml"])

    def test_docker_with_apply_is_refused(self):
        code, _ = self.run_main("--docker", "--apply", "--out", self.tmp)
        self.assertEqual(code, 2)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_a_bad_choice_exits_1_and_writes_nothing(self):
        code, _ = self.run_main("--docker", "--gpu", "cpu", "--profile", "gpu-large", "--out", self.tmp)
        self.assertEqual(code, 1)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_json_prints_the_stack_and_writes_nothing(self):
        code, lines = self.run_main("--docker", "--json", "--out", self.tmp)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(lines[-1])["name"], "lodestone")
        self.assertEqual(os.listdir(self.tmp), [])

    def test_native_dry_run_is_unchanged(self):
        code, lines = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(any("DRY RUN" in l for l in lines))


# ----------------------------------------------------------------------------------------------- with Docker, if any ----
def docker_compose_available():
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(docker_compose_available(), "docker compose is not installed here")
class RealComposeTests(unittest.TestCase):
    def test_docker_compose_accepts_every_file(self):
        for mkind, choices, _profile, _kind in CASES:
            with self.subTest(machine=mkind, choices=choices):
                files, _stack, _comp = render(mkind, **dict(choices))
                with tempfile.TemporaryDirectory() as tmp:
                    for name, text in files.items():
                        with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
                            f.write(text)
                    done = subprocess.run(["docker", "compose", "--project-directory", tmp, "-f", os.path.join(tmp, "docker-compose.yml"),
                                           "config", "--quiet"], capture_output=True, text=True, timeout=60)
                    self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
