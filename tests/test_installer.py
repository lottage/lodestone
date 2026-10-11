"""Offline tests for the Lodestone installer (lodestone/installer.py): the detection parsers on fixture text, the profile
choice and GPU placement, the unit files against the rules the project's engines learned, the config merge, the dry run
(nothing executes) and apply's refusals. No subprocess, no network, no writes outside a temp directory: subprocess.run is
replaced by a function that fails the test if anything calls it."""

import ast
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import installer as inst  # noqa: E402
from lodestone import model_downloader as dl  # noqa: E402

UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'
MEMINFO = "MemTotal:       32859524 kB\nMemFree:        20001234 kB\n"


def gpu_block(n, vendor, device, kind, name, driver="radv"):
    return ("GPU%d:\n\tapiVersion         = 1.3.274\n\tdriverVersion      = 24.0.5\n\tvendorID           = 0x%04x\n"
            "\tdeviceID           = 0x%04x\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_%s\n\tdeviceName         = %s\n"
            "\tdriverID           = DRIVER_ID_MESA_RADV\n\tdriverName         = %s\n\tdriverInfo         = Mesa 24.0.5\n\n"
            % (n, vendor, device, kind, name, driver))


def vulkaninfo(*blocks):
    return ("==========\nVULKANINFO\n==========\n\nVulkan Instance Version: 1.3.275\n\nDevices:\n========\n" + "".join(blocks)
            + "Device Groups:\n=============\nGroup 0:\n\tProperties:\n\t\tphysicalDevicesCount = 1\n\tDevices:\n"
              "\t\tGPU0 (AMD Radeon RX 6750 XT (RADV NAVI22))\n")


RX6750 = gpu_block(0, 0x1002, 0x73DF, "DISCRETE_GPU", "AMD Radeon RX 6750 XT (RADV NAVI22)")
RX6600 = gpu_block(1, 0x1002, 0x73FF, "DISCRETE_GPU", "AMD Radeon RX 6600 (RADV NAVI23)")
LLVMPIPE = gpu_block(2, 0x10005, 0x0, "CPU", "llvmpipe (LLVM 17.0.6, 256 bits)", "llvmpipe")
RTX = gpu_block(0, 0x10DE, 0x2504, "DISCRETE_GPU", "NVIDIA GeForce RTX 3060", "NVIDIA")
IGPU = gpu_block(0, 0x8086, 0x4680, "INTEGRATED_GPU", "Intel(R) UHD Graphics 770 (ADL-S GT1)", "intel open-source Mesa driver")
DRM_AMD2 = [{"vendor_id": 0x1002, "device_id": 0x73DF, "vram_mb": 12272}, {"vendor_id": 0x1002, "device_id": 0x73FF, "vram_mb": 8176}]


def sources(vulkan=None, drm=None, smi=None, **over):
    s = {"system": "Linux", "machine": "x86_64", "cpu_count": 16, "euid": 0, "os_release": UBUNTU, "meminfo": MEMINFO,
         "vulkaninfo": vulkan, "nvidia_smi": smi, "drm": drm or [], "llama_server_path": None,
         "llama_server_version": None, "disk_free_mb": 500000, "config": None}
    s.update(over)
    return s


def machine(kind, **over):
    if kind == "amd2":
        return inst.detect(sources(vulkaninfo(RX6750, RX6600, LLVMPIPE), DRM_AMD2, **over))
    if kind == "amd12":
        return inst.detect(sources(vulkaninfo(RX6750), DRM_AMD2[:1], **over))
    if kind == "amd11_5":      # just inside gpu-large: coordinator and worker fit, the embedder does not
        return inst.detect(sources(vulkaninfo(RX6750), [dict(DRM_AMD2[0], vram_mb=11500)], **over))
    if kind == "amd8":
        return inst.detect(sources(vulkaninfo(gpu_block(0, 0x1002, 0x73FF, "DISCRETE_GPU", "AMD Radeon RX 6600")),
                                   DRM_AMD2[1:], **over))
    if kind == "nvidia":
        return inst.detect(sources(vulkaninfo(RTX), None, "NVIDIA GeForce RTX 3060, 12288\n", **over))
    if kind == "igpu":
        return inst.detect(sources(vulkaninfo(IGPU), [{"vendor_id": 0x8086, "device_id": 0x4680, "vram_mb": 256}], **over))
    return inst.detect(sources(None, **over))      # no vulkaninfo at all


def unit_args(unit):
    """The ExecStart of a unit as a token list."""
    lines = unit.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("ExecStart="))
    joined = lines[start][len("ExecStart="):]
    while joined.endswith("\\"):
        start += 1
        joined = joined[:-1] + " " + lines[start].strip()
    return shlex.split(joined)


def units_of(steps):
    return {os.path.basename(p): c for s in steps for p, c in s.get("files", {}).items() if p.endswith(".service")}


def flag(args, name):
    return args[args.index(name) + 1]


class Fail:
    """Stands in for subprocess.run and friends: calling it fails the test."""

    def __call__(self, *a, **k):
        raise AssertionError("a subprocess was started: %r" % (a[:1],))


class ParserTests(unittest.TestCase):
    def test_os_release(self):
        r = inst.parse_os_release(UBUNTU)
        self.assertEqual((r["id"], r["version_id"], r["pretty_name"]), ("ubuntu", "24.04", "Ubuntu 24.04.1 LTS"))
        self.assertEqual(inst.parse_os_release(None)["id"], "unknown")
        self.assertEqual(inst.parse_os_release("garbage\n=\n")["version_id"], "unknown")

    def test_meminfo(self):
        self.assertEqual(inst.parse_meminfo(MEMINFO), 32089)
        self.assertIsNone(inst.parse_meminfo("nothing here"))
        self.assertIsNone(inst.parse_meminfo(None))

    def test_vulkaninfo_amd_pair_and_software_renderer(self):
        devs = inst.parse_vulkaninfo_summary(vulkaninfo(RX6750, RX6600, LLVMPIPE))
        self.assertEqual([d["index"] for d in devs], [0, 1, 2])
        self.assertEqual([d["type"] for d in devs], ["discrete", "discrete", "cpu"])
        self.assertEqual(devs[0]["name"], "AMD Radeon RX 6750 XT (RADV NAVI22)")
        self.assertEqual((devs[0]["vendor_id"], devs[0]["device_id"]), (0x1002, 0x73DF))
        # the "Device Groups" section repeats GPU0 by name: it must not become a fourth device
        self.assertEqual(len(devs), 3)

    def test_vulkaninfo_old_format_and_garbage(self):
        old = "GPU id : 0 (Some Card):\n\tvendorID = 0x10de\n\tdeviceID = 0x1234\n\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        devs = inst.parse_vulkaninfo_summary(old)
        self.assertEqual((devs[0]["index"], devs[0]["name"], devs[0]["vendor_id"]), (0, "Some Card", 0x10DE))
        self.assertEqual(inst.parse_vulkaninfo_summary("Cannot create Vulkan instance.\nERROR_INCOMPATIBLE_DRIVER"), [])
        self.assertEqual(inst.parse_vulkaninfo_summary(None), [])

    def test_nvidia_smi(self):
        self.assertEqual(inst.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288\nNVIDIA A, B, 24576\nbad line\n"),
                         [{"name": "NVIDIA GeForce RTX 3060", "vram_mb": 12288}, {"name": "NVIDIA A, B", "vram_mb": 24576}])
        self.assertEqual(inst.parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate with the driver"), [])

    def test_llama_version(self):
        self.assertEqual(inst.parse_llama_version("load_backend: ...\nversion: 6123 (8B4B355)\nbuilt with gcc"),
                         {"build": 6123, "commit": "8b4b355"})
        self.assertIsNone(inst.parse_llama_version("no version here"))

    def test_read_drm_from_a_sysfs_tree(self):
        with tempfile.TemporaryDirectory() as root:
            for card, vendor, device, vram in (("card0", "0x1002", "0x73df", str(12272 * 1024 * 1024)),
                                               ("card1", "0x1002", "0x73ff", str(8176 * 1024 * 1024)),
                                               ("card2", "0x8086", "0x4680", None)):
                d = os.path.join(root, card, "device")
                os.makedirs(d)
                for name, val in (("vendor", vendor), ("device", device), ("mem_info_vram_total", vram)):
                    if val is not None:
                        with open(os.path.join(d, name), "w") as f:
                            f.write(val + "\n")
            os.makedirs(os.path.join(root, "card0-HDMI-A-1"))       # a connector, not a card
            got = inst.read_drm(root)
        self.assertEqual(got, DRM_AMD2)       # card2 has no VRAM file and is skipped
        self.assertEqual(inst.read_drm("/does/not/exist"), [])

    def test_gpus_get_vram_from_sysfs_in_order_and_drop_the_software_renderer(self):
        d = machine("amd2")
        self.assertEqual([(g["index"], g["vram_mb"], g["vram_source"]) for g in d["gpus"]],
                         [(0, 12272, "sysfs"), (1, 8176, "sysfs")])
        self.assertEqual(d["vulkan"], "ok")

    def test_two_identical_cards_pair_up_in_order(self):
        two = [{"vendor_id": 0x1002, "device_id": 0x73DF, "vram_mb": 12272}, {"vendor_id": 0x1002, "device_id": 0x73DF, "vram_mb": 8000}]
        gpus = inst.merge_gpus(inst.parse_vulkaninfo_summary(vulkaninfo(RX6750, RX6750.replace("GPU0", "GPU1"))), two, [])
        self.assertEqual([g["vram_mb"] for g in gpus], [12272, 8000])

    def test_nvidia_vram_from_smi(self):
        d = machine("nvidia")
        self.assertEqual((d["gpus"][0]["vendor"], d["gpus"][0]["vram_mb"], d["gpus"][0]["vram_source"]), ("nvidia", 12288, "nvidia-smi"))

    def test_integrated_gpu_is_flagged(self):
        g = machine("igpu")["gpus"][0]
        self.assertTrue(g["integrated"])
        self.assertEqual(g["vendor"], "intel")

    def test_no_gpu_and_nothing_known_never_raises(self):
        d = machine("none")
        self.assertEqual((d["gpus"], d["vulkan"]), ([], "unknown"))
        bare = inst.detect({})
        self.assertEqual((bare["os"]["id"], bare["ram_mb"], bare["cpu_count"], bare["is_root"], bare["gpus"], bare["llama_server"]),
                         ("unknown", None, None, None, [], None))
        self.assertEqual(inst.detect(sources("Cannot create Vulkan instance."))["vulkan"], "no-gpu")
        self.assertTrue(inst.describe(bare))

    def test_llama_server_is_reported(self):
        d = inst.detect(sources(llama_server_path="/opt/x/bin/llama-server", llama_server_version="version: 7 (abc1234)"))
        self.assertEqual(d["llama_server"], {"path": "/opt/x/bin/llama-server", "build": 7, "commit": "abc1234"})

    def test_gather_with_every_probe_missing(self):
        got = inst.gather("/nonexistent/prefix", read=lambda p: None, run=lambda a: None, which=lambda n: None, drm=lambda: [])
        self.assertIsNone(got["vulkaninfo"])
        self.assertIsNone(got["nvidia_smi"])
        self.assertIsNone(got["llama_server_path"])
        self.assertEqual(inst.detect(got)["gpus"], [])

    def test_gather_reads_the_config_and_flags_a_broken_one(self):
        reads = {"/etc/c.json": '\ufeff{"a": 1}', "/etc/bad.json": "{not json"}
        kw = dict(read=reads.get, run=lambda a: None, which=lambda n: None, drm=lambda: [])
        self.assertEqual(inst.gather("/nonexistent", "/etc/c.json", **kw)["config"], {"a": 1})
        self.assertEqual(inst.gather("/nonexistent", "/etc/bad.json", **kw)["config"], "invalid")
        self.assertIsNone(inst.gather("/nonexistent", "/etc/none.json", **kw)["config"])


class ProfileTests(unittest.TestCase):
    def test_auto_picks_by_the_largest_discrete_gpu(self):
        self.assertEqual(inst.choose_profile(machine("amd2")), "gpu-large")
        self.assertEqual(inst.choose_profile(machine("amd12")), "gpu-large")      # 12 GB cards report a little under 12 GiB
        self.assertEqual(inst.choose_profile(machine("nvidia")), "gpu-large")
        self.assertEqual(inst.choose_profile(machine("amd8")), "gpu-small")
        self.assertEqual(inst.choose_profile(machine("none")), "cpu")

    def test_small_unknown_and_integrated_gpus_mean_cpu(self):
        four = inst.detect(sources(vulkaninfo(gpu_block(0, 0x1002, 0x1, "DISCRETE_GPU", "Small")),
                                   [{"vendor_id": 0x1002, "device_id": 0x1, "vram_mb": 4096}]))
        self.assertEqual(inst.choose_profile(four), "cpu")
        self.assertEqual(inst.choose_profile(machine("igpu")), "cpu")
        unknown_vram = inst.detect(sources(vulkaninfo(gpu_block(0, 0x1002, 0x1, "DISCRETE_GPU", "Mystery"))))
        self.assertEqual(inst.choose_profile(unknown_vram), "cpu")

    def test_explicit_profile_wins_and_a_bad_one_is_refused(self):
        self.assertEqual(inst.choose_profile(machine("none"), "gpu-small"), "gpu-small")
        with self.assertRaises(inst.InstallerError):
            inst.choose_profile(machine("none"), "huge")

    def test_two_gpus_layout(self):
        roles, notes = inst.place_roles("gpu-large", machine("amd2"))
        # the coordinator keeps its card to itself; the smaller jobs go where the most room is left
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "worker": 1, "embedder": 1})
        self.assertEqual(notes, [])

    def test_one_12gb_gpu_holds_all_three(self):
        roles, notes = inst.place_roles("gpu-large", machine("amd12"))
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "worker": 0, "embedder": 0})
        self.assertEqual(notes, [])

    def test_when_the_embedder_does_not_fit_it_runs_on_the_cpu(self):
        d = machine("amd11_5")
        self.assertEqual(inst.choose_profile(d), "gpu-large")
        roles, notes = inst.place_roles("gpu-large", d)
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "worker": 0, "embedder": None})
        self.assertTrue(any("embedder runs on the CPU" in n for n in notes))
        steps = inst.plan(d, {})
        self.assertEqual(flag(unit_args(units_of(steps)["lodestone-embedder.service"]), "--device"), "none")

    def test_when_the_worker_does_not_fit_it_is_left_out(self):
        d = inst.detect(sources(vulkaninfo(RX6750), [dict(DRM_AMD2[0], vram_mb=11500)]))
        d["gpus"][0]["vram_mb"] = 9500
        roles, notes = inst.place_roles("gpu-large", d)
        self.assertEqual([r["name"] for r in roles], ["coordinator", "embedder"])
        self.assertTrue(any("worker left out" in n for n in notes))

    def test_gpu_small_has_no_worker(self):
        roles, _ = inst.place_roles("gpu-small", machine("amd8"))
        self.assertEqual([r["name"] for r in roles], ["coordinator", "embedder"])
        self.assertEqual(roles[0]["files"][0]["file"], "Qwen3-8B-Q4_K_M.gguf")

    def test_cpu_profile_roles(self):
        roles, _ = inst.place_roles("cpu", machine("none"))
        self.assertEqual([(r["name"], r["gpu"]) for r in roles], [("coordinator", None), ("embedder", None)])
        self.assertEqual(roles[0]["files"][0]["file"], "granite-4.0-h-tiny-Q4_K_M.gguf")

    def test_vision_needs_a_gpu_of_its_own(self):
        roles, notes = inst.place_roles("gpu-large", machine("amd2"), vision=True)
        by = {r["name"]: r["gpu"] for r in roles}
        self.assertEqual((by["coordinator"], by["vision"]), (0, 1))
        self.assertEqual(len([f for r in roles if r["name"] == "vision" for f in r["files"]]), 2)   # model + mmproj
        roles, notes = inst.place_roles("gpu-large", machine("amd12"), vision=True)
        self.assertNotIn("vision", [r["name"] for r in roles])
        self.assertTrue(any("vision left out" in n for n in notes))
        roles, notes = inst.place_roles("cpu", machine("none"), vision=True)
        self.assertTrue(any("vision skipped" in n for n in notes))

    def test_a_forced_profile_that_cannot_fit_is_refused_with_advice(self):
        with self.assertRaises(inst.InstallerError) as cm:
            inst.place_roles("gpu-large", machine("amd8"))
        self.assertIn("--profile gpu-small", str(cm.exception))
        with self.assertRaises(inst.InstallerError) as cm:
            inst.place_roles("gpu-large", machine("none"))
        self.assertIn("--profile cpu", str(cm.exception))

    def test_forced_profile_on_a_card_of_unknown_size_trusts_the_person(self):
        d = inst.detect(sources(vulkaninfo(gpu_block(0, 0x1002, 0x1, "DISCRETE_GPU", "Mystery"))))
        roles, _ = inst.place_roles("gpu-small", d)
        self.assertEqual(roles[0]["gpu"], 0)

    def test_every_model_in_the_table_is_complete(self):
        for key, m in inst.MODELS.items():
            self.assertRegex(m["repo"], r"^[\w.-]+/[\w.-]+$", key)
            self.assertTrue(m["file"].endswith(".gguf"), key)
            self.assertGreater(m["size_bytes"], 10 ** 8, key)
        for name, p in inst.PROFILES.items():
            for r in p["roles"].values():
                self.assertIn(r["model"], inst.MODELS, name)
        self.assertIn(inst.VISION_ROLE["model"], inst.MODELS)
        self.assertIn(inst.VISION_ROLE["mmproj"], inst.MODELS)


class UnitRuleTests(unittest.TestCase):
    """The rules in the module docstring, asserted on every unit of every profile."""

    PLANS = [("amd2", "gpu-large", False), ("amd2", "gpu-large", True), ("amd12", "gpu-large", False),
             ("amd8", "gpu-small", False), ("nvidia", "auto", False), ("none", "cpu", False), ("amd2", "cpu", False)]

    def each_unit(self):
        for kind, profile, vision in self.PLANS:
            steps = inst.plan(machine(kind), {"profile": profile, "vision": vision})
            for name, text in units_of(steps).items():
                yield kind, profile, name, text, unit_args(text) if name != "lodestone-gateway.service" else None

    def test_there_is_a_unit_per_role_and_one_for_the_gateway(self):
        steps = inst.plan(machine("amd2"), {"profile": "gpu-large", "vision": True})
        self.assertEqual(sorted(units_of(steps)), ["lodestone-coordinator.service", "lodestone-embedder.service",
                                                   "lodestone-gateway.service", "lodestone-vision.service",
                                                   "lodestone-worker.service"])

    def test_device_flag_is_VulkanN_or_none_never_an_integer(self):
        n = 0
        for kind, profile, name, text, args in self.each_unit():
            if args is None:
                continue
            n += 1
            dev = flag(args, "--device")
            self.assertRegex(dev, r"^(Vulkan\d+|none)$", (kind, name))
            self.assertNotRegex(" ".join(args), r"--device \d")
        self.assertGreater(n, 15)

    def test_flash_attn_always_has_a_value(self):
        seen = 0
        for kind, profile, name, text, args in self.each_unit():
            if args and "--flash-attn" in args:
                seen += 1
                self.assertIn(flag(args, "--flash-attn"), ("on", "off", "auto"), (kind, name))
        self.assertGreater(seen, 5)

    def test_chat_engines_use_jinja_and_everything_serves_metrics(self):
        for kind, profile, name, text, args in self.each_unit():
            if args is None:
                continue
            self.assertIn("--metrics", args, name)
            if name in ("lodestone-coordinator.service", "lodestone-worker.service"):
                self.assertIn("--jinja", args, (kind, name))
            if name == "lodestone-embedder.service":
                self.assertIn("--embedding", args)
                self.assertNotIn("--jinja", args)

    def test_quantised_kv_cache_only_with_flash_attention_on(self):
        for kind, profile, name, text, args in self.each_unit():
            if args and "-ctv" in args and flag(args, "-ctv") != "f16":
                self.assertEqual(flag(args, "--flash-attn"), "on", (kind, name))

    def test_gpu_units_are_pinned_and_wait_for_the_gpus(self):
        for kind, profile, name, text, args in self.each_unit():
            if args is None:
                continue
            if flag(args, "--device").startswith("Vulkan"):
                self.assertEqual(flag(args, "--device"), "Vulkan0", "pinned, so the first visible device")
                self.assertRegex(text, r'(?m)^Environment="GGML_VK_VISIBLE_DEVICES=\d+"$')
                self.assertRegex(text, r"(?m)^ExecStartPre=/opt/lodestone/bin/wait-for-gpus\.sh \d+$")
                self.assertIn("-ngl", args)
            else:
                self.assertNotIn("GGML_VK_VISIBLE_DEVICES", text)
                self.assertNotIn("ExecStartPre", text)
                self.assertIn("-t", args)

    def test_pin_matches_the_planned_gpu_and_the_wait_counts_all_gpus(self):
        steps = inst.plan(machine("amd2"), {"profile": "gpu-large"})
        u = units_of(steps)
        self.assertIn('GGML_VK_VISIBLE_DEVICES=0"', u["lodestone-coordinator.service"])
        self.assertIn('GGML_VK_VISIBLE_DEVICES=1"', u["lodestone-embedder.service"])
        self.assertIn("wait-for-gpus.sh 2", u["lodestone-embedder.service"])      # the software renderer is not counted

    def test_cpu_engine_uses_the_detected_cores_up_to_a_cap(self):
        args = unit_args(units_of(inst.plan(machine("none", cpu_count=6), {}))["lodestone-coordinator.service"])
        self.assertEqual((flag(args, "--device"), flag(args, "-t")), ("none", "6"))
        args = unit_args(units_of(inst.plan(machine("none", cpu_count=64), {}))["lodestone-coordinator.service"])
        self.assertEqual(flag(args, "-t"), "8")
        args = unit_args(units_of(inst.plan(machine("none", cpu_count=None), {}))["lodestone-coordinator.service"])
        self.assertEqual(flag(args, "-t"), "4")

    def test_engines_listen_on_loopback_and_run_unprivileged(self):
        for kind, profile, name, text, args in self.each_unit():
            self.assertNotIn("0.0.0.0", text, name)
            self.assertIn("User=lodestone", text)
            self.assertNotIn("User=root", text)
            if args:
                self.assertEqual(flag(args, "--host"), "127.0.0.1")
                self.assertEqual(flag(args, "--alias"), name[len("lodestone-"):-len(".service")])

    def test_ports_match_the_endpoints_defaults(self):
        from lodestone import endpoints
        ports = {name: int(endpoints.SERVICES[name][2].rsplit(":", 1)[1].split("/")[0]) for name in ("coordinator", "worker", "embedder", "vision")}
        for p in inst.PROFILES.values():
            for r in p["roles"].values():
                self.assertEqual(r["port"], ports[[k for k, v in p["roles"].items() if v is r][0]])
        self.assertEqual(inst.VISION_ROLE["port"], ports["vision"])

    def test_gateway_unit(self):
        text = units_of(inst.plan(machine("amd2"), {"prefix": "/srv/lode"}))["lodestone-gateway.service"]
        self.assertIn("ExecStart=/usr/bin/python3 -m lodestone.gateway", text)
        self.assertIn('Environment="PYTHONPATH=/srv/lode/app"', text)
        self.assertIn('Environment="STONESAGE_CONFIG_FILE=/srv/lode/config.json"', text)
        self.assertIn("lodestone-coordinator.service", text.split("Wants=")[1].splitlines()[0])

    def test_wait_script_checks_vulkan_and_fails_loudly(self):
        self.assertIn("vulkaninfo --summary", inst.WAIT_FOR_GPUS)
        self.assertIn("exit 1", inst.WAIT_FOR_GPUS)
        self.assertTrue(inst.WAIT_FOR_GPUS.startswith("#!/bin/sh\n"))
        files = {p: c for s in inst.plan(machine("amd2"), {}) for p, c in s.get("files", {}).items()}
        self.assertEqual(files["/opt/lodestone/bin/wait-for-gpus.sh"], inst.WAIT_FOR_GPUS)
        cpu = {p for s in inst.plan(machine("none"), {}) for p in s.get("files", {})}
        self.assertNotIn("/opt/lodestone/bin/wait-for-gpus.sh", cpu)


class PlanTests(unittest.TestCase):
    def ids(self, steps):
        return [s["id"] for s in steps]

    def test_every_step_is_well_formed_and_the_order_is_right(self):
        for kind in ("amd2", "amd8", "nvidia", "none"):
            steps = inst.plan(machine(kind), {})
            for s in steps:
                self.assertTrue(s["id"] and s["why"], s)
                self.assertTrue(s.get("commands") or s.get("files") or s.get("checks"), s["id"])
                for argv in s.get("commands", []):
                    self.assertTrue(argv and all(isinstance(a, str) for a in argv), (s["id"], argv))
            ids = self.ids(steps)
            self.assertEqual(ids[0], "preflight")
            self.assertLess(ids.index("apt-packages"), ids.index("llama-cpp-build"))
            self.assertLess(ids.index("llama-cpp-build"), ids.index("systemd-units"))
            self.assertLess(max(i for i, x in enumerate(ids) if x.startswith("model-")), ids.index("systemd-units"))
            self.assertLess(ids.index("lodestone-code"), min(i for i, x in enumerate(ids) if x.startswith("model-")))
            self.assertLess(ids.index("systemd-units"), ids.index("config"))
            self.assertEqual(ids[-1], "start")

    def test_vulkan_only_for_gpu_profiles(self):
        gpu = {s["id"]: s for s in inst.plan(machine("amd2"), {})}
        cpu = {s["id"]: s for s in inst.plan(machine("none"), {})}
        self.assertIn("-DGGML_VULKAN=ON", gpu["llama-cpp-build"]["commands"][0])
        self.assertNotIn("-DGGML_VULKAN=ON", cpu["llama-cpp-build"]["commands"][0])
        self.assertIn("libvulkan-dev", gpu["apt-packages"]["commands"][1])
        self.assertNotIn("libvulkan-dev", cpu["apt-packages"]["commands"][1])
        self.assertIn("spirv-headers", gpu)
        self.assertNotIn("spirv-headers", cpu)

    def test_llama_cpp_is_pinned_to_one_documented_commit(self):
        self.assertRegex(inst.LLAMA_CPP_REF, r"^[0-9a-f]{40}$")
        steps = {s["id"]: s for s in inst.plan(machine("amd2"), {})}
        self.assertIn(["git", "-C", "/opt/lodestone/src/llama.cpp", "checkout", "--detach", inst.LLAMA_CPP_REF],
                      steps["llama-cpp-source"]["commands"])
        flat = " ".join(" ".join(a) for s in steps.values() for a in s.get("commands", []))
        self.assertNotRegex(flat, r"git pull|--branch|checkout (master|main)")
        marker = "/opt/lodestone/bin/llama-server.ref"
        self.assertEqual(steps["llama-cpp-build"]["skip_if"]["file_contains"], [marker, inst.LLAMA_CPP_REF])
        self.assertIn(inst.LLAMA_CPP_REF, steps["llama-cpp-build"]["commands"][-1][-1])

    def test_models_go_through_the_downloader_with_a_size_skip(self):
        steps = [s for s in inst.plan(machine("amd2"), {}) if s["id"].startswith("model-")]
        self.assertEqual(len(steps), 3)
        for s in steps:
            argv = s["commands"][0]
            self.assertEqual(argv[:3], ["python3", "-m", "lodestone.model_downloader"])
            repo, file_ = flag(argv, "--repo"), flag(argv, "--file")
            model = next(m for m in inst.MODELS.values() if m["file"] == file_)
            self.assertEqual(repo, model["repo"])
            self.assertEqual(s["skip_if"]["file_size"], ["/opt/lodestone/models/" + file_, model["size_bytes"]])
            self.assertEqual(s["env"], {"PYTHONPATH": "/opt/lodestone/app"})
            self.assertIn("SHA256SUMS", s["why"])

    def test_vision_downloads_both_files_once(self):
        steps = [s for s in inst.plan(machine("amd2"), {"vision": True}) if s["id"].startswith("model-")]
        self.assertEqual(len(steps), 5)
        self.assertEqual(len({s["id"] for s in steps}), 5)

    def test_service_user_and_gpu_groups(self):
        s = next(x for x in inst.plan(machine("amd2"), {}) if x["id"] == "service-user")
        flat = " ".join(" ".join(a) for a in s["commands"])
        self.assertIn("useradd --system", flat)
        self.assertIn("render video", flat)
        self.assertIn("nologin", flat)

    def test_config_step_adds_endpoint_keys_for_the_roles_installed(self):
        step = next(x for x in inst.plan(machine("amd8"), {}) if x["id"] == "config")
        cfg = json.loads(step["files"]["/opt/lodestone/config.json"])
        self.assertEqual(cfg["cluster"], {"coordinator_url": "http://127.0.0.1:8001/v1", "embedder_url": "http://127.0.0.1:8003/v1"})
        self.assertEqual(cfg["gateway"], {"host": "127.0.0.1", "port": 8000})
        self.assertNotIn("api_key", cfg["gateway"])
        self.assertEqual(cfg["lodestone"]["install"]["profile"], "gpu-small")
        self.assertTrue(step["secret"])
        self.assertEqual(step["modes"], {"/opt/lodestone/config.json": 0o640})

    def test_config_keys_are_ones_the_endpoints_module_reads(self):
        from lodestone import endpoints
        cfg = json.loads(next(x for x in inst.plan(machine("amd2"), {"vision": True}) if x["id"] == "config")
                         ["files"]["/opt/lodestone/config.json"])
        for role in ("coordinator", "worker", "embedder", "vision"):
            self.assertEqual(endpoints.url(role, cfg), cfg["cluster"][role + "_url"])

    def test_existing_config_is_merged_never_dropped(self):
        existing = {"server": {"port": 9}, "cluster": {"coordinator_url": "http://10.0.0.5:8001/v1", "odd": [1, 2]},
                    "secret_thing": "keep me", "gateway": {"api_key": "k"}}
        d = machine("amd2", config=existing)
        step = next(x for x in inst.plan(d, {}) if x["id"] == "config")
        cfg = json.loads(step["files"]["/opt/lodestone/config.json"])
        self.assertEqual(cfg["server"], {"port": 9})
        self.assertEqual(cfg["secret_thing"], "keep me")
        self.assertEqual(cfg["cluster"]["odd"], [1, 2])
        self.assertEqual(cfg["cluster"]["coordinator_url"], "http://10.0.0.5:8001/v1")       # the person's value stays
        self.assertEqual(cfg["cluster"]["worker_url"], "http://127.0.0.1:8002/v1")
        self.assertEqual(cfg["gateway"]["api_key"], "k")
        self.assertEqual(step["summary"]["kept"], ["cluster.coordinator_url"])
        self.assertNotIn("cluster.coordinator_url", step["summary"]["added"])
        self.assertNotIn("keep me", " ".join(step["summary"]["added"]))

    def test_unsafe_inputs_are_refused(self):
        for bad in ({"prefix": "relative/dir"}, {"prefix": "/opt/a b"}, {"prefix": "/opt/../etc"}, {"prefix": "/home/x/lode"},
                    {"config": "/root/c.json"}, {"prefix": "/opt/x;rm -rf /"}):
            with self.assertRaises(inst.InstallerError, msg=bad):
                inst.plan(machine("amd2"), bad)
        for broken in ("invalid", [1, 2]):
            with self.assertRaises(inst.InstallerError):
                inst.plan(machine("amd2", config=broken), {})

    def test_prefix_and_config_choices_flow_through(self):
        steps = inst.plan(machine("amd2"), {"prefix": "/srv/lode", "config": "/etc/lode.json"})
        text = json.dumps(steps)
        self.assertNotIn("/opt/lodestone", text)
        self.assertIn("/etc/lode.json", text)
        self.assertIn("/srv/lode/bin/llama-server", text)

    def test_plan_is_pure(self):
        d = machine("amd2")
        before = json.dumps(d, sort_keys=True)
        with mock.patch.object(subprocess, "run", Fail()), mock.patch.object(os, "system", Fail()):
            a = inst.plan(d, {"package_dir": "/x/lodestone"})
            b = inst.plan(d, {"package_dir": "/x/lodestone"})
        self.assertEqual(a, b)
        self.assertEqual(before, json.dumps(d, sort_keys=True))

    def test_preflight_checks(self):
        def checks(kind, **over):
            return {c["id"]: c for c in inst.plan(machine(kind, **over), {})[0]["checks"]}

        good = checks("amd2")
        self.assertTrue(all(c["ok"] for c in good.values()), good)
        self.assertFalse(checks("amd2", system="Windows")["linux"]["ok"])
        self.assertFalse(checks("amd2", euid=1000)["root"]["ok"])
        self.assertFalse(checks("amd2", os_release='ID=fedora\nVERSION_ID="40"\n')["os"]["ok"])
        self.assertTrue(checks("amd2", os_release='ID=fedora\nVERSION_ID="40"\n')["os"]["fatal"])
        self.assertFalse(checks("amd2", os_release='ID=ubuntu\nVERSION_ID="22.04"\n')["os"]["fatal"])      # warn only
        self.assertFalse(checks("amd2", disk_free_mb=1000)["disk"]["ok"])
        self.assertTrue(checks("amd2", disk_free_mb=1000)["disk"]["fatal"])
        self.assertFalse(checks("amd2", disk_free_mb=None)["disk"]["fatal"])
        self.assertFalse(checks("amd2", meminfo="MemTotal: 2000000 kB\n")["ram"]["fatal"])


class ConfigMergeTests(unittest.TestCase):
    def test_merge_rules(self):
        merged, added, kept = inst.merge_config({"a": {"x": 1}, "b": ""}, {"a": {"x": 2, "y": 3}, "b": "now", "c": {"d": 1}})
        self.assertEqual(merged, {"a": {"x": 1, "y": 3}, "b": "now", "c": {"d": 1}})
        self.assertEqual(sorted(added), ["a.y", "b", "c"])
        self.assertEqual(kept, ["a.x"])

    def test_overwrite_paths_replace_whole_subtrees(self):
        merged, added, _ = inst.merge_config({"l": {"install": {"profile": "old", "extra": 1}}}, {"l": {"install": {"profile": "new"}}},
                                             overwrite=("l.install",))
        self.assertEqual(merged, {"l": {"install": {"profile": "new"}}})
        self.assertEqual(added, ["l.install"])

    def test_nothing_in_existing_is_dropped(self):
        existing = {"k%d" % i: {"v": i, "list": [i]} for i in range(20)}
        merged, _, _ = inst.merge_config(existing, {"cluster": {"coordinator_url": "u"}})
        for k, v in existing.items():
            self.assertEqual(merged[k], v)

    def test_same_value_is_neither_added_nor_kept(self):
        merged, added, kept = inst.merge_config({"a": 1}, {"a": 1})
        self.assertEqual((merged, added, kept), ({"a": 1}, [], []))

    def test_write_file_backs_up_a_different_file_and_leaves_an_identical_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "config.json")
            self.assertEqual(inst.write_file(path, '{"a": 1}\n', 0o640, "T1"), "written")
            self.assertEqual(inst.write_file(path, '{"a": 1}\n', 0o640, "T2"), "unchanged")
            self.assertEqual(inst.write_file(path, '{"a": 2}\n', 0o640, "T3"), "backed up and written")
            with open(path + ".bak-T3") as f:
                self.assertEqual(f.read(), '{"a": 1}\n')
            self.assertFalse(os.path.exists(path + ".bak-T2"))
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b'{"a": 2}\n')

    def test_files_are_written_with_lf_even_for_crlf_input_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unit.service")
            inst.write_file(path, "a\nb\n", None, "T")
            with open(path, "rb") as f:
                self.assertNotIn(b"\r", f.read())

    def test_a_bom_in_the_old_file_is_read_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbf{}\n")
            self.assertEqual(inst.write_file(path, "{}\n", None, "T"), "unchanged")


class DryRunAndApplyTests(unittest.TestCase):
    def setUp(self):
        self.steps = inst.plan(machine("amd2"), {})
        patches = [mock.patch.object(subprocess, "run", Fail()), mock.patch.object(subprocess, "Popen", Fail()),
                   mock.patch.object(os, "system", Fail())]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_dry_run_prints_the_plan_and_executes_nothing(self):
        lines = []
        with mock.patch.object(inst, "write_file", Fail()), mock.patch.object(inst, "_run_cmd", Fail()):
            self.assertEqual(inst.apply(self.steps, out=lines.append), [])
            self.assertEqual(inst.apply(self.steps, dry_run=True, out=lines.append), [])
        text = "\n".join(lines)
        self.assertIn("[1/", text)
        self.assertIn("$ apt-get install -y build-essential", text)
        self.assertIn("-DGGML_VULKAN=ON", text)
        self.assertIn("Environment=\"GGML_VK_VISIBLE_DEVICES=0\"", text)
        self.assertIn("write /opt/lodestone/config.json (existing keys kept", text)

    def test_dry_run_does_not_print_the_config_content(self):
        d = machine("amd2", config={"homeassistant": {"token": "TOP-SECRET-VALUE"}})
        lines = []
        inst.apply(inst.plan(d, {}), out=lines.append)
        self.assertNotIn("TOP-SECRET-VALUE", "\n".join(lines))

    def test_main_without_apply_is_a_dry_run(self):
        lines = []
        self.assertEqual(inst.main([], detected=machine("amd2"), out=lines.append), 0)
        text = "\n".join(lines)
        self.assertIn("DRY RUN", text)
        self.assertIn("profile:   gpu-large", text)
        self.assertIn("coordinator", text)

    def test_cpu_profile_says_what_to_expect(self):
        lines = []
        self.assertEqual(inst.main([], detected=machine("none"), out=lines.append), 0)
        text = "\n".join(lines)
        self.assertIn("20-40 tokens/s", text)
        self.assertIn("few tokens/s", text)
        self.assertIn("GPU is strongly recommended", text)
        gpu = []
        inst.main([], detected=machine("amd2"), out=gpu.append)
        self.assertNotIn("tokens/s", "\n".join(gpu))                       # a GPU install is not told this

    def test_cpu_plan_marks_the_coordinator_as_cpu_in_config(self):
        steps = {s["id"]: s for s in inst.plan(machine("none"), {})}
        cfg = json.loads(next(iter(steps["config"]["files"].values())))
        self.assertIs(cfg["cluster"]["coordinator_cpu"], True)
        gpu = {s["id"]: s for s in inst.plan(machine("amd2"), {})}
        self.assertNotIn("coordinator_cpu", json.loads(next(iter(gpu["config"]["files"].values())))["cluster"])

    def test_main_json_prints_the_plan(self):
        lines = []
        self.assertEqual(inst.main(["--json", "--profile", "cpu"], detected=machine("none"), out=lines.append), 0)
        self.assertEqual(json.loads("\n".join(lines))[0]["id"], "preflight")

    def test_main_reports_a_plan_error_and_fails(self):
        with mock.patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertEqual(inst.main(["--prefix", "nope"], detected=machine("amd2"), out=lambda s: None), 1)
        self.assertIn("installer:", err.getvalue())

    def test_apply_refuses_off_linux(self):
        with self.assertRaises(inst.InstallerError) as cm:
            inst.apply(self.steps, dry_run=False, system="Windows", euid=0, out=lambda s: None)
        self.assertIn("Linux", str(cm.exception))

    def test_apply_refuses_without_root(self):
        with self.assertRaises(inst.InstallerError) as cm:
            inst.apply(self.steps, dry_run=False, system="Linux", euid=1000, out=lambda s: None)
        self.assertIn("root", str(cm.exception))
        with self.assertRaises(inst.InstallerError):
            inst.apply(self.steps, dry_run=False, system="Linux", euid=None, out=lambda s: None)

    def test_main_apply_off_linux_fails_cleanly_and_runs_nothing(self):
        with mock.patch.object(sys, "stderr", io.StringIO()) as err, mock.patch.object(inst.platform, "system", lambda: "Windows"):
            self.assertEqual(inst.main(["--apply"], detected=machine("amd2"), out=lambda s: None), 1)
        self.assertIn("only runs on Linux", err.getvalue())

    def test_apply_refuses_a_failing_fatal_check_before_changing_anything(self):
        steps = inst.plan(machine("amd2", disk_free_mb=500), {})
        ran = []
        with mock.patch.object(inst, "write_file", Fail()):
            with self.assertRaises(inst.InstallerError) as cm:
                inst.apply(steps, dry_run=False, system="Linux", euid=0, run=lambda a, env=None: ran.append(a) or 0, out=lambda s: None)
        self.assertIn("preflight", str(cm.exception))
        self.assertEqual(ran, [])

    def test_full_plan_runs_every_command_in_order(self):
        ran, wrote = [], []
        with mock.patch.object(inst, "write_file", lambda p, c, m, s: wrote.append(p) or "written"), \
                mock.patch.object(inst, "_skip", lambda cond: False):
            res = inst.apply(self.steps, dry_run=False, system="Linux", euid=0,
                             run=lambda a, env=None: ran.append((a, env)) or 0, out=lambda s: None)
        want = [a for s in self.steps for a in s.get("commands", [])]
        self.assertEqual([a for a, _ in ran], want)
        self.assertEqual(len(res), len(self.steps))
        self.assertIn("/etc/systemd/system/lodestone-gateway.service", wrote)
        self.assertEqual(next(env for a, env in ran if a[:3] == ["python3", "-m", "lodestone.model_downloader"])["PYTHONPATH"], "/opt/lodestone/app")

    def test_first_failure_stops_and_names_the_step(self):
        calls = []

        def run(argv, env=None):
            calls.append(argv)
            return 3 if argv[:2] == ["cmake", "--build"] else 0

        with mock.patch.object(inst, "write_file", Fail()), mock.patch.object(inst, "_skip", lambda cond: False):
            with self.assertRaises(inst.InstallerError) as cm:
                inst.apply(self.steps, dry_run=False, system="Linux", euid=0, run=run, out=lambda s: None)
        msg = str(cm.exception)
        self.assertIn("llama-cpp-build", msg)
        self.assertIn("cmake --build", msg)
        self.assertIn("exited with 3", msg)
        self.assertEqual(calls[-1][:2], ["cmake", "--build"])
        self.assertFalse(any(a[:2] == ["systemctl", "daemon-reload"] for a in calls))       # nothing after the failure

    def test_a_file_that_cannot_be_written_stops_the_install(self):
        def boom(*a):
            raise PermissionError("denied")

        with mock.patch.object(inst, "write_file", boom), mock.patch.object(inst, "_skip", lambda cond: False):
            with self.assertRaises(inst.InstallerError) as cm:
                inst.apply([{"id": "x", "why": "w", "files": {"/nope": "c"}}], dry_run=False, system="Linux", euid=0,
                           run=lambda a, env=None: 0, out=lambda s: None)
        self.assertIn("cannot write /nope", str(cm.exception))

    def test_finished_steps_are_skipped_on_a_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, marker, model = (os.path.join(tmp, n) for n in ("llama-server", "llama-server.ref", "m.gguf"))
            for path, text in ((binary, "x"), (marker, inst.LLAMA_CPP_REF + "\n"), (model, "12345")):
                with open(path, "w") as f:
                    f.write(text)
            ran = []
            steps = [{"id": "build", "why": "w", "skip_if": {"file_contains": [marker, inst.LLAMA_CPP_REF], "exists": binary},
                      "commands": [["make"]]},
                     {"id": "old-pin", "why": "w", "skip_if": {"file_contains": [marker, "another-commit"], "exists": binary},
                      "commands": [["make", "again"]]},
                     {"id": "model", "why": "w", "skip_if": {"file_size": [model, 5]}, "commands": [["download"]]},
                     {"id": "short-model", "why": "w", "skip_if": {"file_size": [model, 999]}, "commands": [["download", "again"]]},
                     {"id": "any", "why": "w", "skip_if": {"any_exists": [os.path.join(tmp, "no"), binary]}, "commands": [["never"]]},
                     {"id": "none-there", "why": "w", "skip_if": {"exists": os.path.join(tmp, "missing")}, "commands": [["run"]]}]
            res = inst.apply(steps, dry_run=False, system="Linux", euid=0, run=lambda a, env=None: ran.append(a) or 0, out=lambda s: None)
        self.assertEqual(ran, [["make", "again"], ["download", "again"], ["run"]])
        self.assertEqual([s for _, s in res], ["skipped", "done", "skipped", "done", "skipped", "done"])

    def test_files_are_written_before_the_commands_of_their_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unit.service")
            seen = []
            step = {"id": "units", "why": "w", "files": {path: "[Unit]\n"}, "modes": {path: 0o644},
                    "commands": [["systemctl", "daemon-reload"]]}
            inst.apply([step], dry_run=False, system="Linux", euid=0,
                       run=lambda a, env=None: seen.append(os.path.exists(path)) or 0, out=lambda s: None, now=lambda: "T")
            self.assertEqual(seen, [True])
            inst.apply([dict(step, files={path: "[Unit]\nChanged=1\n"})], dry_run=False, system="Linux", euid=0,
                       run=lambda a, env=None: 0, out=lambda s: None, now=lambda: "T9")
            self.assertTrue(os.path.exists(path + ".bak-T9"))


class DownloaderTests(unittest.TestCase):
    """lodestone.model_downloader's plain download (what the installer's model steps run), with a fake opener."""

    class Resp:
        def __init__(self, data, status=200, cut=None):
            self.buf, self.status = io.BytesIO(data if cut is None else data[:cut]), status
            self.headers = {"Content-Length": str(len(data) if status == 200 else len(data))}

        def read(self, n=-1):
            return self.buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def test_url_for_a_hugging_face_file(self):
        self.assertEqual(dl.hf_resolve_url("Owner/Repo", "m-Q4_K_M.gguf"),
                         "https://huggingface.co/Owner/Repo/resolve/main/m-Q4_K_M.gguf?download=true")

    def test_download_then_present_then_resume(self):
        data = bytes(range(256)) * 100
        seen = []

        def opener(req, timeout=None):
            seen.append(dict(req.header_items()))
            return self.Resp(data)

        with tempfile.TemporaryDirectory() as tmp:
            got = dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=opener)
            self.assertEqual((got["status"], got["size"]), ("downloaded", len(data)))
            with open(os.path.join(tmp, "m.gguf"), "rb") as f:
                self.assertEqual(f.read(), data)
            self.assertFalse(os.path.exists(os.path.join(tmp, "m.gguf.part")))
            self.assertEqual(dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=opener)["status"], "present")
            # a half-finished .part resumes with a Range header, and a 206 answer is appended
            os.remove(os.path.join(tmp, "m.gguf"))
            with open(os.path.join(tmp, "m.gguf.part"), "wb") as f:
                f.write(data[:1000])
            seen.clear()
            tail = self.Resp(data[1000:], status=206)
            got = dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=lambda req, timeout=None: (
                seen.append(dict(req.header_items())) or tail))
            self.assertEqual(seen[0].get("Range"), "bytes=1000-")
            with open(os.path.join(tmp, "m.gguf"), "rb") as f:
                self.assertEqual(f.read(), data)

    def test_a_short_download_keeps_the_part_and_raises(self):
        data = b"x" * 5000
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=lambda r, timeout=None: self.Resp(data, cut=2000))
            self.assertFalse(os.path.exists(os.path.join(tmp, "m.gguf")))
            self.assertEqual(os.path.getsize(os.path.join(tmp, "m.gguf.part")), 2000)

    def test_offline_with_the_file_already_there_is_fine_and_without_it_is_an_error(self):
        def offline(req, timeout=None):
            raise urllib.error.URLError("no network")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=offline)
            with open(os.path.join(tmp, "m.gguf"), "wb") as f:
                f.write(b"abc")
            self.assertEqual(dl.download_file("http://example.invalid/m.gguf", tmp, "m.gguf", opener=offline)["status"], "present")

    def test_sha256_is_recorded_in_sha256sum_format_and_replaced_on_redo(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.gguf"), "wb") as f:
                f.write(b"hello")
            digest = dl.sha256_file(os.path.join(tmp, "a.gguf"))
            self.assertEqual(digest, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
            dl.record_sha256(tmp, "a.gguf", digest)
            dl.record_sha256(tmp, "b.gguf", "0" * 64)
            dl.record_sha256(tmp, "a.gguf", "1" * 64)
            with open(os.path.join(tmp, "SHA256SUMS"), "rb") as f:
                raw = f.read()
            self.assertEqual(raw.decode(), "%s  a.gguf\n%s  b.gguf\n" % ("1" * 64, "0" * 64))
            self.assertNotIn(b"\r", raw)

    def test_main_downloads_and_records_the_hash(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sys, "stdout", io.StringIO()) as out:
            def fake(url, dest, name, progress=None):
                with open(os.path.join(dest, name), "wb") as f:
                    f.write(b"hello")
                return {"status": "downloaded", "path": os.path.join(dest, name), "size": 5}

            with mock.patch.object(dl, "download_file", fake):
                self.assertEqual(dl.main(["--repo", "o/r", "--file", "a.gguf", "--dest", tmp]), 0)
            self.assertEqual(dl.read_sums(tmp)["a.gguf"], dl.sha256_file(os.path.join(tmp, "a.gguf")))
            self.assertIn("sha256", out.getvalue())


class StdlibOnlyTests(unittest.TestCase):
    def test_installer_imports_only_the_standard_library(self):
        with open(inst.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names.add((node.module or "").split(".")[0])
        std = getattr(sys, "stdlib_module_names", None)
        if std is not None:
            self.assertEqual(sorted(n for n in names if n not in std), [])
        self.assertNotIn("lodestone", names)       # nothing upward, nothing sideways: it runs before anything is installed

    def test_installer_source_has_no_personal_data(self):
        with open(inst.__file__, encoding="utf-8") as f:
            text = f.read()
        self.assertNotRegex(text, r"\b192\.168\.\d+\.\d+\b")
        self.assertNotIn("StrictHostKeyChecking", text)


if __name__ == "__main__":
    unittest.main()
