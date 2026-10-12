"""Offline tests for the GPU profile ladder of the Lodestone installer (lodestone/installer.py PROFILES): which rung a card of
a given size reaches, what each rung installs, the messages that name the card found (and say why a card was not used), the
old profile names, `--list-profiles`, partial offload, speculative-decoding flags, multi-GPU placement, and the model table
(real repositories, recorded sizes and SHA-256). No subprocess, no network."""

import io
import json
import os
import re
import shlex
import sys
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import installer as inst  # noqa: E402

UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'


def gpu_block(n, vendor, device, kind, name):
    return ("GPU%d:\n\tvendorID           = 0x%04x\n\tdeviceID           = 0x%04x\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_%s\n"
            "\tdeviceName         = %s\n\tdriverName         = x\n\n" % (n, vendor, device, kind, name))


def vulkaninfo(*blocks):
    return "Devices:\n========\n" + "".join(blocks) + "Device Groups:\n=============\n"


def sources(**over):
    s = {"system": "Linux", "machine": "x86_64", "cpu_count": 8, "euid": 0, "os_release": UBUNTU, "meminfo": None,
         "vulkaninfo": None, "nvidia_smi": None, "drm": [], "llama_server_path": None, "llama_server_version": None,
         "disk_free_mb": 900000, "config": None}
    s.update(over)
    return s


def mib(gb):
    return int(gb * 1024)


def amd(*sizes_gb):
    """AMD cards listed by Vulkan, sized by sysfs (the usual Linux case); card i has device id 0x100+i."""
    blocks = [gpu_block(i, 0x1002, 0x100 + i, "DISCRETE_GPU", "AMD Radeon Test %d" % i) for i in range(len(sizes_gb))]
    drm = [{"vendor_id": 0x1002, "device_id": 0x100 + i, "vram_mb": mib(g)} for i, g in enumerate(sizes_gb)]
    return inst.detect(sources(vulkaninfo=vulkaninfo(*blocks), drm=drm))


def nvidia(gb, name="NVIDIA Test Card", vulkan=True, system="Linux"):
    """An NVIDIA card sized by nvidia-smi; Vulkan lists it (vulkan=True) or the host has no vulkaninfo at all (a Windows host,
    a Docker host without the driver's Vulkan part)."""
    vk = vulkaninfo(gpu_block(0, 0x10DE, 0x1, "DISCRETE_GPU", name)) if vulkan else None
    return inst.detect(sources(vulkaninfo=vk, nvidia_smi="%s, %d\n" % (name, mib(gb)), system=system))


def flag(args, name):
    return args[args.index(name) + 1]


SIZES = [(1, "cpu"), (2, "gpu-2gb"), (3, "gpu-2gb"), (3.9, "gpu-4gb"), (4, "gpu-4gb"), (6, "gpu-6gb"), (8, "gpu-8gb"),
         (12, "gpu-12gb"), (16, "gpu-16gb"), (24, "gpu-24gb"), (48, "gpu-24gb")]


class LadderShapeTests(unittest.TestCase):
    def test_the_ladder_runs_from_24gb_down_to_cpu_with_distinct_minimums(self):
        names = list(inst.PROFILES)
        self.assertEqual(names, ["gpu-24gb", "gpu-16gb", "gpu-12gb", "gpu-8gb", "gpu-6gb", "gpu-4gb", "gpu-2gb", "cpu"])
        mins = [inst.PROFILES[n]["min_vram_mb"] for n in names]
        self.assertEqual(mins, sorted(mins, reverse=True))
        self.assertEqual(len(set(mins)), len(mins))
        self.assertEqual(mins[-1], 0)

    def test_the_boundaries(self):
        for name, p in inst.PROFILES.items():
            if name == "cpu":
                continue
            below = inst.tier_for(p["min_vram_mb"] - 1)
            self.assertEqual(inst.tier_for(p["min_vram_mb"]), name)
            self.assertNotEqual(below, name)
        self.assertEqual(inst.tier_for(None), "cpu")
        self.assertEqual(inst.tier_for(0), "cpu")

    def test_aliases_point_at_the_rungs_they_replaced(self):
        self.assertEqual(inst.PROFILE_ALIASES, {"gpu-large": "gpu-12gb", "gpu-small": "gpu-8gb"})
        self.assertEqual(inst.canonical_profile("gpu-large"), "gpu-12gb")
        self.assertEqual(inst.canonical_profile("gpu-small"), "gpu-8gb")
        self.assertEqual(inst.canonical_profile("gpu-4gb"), "gpu-4gb")
        with self.assertRaises(inst.InstallerError):
            inst.canonical_profile("gpu-huge")
        self.assertEqual(inst.choose_profile(amd(12), "gpu-large"), "gpu-12gb")
        # the old words keep working on the command line, and the config records the new name
        out = []
        code = inst.main(["--profile", "gpu-small", "--json"], detected=amd(8), out=out.append)
        self.assertEqual(code, 0)
        cfg = next(json.loads(s["files"]["/opt/lodestone/config.json"]) for s in json.loads(out[-1]) if s["id"] == "config")
        self.assertEqual(cfg["lodestone"]["install"]["profile"], "gpu-8gb")

    def test_headroom_each_rungs_coordinator_fits_its_minimum_under_the_margin(self):
        for name, p in inst.PROFILES.items():
            if name == "cpu":
                continue
            need = p["roles"]["coordinator"]["vram_mb"]
            self.assertLessEqual(need, p["min_vram_mb"] * inst.VRAM_MARGIN, "%s: %d MB does not fit %d MB x 0.95" % (name, need, p["min_vram_mb"]))
            # and a card exactly at the minimum really gets the coordinator on the GPU, with no refusal
            d = inst.detect(sources(vulkaninfo=vulkaninfo(gpu_block(0, 0x1002, 0x1, "DISCRETE_GPU", "Edge")),
                                    drm=[{"vendor_id": 0x1002, "device_id": 0x1, "vram_mb": p["min_vram_mb"]}]))
            self.assertEqual(inst.choose_profile(d), name)
            roles, _ = inst.place_roles(name, d)
            self.assertEqual(roles[0]["name"], "coordinator")
            self.assertEqual(roles[0]["gpu"], 0)

    def test_every_model_is_a_real_repository_file_with_a_recorded_size_and_digest(self):
        for key, m in inst.MODELS.items():
            self.assertRegex(m["repo"], r"^[\w.-]+/[\w.-]+$", key)
            self.assertTrue(m["file"].endswith(".gguf"), key)
            self.assertGreater(m["size_bytes"], 10 ** 8, key)
            self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$", key)
        self.assertEqual(len({m["sha256"] for m in inst.MODELS.values()}), len(inst.MODELS))
        used = set()
        for p in inst.PROFILES.values():
            for r in p["roles"].values():
                for k in ("model", "mmproj", "draft"):
                    if r.get(k):
                        self.assertIn(r[k], inst.MODELS)
                        used.add(r[k])
        used |= {inst.VISION_ROLE["model"], inst.VISION_ROLE["mmproj"]}
        self.assertEqual(used, set(inst.MODELS), "a model in the table that no rung uses, or the other way round")

    def test_what_each_rung_installs(self):
        want = {   # rung -> (coordinator model, ngl, kv, draft, vision?)
            "gpu-24gb": ("Qwen3-32B-Q4_K_M.gguf", 99, "q4_0", None, False),
            "gpu-16gb": ("Qwen3-14B-Q4_K_M.gguf", 99, "q4_0", "Qwen3-0.6B-Q8_0.gguf", True),
            "gpu-12gb": ("Qwen3-14B-Q4_K_M.gguf", 99, "q4_0", "Qwen3-0.6B-Q8_0.gguf", False),
            "gpu-8gb": ("Qwen3-8B-Q4_K_M.gguf", 99, "q8_0", None, False),
            "gpu-6gb": ("granite-4.0-h-tiny-Q4_K_M.gguf", 99, "q8_0", None, False),
            "gpu-4gb": ("Qwen3-4B-Q4_K_M.gguf", 99, "q8_0", None, False),
            "gpu-2gb": ("granite-4.0-h-tiny-Q4_K_M.gguf", 12, "q8_0", None, False),
            "cpu": ("granite-4.0-h-tiny-Q4_K_M.gguf", 99, "q8_0", None, False),
        }
        for name, (model, ngl, kv, draft, vision) in want.items():
            roles = inst.PROFILES[name]["roles"]
            c = roles["coordinator"]
            self.assertEqual(inst.MODELS[c["model"]]["file"], model, name)
            self.assertEqual(c.get("ngl", 99), ngl, name)
            self.assertEqual(c["kv"], kv, name)
            self.assertEqual(inst.MODELS[c["draft"]]["file"] if c.get("draft") else None, draft, name)
            self.assertEqual("vision" in roles, vision, name)
            self.assertIn("embedder", roles, name)
            self.assertTrue(inst.PROFILES[name]["examples"] and inst.PROFILES[name]["note"] and inst.PROFILES[name]["speed"], name)

    def test_the_measured_numbers_are_recorded(self):
        self.assertIn("13.1 tok/s", inst.PROFILES["gpu-4gb"]["speed"])
        self.assertIn("75 tok/s", inst.PROFILES["gpu-6gb"]["speed"])
        self.assertIn("41 tok/s", inst.PROFILES["gpu-12gb"]["speed"])
        self.assertIn("44/46", inst.PROFILES["gpu-6gb"]["note"])
        self.assertIn("NOT measured", inst.PROFILES["gpu-2gb"]["note"])
        self.assertIn("NOT measured", inst.PROFILES["gpu-24gb"]["note"])


class ChoiceBySizeTests(unittest.TestCase):
    def test_amd_and_nvidia_cards_of_every_size_get_their_rung(self):
        for gb, want in SIZES:
            for label, d in (("amd", amd(gb)), ("nvidia", nvidia(gb))):
                self.assertEqual(inst.choose_profile(d), want, "%s %s GB" % (label, gb))

    def test_every_rung_plans_natively_and_in_docker_without_error(self):
        for gb, want in SIZES:
            steps = inst.plan(amd(gb), {})
            cfg = json.loads(next(s for s in steps if s["id"] == "config")["files"]["/opt/lodestone/config.json"])
            self.assertEqual(cfg["lodestone"]["install"]["profile"], want, gb)
            for kind in ("cuda",):
                stack = inst.docker_stack(nvidia(gb, vulkan=False), {"gpu": kind} if want != "cpu" else {})
                self.assertEqual(stack["profile"], want, gb)

    def test_the_largest_card_sets_the_rung_in_any_order(self):
        for sizes in ((8, 24), (24, 8), (4, 16, 6)):
            d = amd(*sizes)
            self.assertEqual(inst.choose_profile(d), inst.tier_for(mib(max(sizes))))
            roles, _ = inst.place_roles(inst.choose_profile(d), d)
            big = sizes.index(max(sizes))
            self.assertEqual(next(r for r in roles if r["name"] == "coordinator")["gpu"], big, sizes)

    def test_a_24gb_card_with_a_second_card_puts_the_embedder_on_the_second(self):
        d = amd(24, 8)
        roles, notes = inst.place_roles("gpu-24gb", d)
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "embedder": 1})
        self.assertEqual(notes, [])

    def test_a_16gb_card_holds_the_14b_and_vision_and_the_embedder_goes_to_the_cpu(self):
        roles, notes = inst.place_roles("gpu-16gb", amd(16))
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "vision": 0, "embedder": None})
        self.assertTrue(any("embedder runs on the CPU" in n for n in notes))
        roles, _ = inst.place_roles("gpu-16gb", amd(16, 8))
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "vision": 1, "embedder": 0})

    def test_a_4gb_card_runs_the_4b_whole_and_the_embedder_on_the_cpu(self):
        roles, notes = inst.place_roles("gpu-4gb", amd(4))
        self.assertEqual({r["name"]: r["gpu"] for r in roles}, {"coordinator": 0, "embedder": None})
        self.assertTrue(any("embedder runs on the CPU" in n for n in notes))

    def test_vision_is_one_flag_on_any_rung_that_has_room(self):
        roles, notes = inst.place_roles("gpu-24gb", amd(24, 8), vision=True)
        self.assertEqual(next(r for r in roles if r["name"] == "vision")["gpu"], 1)
        roles, notes = inst.place_roles("gpu-24gb", amd(24), vision=True)
        self.assertNotIn("vision", [r["name"] for r in roles])
        self.assertTrue(any("vision left out" in n for n in notes))


class MessageTests(unittest.TestCase):
    def test_a_4gb_quadro_is_named_with_its_size_and_rung(self):
        d = nvidia(4, "Quadro M2000")
        self.assertEqual(inst.profile_notes(d), ["found Quadro M2000, 4.0 GB -> gpu-4gb"])
        stack = inst.docker_stack(nvidia(4, "Quadro M2000", vulkan=False), {})      # a host with nvidia-smi and no vulkaninfo
        self.assertEqual((stack["profile"], stack["gpu"]), ("gpu-4gb", "cuda"))
        self.assertEqual(stack["notes"][0], "found Quadro M2000, 4.0 GB -> gpu-4gb")
        self.assertTrue(any("embedder runs on the CPU" in n for n in stack["notes"]))

    def test_a_card_under_2gb_says_so_and_uses_the_cpu(self):
        d = nvidia(1, "GT 710")
        self.assertEqual(inst.profile_notes(d), ["found GT 710, 1.0 GB: below the 2 GB minimum, using the CPU"])
        stack = inst.docker_stack(nvidia(1, "GT 710", vulkan=False), {})
        self.assertEqual((stack["profile"], stack["gpu"]), ("cpu", "cpu"))
        self.assertIn("found GT 710, 1.0 GB: below the 2 GB minimum, using the CPU", stack["notes"])

    def test_a_nvidia_only_host_never_reports_no_gpu_found(self):
        d = nvidia(4, "Quadro M2000", vulkan=False)
        text = "\n".join(inst.describe(d))
        self.assertNotIn("no GPU found", text)
        self.assertNotIn("none found", text)
        self.assertIn("Quadro M2000, 4.0 GB (nvidia-smi", text)
        self.assertIn("not installed (vulkaninfo missing)", text)
        self.assertEqual(len(d["cards"]), 1)
        # the native install runs on Vulkan: it says why it cannot use that card, naming it, and what to do
        notes = inst.profile_notes(d)
        self.assertEqual(inst.choose_profile(d), "cpu")
        self.assertEqual(len(notes), 1)
        self.assertIn("found Quadro M2000, 4.0 GB (nvidia-smi): Vulkan does not list it", notes[0])
        self.assertIn("--docker", notes[0])
        self.assertTrue(notes[0].endswith("using the CPU"))

    def test_sysfs_alone_names_the_card_too(self):
        d = inst.detect(sources(drm=[{"vendor_id": 0x1002, "device_id": 0x73FF, "vram_mb": 8176}]))
        self.assertEqual(len(d["cards"]), 1)
        self.assertEqual(d["cards"][0]["source"], "sysfs")
        self.assertIn("AMD GPU (device 0x73ff), 8.0 GB (sysfs", "\n".join(inst.describe(d)))

    def test_no_card_at_all_is_the_only_time_it_says_none_found(self):
        d = inst.detect(sources())
        self.assertIn("none found (looked at nvidia-smi, sysfs and vulkaninfo)", "\n".join(inst.describe(d)))
        self.assertEqual(inst.choose_profile(d), "cpu")

    def test_a_card_of_unknown_size_is_named_not_used_and_can_be_forced(self):
        d = inst.detect(sources(vulkaninfo=vulkaninfo(gpu_block(0, 0x1002, 0x1, "DISCRETE_GPU", "Mystery Card"))))
        notes = inst.profile_notes(d)
        self.assertEqual(inst.choose_profile(d), "cpu")
        self.assertIn("found Mystery Card: VRAM size unknown, not used", notes[0])
        self.assertIn("--profile", notes[0])
        # forcing a rung trusts the person: one card of that rung is assumed
        roles, _ = inst.place_roles("gpu-4gb", d)
        self.assertEqual(roles[0]["gpu"], 0)
        stack = inst.docker_stack(d, {"gpu": "vulkan", "profile": "gpu-4gb"})
        self.assertEqual((stack["profile"], stack["gpu"]), ("gpu-4gb", "vulkan"))
        stack = inst.docker_stack(nvidia(4, vulkan=False), {"gpu": "cuda", "profile": "gpu-2gb"})
        self.assertEqual(stack["profile"], "gpu-2gb")

    def test_integrated_graphics_are_named_and_not_used(self):
        d = inst.detect(sources(vulkaninfo=vulkaninfo(gpu_block(0, 0x8086, 0x4680, "INTEGRATED_GPU", "Intel UHD Graphics 770")),
                                drm=[{"vendor_id": 0x8086, "device_id": 0x4680, "vram_mb": 256}]))
        self.assertEqual(inst.choose_profile(d), "cpu")
        self.assertIn("found Intel UHD Graphics 770, 0.2 GB (integrated graphics share system RAM): not used", inst.profile_notes(d))

    def test_two_cards_are_both_named_and_the_larger_sets_the_rung(self):
        notes = inst.profile_notes(amd(12, 8))
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0], "found AMD Radeon Test 0, 12.0 GB -> gpu-12gb")
        self.assertIn("found AMD Radeon Test 1, 8.0 GB (second card", notes[1])

    def test_the_dry_run_prints_the_notes_and_the_profile(self):
        out = []
        code = inst.main([], detected=nvidia(4, "Quadro M2000"), out=out.append)
        text = "\n".join(out)
        self.assertEqual(code, 0)
        self.assertIn("note: found Quadro M2000, 4.0 GB -> gpu-4gb", text)
        self.assertIn("profile:   gpu-4gb", text)
        self.assertIn("Qwen3-4B-Q4_K_M.gguf on GPU 0", text)


class ListProfilesTests(unittest.TestCase):
    def run_main(self, *argv):
        out = []
        code = inst.main(list(argv), detected=None, out=out.append)       # detected=None would probe the machine: must not happen
        return code, "\n".join(out)

    def test_lists_every_rung_with_cards_models_and_speed(self):
        code, text = self.run_main("--list-profiles")
        self.assertEqual(code, 0)
        for name in inst.PROFILES:
            self.assertRegex(text, r"(?m)^%s  \(" % re.escape(name))
        for needle in ("Quadro M2000", "RTX 3090", "GT 1030", "Qwen3-32B-Q4_K_M.gguf", "granite-4.0-h-tiny-Q4_K_M.gguf",
                       "12 layers on the GPU", "(draft)", "13.1 tok/s", "75 tok/s", "gpu-large = gpu-12gb"):
            self.assertIn(needle, text)
        self.assertIn("(2 to 3.5 GB)", text)
        self.assertIn("(23 GB and up)", text)

    def test_json_rows(self):
        code, text = self.run_main("--list-profiles", "--json")
        rows = json.loads(text)
        self.assertEqual([r["name"] for r in rows], list(inst.PROFILES))
        self.assertIsNone(rows[0]["max_vram_mb"])
        self.assertEqual(rows[1]["max_vram_mb"], rows[0]["min_vram_mb"])

    def test_the_command_line_accepts_every_rung_and_the_old_names(self):
        for name in inst.PROFILES:
            out = []
            self.assertEqual(inst.main(["--profile", name, "--json"], detected=amd(24), out=out.append), 0 if name != "cpu" else 0, name)
        for old in inst.PROFILE_ALIASES:
            self.assertEqual(inst.main(["--profile", old, "--json"], detected=amd(24), out=lambda s: None), 0)


class EngineFlagTests(unittest.TestCase):
    def roles(self, profile, d):
        roles, _ = inst.place_roles(profile, d)
        return {r["name"]: r for r in roles}

    def test_the_2gb_rung_offloads_12_layers_and_gives_the_rest_threads(self):
        d = amd(2)
        r = self.roles("gpu-2gb", d)["coordinator"]
        args = inst.engine_argv(r, "/opt/lodestone", 6)
        self.assertEqual(flag(args, "-ngl"), "12")
        self.assertEqual(flag(args, "-t"), "6")
        self.assertEqual(flag(args, "--device"), "Vulkan0")
        self.assertEqual(flag(args, "--flash-attn"), "on")
        self.assertIn("--jinja", args)
        emb = inst.engine_argv(self.roles("gpu-2gb", d)["embedder"], "/opt/lodestone", 6)
        self.assertEqual(flag(emb, "--device"), "none")
        cuda = inst.engine_argv(r, "", 6, backend="cuda", host="0.0.0.0", binary=False, models_dir="/models")
        self.assertEqual((flag(cuda, "--device"), flag(cuda, "-ngl")), ("CUDA0", "12"))

    def test_whole_offload_everywhere_else(self):
        for name in ("gpu-24gb", "gpu-16gb", "gpu-12gb", "gpu-8gb", "gpu-6gb", "gpu-4gb"):
            r = self.roles(name, amd(inst.PROFILES[name]["min_vram_mb"] / 1024.0 + 0.5))["coordinator"]
            args = inst.engine_argv(r, "/opt/lodestone", 6)
            self.assertEqual(flag(args, "-ngl"), "99", name)
            self.assertNotIn("-t", args, name)

    def test_the_14b_carries_its_draft_model_flags_and_only_it(self):
        for name in ("gpu-16gb", "gpu-12gb"):
            r = self.roles(name, amd(16))["coordinator"]
            args = inst.engine_argv(r, "/opt/lodestone", 6)
            self.assertEqual(flag(args, "--spec-draft-model"), "/opt/lodestone/models/Qwen3-0.6B-Q8_0.gguf", name)
            self.assertEqual(flag(args, "--spec-draft-ngl"), "99")
            self.assertEqual((flag(args, "--spec-draft-n-max"), flag(args, "--spec-draft-n-min"), flag(args, "--spec-draft-p-min")),
                             ("12", "2", "0.65"))
            self.assertEqual([f["file"] for f in r["files"]], ["Qwen3-14B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf"], "the draft is downloaded too")
            self.assertEqual((flag(args, "-ctk"), flag(args, "-ctv"), flag(args, "-c"), flag(args, "-np")), ("q4_0", "q4_0", "12288", "2"))
        for name in ("gpu-24gb", "gpu-8gb", "gpu-6gb", "gpu-4gb"):
            r = self.roles(name, amd(24))["coordinator"]
            self.assertFalse(any(a.startswith("--spec-draft") for a in inst.engine_argv(r, "/opt/lodestone", 6)), name)

    def test_the_vision_engine_still_carries_its_projector_after_the_draft_change(self):
        r = self.roles("gpu-16gb", amd(16))["vision"]
        args = inst.engine_argv(r, "/opt/lodestone", 6)
        self.assertEqual(flag(args, "--mmproj"), "/opt/lodestone/models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")
        self.assertEqual(flag(args, "--model"), "/opt/lodestone/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")

    def test_a_partly_offloaded_model_is_a_slow_brain_with_its_own_expectation(self):
        roles, _ = inst.place_roles("gpu-2gb", amd(2))
        self.assertTrue(inst.coordinator_on_cpu(roles))
        self.assertEqual(inst.expectation_lines(roles), inst.PARTIAL_EXPECTATION)
        roles, _ = inst.place_roles("cpu", amd(2))
        self.assertEqual(inst.expectation_lines(roles), inst.CPU_EXPECTATION)
        roles, _ = inst.place_roles("gpu-4gb", amd(4))
        self.assertFalse(inst.coordinator_on_cpu(roles))
        self.assertEqual(inst.expectation_lines(roles), [])

    def test_compose_carries_each_rungs_flags(self):
        for gb, want in SIZES:
            if want == "cpu":
                continue
            stack = inst.docker_stack(nvidia(gb, vulkan=False), {"gpu": "cuda"})
            comp = inst.compose_dict(stack)
            self.assertEqual(stack["profile"], want)
            spec = inst.PROFILES[want]["roles"]["coordinator"]
            cmd = comp["services"]["coordinator"]["command"]
            self.assertEqual(flag(cmd, "--model"), "/models/" + inst.MODELS[spec["model"]]["file"], want)
            self.assertEqual(flag(cmd, "-ngl"), str(spec.get("ngl", 99)), want)
            self.assertEqual(flag(cmd, "-c"), str(spec["ctx"]), want)
            self.assertEqual(flag(cmd, "-ctk"), spec["kv"], want)
            self.assertEqual(("--spec-draft-model" in cmd), bool(spec.get("draft")), want)
            self.assertEqual("-t" in cmd, spec.get("ngl", 99) < 99, want)
            fetched = comp["services"]["models"]["command"][2]
            for role in stack["roles"]:
                for f in role["files"]:
                    self.assertIn("--repo %s --file %s --dest /models" % (f["repo"], f["file"]), fetched, (want, f["file"]))
            if "vision" in inst.PROFILES[want]["roles"] and any(r["name"] == "vision" for r in stack["roles"]):
                self.assertIn("vision", comp["services"])

    def test_native_units_carry_the_flags_too(self):
        steps = inst.plan(amd(2), {})
        unit = next(c for s in steps for p, c in s.get("files", {}).items() if p.endswith("lodestone-coordinator.service"))
        self.assertIn("-ngl 12", unit.replace("\\\n    ", " "))
        steps = inst.plan(amd(12), {})
        unit = next(c for s in steps for p, c in s.get("files", {}).items() if p.endswith("lodestone-coordinator.service"))
        self.assertIn("--spec-draft-model /opt/lodestone/models/Qwen3-0.6B-Q8_0.gguf", unit.replace("\\\n    ", " "))


class DockerMainTests(unittest.TestCase):
    def test_docker_prints_the_card_and_the_rung_and_the_right_folder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = []
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = inst.main(["--docker", "--out", tmp, "--no-build"], detected=nvidia(4, "Quadro M2000", vulkan=False),
                                 out=out.append)
            self.assertEqual(code, 0)
            text = "\n".join(out)
            self.assertIn("note: found Quadro M2000, 4.0 GB -> gpu-4gb", text)
            self.assertIn("profile gpu-4gb, cuda engine images", text)
            nxt = [l for l in out if l.startswith("Next:")][0]
            self.assertIn("cd %s && docker compose up -d`" % shlex.quote(os.path.abspath(tmp)), nxt)
            self.assertNotIn(" ..", nxt)

    def test_the_2gb_rung_tells_the_person_what_to_expect(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = []
            inst.main(["--docker", "--out", tmp, "--profile", "gpu-2gb", "--gpu", "cuda"], detected=nvidia(2, "GT 1030", vulkan=False),
                      out=out.append)
            self.assertIn("Partial-offload profile", "\n".join(out))


if __name__ == "__main__":
    unittest.main()
