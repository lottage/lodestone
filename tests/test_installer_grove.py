"""Offline tests for `python -m lodestone.installer --with-garden` (Grove, the optional agents add-on): the Docker stack it adds
(networks, hardening, no secrets in the Compose file, the firewall script) and the bare-metal steps (users, env files, units,
the nftables table), plus the rule that nothing is applied without an explicit yes. Nothing needs Docker, root or a network."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import installer as inst  # noqa: E402
from lodestone import installer_grove as ig  # noqa: E402

UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'
TAG = "v0.1.0-rc.9"


def machine(system="Linux", config=None):
    return inst.detect({"system": system, "machine": "x86_64", "cpu_count": 8, "euid": 0, "os_release": UBUNTU, "meminfo": None,
                        "vulkaninfo": None, "nvidia_smi": None, "drm": [], "llama_server_path": None, "llama_server_version": None,
                        "disk_free_mb": 500000, "config": config})


def stack(build=True, **extra):
    keys = inst.api_keys(os.path.join(tempfile.gettempdir(), "no-such-folder-for-grove-tests"), True)
    choices = dict({"gpu": "cpu", "profile": "cpu", "stonesage": True, "garden": True, "keys": keys, "tag": TAG, "build": build}, **extra)
    files, st = inst.docker_files(machine(), choices)
    return files, st, keys


class DockerStack(unittest.TestCase):
    def setUp(self):
        self.files, self.stack, self.keys = stack()
        self.comp = inst.compose_dict(self.stack)
        self.svc = self.comp["services"]

    def test_the_services_and_where_each_one_sits(self):
        for name in ("grove-worker", "grove-fetcher", "grove-searxng", "stonesage", "hearth-image"):
            self.assertIn(name, self.svc)
        nets = {n: s.get("networks") for n, s in self.svc.items() if n.startswith("grove") or n == "stonesage"}
        self.assertEqual(nets["grove-worker"], ["grove-model", "grove-web"])
        self.assertEqual(nets["grove-fetcher"], ["grove-web", "grove-egress"])
        self.assertEqual(nets["grove-searxng"], ["grove-egress"])
        self.assertEqual(nets["stonesage"], ["default", "grove-model"])

    def test_the_worker_has_no_route_out(self):
        internal = {n for n, v in self.comp["networks"].items() if v.get("internal")}
        self.assertEqual(internal, {"grove-model", "grove-web"})
        self.assertTrue(set(self.svc["grove-worker"]["networks"]) <= internal, "the worker may sit on internal networks only")
        egress = self.comp["networks"]["grove-egress"]
        self.assertNotIn("internal", egress)
        self.assertEqual(egress["driver_opts"]["com.docker.network.bridge.name"], ig.EGRESS_BRIDGE)
        self.assertIs(egress["enable_ipv6"], False)
        # only the fetcher and SearXNG are on the network with a way out
        self.assertEqual(sorted(n for n, s in self.svc.items() if "grove-egress" in (s.get("networks") or [])),
                         ["grove-fetcher", "grove-searxng"])

    def test_no_host_network_no_privilege_no_published_ports_for_grove(self):
        for name, s in self.svc.items():
            self.assertNotIn("network_mode", s, name)
            self.assertNotIn("privileged", s, name)
            self.assertNotIn("pid", s, name)
            if name.startswith("grove"):
                self.assertNotIn("ports", s, name)
        self.assertNotIn("host", json.dumps(self.comp["networks"]))
        for name in ("grove-worker", "grove-fetcher"):
            s = self.svc[name]
            self.assertIs(s["read_only"], True)
            self.assertEqual(s["cap_drop"], ["ALL"])
            self.assertIn("no-new-privileges:true", s["security_opt"])

    def test_no_secret_is_in_the_compose_file(self):
        text = self.files["docker-compose.yml"]
        for name, value in self.keys.items():
            self.assertNotIn(value, text, name)
        for value in self.keys.values():
            self.assertIn(value, self.files[".env"])
        for var in ig.DOCKER_KEYS:
            self.assertRegex(text, r"\$\{%s:\?" % var if var != "GROVE_SEARXNG_SECRET" else r"\$\{GROVE_SEARXNG_SECRET:\?")
        self.assertNotRegex(text, r"192\.168\.|10\.\d+\.\d+\.\d+|eyJ[A-Za-z0-9_-]{8,}")

    def test_keys_pair_the_cockpit_the_worker_and_the_reader(self):
        stone, worker, fetcher = (self.svc[n]["environment"] for n in ("stonesage", "grove-worker", "grove-fetcher"))
        self.assertEqual(stone["STONESAGE_GROVE_CONTROL_KEY"], worker["GROVE_CONTROL_KEY"].replace("GROVE_CONTROL_KEY", "GROVE_CONTROL_KEY"))
        self.assertIn("GROVE_CONTROL_KEY", stone["STONESAGE_GROVE_CONTROL_KEY"])
        self.assertIn("GROVE_MODEL_KEY", stone["STONESAGE_GROVE_MODEL_KEY"])
        self.assertIn("GROVE_MODEL_KEY", worker["GROVE_MODEL_KEY"])
        self.assertIn("GROVE_FETCHER_KEY", worker["GROVE_FETCHER_KEY"])
        self.assertIn("GROVE_FETCHER_KEY", fetcher["FETCHER_KEY"])
        self.assertEqual(worker["GROVE_MODEL_URL"], "http://stonesage:8888/api/garden/v1")
        self.assertEqual(worker["GROVE_FETCHER_URL"], "http://grove-fetcher:8090")
        self.assertEqual(fetcher["READER_SEARX_URL"], "http://grove-searxng:8080")

    def test_the_cockpit_loads_the_app_and_the_first_resident_is_unnamed_with_the_scheduler_off(self):
        env = self.svc["stonesage"]["environment"]
        self.assertEqual(env["STONESAGE_APPS"], "hearth,grove")
        self.assertEqual(env["GROVE_WORKER_URL"], "http://grove-worker:8780")
        self.assertEqual(env["GROVE_ENGINE_URL"], "http://coordinator:8001/v1")                      # the profile's coordinator
        self.assertEqual(env["GROVE_RESIDENT"], "resident-1")
        self.assertEqual(self.svc["grove-worker"]["environment"]["GROVE_FIRST_RESIDENT"], "resident-1")
        self.assertEqual(self.stack["garden"]["model"], "granite-4.0-h-tiny-Q4_K_M")
        cfg = ig.grove_config(self.stack["garden"])
        self.assertEqual(cfg["schedule"], {"enabled": False})
        self.assertNotIn("control_key", cfg, "Docker passes the keys in the environment, not in the file")

    def test_the_user_may_pick_the_model_endpoint(self):
        _f, st, _k = stack(garden_engine_url="http://my-engine:9000/v1", garden_model="my-model", garden_resident="fern")
        self.assertEqual((st["garden"]["engine_url"], st["garden"]["model"], st["garden"]["resident"]), ("http://my-engine:9000/v1", "my-model", "fern"))
        _f, st, _k = stack(garden_engine_url="http://my-engine:9000/v1")
        self.assertEqual(st["garden"]["model"], "", "a model the installer did not choose is not named after the coordinator's")
        with self.assertRaises(inst.InstallerError):
            stack(garden_resident="Not Valid")

    def test_images_are_built_in_order_and_named_by_version(self):
        build = self.svc["stonesage"]["build"]
        self.assertEqual(build["context"], "https://github.com/lottage/grove.git#" + TAG)
        self.assertEqual(build["additional_contexts"], {"hearth-image": "service:hearth-image"})
        self.assertEqual(self.svc["hearth-image"]["build"]["context"], "https://github.com/lottage/hearth.git#" + TAG)
        self.assertEqual(self.svc["grove-worker"]["build"]["dockerfile"], "docker/worker.Dockerfile")
        self.assertEqual(self.svc["grove-fetcher"]["build"]["dockerfile"], "docker/fetcher.Dockerfile")
        self.assertIn("grove:0.1.0-rc.9", self.svc["stonesage"]["image"])
        self.assertNotIn(":latest", self.files["docker-compose.yml"])

    def test_a_published_release_pulls_instead_of_building(self):
        _f, st, _k = stack(build=False, tag="v0.2.0")
        svc = inst.compose_dict(st)["services"]
        self.assertNotIn("hearth-image", svc)
        for name in ("stonesage", "grove-worker", "grove-fetcher"):
            self.assertNotIn("build", svc[name], name)
        self.assertIn("ghcr.io/lottage/grove-worker:0.2.0", svc["grove-worker"]["image"])

    def test_searxng_is_pinned_by_tag_and_json_is_on_without_a_secret(self):
        self.assertRegex(ig.SEARXNG_IMAGE, r"^docker\.io/searxng/searxng:\d{4}\.\d+\.\d+-[0-9a-f]+$")
        self.assertIn(ig.SEARXNG_IMAGE, self.files[".env"])
        settings = self.files[ig.SEARXNG_SETTINGS_PATH]
        self.assertIn("- json", settings)
        self.assertNotRegex(settings, r"(?i)secret_key|ultrasecretkey")
        self.assertIn("./grove/searxng-settings.yml:/etc/searxng/settings.yml:ro", self.svc["grove-searxng"]["volumes"])

    def test_the_rendered_file_names_the_networks_and_the_volume(self):
        text = self.files["docker-compose.yml"]
        self.assertRegex(text, r"(?m)^networks:\n  grove-model:\n    internal: true\n  grove-web:\n    internal: true\n  grove-egress:")
        self.assertIn("  grove-data: {}", text)
        self.assertIn('com.docker.network.bridge.name: "br-grove-egress"', text)

    def test_without_garden_nothing_of_grove_is_written(self):
        files, st = inst.docker_files(machine(), {"gpu": "cpu", "profile": "cpu", "stonesage": True, "tag": TAG, "build": True})
        self.assertIsNone(st["garden"])
        self.assertEqual(sorted(files), [".env", "docker-compose.yml"])
        self.assertNotRegex(files["docker-compose.yml"] + files[".env"], r"(?i)grove|searxng")

    def test_grove_needs_the_cockpit(self):
        with self.assertRaises(inst.InstallerError):
            inst.docker_files(machine(), {"gpu": "cpu", "profile": "cpu", "stonesage": False, "garden": True, "tag": TAG})

    def test_the_worker_host_name_is_always_an_allowed_host(self):
        # the worker calls http://stonesage:8888 and StoneSage answers 421 to a Host it does not know (the rc.5 dry run)
        env = self.svc["stonesage"]["environment"]["STONESAGE_ALLOWED_HOSTS"]
        self.assertEqual(env, "stonesage,${STONESAGE_ALLOWED_HOSTS:-}")
        self.assertIn("STONESAGE_ALLOWED_HOSTS: stonesage,${STONESAGE_ALLOWED_HOSTS:-}", self.files["docker-compose.yml"].replace('"', ""))
        # a host the user lists in .env comes after it, and the entrypoint drops the empty item
        self.assertEqual(env.split(",")[0], ig.SERVICE_HOST)

    def test_without_garden_the_allowed_hosts_are_what_they_were(self):
        files, st = inst.docker_files(machine(), {"gpu": "cpu", "profile": "cpu", "stonesage": True, "tag": TAG, "build": True})
        comp = inst.compose_dict(st)
        self.assertEqual(comp["services"]["stonesage"]["environment"]["STONESAGE_ALLOWED_HOSTS"], "${STONESAGE_ALLOWED_HOSTS:-}")
        self.assertNotIn("stonesage,", files["docker-compose.yml"])

    def test_rerunning_keeps_the_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = inst.api_keys(tmp, True)
            with open(os.path.join(tmp, ".env"), "w") as f:
                f.write("".join("%s=%s\n" % kv for kv in first.items()))
            self.assertEqual(inst.api_keys(tmp, True), first)
            self.assertEqual(sorted(first), sorted(("STONESAGE_API_KEY", "LODESTONE_API_KEY") + ig.DOCKER_KEYS))
            self.assertEqual(len(set(first.values())), len(first), "every key is its own value")


class FirewallScript(unittest.TestCase):
    def setUp(self):
        self.text = ig.firewall_script()

    def test_it_filters_the_reader_bridge_by_name_and_drops_private_ranges(self):
        self.assertIn("BRIDGE=br-grove-egress", self.text)
        self.assertIn("DOCKER-USER", self.text)
        for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16", "127.0.0.0/8"):
            self.assertIn(net, self.text)
        self.assertEqual(sorted(ig.PRIVATE_V4), sorted(re.search(r'PRIVATE="([^"]+)"', self.text).group(1).split()))
        self.assertIn('iptables -A "$CHAIN" -i "$BRIDGE" -d "$net" -j DROP', self.text)
        self.assertIn('ip6tables -A "$CHAIN" -i "$BRIDGE" -j DROP', self.text, "all IPv6 from that bridge is dropped")

    def test_traffic_inside_the_network_is_allowed_before_the_drops(self):
        self.assertLess(self.text.index('-i "$BRIDGE" -o "$BRIDGE" -j ACCEPT'), self.text.index('-d "$net" -j DROP'))

    def test_the_host_itself_is_closed_to_that_bridge(self):
        self.assertIn('INPUT 1 -i "$BRIDGE" -m conntrack --ctstate NEW -j DROP', self.text)

    def test_it_needs_root_is_undoable_and_touches_only_its_chain(self):
        self.assertIn('"$(id -u)" = 0', self.text)
        self.assertIn("--remove", self.text)
        self.assertNotIn("-F INPUT", self.text)
        self.assertNotIn("-F DOCKER-USER", self.text)
        self.assertNotRegex(self.text, r"-P \w+ (DROP|ACCEPT)")
        self.assertNotIn("iptables -F\n", self.text)

    @unittest.skipUnless(shutil.which("sh"), "sh is not installed")
    def test_it_parses_as_a_shell_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grove-firewall.sh")
            with open(path, "w", newline="\n") as f:
                f.write(self.text)
            r = subprocess.run(["sh", "-n", path], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_boot_unit_runs_before_docker(self):
        unit = ig.firewall_unit()
        self.assertIn("Before=docker.service", unit)
        self.assertIn("ExecStop=/bin/sh /etc/grove/grove-firewall.sh --remove", unit)


class DockerMain(unittest.TestCase):
    def run_main(self, *argv, ask=None, machine_=None, run=None):
        out = []
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        code = inst.main(["--docker", "--out", tmp] + list(argv), detected=machine_ or machine(), out=out.append, ask=ask)
        return code, "\n".join(out), tmp

    def test_the_files_are_written_and_the_firewall_is_shown_but_not_applied(self):
        code, text, tmp = self.run_main("--with-garden", "--gpu", "cpu")
        self.assertEqual(code, 0, text)
        self.assertTrue(os.path.isfile(os.path.join(tmp, "grove-firewall.sh")))
        self.assertTrue(os.path.isfile(os.path.join(tmp, "grove", "searxng-settings.yml")))
        self.assertIn("iptables -A", text)                                            # the script is printed
        self.assertIn("Not applied", text)
        self.assertIn("sudo sh", text)
        if os.name == "posix":
            self.assertEqual(os.stat(os.path.join(tmp, ".env")).st_mode & 0o777, 0o600)
        env = open(os.path.join(tmp, ".env"), encoding="utf-8").read()
        self.assertRegex(env, r"(?m)^GROVE_CONTROL_KEY=\S{30,}$")

    def test_nothing_is_asked_or_added_when_no_one_is_at_a_terminal(self):
        code, text, tmp = self.run_main("--gpu", "cpu")                              # no flag, no ask callback
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(os.path.join(tmp, "grove-firewall.sh")))
        self.assertNotIn("grove", open(os.path.join(tmp, "docker-compose.yml"), encoding="utf-8").read().lower())

    def test_a_person_at_a_terminal_is_asked_and_can_say_no_or_yes(self):
        asked = []

        def no(question, default):
            asked.append(question)
            return False
        code, _t, tmp = self.run_main("--gpu", "cpu", ask=no)
        self.assertEqual(len(asked), 1)
        self.assertIn("Grove", asked[0])
        self.assertFalse(os.path.exists(os.path.join(tmp, "grove-firewall.sh")))
        answers = iter([True, False])                                                  # add Grove: yes; apply the firewall: no
        code, text, tmp = self.run_main("--gpu", "cpu", ask=lambda q, d: next(answers))
        self.assertTrue(os.path.exists(os.path.join(tmp, "grove-firewall.sh")))
        self.assertIn("Not applied", text)

    def test_no_garden_flag_means_no_question(self):
        asked = []
        self.run_main("--no-garden", "--gpu", "cpu", ask=lambda q, d: asked.append(q) or True)
        self.assertEqual(asked, [])

    def test_the_firewall_is_applied_only_with_a_yes_and_root_on_linux(self):
        ran = []
        d = machine()

        def attempt(args, euid, ask=None, system_d=None):
            ran.clear()
            ns = inst.argparse.Namespace(apply_firewall=args)
            out = []
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, ig.FIREWALL_SCRIPT), "w") as f:
                    f.write(ig.firewall_script())
                ok = inst.firewall_flow(ns, system_d or d, tmp, out.append, ask, run=lambda argv, **k: ran.append(argv) or 0, euid=euid)
            return ok, "\n".join(out)
        ok, text = attempt(False, 0)
        self.assertEqual((ok, ran), (False, []))
        self.assertIn("Not applied", text)
        ok, text = attempt(True, 1000)
        self.assertEqual((ok, ran), (False, []))
        self.assertIn("needs root", text)
        ok, _t = attempt(True, 0)
        self.assertTrue(ok)
        self.assertEqual(len(ran), 1)
        self.assertEqual(ran[0][0], "sh")
        ok, _t = attempt(False, 0, ask=lambda q, default: True)                       # a yes at the prompt counts
        self.assertTrue(ok)
        ok, _t = attempt(False, 0, ask=lambda q, default: False)
        self.assertFalse(ok)
        ok, text = attempt(True, 0, system_d=machine("Windows"))                       # Docker Desktop: iptables of the host is not the wall
        self.assertEqual((ok, ran), (False, []))
        self.assertIn("NOT in place", text)

    def test_a_failing_script_is_reported_not_hidden(self):
        ns = inst.argparse.Namespace(apply_firewall=True)
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, ig.FIREWALL_SCRIPT), "w") as f:
                f.write(ig.firewall_script())
            ok = inst.firewall_flow(ns, machine(), tmp, out.append, None, run=lambda argv, **k: 1, euid=0)
        self.assertFalse(ok)
        self.assertIn("failed", "\n".join(out))

    def test_help_lists_the_options(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as stop:
            inst.main(["--help"])
        self.assertEqual(stop.exception.code, 0)
        text = buf.getvalue()
        for flag in ("--with-garden", "--no-garden", "--garden-resident", "--garden-engine-url", "--apply-firewall"):
            self.assertIn(flag, text)


class BareMetal(unittest.TestCase):
    KEYS = {"GROVE_CONTROL_KEY": "c" * 40, "GROVE_MODEL_KEY": "m" * 40, "GROVE_FETCHER_KEY": "f" * 40}

    def plan(self, config=None, **extra):
        choices = dict({"profile": "cpu", "garden": True, "garden_keys": self.KEYS, "tag": TAG}, **extra)
        return inst.plan(machine(config=config), choices)

    def steps(self, **kw):
        return {s["id"]: s for s in self.plan(**kw)}

    def test_the_steps_exist_only_with_garden(self):
        self.assertFalse([s for s in inst.plan(machine(), {"profile": "cpu"}) if s["id"].startswith("grove")])
        ids = [s["id"] for s in self.plan()]
        self.assertEqual([i for i in ids if i.startswith("grove")], ["grove-users", "grove-package", "grove-files", "grove-start"])
        self.assertLess(ids.index("start"), ids.index("grove-users"), "the engines first")

    def test_two_users_the_worker_and_the_reader(self):
        cmds = " ".join(" ".join(c) for c in self.steps()["grove-users"]["commands"])
        self.assertIn("useradd --system", cmds)
        self.assertIn(ig.WORKER_USER, cmds)
        self.assertIn(ig.READER_USER, cmds)
        self.assertIn("--shell /usr/sbin/nologin", cmds)

    def test_pip_installs_grove_at_the_tag_without_dependencies(self):
        pip = [c for c in self.steps()["grove-package"]["commands"] if "pip" in c[0]][0]
        self.assertIn("--no-deps", pip)
        self.assertEqual(pip[-1], "grove @ git+https://github.com/lottage/grove@" + TAG)

    def test_env_files_are_owner_only_and_hold_the_keys_the_config_holds(self):
        files = self.steps()["grove-files"]
        modes = files["modes"]
        for name in ("worker.env", "reader.env"):
            path = "/etc/grove/" + name
            self.assertEqual(modes[path], 0o600)
        worker, reader = files["files"]["/etc/grove/worker.env"], files["files"]["/etc/grove/reader.env"]
        self.assertIn("GROVE_CONTROL_KEY=" + "c" * 40, worker)
        self.assertIn("GROVE_FIRST_RESIDENT=resident-1", worker)
        self.assertIn("GROVE_BIND=127.0.0.1", worker)
        self.assertIn("FETCHER_KEY=" + "f" * 40, reader)
        self.assertNotIn("c" * 40, reader)
        self.assertTrue(files["secret"], "a dry run prints a summary, not the keys")
        out = []
        inst.show([files], out.append)
        self.assertNotIn("c" * 40, "\n".join(out))

    def test_the_units_run_as_the_two_users_with_the_venv_python(self):
        units = self.steps()["grove-files"]["files"]
        worker, reader = units["/etc/systemd/system/grove-worker.service"], units["/etc/systemd/system/grove-reader.service"]
        self.assertIn("User=grove\n", worker)
        self.assertIn("User=grovereader\n", reader)
        self.assertIn("ExecStart=/opt/grove/venv/bin/python -m grove.worker", worker)
        self.assertIn("ExecStart=/opt/grove/venv/bin/python -m grove.reader", reader)
        for unit in (worker, reader):
            for hardening in ("NoNewPrivileges=true", "ProtectSystem=strict", "ProtectHome=true", "PrivateTmp=true", "CapabilityBoundingSet="):
                self.assertIn(hardening, unit)
        self.assertIn("ReadWritePaths=/var/lib/grove", worker)
        self.assertNotIn("ReadWritePaths", reader)

    def test_the_nft_table_confines_each_user_by_uid(self):
        conf = self.steps()["grove-files"]["files"]["/etc/grove/grove-nft.conf"]
        self.assertIn('meta skuid "grovereader" ip daddr { ' + ", ".join(ig.PRIVATE_V4) + " } counter drop", conf)
        self.assertIn('meta skuid "grovereader" meta nfproto ipv6 counter drop', conf)
        self.assertIn('meta skuid "grovereader" ip daddr 127.0.0.1 tcp dport 8080 accept', conf)            # SearXNG on this machine
        self.assertIn('meta skuid "grove" ip daddr 127.0.0.1 tcp dport 8890 accept'.replace("8890", "8090"), conf)   # the reader
        self.assertIn('meta skuid "grove" ip daddr 127.0.0.1 tcp dport 8888 accept', conf)                # StoneSage
        self.assertTrue(conf.rstrip().splitlines()[-3].strip().endswith('meta skuid "grove" counter drop'))
        self.assertLess(conf.index("tcp dport 8888 accept"), conf.index('meta skuid "grove" counter drop'))
        self.assertIn("delete table inet grove", conf)
        self.assertIn("weaker than separate machines or containers", conf)

    def test_the_workers_stonesage_may_be_another_machine_but_must_be_an_address(self):
        conf = self.steps(garden_stonesage_url="http://10.0.0.5:9000")["grove-files"]["files"]
        self.assertIn('meta skuid "grove" ip daddr 10.0.0.5 tcp dport 9000 accept', conf["/etc/grove/grove-nft.conf"])
        self.assertIn("GROVE_MODEL_URL=http://10.0.0.5:9000/api/garden/v1", conf["/etc/grove/worker.env"])
        with self.assertRaises(inst.InstallerError):
            self.plan(garden_stonesage_url="http://stonesage.example:8888")

    def test_the_firewall_is_loaded_only_after_an_explicit_yes(self):
        self.assertNotIn("grove-firewall", self.steps())
        self.assertNotIn(["nft", "-f", "/etc/grove/grove-nft.conf"], [c for s in self.plan() for c in s.get("commands", [])])
        yes = self.steps(apply_firewall=True)
        self.assertEqual(yes["grove-firewall"]["commands"], [["nft", "-f", "/etc/grove/grove-nft.conf"]])

    def test_the_config_gets_the_section_the_app_and_the_keys(self):
        config = [s for s in self.plan() if s["id"] == "config"][0]
        merged = json.loads(config["files"][inst._safe_path("--config", "/opt/lodestone/config.json")])
        self.assertEqual(merged["apps"], ["hearth", "grove"], "no apps key means StoneSage's default (hearth), which must stay on")
        g = merged["grove"]
        self.assertEqual(g["schedule"], {"enabled": False})
        self.assertEqual((g["worker_url"], g["resident"], g["control_key"], g["model_key"]),
                         ("http://127.0.0.1:8780", "resident-1", "c" * 40, "m" * 40))
        self.assertTrue(g["engine_url"].startswith("http://127.0.0.1:") and g["engine_url"].endswith("/v1"))

    def test_an_existing_apps_list_and_grove_section_are_respected(self):
        cfg = {"apps": ["other"], "grove": {"resident": "mine", "schedule": {"enabled": True}}}
        config = [s for s in self.plan(config=cfg) if s["id"] == "config"][0]
        merged = json.loads(list(config["files"].values())[0])
        self.assertEqual(merged["apps"], ["other", "grove"])
        self.assertEqual(merged["grove"]["resident"], "mine")
        self.assertTrue(merged["grove"]["schedule"]["enabled"], "a schedule the person switched on is not switched off")
        again = [s for s in self.plan(config={"apps": ["hearth", "grove"]}) if s["id"] == "config"][0]
        self.assertEqual(json.loads(list(again["files"].values())[0])["apps"], ["hearth", "grove"])

    def test_keys_of_an_earlier_install_are_reused(self):
        files = {"/etc/grove/worker.env": "GROVE_CONTROL_KEY=old-control\nGROVE_MODEL_KEY=old-model\nGROVE_FETCHER_KEY=old-fetch\n",
                 "/etc/grove/reader.env": "FETCHER_KEY=old-fetch\n"}
        keys = inst.existing_grove_keys(machine(), read=lambda p: files.get(p))
        self.assertEqual(keys, {"GROVE_CONTROL_KEY": "old-control", "GROVE_MODEL_KEY": "old-model", "GROVE_FETCHER_KEY": "old-fetch"})
        self.assertIsNone(inst.existing_grove_keys(machine(), read=lambda p: None))
        from_config = inst.existing_grove_keys(machine(config={"grove": {"control_key": "cc", "model_key": "mm"}}), read=lambda p: None)
        self.assertEqual((from_config["GROVE_CONTROL_KEY"], from_config["GROVE_MODEL_KEY"]), ("cc", "mm"))

    def test_the_dry_run_via_main_prints_the_plan_and_the_notes_and_writes_nothing(self):
        out = []
        code = inst.main(["--with-garden", "--profile", "cpu"], detected=machine(), out=out.append)
        text = "\n".join(out)
        self.assertEqual(code, 0, text)
        self.assertIn("grove-users", text)
        self.assertIn("DRY RUN", text)
        self.assertIn("NOT loaded", text)
        self.assertIn("pip install \"grove @ git+https://github.com/lottage/grove@", text)
        self.assertNotRegex(text, r"[A-Za-z0-9_-]{40,}=?$", "no generated key is printed")

    def test_without_the_flag_and_without_a_terminal_the_plan_has_no_grove(self):
        out = []
        inst.main(["--profile", "cpu"], detected=machine(), out=out.append)
        self.assertNotIn("grove", "\n".join(out).lower())


class Pieces(unittest.TestCase):
    def test_apps_with_grove(self):
        self.assertEqual(ig.apps_with_grove(None), ["hearth", "grove"])
        self.assertEqual(ig.apps_with_grove([]), ["hearth", "grove"])
        self.assertEqual(ig.apps_with_grove(["x"]), ["x", "grove"])
        self.assertEqual(ig.apps_with_grove(["grove", "x"]), ["grove", "x"])

    def test_resident_ids(self):
        self.assertEqual(ig.check_resident(""), "resident-1")
        for bad in ("A", "has space", "x" * 65, "../x"):
            with self.assertRaises(ig.GroveError):
                ig.check_resident(bad)

    def test_parse_stonesage(self):
        self.assertEqual(ig.parse_stonesage("")[1:], ("127.0.0.1", 8888))
        self.assertEqual(ig.parse_stonesage("https://192.0.2.7")[1:], ("192.0.2.7", 443))
        for bad in ("ftp://127.0.0.1", "http://localhost:8888", "nonsense"):
            with self.assertRaises(ig.GroveError):
                ig.parse_stonesage(bad)

    def test_release_images_include_grove(self):
        self.assertIn("grove", inst.IMAGE_REPOS)
        self.assertEqual(inst.layer_image("grove-worker", TAG), "ghcr.io/lottage/grove-worker:0.1.0-rc.9")


if __name__ == "__main__":
    unittest.main()
