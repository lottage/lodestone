"""
Test runner with a hard unit/live split.

    python tests/run_tests.py          # unit: tests/test_*.py with every non-localhost connection blocked
    python tests/run_tests.py live     # live: tests/live/test_*.py against the real stack (read-only checks)
    python tests/run_tests.py unit "test_household*.py"   # only the test files matching a glob

Unit tests must never reach the LAN: several harness modules (Valkey A-MEM, the MCP bridge,
edge-fleet OTA) connect to live services by default, and a unit run on the LAN used to write
test cards into Courage's live Valkey. The guard below turns any such connection into an
OSError so modules fall back to their offline paths, and lists every attempt at the end.
"""

import os
import shutil
import socket
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)  # same import root as `python -m unittest` from the repo root
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
blocked_attempts = []


def _is_local(host) -> bool:
    host = str(host)
    return host in LOCAL_HOSTS or host.startswith("127.")


def install_network_guard():
    orig_connect = socket.socket.connect
    orig_create = socket.create_connection

    def guarded_connect(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if not _is_local(host):
            blocked_attempts.append(addr)
            raise OSError(f"unit tests may not open network connections (blocked {addr})")
        return orig_connect(self, addr)

    def guarded_create(address, *args, **kwargs):
        if not _is_local(address[0]):
            blocked_attempts.append(address)
            raise OSError(f"unit tests may not open network connections (blocked {address})")
        return orig_create(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create

    # Sockets are not the only way out: several modules shell out to ssh/scp (the engine host scripts, some of which CHANGE
    # things, e.g. wildlife_admin.py relabels sightings). Refuse network programs the same way (2026-10-09: the route
    # golden test found /api/presence/snapshot really SSHing to the engine host from a unit test).
    import subprocess
    orig_popen = subprocess.Popen

    class GuardedPopen(orig_popen):
        def __init__(self, args, *a, **kw):
            first = args if isinstance(args, (str, bytes)) else (args[0] if args else "")
            first = first.decode() if isinstance(first, bytes) else str(first)
            prog = os.path.basename(first.strip().split()[0] if first.strip() else "").lower()
            prog = prog[:-4] if prog.endswith(".exe") else prog
            if prog in NETWORK_PROGRAMS:
                blocked_attempts.append(f"{prog} (subprocess)")
                raise OSError(f"unit tests may not run network programs (blocked {prog})")
            super().__init__(args, *a, **kw)

    subprocess.Popen = GuardedPopen


NETWORK_PROGRAMS = {"ssh", "scp", "sftp", "rsync", "curl", "wget", "nc", "ncat", "telnet", "plink", "pscp", "ping"}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "test_*.py"
    if mode == "unit":
        os.environ.setdefault("STONESAGE_OFFLINE", "1")
        # modules that persist state (agent profiles, custom tools) write here instead of into the repo
        # (seeded with copies of the committed files, which some tests read)
        scratch = tempfile.mkdtemp(prefix="stonesage-unit-")
        profiles = os.path.join(scratch, "agent_profiles")
        committed = os.path.join(REPO_ROOT, "server setup", "cluster-bridge", "agent_profiles")
        # the real profiles are private (absent from the public snapshot); tests/fixtures holds the two base agents tests use
        fixtures = os.path.join(TESTS_DIR, "fixtures", "agent_profiles")
        shutil.copytree(committed if os.path.isdir(committed) else fixtures, profiles)
        data = os.path.join(scratch, "data")
        os.makedirs(data)
        tools_json = os.path.join(REPO_ROOT, "data", "custom_tools.json")
        if os.path.exists(tools_json):
            shutil.copy2(tools_json, data)
        os.environ["STONESAGE_AGENT_PROFILES_DIR"] = profiles
        os.environ["STONESAGE_DATA_DIR"] = data
        # Several tests save the config (workspace roots, active instance). They write to scratch, never to the real
        # config.json (which holds the operator's secrets); absent at first, so reads fall back to config.example.json.
        os.environ["STONESAGE_CONFIG_FILE"] = os.path.join(scratch, "config.json")
        # Services nobody configured would default to localhost (lodestone/endpoints.py), and on Windows a refused
        # localhost connect takes ~2 s. Point them at TEST-NET instead: the guard refuses it at once, and nothing local
        # (a developer's own llama-server) is ever touched. A test's own config still wins over these.
        sys.path.insert(0, os.path.join(REPO_ROOT, "StoneSage", "backend"))
        from lodestone import endpoints
        for name, (_, env, default) in endpoints.SERVICES.items():
            if default and env not in os.environ:
                os.environ[env] = default.replace("127.0.0.1", "192.0.2.1").replace("homeassistant.local", "192.0.2.1")
        install_network_guard()
        suite = unittest.defaultTestLoader.discover(TESTS_DIR, pattern=pattern, top_level_dir=TESTS_DIR)
    elif mode == "live":
        suite = unittest.defaultTestLoader.discover(os.path.join(TESTS_DIR, "live"), pattern=pattern)
    else:
        print(f"unknown mode {mode!r}; use 'unit' or 'live'")
        return 2

    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if blocked_attempts:
        uniq = sorted({f"{a[0]}:{a[1]}" if isinstance(a, tuple) else str(a) for a in blocked_attempts})
        print(f"\nNetwork guard blocked {len(blocked_attempts)} connection attempt(s) to: {', '.join(uniq)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
