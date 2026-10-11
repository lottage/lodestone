#!/usr/bin/env python3
"""
StoneSage Model Downloader Engine.
Supports direct GGUF and Hugging Face model downloads with automatic URL resolution,
atomic writes, streaming progress metrics, and remote execution on compute nodes.
"""

import os
import sys
import time
import json
import shlex
import re
import urllib.error
import urllib.parse
import urllib.request
import threading
from lodestone import endpoints
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("StoneSage.Downloader")


VALID_MODEL_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf", re.IGNORECASE)


def normalize_download_url(raw_url: str, custom_filename: Optional[str] = None) -> Tuple[str, str]:
    """
    Normalizes Hugging Face and direct URLs.
    Converts /blob/ to /resolve/ and extracts a clean GGUF filename.
    Automatically queries Hugging Face API to resolve repo URLs to the optimal GGUF file.
    """
    url = (raw_url or "").strip()
    if not url:
        return "", ""

    # Parse and normalize Hugging Face host and blob/resolve paths
    if "hf.co" in url or "huggingface.co" in url:
        url = url.replace("hf.co/", "huggingface.co/")

        # Check if URL is a repository root (e.g. https://huggingface.co/owner/repo)
        parsed = urllib.parse.urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(path_parts) == 2 and "resolve" not in path_parts and "blob" not in path_parts:
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            try:
                api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
                req = urllib.request.Request(api_url, headers={"User-Agent": "StoneSage/4.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    tree = json.loads(resp.read().decode("utf-8"))
                    ggufs = [f for f in tree if f.get("path", "").lower().endswith(".gguf")]
                    if ggufs:
                        chosen = None
                        if custom_filename:
                            custom_clean = custom_filename.lower().replace(".gguf", "")
                            for g in ggufs:
                                if custom_clean in g["path"].lower():
                                    chosen = g["path"]
                                    break
                        if not chosen:
                            # Filter out mmproj vision projectors and heavy unquantized F16 unless necessary
                            candidates = [g["path"] for g in ggufs if not g["path"].lower().startswith("mmproj") and "-f16" not in g["path"].lower()]
                            if not candidates:
                                candidates = [g["path"] for g in ggufs]
                            # Quantization priority list for optimal accuracy and local VRAM fit
                            for pref in ["PQ2_0", "Q4_K_M", "PTQ1_0", "Q5_K_M", "Q4_0", "Q8_0", "Q6_K", "IQ4_NL"]:
                                for c in candidates:
                                    if pref.lower() in c.lower():
                                        chosen = c
                                        break
                                if chosen:
                                    break
                            if not chosen and candidates:
                                chosen = candidates[0]

                        if chosen:
                            resolved_url = f"https://huggingface.co/{repo_id}/resolve/main/{chosen}?download=true"
                            return resolved_url, chosen
            except Exception as e:
                logger.warning(f"Failed to auto-resolve Hugging Face repo tree for '{repo_id}': {e}")

        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        if "?download=true" not in url and not url.endswith(".gguf"):
            if "?" in url:
                url += "&download=true"
            else:
                url += "?download=true"

    # Extract filename from URL path
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    raw_name = path.split("/")[-1] if "/" in path else "model.gguf"
    filename = urllib.parse.unquote(raw_name).strip()

    # Strip any trailing query or fragments
    filename = filename.split("?")[0].split("#")[0].strip()
    if not filename or not filename.lower().endswith(".gguf"):
        if not filename:
            filename = "downloaded_model.gguf"
        else:
            filename = f"{filename}.gguf"

    return url, filename


def _on_engine_host(target: str) -> bool:
    """Is a download target one of the engine host's node ids (or named by its address)? Anything else downloads locally."""
    host = endpoints.host("coordinator") or ""
    return (target or "") in ("node1_primary", "node1_secondary", "dual_gpu") or bool(host and host in (target or ""))


class ModelDownloadManager:
    """Thread-safe orchestrator for local and remote model downloads."""

    def _target(self) -> str:
        return f"{self._user or endpoints.ssh_user()}@{self._host or endpoints.host('coordinator')}"

    def __init__(self, engine_host: str = "", engine_user: str = ""):
        # the engines' host (lodestone/endpoints.py), read when a download starts so config changes apply
        self._host, self._user = engine_host, engine_user
        self.lock = threading.Lock()
        self._local_thread: Optional[threading.Thread] = None
        self._cancel_flag = threading.Event()

        self._state: Dict[str, Any] = {
            "active": False,
            "status": "idle",  # "idle", "starting", "downloading", "completed", "error", "cancelled"
            "url": "",
            "filename": "",
            "target_node": "node1_primary",
            "bytes_downloaded": 0,
            "total_bytes": 0,
            "percent": 0.0,
            "speed_mb": 0.0,
            "eta_seconds": 0,
            "error": None,
            "started_at": 0.0,
            "updated_at": 0.0
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns current download status, polling remote host if active on the engine host."""
        with self.lock:
            try:
                remote_status = self._read_remote_status()
                if remote_status and remote_status.get("active"):
                    self._state.update(remote_status)
                elif remote_status and remote_status.get("status") in ("completed", "error", "cancelled"):
                    if self._state.get("active") or self._state.get("status") in ("starting", "connecting"):
                        self._state.update(remote_status)
                        self._state["active"] = False
            except Exception:
                pass

            return dict(self._state)

    def start_download(self, raw_url: str, custom_filename: Optional[str] = None, target_node: str = "node1_primary") -> Dict[str, Any]:
        """Initiates a background model download."""
        url, default_fn = normalize_download_url(raw_url, custom_filename=custom_filename)
        if not url:
            return {"ok": False, "error": "Invalid or empty download URL provided."}

        filename = (custom_filename or "").strip() or default_fn
        if not filename.lower().endswith(".gguf"):
            filename = f"{filename}.gguf"
        if not VALID_MODEL_FILE.fullmatch(filename):
            # the name becomes a path on the engine host (and part of a remote rm): plain characters only, no folders
            return {"ok": False, "error": "Invalid model filename (letters, digits, . _ - and ending in .gguf; no folders)."}
        try:
            from lodestone.urlcheck import check_fetch_url
            check_fetch_url(url, resolve=False)          # http(s) only; never file://, loopback or a metadata address
        except ValueError as e:
            return {"ok": False, "error": f"That URL is not fetched: {e}"}

        with self.lock:
            if self._state.get("active"):
                return {"ok": False, "error": f"A download is already in progress for '{self._state.get('filename')}'. Cancel it first."}

            self._cancel_flag.clear()
            self._state = {
                "active": True,
                "status": "starting",
                "url": url,
                "filename": filename,
                "target_node": target_node,
                "bytes_downloaded": 0,
                "total_bytes": 0,
                "percent": 0.0,
                "speed_mb": 0.0,
                "eta_seconds": 0,
                "error": None,
                "started_at": time.time(),
                "updated_at": time.time()
            }

            if _on_engine_host(target_node):
                success, err = self._start_remote_download(url, filename)
                if not success:
                    self._state["active"] = False
                    self._state["status"] = "error"
                    self._state["error"] = err
                    return {"ok": False, "error": err}
            else:
                self._local_thread = threading.Thread(target=self._run_local_download, args=(url, filename), daemon=True)
                self._local_thread.start()

            return {
                "ok": True,
                "status": "starting",
                "filename": filename,
                "url": url,
                "target_node": target_node
            }

    def cancel_download(self) -> Dict[str, Any]:
        """Cancels any running download and removes partial files."""
        with self.lock:
            if not self._state.get("active"):
                return {"ok": True, "message": "No active download to cancel."}

            target = self._state.get("target_node", "")
            filename = self._state.get("filename", "")
            if _on_engine_host(target):
                self._cancel_remote_download(filename)
            else:
                self._cancel_flag.set()

            self._state["active"] = False
            self._state["status"] = "cancelled"
            self._state["updated_at"] = time.time()
            return {"ok": True, "status": "cancelled", "filename": filename}

    # =========================================================================
    # Remote Execution on the engine host
    # =========================================================================

    def _start_remote_download(self, url: str, filename: str) -> Tuple[bool, Optional[str]]:
        """Deploys and launches the background downloader on the engine host."""
        dest_dir = "/opt/models"
        final_path = f"{dest_dir}/{filename}"
        part_path = f"{final_path}.part"
        status_file = "/tmp/stonesage_dl_status.json"
        cancel_file = "/tmp/stonesage_dl.cancel"

        # Python script to run in background on the engine host
        remote_script = f"""
import os, sys, time, json, urllib.request, socket

# Ensure IPv4 preference to avoid IPv6 SYN drops on homelab bridges
orig_gai = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_gai(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv4

url = {json.dumps(url)}
final_path = {json.dumps(final_path)}
part_path = {json.dumps(part_path)}
status_file = {json.dumps(status_file)}
cancel_file = {json.dumps(cancel_file)}

if os.path.exists(cancel_file):
    try: os.remove(cancel_file)
    except Exception: pass

state = {{
    "active": True,
    "status": "connecting",
    "filename": os.path.basename(final_path),
    "bytes_downloaded": 0,
    "total_bytes": 0,
    "percent": 0.0,
    "speed_mb": 0.0,
    "eta_seconds": 0,
    "error": None,
    "started_at": time.time(),
    "updated_at": time.time()
}}

def save_state():
    state["updated_at"] = time.time()
    try:
        tmp = status_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, status_file)
    except Exception:
        pass

save_state()

try:
    headers = {{"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StoneSage/4.0"}}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        state["total_bytes"] = total
        state["status"] = "downloading"
        save_state()

        downloaded = 0
        last_time = time.time()
        start_time = last_time
        last_bytes = 0

        with open(part_path, "wb") as out_f:
            while True:
                if os.path.exists(cancel_file):
                    state["status"] = "cancelled"
                    state["active"] = False
                    save_state()
                    try: os.remove(part_path)
                    except Exception: pass
                    sys.exit(0)

                chunk = resp.read(1048576) # 1MB chunk
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_time >= 0.5:
                    elapsed = now - last_time
                    speed_bps = (downloaded - last_bytes) / elapsed if elapsed > 0 else 0
                    speed_mb = round(speed_bps / (1024 * 1024), 2)
                    percent = round((downloaded / total * 100), 2) if total > 0 else 0.0
                    remaining_bytes = max(0, total - downloaded)
                    eta = int(remaining_bytes / speed_bps) if speed_bps > 0 else 0

                    state["bytes_downloaded"] = downloaded
                    state["percent"] = percent
                    state["speed_mb"] = speed_mb
                    state["eta_seconds"] = eta
                    save_state()

                    last_time = now
                    last_bytes = downloaded

        # Atomic move to final location
        os.replace(part_path, final_path)
        state["bytes_downloaded"] = downloaded
        state["percent"] = 100.0
        state["status"] = "completed"
        state["active"] = False
        save_state()

except Exception as ex:
    state["status"] = "error"
    state["active"] = False
    state["error"] = str(ex)
    save_state()
    try:
        if os.path.exists(part_path):
            os.remove(part_path)
    except Exception:
        pass
    sys.exit(1)
"""
        try:
            # Write script to remote runner file via SSH stdin to prevent shell escaping bugs
            script_path = "/tmp/stonesage_dl_runner.py"
            write_cmd = endpoints.ssh_cmd(None, f"cat > {script_path}", target=self._target())
            sub_w = subprocess.run(write_cmd, input=remote_script, text=True, timeout=10, capture_output=True)
            if sub_w.returncode != 0:
                return False, f"SSH script upload failed: {sub_w.stderr.strip()}"

            # Clean old cancel / status files and launch background runner
            launcher = (
                f"rm -f {cancel_file} {status_file} && "
                f"nohup python3 -u {script_path} < /dev/null > /tmp/stonesage_dl.log 2>&1 &"
            )
            cmd = endpoints.ssh_cmd(None, launcher, target=self._target(), no_stdin=True)
            sub = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if sub.returncode != 0:
                return False, f"SSH launch failed: {sub.stderr.strip()}"
            return True, None
        except Exception as e:
            return False, str(e)

    def _read_remote_status(self) -> Optional[Dict[str, Any]]:
        """Reads remote status JSON file from the engine host via SSH."""
        cmd = endpoints.ssh_cmd(None, "cat /tmp/stonesage_dl_status.json 2>/dev/null", target=self._target())
        try:
            sub = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if sub.returncode == 0 and sub.stdout.strip():
                return json.loads(sub.stdout.strip())
        except Exception:
            pass
        return None

    def _cancel_remote_download(self, filename: str):
        """Signals the engine host downloader to abort and clean up."""
        if not VALID_MODEL_FILE.fullmatch(str(filename or "")):
            logger.warning("Remote cancel: refusing to remove a file whose name is not plain")
            filename = "none.gguf"
        part = shlex.quote(f"/opt/models/{filename}.part")
        cancel_cmd = (
            "touch /tmp/stonesage_dl.cancel && "
            "pkill -f 'stonesage_dl' || true && "
            f"rm -f -- {part} /tmp/stonesage_dl_status.json"
        )
        cmd = endpoints.ssh_cmd(None, cancel_cmd, target=self._target())
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except Exception as e:
            logger.warning(f"Error sending remote cancel: {e}")

    # =========================================================================
    # Local Workstation Execution
    # =========================================================================

    def _run_local_download(self, url: str, filename: str):
        """Runs the download locally in a worker thread."""
        local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")
        try:
            os.makedirs(local_dir, exist_ok=True)
        except Exception:
            local_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(local_dir, exist_ok=True)

        final_path = os.path.join(local_dir, filename)
        part_path = f"{final_path}.part"

        with self.lock:
            self._state["status"] = "connecting"

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StoneSage/4.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                with self.lock:
                    self._state["total_bytes"] = total
                    self._state["status"] = "downloading"

                downloaded = 0
                last_time = time.time()
                last_bytes = 0

                with open(part_path, "wb") as out_f:
                    while True:
                        if self._cancel_flag.is_set():
                            with self.lock:
                                self._state["status"] = "cancelled"
                                self._state["active"] = False
                            if os.path.exists(part_path):
                                os.remove(part_path)
                            return

                        chunk = resp.read(1048576)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_time >= 0.5:
                            elapsed = now - last_time
                            speed_bps = (downloaded - last_bytes) / elapsed if elapsed > 0 else 0
                            speed_mb = round(speed_bps / (1024 * 1024), 2)
                            percent = round((downloaded / total * 100), 2) if total > 0 else 0.0
                            remaining_bytes = max(0, total - downloaded)
                            eta = int(remaining_bytes / speed_bps) if speed_bps > 0 else 0

                            with self.lock:
                                self._state["bytes_downloaded"] = downloaded
                                self._state["percent"] = percent
                                self._state["speed_mb"] = speed_mb
                                self._state["eta_seconds"] = eta
                                self._state["updated_at"] = now

                            last_time = now
                            last_bytes = downloaded

                # Atomic replace
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.replace(part_path, final_path)

                with self.lock:
                    self._state["bytes_downloaded"] = downloaded
                    self._state["percent"] = 100.0
                    self._state["status"] = "completed"
                    self._state["active"] = False
                    self._state["updated_at"] = time.time()

        except Exception as ex:
            logger.error(f"Local download failed: {ex}")
            with self.lock:
                self._state["status"] = "error"
                self._state["active"] = False
                self._state["error"] = str(ex)
                self._state["updated_at"] = time.time()
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass


# Global singleton instance
model_download_manager = ModelDownloadManager()


# =========================================================================
# Plain file download: no state, no SSH. lodestone.installer runs this as
#   python3 -m lodestone.model_downloader --repo OWNER/NAME --file FILE.gguf --dest DIR
# =========================================================================

SUMS_NAME = "SHA256SUMS"


def hf_resolve_url(repo: str, filename: str) -> str:
    """The direct download URL of one file in a Hugging Face repository (main branch)."""
    return "https://huggingface.co/%s/resolve/main/%s?download=true" % (repo.strip("/"), urllib.parse.quote(filename))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def read_sums(dest_dir: str) -> Dict[str, str]:
    """{filename: sha256} from DEST/SHA256SUMS (the `sha256sum` format: hash, two spaces, name)."""
    out: Dict[str, str] = {}
    try:
        with open(os.path.join(dest_dir, SUMS_NAME), encoding="utf-8") as f:
            for line in f:
                digest, _, name = line.strip().partition("  ")
                if digest and name:
                    out[name] = digest
    except OSError:
        pass
    return out


def record_sha256(dest_dir: str, filename: str, digest: str) -> None:
    """Add or replace one line of DEST/SHA256SUMS, so `sha256sum -c` works there."""
    sums = read_sums(dest_dir)
    sums[filename] = digest
    path = os.path.join(dest_dir, SUMS_NAME)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for name in sorted(sums):
            f.write("%s  %s\n" % (sums[name], name))
    os.replace(path + ".tmp", path)


def download_file(url: str, dest_dir: str, filename: str, opener=None, timeout: float = 60,
                  progress=None) -> Dict[str, Any]:
    """Download url to dest_dir/filename. Atomic (written as .part, renamed when complete), resumes a .part, and does
    nothing when the file is already there with the size the server reports. Returns
    {"status": "present" | "downloaded", "path", "size"}; raises OSError when it cannot finish."""
    opener = opener or urllib.request.urlopen
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(dest_dir, filename)
    part = final + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {"User-Agent": "Lodestone/1"}
    if have:
        headers["Range"] = "bytes=%d-" % have
    try:
        resp = opener(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:        # the .part already holds everything
            os.replace(part, final)
            return {"status": "downloaded", "path": final, "size": os.path.getsize(final)}
        if os.path.isfile(final):
            return {"status": "present", "path": final, "size": os.path.getsize(final)}
        raise OSError("download of %s failed: HTTP %s" % (filename, e.code))
    except Exception as e:
        if os.path.isfile(final):         # offline: trust what is there
            return {"status": "present", "path": final, "size": os.path.getsize(final)}
        raise OSError("download of %s failed: %s" % (filename, e))
    with resp:
        length = int(resp.headers.get("Content-Length") or 0)
        resumed = getattr(resp, "status", 200) == 206
        total = (have + length) if resumed else length
        if os.path.isfile(final) and total and os.path.getsize(final) == total:
            return {"status": "present", "path": final, "size": total}
        written = have if resumed else 0
        with open(part, "ab" if resumed else "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total)
    if total and written != total:
        raise OSError("download of %s stopped at %d of %d bytes (run again to resume)" % (filename, written, total))
    os.replace(part, final)
    return {"status": "downloaded", "path": final, "size": written}


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m lodestone.model_downloader",
                                 description="Download one GGUF file and record its SHA-256 in DEST/SHA256SUMS.")
    ap.add_argument("--repo", help="Hugging Face repository, OWNER/NAME")
    ap.add_argument("--file", help="file name inside the repository (with --repo)")
    ap.add_argument("--url", help="a direct URL instead of --repo/--file")
    ap.add_argument("--dest", required=True, help="directory to put the file in")
    ap.add_argument("--filename", help="save under this name (default: the file's own)")
    args = ap.parse_args(argv)
    if args.url:
        url, name = normalize_download_url(args.url, args.filename)
    elif args.repo and args.file:
        url, name = hf_resolve_url(args.repo, args.file), args.file
    else:
        ap.error("give --repo and --file, or --url")
    name = args.filename or name
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] >= 5:
            last[0] = time.time()
            print("  %s: %.0f of %.0f MB" % (name, done / 1048576, total / 1048576), flush=True)

    try:
        got = download_file(url, args.dest, name, progress=progress)
    except OSError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    digest = read_sums(args.dest).get(name) if got["status"] == "present" else None
    if not digest:
        digest = sha256_file(got["path"])
        record_sha256(args.dest, name, digest)
    print("%s: %s (%.0f MB), sha256 %s" % (name, got["status"], got["size"] / 1048576, digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
