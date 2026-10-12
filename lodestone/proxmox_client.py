"""
Proxmox VE REST API Client for StoneSage
Queries node telemetry (CPU, RAM, storage, uptime), lists LXCs and VMs,
and executes safe lifecycle operations (reboot, start, shutdown).
Falls back to TCP socket health checks when API token is unconfigured.
"""

import logging
import ssl
import json
from lodestone import endpoints, tls
import urllib.request
import urllib.error
import socket
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger("stonesage.proxmox")
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    """The cockpit polls often: say a failure once per node and kind, not on every poll."""
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


class ProxmoxClient:
    def __init__(self, node_configs: Dict[str, Any]):
        self.node_configs = node_configs

    def _ssl_ctx(self, node_config: Dict[str, Any]) -> ssl.SSLContext:
        """Verified by default (lodestone/tls.py). A self-signed Proxmox sets `ca_file` (its PEM certificate) or, knowingly,
        `verify_tls: false`, in the "proxmox" section or on the node's own entry (the node wins)."""
        return tls.context(self.node_configs, node_config if isinstance(node_config, dict) else None, name="proxmox")

    def _get_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        token_id = config.get("token_id", "").strip()
        token_secret = config.get("token_secret", "").strip()

        # Fallback to token_value in node config or root proxmox config
        if not (token_id and token_secret):
            raw = config.get("token_value", "").strip() or self.node_configs.get("token_value", "").strip()
            if raw:
                if raw.startswith("PVEAPIToken="):
                    raw = raw[len("PVEAPIToken="):].strip()
                if "=" in raw:
                    parts = raw.split("=", 1)
                    token_id = parts[0].strip()
                    token_secret = parts[1].strip()

        if token_id and token_secret:
            headers["Authorization"] = f"PVEAPIToken={token_id}={token_secret}"
        return headers

    def ping_node(self, host: str, port: int = 8006, timeout: float = 1.5) -> Dict[str, Any]:
        """Fast TCP probe to verify if Proxmox host port is open and measure latency."""
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                latency = round((time.perf_counter() - start) * 1000, 1)
                return {"online": True, "latency_ms": latency}
        except Exception as e:
            return {"online": False, "error": str(e)}

    def get_node_status(self, node_key: str) -> Dict[str, Any]:
        """Fetch node CPU, RAM, kernel version, and storage from /api2/json/nodes/{node}/status."""
        config = self.node_configs.get(node_key, {})
        default_host = self.node_configs.get("host") or endpoints.host("proxmox")
        host = config.get("host", default_host)
        port = config.get("port", 8006)
        node_name = config.get("name", node_key.replace("_node", ""))
        
        # Ping first
        ping_res = self.ping_node(host, port)
        if not ping_res.get("online"):
            return {
                "node": node_name,
                "online": False,
                "error": ping_res.get("error", "Host unreachable")
            }

        headers = self._get_headers(config)
        if "Authorization" not in headers:
            # Fallback when API token is not yet configured
            return {
                "node": node_name,
                "online": True,
                "latency_ms": ping_res.get("latency_ms"),
                "token_configured": False,
                "cpu": None,
                "cpu_pct": None,
                "memory": None,
                "memory_pct": None,
                "message": "Node reachable over HTTPS. Add Proxmox API token in Settings to view CPU/RAM gauges."
            }

        url = f"https://{host}:{port}/api2/json/nodes/{node_name}/status"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, context=self._ssl_ctx(config), timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8")).get("data", {})
                cpu_usage = round(data.get("cpu", 0) * 100, 1)
                mem_data = data.get("memory", {})
                mem_used = mem_data.get("used", 0)
                mem_total = mem_data.get("total", 1)
                mem_pct = round((mem_used / mem_total) * 100, 1) if mem_total else 0

                rootfs = data.get("rootfs", {})
                disk_used = rootfs.get("used", 0)
                disk_total = rootfs.get("total", 1)
                disk_pct = round((disk_used / disk_total) * 100, 1) if disk_total else 0

                return {
                    "node": node_name,
                    "online": True,
                    "latency_ms": ping_res.get("latency_ms"),
                    "token_configured": True,
                    "cpu": data.get("cpu", 0),
                    "cpu_pct": cpu_usage,
                    "memory": mem_data,
                    "memory_pct": mem_pct,
                    "memory_used_gb": round(mem_used / (1024**3), 2),
                    "memory_total_gb": round(mem_total / (1024**3), 2),
                    "disk": rootfs,
                    "disk_pct": disk_pct,
                    "disk_used_gb": round(disk_used / (1024**3), 2),
                    "disk_total_gb": round(disk_total / (1024**3), 2),
                    "uptime_sec": data.get("uptime", 0),
                    "kernel": data.get("kversion", "unknown")
                }
        except urllib.error.HTTPError as he:
            err_msg = str(he)
            try:
                err_data = json.loads(he.read().decode("utf-8"))
                if err_data.get("message"):
                    err_msg = err_data["message"].strip()
            except Exception:
                pass
            
            if "Permission check failed" in err_msg or he.code == 403:
                err_msg = "Permission check failed (Privilege Separation enabled). In Proxmox, assign Role 'Administrator' or 'PVEAuditor' to token under Datacenter -> Permissions, or recreate token with Privilege Separation unchecked."

            return {
                "node": node_name,
                "online": True,
                "latency_ms": ping_res.get("latency_ms"),
                "token_configured": True,
                "error": err_msg
            }
        except Exception as e:
            return {
                "node": node_name,
                "online": True,
                "latency_ms": ping_res.get("latency_ms"),
                "token_configured": True,
                "error": tls.explain(e, "Proxmox")
            }

    def get_guests(self, node_key: str) -> Dict[str, Any]:
        """List LXC containers and QEMU VMs running on the target node."""
        config = self.node_configs.get(node_key, {})
        default_host = self.node_configs.get("host") or endpoints.host("proxmox")
        host = config.get("host", default_host)
        port = config.get("port", 8006)
        node_name = config.get("name", node_key.replace("_node", ""))
        headers = self._get_headers(config)

        if "Authorization" not in headers:
            return {"ok": False, "guests": [], "message": "Proxmox API Token required for guest listing"}

        guests = []
        # Query LXCs
        try:
            lxc_url = f"https://{host}:{port}/api2/json/nodes/{node_name}/lxc"
            req = urllib.request.Request(lxc_url, headers=headers)
            with urllib.request.urlopen(req, context=self._ssl_ctx(config), timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
                for item in data:
                    item["type"] = "lxc"
                    item["node"] = node_name
                    guests.append(item)
        except Exception as e:
            _warn_once("lxc:" + node_name, f"Proxmox: cannot list containers on {node_name}: {tls.explain(e, 'Proxmox')}")

        # Query VMs (QEMU)
        try:
            vm_url = f"https://{host}:{port}/api2/json/nodes/{node_name}/qemu"
            req = urllib.request.Request(vm_url, headers=headers)
            with urllib.request.urlopen(req, context=self._ssl_ctx(config), timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
                for item in data:
                    item["type"] = "qemu"
                    item["node"] = node_name
                    guests.append(item)
        except Exception as e:
            _warn_once("qemu:" + node_name, f"Proxmox: cannot list VMs on {node_name}: {tls.explain(e, 'Proxmox')}")

        return {"ok": True, "guests": guests}

    def reboot_guest(self, node_key: str, guest_type: str, vmid: str) -> Dict[str, Any]:
        """Reboot an LXC container or QEMU VM."""
        config = self.node_configs.get(node_key, {})
        default_host = self.node_configs.get("host") or endpoints.host("proxmox")
        host = config.get("host", default_host)
        port = config.get("port", 8006)
        node_name = config.get("name", node_key.replace("_node", ""))
        headers = self._get_headers(config)

        if "Authorization" not in headers:
            return {"ok": False, "error": "Proxmox API token required for power operations"}

        path_type = "lxc" if guest_type == "lxc" else "qemu"
        url = f"https://{host}:{port}/api2/json/nodes/{node_name}/{path_type}/{vmid}/status/reboot"
        
        req = urllib.request.Request(url, data=b"", headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, context=self._ssl_ctx(config), timeout=5) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return {"ok": True, "upid": res.get("data"), "message": f"Reboot initiated for {path_type} {vmid}"}
        except Exception as e:
            return {"ok": False, "error": tls.explain(e, "Proxmox")}
