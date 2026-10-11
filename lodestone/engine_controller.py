import json
import logging
import re
import socket
import subprocess
import time
import urllib.request
import urllib.error
from lodestone import endpoints
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Dict, Any, List

logger = logging.getLogger("StoneSage.EngineController")

def engine_registry() -> Dict[str, Dict[str, Any]]:
    """Engines from config.json (URLs) and the live system profile (GPU, model, systemd unit). Nothing hardcoded."""
    from lodestone import system_profile
    from urllib.parse import urlparse
    cfg = system_profile.load_cfg()
    prof = system_profile.get_profile(cfg)
    reg = {}
    for role, key, purpose in system_profile.ROLES:
        url = cfg.get("cluster", {}).get(key)
        if not url:
            continue
        u = urlparse(url)
        eng = prof.get("engines", {}).get(role) or {}
        gpu = eng.get("gpu") or {}
        device = gpu.get("name") or ("CPU" if eng.get("offloaded") else "unknown")
        reg[role] = {
            "port": u.port, "host": u.hostname,
            "service": eng.get("unit") or f"llama-{role}",
            "device": device,
            "role": f"{purpose} ({eng['model']})" if eng.get("model") else purpose,
            "is_local": True,
        }
    return reg


def _is_valid_service_name(service_name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9\-]+$", service_name))

def probe_engine_health(host: str, port: int, timeout: float = 1.0) -> dict:
    """Probes engine health via TCP and HTTP GET /health."""
    start_time = time.time()
    try:
        url = f"http://{host}:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.getcode()
            if status == 200:
                latency_ms = (time.time() - start_time) * 1000
                return {"online": True, "status": "ok", "latency_ms": latency_ms}
    except Exception as e:
        logger.debug(f"Health probe failed for {host}:{port}: {e}")
        pass
    return {"online": False, "status": "error", "latency_ms": 0.0}

def fetch_engine_props(host: str, port: int, timeout: float = 1.5) -> dict:
    """Fetches engine properties via GET /props."""
    try:
        url = f"http://{host}:{port}/props"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                return {
                    "model_path": data.get("model_path", ""),
                    "model_alias": data.get("model_alias", ""),
                    "model_ftype": data.get("model_ftype", ""),
                    "total_slots": data.get("total_slots", 0),
                    "modalities": data.get("modalities", {}),
                    "chat_template_caps": data.get("chat_template_caps", {}),
                    "n_ctx": data.get("default_generation_settings", {}).get("n_ctx", 0),
                    "raw": data
                }
    except Exception as e:
        logger.debug(f"Props fetch failed for {host}:{port}: {e}")
        return {"error": str(e)}
    return {"error": "Unknown error"}

def fetch_engine_slots(host: str, port: int, timeout: float = 1.5) -> list:
    """Fetches engine slots via GET /slots."""
    try:
        url = f"http://{host}:{port}/slots"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                if isinstance(data, list):
                    return data
    except Exception as e:
        logger.debug(f"Slots fetch failed for {host}:{port}: {e}")
    return []

def parse_prometheus_metrics(text: str) -> dict:
    """Parses Prometheus exposition format text into a flat dict."""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            key, val_str = parts
            try:
                if "." in val_str or "e" in val_str.lower():
                    metrics[key] = float(val_str)
                else:
                    metrics[key] = int(val_str)
            except ValueError:
                pass
    return metrics

def fetch_engine_metrics(host: str, port: int, timeout: float = 1.5) -> dict:
    """Fetches engine metrics via GET /metrics."""
    try:
        url = f"http://{host}:{port}/metrics"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() == 200:
                text = response.read().decode('utf-8')
                return parse_prometheus_metrics(text)
    except Exception as e:
        logger.debug(f"Metrics fetch failed for {host}:{port}: {e}")
    return {}

def build_engine_state(engine_key: str) -> dict:
    """Builds a unified engine state dictionary."""
    registry = engine_registry()
    if engine_key not in registry:
        return {"error": f"Unknown engine key: {engine_key}"}
    
    config = registry[engine_key]
    host = config["host"]
    port = config["port"]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_health = executor.submit(probe_engine_health, host, port)
        f_props = executor.submit(fetch_engine_props, host, port)
        f_slots = executor.submit(fetch_engine_slots, host, port)
        f_metrics = executor.submit(fetch_engine_metrics, host, port)
        
        health = f_health.result()
        props = f_props.result()
        slots = f_slots.result()
        metrics = f_metrics.result()
        
    return {
        "key": engine_key,
        "host": host,
        "port": port,
        "device": config["device"],
        "role": config["role"],
        "service": config["service"],
        "online": health.get("online", False),
        "latency_ms": health.get("latency_ms", 0.0),
        "model": {
            "path": props.get("model_path", ""),
            "alias": props.get("model_alias", ""),
            "quantization": props.get("model_ftype", ""),
            "modalities": props.get("modalities", {}),
        },
        "slots": slots,
        "metrics": metrics,
        "props": props,
    }

def build_all_engine_states() -> dict:
    """Builds state for all registered engines in parallel."""
    states = {}
    registry = engine_registry()
    with ThreadPoolExecutor(max_workers=max(len(registry), 1)) as executor:
        futures = {executor.submit(build_engine_state, key): key for key in registry}
        for future in futures:
            key = futures[future]
            states[key] = future.result()
            
    return {
        "engines": states,
        "timestamp": time.time()
    }

def _configured_device(host: str, port: int) -> Dict[str, Any]:
    """The user's own description of a node (config.json harness_instances[].hardware), matched by host:port."""
    from lodestone import system_profile
    for inst in system_profile.load_cfg().get("harness_instances", []):
        u = inst.get("url", "")
        if f"//{host}:{port}" in u:
            return {"name": inst.get("name"), "hardware": inst.get("hardware") or {}}
    return {}


def _lmstudio_state(host: str, port: int, name: str, data: dict, latency_ms: float) -> dict:
    """LM Studio REST API v1 (/api/v1/models) state. LM Studio has no /props, /slots, /health or /metrics:
    calling those only fills its developer log with 'Unexpected endpoint' errors."""
    models = [m for m in data.get("models", []) if m.get("type", "llm") == "llm"]
    loaded = next((m for m in models if m.get("loaded_instances")), None)
    inst = (loaded or {}).get("loaded_instances", [{}])[0] if loaded else {}
    cfg = inst.get("config", {}) if isinstance(inst, dict) else {}
    dev = _configured_device(host, port)
    hw = dev.get("hardware") or {}
    device = ", ".join(str(x) for x in (hw.get("device"), hw.get("cpu"), f"{hw['ram_gb']} GB" if hw.get("ram_gb") else None) if x)
    return {
        "key": name, "host": host, "port": port,
        "device": device or "LM Studio host",
        "role": dev.get("name") or "LM Studio",
        "service": "lm-studio", "api": "lm-studio",
        "online": True, "latency_ms": latency_ms,
        "model": {
            "path": (loaded or {}).get("key", ""),
            "alias": (loaded or {}).get("display_name") or (loaded or {}).get("key", ""),
            "quantization": ((loaded or {}).get("quantization") or {}).get("name", ""),
            "params": (loaded or {}).get("params_string", ""),
            "architecture": (loaded or {}).get("architecture", ""),
            "context": cfg.get("context_length") or 0,
            "max_context": (loaded or {}).get("max_context_length") or 0,
            "modalities": {"vision": bool(((loaded or {}).get("capabilities") or {}).get("vision"))},
        },
        "status_note": None if loaded else "online, no model loaded",
        "available_models": [{"key": m.get("key"), "name": m.get("display_name") or m.get("key"),
                              "params": m.get("params_string"), "quant": (m.get("quantization") or {}).get("name"),
                              "max_context": m.get("max_context_length"), "size_gb": round((m.get("size_bytes") or 0) / 1024**3, 2)}
                             for m in models],
        "slots": [], "metrics": {}, "props": {}, "is_dynamic": True,
    }


def build_dynamic_engine_state(host: str, port: int, name: str = "dynamic") -> dict:
    """State for a roaming/dynamic engine. LM Studio is detected by its REST API and polled only through it;
    llama-server gets the /health, /props, /slots, /metrics probes; other OpenAI servers just /v1/models."""
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/v1/models", headers={"User-Agent": "StoneSage-EngineConsole"})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                return _lmstudio_state(host, port, name, data, round((time.time() - t0) * 1000, 1))
    except Exception:
        pass

    # Not LM Studio. llama-server answers /props with JSON; only then use its other endpoints.
    props = fetch_engine_props(host, port)
    if props:
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_health = executor.submit(probe_engine_health, host, port)
            f_slots = executor.submit(fetch_engine_slots, host, port)
            f_metrics = executor.submit(fetch_engine_metrics, host, port)
            health, slots, metrics = f_health.result(), f_slots.result(), f_metrics.result()
        api = "llama-server"
    else:
        health, slots, metrics, api = {"online": False, "latency_ms": 0.0}, [], {}, "openai"

    model_alias = props.get("model_alias", "")
    if not props:  # plain OpenAI-compatible server (Ollama, vLLM...)
        try:
            req = urllib.request.Request(f"http://{host}:{port}/v1/models", headers={"User-Agent": "StoneSage-EngineConsole"})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                model_list = json.loads(response.read().decode("utf-8")).get("data", [])
                health = {"online": True, "latency_ms": round((time.time() - t0) * 1000, 1)}
                model_alias = model_list[0].get("id", "") if model_list else ""
        except Exception:
            pass

    dev = _configured_device(host, port)
    return {
        "key": name, "host": host, "port": port,
        "device": (dev.get("hardware") or {}).get("device") or "unknown",
        "role": dev.get("name") or "Dynamic Node",
        "service": "unknown", "api": api,
        "online": health.get("online", False),
        "latency_ms": health.get("latency_ms", 0.0),
        "model": {
            "path": props.get("model_path", ""),
            "alias": model_alias,
            "quantization": props.get("model_ftype", ""),
            "modalities": props.get("modalities", {}),
            "context": (props.get("default_generation_settings") or {}).get("n_ctx", 0),
        },
        "slots": slots, "metrics": metrics, "props": props, "is_dynamic": True,
    }


def reload_service(service_name: str, ssh_host: str = "", ssh_user: str = "") -> dict:
    """Restarts a systemd service via SSH and polls for health. Host/user: the engines' host (lodestone/endpoints.py)."""
    ssh_host, ssh_user = ssh_host or endpoints.host("coordinator"), ssh_user or endpoints.ssh_user()
    if not _is_valid_service_name(service_name):
        return {"ok": False, "error": "Invalid service name"}
        
    start_time = time.time()
    cmd = endpoints.ssh_cmd(None, f"sudo systemctl restart {service_name}.service", target=f"{ssh_user}@{ssh_host}")
    
    try:
        subprocess.run(cmd, timeout=35, check=True, capture_output=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SSH command timed out"}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"SSH command failed: {e.stderr.decode()}"}
        
    # Poll for health
    engine_key = None
    registry = engine_registry()
    for k, v in registry.items():
        if v["service"] == service_name and v["host"] == ssh_host:
            engine_key = k
            break
            
    if not engine_key:
        return {"ok": True, "restart_time_ms": (time.time() - start_time) * 1000, "new_state": {}}
        
    config = registry[engine_key]
    host = config["host"]
    port = config["port"]
    
    # Poll up to 30 seconds
    for _ in range(60):
        health = probe_engine_health(host, port, timeout=1.0)
        if health["online"]:
            break
        time.sleep(0.5)
        
    new_state = build_engine_state(engine_key)
    return {
        "ok": True,
        "restart_time_ms": (time.time() - start_time) * 1000,
        "new_state": new_state
    }

def write_systemd_override(service_name: str, overrides: dict, ssh_host: str = "", ssh_user: str = "") -> dict:
    """Writes a systemd override file via SSH."""
    ssh_host, ssh_user = ssh_host or endpoints.host("coordinator"), ssh_user or endpoints.ssh_user()
    # TODO: Implement this stub
    if not _is_valid_service_name(service_name):
        return {"ok": False, "error": "Invalid service name"}
    return {"ok": False, "error": "Not implemented"}

def stream_debug_completion(host: str, port: int, prompt: str, params: dict = None) -> Generator[dict, None, None]:
    """Streams a completion request yielding SSE data."""
    if params is None:
        params = {}
        
    payload = {
        "prompt": prompt,
        "stream": True,
        "timings_per_token": True,
        "post_sampling_probs": True,
        "n_probs": 5,
        "n_predict": 512,
    }
    payload.update(params)
    
    url = f"http://{host}:{port}/completion"
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                if line_str.startswith("data:"):
                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logger.error(f"Stream completion failed: {e}")

def fetch_journal_logs(service_name: str, lines: int = 100, ssh_host: str = "", ssh_user: str = "") -> list:
    """Fetches journalctl logs via SSH."""
    ssh_host, ssh_user = ssh_host or endpoints.host("coordinator"), ssh_user or endpoints.ssh_user()
    if not _is_valid_service_name(service_name):
        return ["Invalid service name"]
        
    cmd = [
        "ssh", "-o", "BatchMode=yes", "--",
        endpoints.check_ssh_target(f"{ssh_user}@{ssh_host}"),
        f"journalctl -u {service_name}.service -n {lines} --no-pager"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except subprocess.TimeoutExpired:
        return ["Error: SSH command timed out"]
    except subprocess.CalledProcessError as e:
        return [f"Error: SSH command failed: {e.stderr}"]
