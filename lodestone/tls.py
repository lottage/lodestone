"""TLS for the https services this server calls (Proxmox, Immich, a service on the health list...): verified by default.

A home lab often runs these with a self-signed certificate. The old clients switched certificate checking off for
everything, so anybody on the network could pose as the Proxmox API and read the API token sent to it. Now:

  - the default is a verified connection (certificate chain and host name, against the system's trusted roots);
  - `ca_file` in the service's config section names a PEM file with the certificate (or CA) to trust: the connection is still
    verified, against that file alone, which is the right answer for a self-signed certificate;
  - `verify_tls: false` switches the check off for that service, explicitly, with one warning in the log at the first use.

    config.json:  "proxmox": {"ca_file": "/etc/stonesage/pve-ca.pem"}      or      "proxmox": {"verify_tls": false}

The setting can be given on the service's section (`proxmox`, `immich`) and, for a Proxmox node or a health-list entry, on that
entry too (the entry wins). A `ca_file` that cannot be read is an error (TlsConfigError), never a quiet fall back to "no
checking"; callers show it as a connection failure. Standard library only."""

import logging
import os
import ssl
from typing import Any, Dict, Mapping, Optional, Set

logger = logging.getLogger("stonesage.tls")

_warned: Set[str] = set()


class TlsConfigError(ValueError):
    """The TLS settings cannot be used (for example `ca_file` points at nothing). The message says what to fix."""


def _truthy(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("0", "false", "no", "off"):
            return False
        if v in ("1", "true", "yes", "on"):
            return True
    return default            # anything odd keeps the safe default: checking stays on


def settings(*sections: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """{'verify': bool, 'ca_file': str} from config sections, the LAST one that sets a key winning (service, then entry)."""
    verify, ca = True, ""
    for sec in sections:
        if not isinstance(sec, Mapping):
            continue
        if "verify_tls" in sec:
            verify = _truthy(sec.get("verify_tls"), True)
        if str(sec.get("ca_file") or "").strip():
            ca = str(sec["ca_file"]).strip()
    return {"verify": verify, "ca_file": ca}


def context(*sections: Optional[Mapping[str, Any]], name: str = "service") -> ssl.SSLContext:
    """An SSL context for a client request, from the config section(s) of the service (see the module doc).
    `name` is only for the log line. Raises TlsConfigError for an unusable `ca_file`."""
    s = settings(*sections)
    if s["ca_file"]:
        path = os.path.expanduser(s["ca_file"])
        try:
            ctx = ssl.create_default_context(cafile=path)          # only this file is trusted, host name still checked
        except (OSError, ssl.SSLError) as e:
            raise TlsConfigError(f"{name}: ca_file {s['ca_file']!r} cannot be used ({e.__class__.__name__}: {e}). "
                                 f"Point it at the service's PEM certificate, or set verify_tls to false.") from e
        if not s["verify"]:
            logger.debug("%s: ca_file given and verify_tls false: verification stays off", name)
        else:
            return ctx
    ctx = ssl.create_default_context()
    if not s["verify"]:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        if name not in _warned:
            _warned.add(name)
            logger.warning("TLS certificate checking is OFF for %s (verify_tls is false): anyone on the network can pose as it. "
                           "Prefer ca_file with its certificate.", name)
    return ctx


def explain(error: BaseException, name: str = "the service") -> str:
    """The error text, with the fix added when the failure is a certificate the client does not trust."""
    text = str(error)
    reason = getattr(error, "reason", error)
    if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in text:
        return (f"{text} -- the certificate of {name} is not trusted. Set ca_file in its config section to its PEM certificate "
                f"(best), or verify_tls to false (accepts anybody who answers).")
    return text


def audit(config: Any, _path: str = "") -> list:
    """Where config.json switches certificate checking off: ['proxmox', 'proxmox.pve_node', 'services[2]'...]. The server logs these
    once at start, so a forgotten `verify_tls: false` is not invisible."""
    found = []
    if isinstance(config, Mapping):
        if "verify_tls" in config and not _truthy(config.get("verify_tls"), True):
            found.append(_path or "(top level)")
        for k, v in config.items():
            if isinstance(v, (Mapping, list)):
                found.extend(audit(v, f"{_path}.{k}" if _path else str(k)))
    elif isinstance(config, list):
        for i, v in enumerate(config):
            if isinstance(v, (Mapping, list)):
                found.extend(audit(v, f"{_path}[{i}]"))
    return found
