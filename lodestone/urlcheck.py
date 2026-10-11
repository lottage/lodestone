"""Checks for a URL a client asks this server to fetch (server-side request forgery).

A route that takes a URL from the browser and fetches it from the server's own network position can be pointed at
things the browser could never reach: `file:///etc/passwd`, `http://127.0.0.1:<admin port>`, a cloud metadata address.
`check_fetch_url` refuses the schemes and the places that have no business being fetched this way; a normal LAN address
(10.x, 172.16-31.x, 192.168.x) is allowed, because a home server legitimately fetches from its own network.
Layer 0: stdlib only, imports nothing from the platform."""

import ipaddress
import socket
import urllib.parse
from typing import Iterable, Optional

METADATA_NAMES = {"metadata.google.internal", "metadata", "instance-data"}


def _blocked_address(ip, allow_private: bool) -> Optional[str]:
    ip = getattr(ip, "ipv4_mapped", None) or ip
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local (cloud metadata lives here)"
    if ip.is_unspecified or ip.is_multicast or ip.is_reserved:
        return "not a normal host address"
    if ip.is_private and not allow_private:
        return "private network"
    return None


def check_fetch_url(url: str, schemes: Iterable[str] = ("http", "https"), allow_private: bool = True,
                    resolve: bool = True) -> str:
    """The URL back when it is fine to fetch; ValueError (with the reason) when it is not. Names are resolved, and every
    address they resolve to is checked (a name that points at 127.0.0.1 is as bad as 127.0.0.1)."""
    raw = str(url or "").strip()
    if not raw or any(c in raw for c in "\r\n\0 "):
        raise ValueError("not a usable URL")
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme.lower() not in tuple(schemes):
        raise ValueError(f"only {'/'.join(schemes)} URLs are fetched (not '{parts.scheme or 'none'}')")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("the URL has no host")
    if host in METADATA_NAMES or host.endswith(".internal"):
        raise ValueError("that host name is not fetched from here")
    try:
        addrs = [ipaddress.ip_address(host.split("%", 1)[0])]
    except ValueError:
        addrs = []
        if resolve:
            try:
                for info in socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM):
                    addrs.append(ipaddress.ip_address(info[4][0].split("%", 1)[0]))
            except (OSError, ValueError) as e:
                raise ValueError(f"cannot resolve '{host}' ({e})") from e
        if host == "localhost" or host.endswith(".localhost"):
            raise ValueError("loopback is not fetched from here")
    for ip in addrs:
        why = _blocked_address(ip, allow_private)
        if why:
            raise ValueError(f"{ip} is {why}")
    return raw
