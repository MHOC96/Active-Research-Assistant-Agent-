"""URL validation for secure PDF downloads."""

from __future__ import annotations

from urllib.parse import urlparse


class UrlValidationError(ValueError):
    """Raised when a URL fails security policy checks."""


def validate_https_url(url: str, allowed_domains: frozenset[str]) -> None:
    """Ensure URL uses HTTPS and resolves to an allowed domain."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise UrlValidationError("HTTPS-only sources are required")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlValidationError("URL must include a hostname")

    if not _host_allowed(host, allowed_domains):
        raise UrlValidationError(f"domain '{host}' is not in the allowed download policy")


def _host_allowed(host: str, allowed_domains: frozenset[str]) -> bool:
    for domain in allowed_domains:
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False
