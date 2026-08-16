"""Publisher and date helpers for web discovery results."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

SKIP_DOMAIN_LABELS = frozenset(
    {
        "www",
        "docs",
        "support",
        "learn",
        "help",
        "developer",
        "developers",
        "blog",
        "en",
        "www2",
    }
)

KNOWN_PUBLISHERS = {
    "servicenow": "ServiceNow",
    "microsoft": "Microsoft",
    "salesforce": "Salesforce",
    "google": "Google",
    "amazon": "Amazon",
    "ibm": "IBM",
    "oracle": "Oracle",
}


def unwrap_redirect_url(href: str) -> str:
    """Resolve DuckDuckGo redirect links to the target URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "uddg" in parsed.query:
        values = parse_qs(parsed.query).get("uddg", [])
        if values:
            return unquote(values[0])
    return href


def publisher_from_url(url: str) -> str:
    """Derive a human-readable organization name from a URL hostname."""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if not host:
        return "Unknown"

    parts = [part for part in host.split(".") if part]
    if not parts:
        return "Unknown"

    candidate = parts[0]
    if candidate in SKIP_DOMAIN_LABELS and len(parts) >= 2:
        candidate = parts[-2] if parts[-1] in {"com", "org", "net", "io", "co", "edu", "gov"} else parts[1]

    label = candidate.replace("-", " ").replace("_", " ").strip()
    if not label:
        return "Unknown"
    known = KNOWN_PUBLISHERS.get(label.lower())
    if known:
        return known
    if label.isupper():
        return label
    if label.isalpha() and len(label) <= 4:
        return label.upper()
    return label.title()


def extract_year(*texts: str | None) -> str | None:
    """Return the first 4-digit year found in the given strings."""
    for text in texts:
        if not text:
            continue
        match = YEAR_PATTERN.search(text)
        if match:
            return match.group(0)
    return None
