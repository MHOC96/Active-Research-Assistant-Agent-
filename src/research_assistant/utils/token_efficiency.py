"""Helpers for reducing LLM prompt size and skipping unnecessary model calls."""

from __future__ import annotations

import re

_COMPLEX_QUERY_PATTERN = re.compile(
    r"\b("
    r"compare|comparison|versus|vs\.?|difference|differences|contrast|"
    r"each of|pros and cons|advantages and disadvantages|"
    r"relationship between|similarities and differences|"
    r"how do .+ differ|in terms of"
    r")\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_PREFIX = re.compile(
    r"^(?:"
    r"can you|could you|would you|please|"
    r"i want to know|i need to know|tell me|help me|"
    r"explain|describe|what is|what are|how does|how do|why does|why do"
    r")\s+",
    re.IGNORECASE,
)


def heuristic_normalize_query(query: str) -> str:
    """Strip conversational filler without calling an LLM."""
    text = " ".join(query.strip().split())
    changed = True
    while changed:
        changed = False
        stripped = _CONVERSATIONAL_PREFIX.sub("", text).strip()
        if stripped != text:
            text = stripped
            changed = True
    text = text.rstrip("?.!").strip()
    return text or query.strip()


def is_likely_simple_query(query: str) -> bool:
    """Heuristic for queries that do not need LLM decomposition."""
    text = query.strip()
    if not text:
        return False
    if len(text) > 400:
        return False
    if text.count("?") > 1:
        return False
    if text.count("\n") > 2:
        return False
    if re.search(r"(?:^|\n)\s*\d+[\).\]]\s+", text):
        return False
    if _COMPLEX_QUERY_PATTERN.search(text):
        return False
    return True


def cap_subqueries(subqueries: list[str], max_subqueries: int, fallback: str) -> list[str]:
    """Limit decomposition breadth to control retrieval cost."""
    cleaned = [item.strip() for item in subqueries if item.strip()]
    if not cleaned:
        return [fallback]
    if len(cleaned) <= max_subqueries:
        return cleaned
    return cleaned[:max_subqueries]


def truncate_passage(text: str, max_chars: int) -> str:
    """Truncate long passages while preserving readable boundaries."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()
    for separator in (". ", "; ", ", ", " "):
        boundary = clipped.rfind(separator)
        if boundary >= int(max_chars * 0.6):
            return clipped[: boundary + len(separator)].rstrip() + " [...]"

    return clipped + " [...]"
