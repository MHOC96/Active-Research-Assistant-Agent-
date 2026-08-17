"""Extract focused bibliographic search queries from pasted prose."""

from __future__ import annotations

import re

from research_assistant.utils.token_efficiency import cap_subqueries

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "by",
        "for",
        "with",
        "without",
        "from",
        "into",
        "to",
        "of",
        "in",
        "on",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "that",
        "this",
        "these",
        "those",
        "their",
        "they",
        "them",
        "it",
        "its",
        "can",
        "may",
        "will",
        "would",
        "should",
        "could",
        "such",
        "than",
        "then",
        "when",
        "where",
        "which",
        "while",
        "who",
        "what",
        "how",
        "why",
        "all",
        "any",
        "each",
        "both",
        "more",
        "most",
        "other",
        "some",
        "only",
        "also",
        "not",
        "over",
        "under",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "about",
        "against",
        "across",
        "among",
        "within",
        "ensure",
        "ensuring",
        "enable",
        "enables",
        "require",
        "requiring",
        "complete",
        "rapid",
        "lightweight",
        "automated",
        "manual",
        "underlying",
        "disparate",
        "compared",
        "traditional",
        "manage",
        "handle",
        "rely",
        "rely",
        "behavior",
        "mechanisms",
        "instances",
        "layers",
        "scale",
        "across",
        "environments",
    }
)

_PRIORITY_PHRASES = (
    "cloud computing",
    "container orchestration",
    "horizontal scaling",
    "rolling updates",
    "service discovery",
    "self-healing",
    "resource utilization",
    "runtime dependencies",
    "operating system",
    "container engines",
    "containerization",
    "virtualization",
    "kubernetes",
    "docker",
    "microservices",
    "devops",
    "ephemeral",
    "cluster administration",
    "entrepreneur manager transition",
    "founder manager",
    "entrepreneurial leadership",
    "small business management",
    "managerial skills",
    "owner operator",
)

_GENERIC_QUERY_TERMS = frozenset(
    {
        "management",
        "manager",
        "managers",
        "managerial",
        "business",
        "owners",
        "owner",
        "simple",
        "task",
        "experience",
        "knowledge",
        "problems",
        "problem",
        "organizational",
        "structure",
        "frustrated",
        "become",
        "until",
        "because",
        "needs",
        "need",
        "feel",
        "deal",
        "they",
        "will",
        "like",
        "when",
        "entrepreneurs",
        "entrepreneur",
    }
)

_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bentrepreneur\w*\b.*\bmanager\w*\b", re.IGNORECASE),
        "entrepreneur manager transition",
    ),
    (
        re.compile(r"\bfounder\w*\b.*\bmanager\w*\b", re.IGNORECASE),
        "founder manager transition",
    ),
    (
        re.compile(r"\bmanagerial\s+skills?\b", re.IGNORECASE),
        "managerial skills development",
    ),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_paste_to_cite(text: str) -> bool:
    """Return True when the user pasted prose to find citations for, not a question."""
    stripped = text.strip()
    if len(stripped) < 120:
        return False
    if stripped.endswith("?"):
        return False
    if re.match(r"^(what|how|why|when|where|who|which|can|could|should|is|are|do|does)\b", stripped, re.I):
        if stripped.endswith("?"):
            return False
    sentences = _split_sentences(stripped)
    if len(sentences) >= 2:
        return True
    return len(stripped) >= 280 and not stripped.endswith("?")


def heuristic_citation_queries(text: str, max_queries: int = 4) -> list[str]:
    """Build short search queries from each sentence or claim in pasted prose."""
    sentences = _split_sentences(text.strip())
    queries: list[str] = []
    seen: set[str] = set()

    for sentence in sentences:
        query = _sentence_to_search_query(sentence)
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)

    if not queries:
        queries = [_sentence_to_search_query(text)]

    fallback = queries[0] if queries else text[:120]
    return cap_subqueries(queries, max_queries, fallback)


def summarize_paste_topic(text: str, subqueries: list[str]) -> str:
    """Short label for a pasted paragraph."""
    if subqueries:
        return subqueries[0]
    return text[:120].strip()


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return parts or [text.strip()]


def sentence_spans_in_text(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence, start, end) offsets for each sentence in the original text."""
    stripped = text.strip()
    if not stripped:
        return []

    lead = text.find(stripped)
    if lead < 0:
        lead = 0

    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for part in _SENTENCE_SPLIT.split(stripped):
        sentence = part.strip()
        if not sentence:
            continue

        rel_start = stripped.find(sentence, cursor)
        if rel_start < 0:
            rel_start = stripped.lower().find(sentence.lower(), cursor)
        if rel_start < 0:
            continue

        rel_end = rel_start + len(sentence)
        spans.append((sentence, lead + rel_start, lead + rel_end))
        cursor = rel_end

    return spans


def _sentence_to_search_query(sentence: str) -> str:
    lower = sentence.lower()
    selected: list[str] = []

    for pattern, phrase in _CLAIM_PATTERNS:
        if pattern.search(sentence) and phrase not in selected:
            selected.append(phrase)

    for phrase in _PRIORITY_PHRASES:
        if phrase in lower and phrase not in selected:
            selected.append(phrase)

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", sentence)
    ranked: list[str] = []
    for word in words:
        token = word.lower()
        if len(token) <= 3 or token in _STOPWORDS or token in _GENERIC_QUERY_TERMS:
            continue
        if token in selected or any(token in phrase for phrase in selected):
            continue
        ranked.append(token)

    ranked.sort(key=len, reverse=True)
    for token in ranked:
        if len(selected) >= 6:
            break
        if token not in selected:
            selected.append(token)

    return " ".join(selected[:6])
