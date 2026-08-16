"""Discovery source registry and labels."""

from __future__ import annotations

SOURCE_LABELS: dict[str, str] = {
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "web": "Web",
}

DEFAULT_DISCOVERY_SOURCES: tuple[str, ...] = (
    "arxiv",
    "openalex",
    "semantic_scholar",
    "web",
)


def parse_discovery_sources(value: str) -> list[str]:
    sources = [token.strip().lower() for token in value.split(",") if token.strip()]
    valid = [source for source in sources if source in SOURCE_LABELS]
    return valid or list(DEFAULT_DISCOVERY_SOURCES)


def source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())
