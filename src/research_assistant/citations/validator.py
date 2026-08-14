"""Citation provenance and validation."""

from __future__ import annotations

import re

from research_assistant.models import RetrievalHit

CITATION_PATTERN = re.compile(
    r"\[arXiv:([0-9]{4}\.[0-9]{4,5})\s*\|\s*Chunk\s*(\d+)\]",
    re.IGNORECASE,
)


def format_provenance(arxiv_id: str, chunk_index: int) -> str:
    return f"[arXiv:{arxiv_id} | Chunk {chunk_index}]"


def build_citation_map(hits: list[RetrievalHit]) -> dict[str, RetrievalHit]:
    """Map internal provenance strings to retrieval hits."""
    return {hit.provenance: hit for hit in hits}


def extract_citations(text: str) -> list[tuple[str, int]]:
    """Extract (arxiv_id, chunk_index) pairs from text."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_PATTERN.finditer(text)]


def validate_citations(text: str, hits: list[RetrievalHit]) -> tuple[bool, list[str]]:
    """Verify all citations in text reference actual retrieved chunks."""
    valid_keys = {(h.arxiv_id, h.chunk_index) for h in hits}
    errors: list[str] = []

    for arxiv_id, chunk_index in extract_citations(text):
        if (arxiv_id, chunk_index) not in valid_keys:
            errors.append(
                f"Invalid citation: [arXiv:{arxiv_id} | Chunk {chunk_index}] "
                "not in retrieval context"
            )

    return len(errors) == 0, errors
