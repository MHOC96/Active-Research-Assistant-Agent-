"""Tests for citation validation."""

from research_assistant.citations.validator import extract_citations, validate_citations
from research_assistant.models import RetrievalHit


def _hit(arxiv_id: str = "2407.08608", chunk_index: int = 12) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{arxiv_id}:{chunk_index}",
        passage="evidence",
        document_id=arxiv_id,
        arxiv_id=arxiv_id,
        title="Paper",
        chunk_index=chunk_index,
        rerank_score=0.9,
    )


def test_extract_citations():
    text = "Claim [arXiv:2407.08608 | Chunk 12] and [arXiv:2310.12345 | Chunk 3]."
    citations = extract_citations(text)
    assert ("2407.08608", 12) in citations
    assert ("2310.12345", 3) in citations


def test_validate_valid_citations():
    hits = [_hit("2407.08608", 12)]
    text = "Supported claim [arXiv:2407.08608 | Chunk 12]."
    ok, errors = validate_citations(text, hits)
    assert ok is True
    assert errors == []


def test_validate_rejects_invented_citation():
    hits = [_hit("2407.08608", 12)]
    text = "Unsupported [arXiv:9999.99999 | Chunk 99]."
    ok, errors = validate_citations(text, hits)
    assert ok is False
    assert len(errors) == 1
