"""Tests for grounded synthesis."""

from research_assistant.models import RetrievalHit, SufficiencyResult
from research_assistant.orchestrator.synthesis import GroundedSynthesizer


class MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="2407.08608:0",
        passage="Transformers use multi-head self-attention.",
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Attention Paper",
        chunk_index=0,
        rerank_score=0.91,
    )


def test_synthesize_returns_insufficient_when_not_sufficient():
    synthesizer = GroundedSynthesizer(MockLLM([]))
    answer, valid, errors = synthesizer.synthesize(
        "transformer attention",
        [],
        sufficiency=SufficiencyResult(
            sufficient=False,
            candidate_count=0,
            top_score=None,
            reason="candidate_count (0) < MIN_CANDIDATES (1)",
        ),
    )
    assert answer.startswith("INSUFFICIENT_EVIDENCE:")
    assert valid is True
    assert errors == []


def test_synthesize_validates_citations():
    llm = MockLLM(
        [
            "Attention enables parallel sequence modeling [arXiv:9999.99999 | Chunk 99].",
            "Attention enables parallel sequence modeling [arXiv:2407.08608 | Chunk 0].",
        ]
    )
    synthesizer = GroundedSynthesizer(llm, max_regenerations=1)
    answer, valid, errors = synthesizer.synthesize(
        "transformer attention",
        [_hit()],
        sufficiency=SufficiencyResult(sufficient=True, candidate_count=1, top_score=0.91),
    )
    assert valid is True
    assert "[arXiv:2407.08608 | Chunk 0]" in answer
    assert errors == []


def test_synthesize_reports_invalid_citations_after_retry():
    llm = MockLLM(
        [
            "Unsupported [arXiv:9999.99999 | Chunk 99].",
            "Still unsupported [arXiv:9999.99999 | Chunk 99].",
        ]
    )
    synthesizer = GroundedSynthesizer(llm, max_regenerations=1)
    answer, valid, errors = synthesizer.synthesize(
        "transformer attention",
        [_hit()],
        sufficiency=SufficiencyResult(sufficient=True, candidate_count=1, top_score=0.91),
    )
    assert valid is False
    assert len(errors) == 1
