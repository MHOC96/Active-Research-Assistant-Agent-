"""Tests for evidence sufficiency gate."""

from research_assistant.models import RetrievalHit
from research_assistant.sufficiency.gate import evaluate_sufficiency


def _hit(chunk_id: str, score: float | None) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        passage="text",
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Paper",
        chunk_index=1,
        rerank_score=score,
    )


def test_sufficient_when_count_and_score_met():
    result = evaluate_sufficiency([_hit("a", 0.85)], min_candidates=1, min_rerank_score=0.70)
    assert result.sufficient is True
    assert result.top_score == 0.85


def test_insufficient_when_no_candidates():
    result = evaluate_sufficiency([], min_candidates=1, min_rerank_score=0.70)
    assert result.sufficient is False
    assert "candidate_count" in (result.reason or "")


def test_insufficient_when_score_below_threshold():
    result = evaluate_sufficiency([_hit("a", 0.55)], min_candidates=1, min_rerank_score=0.70)
    assert result.sufficient is False
    assert "top_score" in (result.reason or "")


def test_insufficient_when_no_rerank_scores():
    result = evaluate_sufficiency([_hit("a", None)], min_candidates=1, min_rerank_score=0.70)
    assert result.sufficient is False
    assert result.reason == "no rerank scores available"
