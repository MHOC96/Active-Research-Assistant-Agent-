"""Evidence sufficiency gate using FlashRank reranking scores only."""

from __future__ import annotations

from research_assistant.models import RetrievalHit, SufficiencyResult


def evaluate_sufficiency(
    candidates: list[RetrievalHit],
    *,
    min_candidates: int = 1,
    min_rerank_score: float = 0.70,
) -> SufficiencyResult:
    """Determine whether retrieved evidence is sufficient for synthesis.

    Uses only FlashRank rerank_score — never BM25, RRF, or ChromaDB distance.
    """
    candidate_count = len(candidates)
    top_score = max((c.rerank_score for c in candidates if c.rerank_score is not None), default=None)

    if candidate_count < min_candidates:
        return SufficiencyResult(
            sufficient=False,
            candidate_count=candidate_count,
            top_score=top_score,
            reason=f"candidate_count ({candidate_count}) < MIN_CANDIDATES ({min_candidates})",
        )

    if top_score is None:
        return SufficiencyResult(
            sufficient=False,
            candidate_count=candidate_count,
            top_score=None,
            reason="no rerank scores available",
        )

    if top_score < min_rerank_score:
        return SufficiencyResult(
            sufficient=False,
            candidate_count=candidate_count,
            top_score=top_score,
            reason=f"top_score ({top_score:.4f}) < MIN_RERANK_SCORE ({min_rerank_score})",
        )

    return SufficiencyResult(
        sufficient=True,
        candidate_count=candidate_count,
        top_score=top_score,
        reason=None,
    )
