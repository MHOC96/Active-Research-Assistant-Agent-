"""Tests for FlashRank reranker."""

from research_assistant.models import RetrievalHit
from research_assistant.reranking.flashrank_reranker import FlashRankReranker


def _hit(chunk_id: str, passage: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        passage=passage,
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Paper",
        chunk_index=int(chunk_id.split(":")[-1]),
    )


def test_flashrank_reranks_by_relevance():
    reranker = FlashRankReranker()
    query = "What is the transformer attention mechanism?"
    candidates = [
        _hit("2407.08608:0", "The transformer uses multi-head self-attention."),
        _hit("2407.08608:1", "Gradient descent on convex functions."),
    ]

    results = reranker.rerank(query, candidates, top_k=2)

    assert len(results) == 2
    assert all(r.rerank_score is not None for r in results)
    assert results[0].passage.startswith("The transformer")


def test_flashrank_empty_candidates():
    reranker = FlashRankReranker()
    assert reranker.rerank("query", []) == []
