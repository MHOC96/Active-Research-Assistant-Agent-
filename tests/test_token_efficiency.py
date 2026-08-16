"""Tests for token efficiency helpers and settings."""

import json

from research_assistant.config import Settings
from research_assistant.orchestrator.query_processor import QueryProcessor
from research_assistant.orchestrator.synthesis import _build_prompt
from research_assistant.utils.token_efficiency import (
    cap_subqueries,
    heuristic_normalize_query,
    is_likely_simple_query,
    truncate_passage,
)


def test_heuristic_normalize_strips_conversational_prefix():
    assert heuristic_normalize_query("Can you explain transformer attention?") == "transformer attention"


def test_is_likely_simple_query_detects_comparisons():
    assert is_likely_simple_query("What is transformer attention?") is True
    assert is_likely_simple_query("Compare RAG and GraphRAG on accuracy") is False


def test_cap_subqueries_limits_breadth():
    subqueries = ["a", "b", "c", "d", "e", "f"]
    assert cap_subqueries(subqueries, 4, "fallback") == ["a", "b", "c", "d"]


def test_truncate_passage_adds_ellipsis():
    text = "A" * 100
    result = truncate_passage(text, 40)
    assert result.endswith("[...]")
    assert len(result) < len(text)


def test_simple_query_skips_llm_for_simple_queries():
    class FailingLLM:
        def complete(self, **_kwargs):
            raise AssertionError("LLM should not be called for simple queries")

    settings = Settings(skip_query_llm_for_simple=True)
    processor = QueryProcessor(FailingLLM(), settings=settings)
    analysis = processor.analyze("What is transformer self-attention?")
    assert analysis.query_type == "simple"
    assert analysis.subqueries == [analysis.normalized_query]


def test_query_analysis_is_cached():
    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **_kwargs) -> str:
            self.calls += 1
            return json.dumps(
                {
                    "normalized_query": "graph rag latency",
                    "query_type": "simple",
                    "subqueries": ["graph rag latency"],
                }
            )

    llm = CountingLLM()
    settings = Settings(query_analysis_cache_size=8, skip_query_llm_for_simple=False)
    processor = QueryProcessor(llm, settings=settings)

    first = processor.analyze("Compare RAG and GraphRAG latency")
    second = processor.analyze("Compare RAG and GraphRAG latency")

    assert first.normalized_query == second.normalized_query
    assert llm.calls == 1


def test_synthesis_prompt_truncates_long_passages():
    from research_assistant.models import RetrievalHit

    hit = RetrievalHit(
        chunk_id="2407.08608:0",
        passage="x" * 5000,
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Paper",
        chunk_index=0,
        rerank_score=0.9,
    )
    settings = Settings(synthesis_max_passage_chars=500)
    prompt = _build_prompt("latency", [hit], None, settings)
    assert "[...]" in prompt
    assert "x" * 5000 not in prompt
