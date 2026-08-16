"""Tests for query processor."""

import json

from research_assistant.config import Settings
from research_assistant.orchestrator.query_processor import QueryProcessor


class MockLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.response


def test_analyze_simple_query():
    payload = json.dumps(
        {
            "normalized_query": "transformer attention mechanism",
            "query_type": "simple",
            "subqueries": ["transformer attention mechanism"],
        }
    )
    processor = QueryProcessor(
        MockLLM(payload),
        settings=Settings(skip_query_llm_for_simple=False),
    )

    analysis = processor.analyze("Can you explain the transformer attention mechanism?")

    assert analysis.normalized_query == "transformer attention mechanism"
    assert analysis.query_type == "simple"
    assert analysis.subqueries == ["transformer attention mechanism"]


def test_analyze_complex_query():
    payload = json.dumps(
        {
            "normalized_query": "compare RAG and GraphRAG accuracy and latency",
            "query_type": "complex",
            "subqueries": ["RAG accuracy", "GraphRAG accuracy", "RAG latency", "GraphRAG latency"],
        }
    )
    processor = QueryProcessor(MockLLM(payload))

    analysis = processor.analyze("Compare RAG and GraphRAG on accuracy and latency")

    assert analysis.query_type == "complex"
    assert len(analysis.subqueries) == 4
