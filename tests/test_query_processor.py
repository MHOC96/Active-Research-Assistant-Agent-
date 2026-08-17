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


def test_analyze_pasted_paragraph_uses_citation_queries():
    payload = json.dumps(
        {
            "normalized_query": "cloud containerization orchestration",
            "query_type": "complex",
            "subqueries": [
                "cloud computing containerization deployment",
                "container engines virtualization resource utilization",
                "kubernetes orchestration rolling updates",
            ],
        }
    )
    llm = MockLLM(payload)
    processor = QueryProcessor(llm, settings=Settings(skip_query_llm_for_simple=False))

    paragraph = (
        "Cloud computing architectures rely on containerization to package applications "
        "with their complete runtime dependencies. Container engines enable horizontal "
        "scaling compared to virtualization. Orchestration frameworks handle rolling updates."
    )
    analysis = processor.analyze(paragraph)

    assert analysis.query_type == "complex"
    assert len(analysis.subqueries) == 3
    assert llm.calls[0]["system"].startswith("Return ONLY JSON")
    assert len(analysis.subqueries[0]) < len(paragraph)


def test_analyze_pasted_paragraph_falls_back_to_heuristics():
    class FailingLLM:
        def complete(self, **_kwargs):
            raise RuntimeError("offline")

    processor = QueryProcessor(FailingLLM(), settings=Settings(skip_query_llm_for_simple=False))
    paragraph = (
        "Cloud computing architectures rely on containerization to package applications "
        "with their complete runtime dependencies. Container engines enable horizontal "
        "scaling compared to virtualization. Orchestration frameworks handle rolling updates."
    )
    analysis = processor.analyze(paragraph)

    assert analysis.query_type == "complex"
    assert len(analysis.subqueries) >= 2
    assert all(len(q) < len(paragraph) for q in analysis.subqueries)
