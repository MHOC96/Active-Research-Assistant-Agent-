"""Tests for research orchestrator."""

from unittest.mock import MagicMock

from research_assistant.citations.styles import CitationStyle
from research_assistant.models import (
    ActiveResearchResult,
    HybridRetrieveResult,
    ResearchResponse,
    RetrievalHit,
    SufficiencyResult,
)
from research_assistant.orchestrator.agent import ResearchOrchestrator
from research_assistant.orchestrator.query_processor import QueryAnalysis


def _active_result(query: str, sufficient: bool) -> ActiveResearchResult:
    candidates = []
    if sufficient:
        candidates = [
            RetrievalHit(
                chunk_id="2407.08608:0",
                passage="Graph RAG accuracy results on benchmark tasks.",
                document_id="2407.08608",
                arxiv_id="2407.08608",
                title="RAG vs GraphRAG accuracy study",
                chunk_index=0,
                rerank_score=0.88,
            )
        ]
    return ActiveResearchResult(
        query=query,
        request_id="req-1",
        retrieval=HybridRetrieveResult(
            query=query,
            candidates=candidates,
            sufficiency=SufficiencyResult(
                sufficient=sufficient,
                candidate_count=len(candidates),
                top_score=0.88 if sufficient else 0.3,
            ),
        ),
        insufficient_message=None if sufficient else "INSUFFICIENT_EVIDENCE: low score",
    )


def test_orchestrator_runs_subqueries_and_returns_references():
    pipeline = MagicMock()
    pipeline.settings.min_candidates = 1
    pipeline.settings.min_rerank_score = 0.7
    pipeline.settings.min_external_relevance_score = 0.35
    pipeline.settings.min_indexed_topic_score = 0.30
    pipeline.settings.citation_style = "mla9"
    pipeline.settings.final_top_k = 3
    pipeline.settings.discovery_source_list = ["arxiv", "openalex", "semantic_scholar", "web"]
    pipeline.run.side_effect = [
        _active_result("RAG accuracy", True),
        _active_result("GraphRAG accuracy", True),
    ]

    query_processor = MagicMock()
    query_processor.analyze.return_value = QueryAnalysis(
        original_query="Compare RAG and GraphRAG accuracy",
        normalized_query="compare RAG and GraphRAG accuracy",
        query_type="complex",
        subqueries=["RAG accuracy", "GraphRAG accuracy"],
    )

    orchestrator = ResearchOrchestrator(
        pipeline=pipeline,
        llm=MagicMock(),
        query_processor=query_processor,
    )

    response = orchestrator.answer(
        "Compare RAG and GraphRAG accuracy",
        citation_style=CitationStyle.MLA9,
    )

    assert isinstance(response, ResearchResponse)
    assert pipeline.run.call_count == 2
    assert response.citations_valid is True
    assert response.answer.startswith("References")
    assert len(response.evidence_hits) == 1


def test_orchestrator_marks_insufficient_when_subquery_fails():
    pipeline = MagicMock()
    pipeline.settings.min_candidates = 1
    pipeline.settings.min_rerank_score = 0.7
    pipeline.settings.min_external_relevance_score = 0.35
    pipeline.settings.min_indexed_topic_score = 0.30
    pipeline.settings.citation_style = "mla9"
    pipeline.settings.final_top_k = 3
    pipeline.settings.discovery_source_list = ["arxiv", "openalex", "semantic_scholar", "web"]
    pipeline.run.return_value = _active_result("unknown topic", False)

    query_processor = MagicMock()
    query_processor.analyze.return_value = QueryAnalysis(
        original_query="unknown topic",
        normalized_query="unknown topic",
        query_type="simple",
        subqueries=["unknown topic"],
    )

    orchestrator = ResearchOrchestrator(
        pipeline=pipeline,
        llm=MagicMock(),
        query_processor=query_processor,
    )

    response = orchestrator.answer("unknown topic")

    assert response.sufficient is False
    assert response.insufficient_message.startswith("INSUFFICIENT_EVIDENCE:")
