"""Tests for orchestrator web-only citation fallback."""

from unittest.mock import MagicMock

from research_assistant.citations.styles import CitationStyle
from research_assistant.models import (
    ActiveResearchResult,
    ExternalCitation,
    HybridRetrieveResult,
    SufficiencyResult,
)
from research_assistant.orchestrator.agent import ResearchOrchestrator
from research_assistant.orchestrator.query_processor import QueryAnalysis


def test_orchestrator_returns_web_citations_when_local_index_empty():
    pipeline = MagicMock()
    pipeline.settings.min_candidates = 1
    pipeline.settings.min_rerank_score = 0.7
    pipeline.settings.min_external_relevance_score = 0.35
    pipeline.settings.min_indexed_topic_score = 0.30
    pipeline.settings.citation_style = "apa7"
    pipeline.settings.final_top_k = 3
    pipeline.settings.discovery_source_list = ["web"]

    web_citation = ExternalCitation(
        source="web",
        source_label="Web",
        title="ServiceNow ITSM Platform Overview",
        publisher="ServiceNow",
        published_date="2023",
        url="https://www.servicenow.com/products/itsm.html",
        relevance_score=0.9,
    )
    pipeline.run.return_value = ActiveResearchResult(
        query="ServiceNow ITSM multi-instance architecture",
        request_id="req-web",
        retrieval=HybridRetrieveResult(
            query="ServiceNow ITSM multi-instance architecture",
            candidates=[],
            sufficiency=SufficiencyResult(
                sufficient=False,
                candidate_count=0,
                top_score=0.1,
                reason="top_score below threshold",
            ),
        ),
        external_citations=[web_citation],
        insufficient_message="INSUFFICIENT_EVIDENCE: top_score below threshold",
    )

    query_processor = MagicMock()
    query_processor.analyze.return_value = QueryAnalysis(
        original_query="ServiceNow ITSM multi-instance architecture",
        normalized_query="ServiceNow ITSM multi-instance architecture",
        query_type="simple",
        subqueries=["ServiceNow ITSM multi-instance architecture"],
    )

    orchestrator = ResearchOrchestrator(
        pipeline=pipeline,
        llm=MagicMock(),
        query_processor=query_processor,
    )

    response = orchestrator.answer(
        "ServiceNow ITSM multi-instance architecture",
        citation_style=CitationStyle.APA7,
    )

    assert response.answer.startswith("References")
    assert "From Web" in response.answer
    assert "(ServiceNow, 2023)" in response.answer
    assert response.sufficient is False
