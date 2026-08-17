"""Tests for cooperative request cancellation."""

from unittest.mock import MagicMock

import pytest

from research_assistant.models import (
    ActiveResearchResult,
    HybridRetrieveResult,
    SufficiencyResult,
)
from research_assistant.orchestrator.agent import ResearchOrchestrator
from research_assistant.orchestrator.query_processor import QueryAnalysis
from research_assistant.pipeline.active_loop import ActiveLiteraturePipeline
from research_assistant.utils.cancellation import (
    CancellationRegistry,
    CancellationToken,
    RequestCancelledError,
    cancellation_registry,
)


def test_token_raises_when_cancelled():
    token = CancellationToken("req-1")
    token.cancel()
    with pytest.raises(RequestCancelledError) as exc_info:
        token.raise_if_cancelled("retrieval")
    assert exc_info.value.request_id == "req-1"
    assert exc_info.value.stage == "retrieval"


def test_registry_register_cancel_unregister():
    registry = CancellationRegistry()
    token = registry.register("req-abc")
    assert registry.cancel("req-abc") is True
    assert token.is_cancelled is True
    registry.unregister("req-abc")
    assert registry.cancel("req-abc") is False


def test_orchestrator_stops_before_subquery_when_cancelled():
    pipeline = MagicMock()
    pipeline.settings.min_candidates = 1
    pipeline.settings.min_rerank_score = 0.7
    pipeline.settings.min_external_relevance_score = 0.35
    pipeline.settings.min_indexed_topic_score = 0.30
    pipeline.settings.citation_style = "internal"
    pipeline.settings.final_top_k = 3

    query_processor = MagicMock()
    query_processor.analyze.return_value = QueryAnalysis(
        original_query="Compare RAG and GraphRAG",
        normalized_query="compare RAG and GraphRAG",
        query_type="complex",
        subqueries=["RAG accuracy", "GraphRAG accuracy"],
    )

    token = CancellationToken("req-cancel")
    token.cancel()

    orchestrator = ResearchOrchestrator(
        pipeline=pipeline,
        llm=MagicMock(),
        query_processor=query_processor,
    )

    with pytest.raises(RequestCancelledError):
        orchestrator.answer("Compare RAG and GraphRAG", cancellation=token)

    pipeline.run.assert_not_called()


def test_active_loop_stops_before_discovery_when_cancelled():
    retriever = MagicMock()
    retriever.retrieve.return_value = HybridRetrieveResult(
        query="unknown topic",
        candidates=[],
        sufficiency=SufficiencyResult(
            sufficient=False,
            candidate_count=0,
            top_score=0.2,
            reason="top_score below threshold",
        ),
    )

    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=MagicMock(),
        ingestion_worker=MagicMock(),
        metadata=MagicMock(),
    )

    token = CancellationToken("req-loop")
    token.cancel()

    with pytest.raises(RequestCancelledError):
        pipeline.run("unknown topic", cancellation=token)

    pipeline.discovery.search_by_source.assert_not_called()


def test_active_loop_stops_before_ingestion_when_cancelled_mid_round():
    from research_assistant.models import ArxivPaper, DiscoveredPaper

    def _arxiv_discovered(paper: ArxivPaper) -> DiscoveredPaper:
        return DiscoveredPaper(
            source="arxiv",
            external_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            published_date=paper.published_date,
            pdf_url=paper.pdf_url,
            landing_url=f"https://arxiv.org/abs/{paper.arxiv_id}",
            arxiv_id=paper.arxiv_id,
            categories=paper.categories,
        )

    retriever = MagicMock()
    retriever.retrieve.return_value = HybridRetrieveResult(
        query="graph rag",
        candidates=[],
        sufficiency=SufficiencyResult(
            sufficient=False,
            candidate_count=0,
            top_score=0.2,
        ),
    )

    discovery = MagicMock()
    arxiv_paper = ArxivPaper(
        arxiv_id="2407.08608",
        title="Paper A",
        abstract="A",
        pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
    )
    discovery.search_by_source.return_value = {"arxiv": [_arxiv_discovered(arxiv_paper)]}

    ingestion_worker = MagicMock()
    metadata = MagicMock()
    metadata.is_ingested.return_value = False

    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=discovery,
        ingestion_worker=ingestion_worker,
        metadata=metadata,
    )

    token = CancellationToken("req-ingest")
    token.cancel()

    with pytest.raises(RequestCancelledError):
        pipeline.run("graph rag", cancellation=token)

    ingestion_worker.ingest_pdf_document.assert_not_called()


def test_global_registry_is_singleton():
    token = cancellation_registry.register("singleton-test")
    assert cancellation_registry.get("singleton-test") is token
    cancellation_registry.unregister("singleton-test")
