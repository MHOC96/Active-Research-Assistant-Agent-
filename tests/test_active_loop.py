"""Tests for active literature discovery loop."""

from unittest.mock import MagicMock

import pytest

from research_assistant.models import (
    ArxivPaper,
    DocumentStatus,
    HybridRetrieveResult,
    IngestionResult,
    RetrievalHit,
    SufficiencyResult,
)
from research_assistant.pipeline.active_loop import ActiveLiteraturePipeline


def _retrieval(query: str, sufficient: bool, top_score: float = 0.8) -> HybridRetrieveResult:
    candidates = []
    if sufficient:
        candidates = [
            RetrievalHit(
                chunk_id="2407.08608:0",
                passage="Evidence about transformer attention.",
                document_id="2407.08608",
                arxiv_id="2407.08608",
                title="Paper",
                chunk_index=0,
                rerank_score=top_score,
            )
        ]
    return HybridRetrieveResult(
        query=query,
        candidates=candidates,
        sufficiency=SufficiencyResult(
            sufficient=sufficient,
            candidate_count=len(candidates),
            top_score=top_score if sufficient else 0.4,
            reason=None if sufficient else "top_score below threshold",
        ),
    )


def test_pipeline_returns_immediately_when_sufficient():
    retriever = MagicMock()
    retriever.retrieve.return_value = _retrieval("transformer attention", sufficient=True)

    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=MagicMock(),
        ingestion_worker=MagicMock(),
        metadata=MagicMock(),
    )

    result = pipeline.run("transformer attention")

    assert result.sufficient is True
    assert result.discovery_rounds == []
    retriever.retrieve.assert_called_once()
    pipeline.discovery.search_arxiv.assert_not_called()


def test_pipeline_discovers_ingests_and_reruns():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        _retrieval("graph rag latency", sufficient=False),
        _retrieval("graph rag latency", sufficient=True),
    ]

    discovery = MagicMock()
    discovery.search_arxiv.return_value = [
        ArxivPaper(
            arxiv_id="2407.08608",
            title="Graph RAG Survey",
            abstract="Graph RAG latency tradeoffs.",
            pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
        )
    ]

    ingestion_worker = MagicMock()
    ingestion_worker.ingest_pdf_document.return_value = IngestionResult(
        document_id="2407.08608",
        arxiv_id="2407.08608",
        status=DocumentStatus.INGESTED,
        chunk_count=3,
    )

    metadata = MagicMock()
    metadata.is_ingested.return_value = False

    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=discovery,
        ingestion_worker=ingestion_worker,
        metadata=metadata,
    )

    result = pipeline.run("graph rag latency")

    assert result.sufficient is True
    assert len(result.discovery_rounds) == 1
    assert result.papers_discovered == 1
    assert result.papers_ingested == 1
    ingestion_worker.ingest_pdf_document.assert_called_once()
    assert retriever.retrieve.call_count == 2


def test_pipeline_reports_insufficient_after_max_rounds():
    retriever = MagicMock()
    retriever.retrieve.return_value = _retrieval("unknown topic", sufficient=False, top_score=0.2)

    discovery = MagicMock()
    discovery.search_arxiv.return_value = []

    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=discovery,
        ingestion_worker=MagicMock(),
        metadata=MagicMock(),
    )

    from research_assistant.config import Settings

    pipeline.settings = Settings(MAX_DISCOVERY_ROUNDS=2)

    result = pipeline.run("unknown topic")

    assert result.sufficient is False
    assert len(result.discovery_rounds) == 2
    assert result.insufficient_message.startswith("INSUFFICIENT_EVIDENCE:")
