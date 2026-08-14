"""Tests for arXiv discovery."""

from datetime import datetime
from unittest.mock import MagicMock

import arxiv
import pytest

from research_assistant.discovery.arxiv import (
    ArxivDiscoveryService,
    deduplicate_papers,
    select_papers_for_ingestion,
)
from research_assistant.models import ArxivPaper, DocumentRecord, DocumentStatus
from research_assistant.storage.metadata_store import MetadataStore


def _result(
    entry_id: str,
    title: str,
    summary: str,
    pdf_url: str = "https://arxiv.org/pdf/2407.08608.pdf",
) -> arxiv.Result:
    return arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        published=datetime(2024, 7, 11),
        updated=datetime(2024, 7, 11),
        authors=[arxiv.Result.Author("Author One")],
        categories=["cs.AI"],
        links=[
            arxiv.Result.Link(
                href=pdf_url,
                rel="related",
                content_type="application/pdf",
            )
        ],
    )


def test_search_arxiv_maps_results():
    client = MagicMock()
    client.results.return_value = [
        _result("https://arxiv.org/abs/2407.08608", "Attention Paper", "Transformer attention.")
    ]
    service = ArxivDiscoveryService(client=client)

    papers = service.search_arxiv("transformer attention", max_results=5)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2407.08608"
    assert papers[0].pdf_url.startswith("https://")
    assert papers[0].authors == ["Author One"]


def test_search_arxiv_raises_on_client_failure():
    client = MagicMock()
    client.results.side_effect = RuntimeError("network")
    service = ArxivDiscoveryService(client=client)

    with pytest.raises(RuntimeError, match="ARXIV_SEARCH_FAILED"):
        service.search_arxiv("transformer attention")


def test_deduplicate_papers(metadata_store: MetadataStore):
    metadata_store.upsert_document(
        DocumentRecord(
            document_id="2407.08608",
            arxiv_id="2407.08608",
            title="Existing",
            status=DocumentStatus.INGESTED,
        )
    )
    papers = [
        ArxivPaper(
            arxiv_id="2407.08608",
            title="Existing",
            pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
        ),
        ArxivPaper(
            arxiv_id="2310.12345",
            title="New Paper",
            pdf_url="https://arxiv.org/pdf/2310.12345.pdf",
        ),
        ArxivPaper(
            arxiv_id="2310.12345",
            title="Duplicate in batch",
            pdf_url="https://arxiv.org/pdf/2310.12345.pdf",
        ),
    ]

    result = deduplicate_papers(papers, metadata_store)

    assert len(result) == 1
    assert result[0].arxiv_id == "2310.12345"


def test_select_papers_for_ingestion_ranks_by_relevance():
    papers = [
        ArxivPaper(
            arxiv_id="1",
            title="Unrelated chemistry",
            abstract="Organic synthesis methods.",
            pdf_url="https://arxiv.org/pdf/1.pdf",
        ),
        ArxivPaper(
            arxiv_id="2",
            title="Transformer attention for NLP",
            abstract="We study attention mechanisms.",
            pdf_url="https://arxiv.org/pdf/2.pdf",
        ),
    ]

    selected = select_papers_for_ingestion(
        "transformer attention mechanism",
        papers,
        max_select=1,
    )

    assert len(selected) == 1
    assert selected[0].arxiv_id == "2"
