"""Tests for multi-source literature discovery."""

from unittest.mock import MagicMock

from research_assistant.discovery.multi import MultiSourceDiscoveryService
from research_assistant.models import ArxivPaper, DiscoveredPaper


def _arxiv_paper() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="2407.08608",
        title="Graph RAG Survey",
        abstract="Graph RAG latency tradeoffs.",
        pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
    )


def _openalex_paper() -> DiscoveredPaper:
    return DiscoveredPaper(
        source="openalex",
        external_id="W123",
        title="Knowledge Graph RAG",
        authors=["Alice Smith"],
        abstract="OpenAlex graph retrieval paper.",
        published_date="2024",
        landing_url="https://openalex.org/W123",
        doi="10.1234/example",
    )


def test_search_by_source_returns_top_per_source():
    settings = MagicMock()
    settings.discovery_source_list = ["arxiv", "openalex"]
    settings.discovery_per_source_max = 1
    settings.discovery_max_results = 5
    settings.min_external_relevance_score = 0.0

    arxiv = MagicMock()
    arxiv.search_arxiv.return_value = [_arxiv_paper()]

    openalex = MagicMock()
    openalex.search.return_value = [_openalex_paper()]

    service = MultiSourceDiscoveryService(
        settings=settings,
        arxiv=arxiv,
        openalex=openalex,
        semantic_scholar=MagicMock(),
    )

    results = service.search_by_source("graph rag latency")

    assert set(results.keys()) == {"arxiv", "openalex"}
    assert results["arxiv"][0].arxiv_id == "2407.08608"
    assert results["openalex"][0].source == "openalex"


def test_search_by_source_skips_empty_sources():
    settings = MagicMock()
    settings.discovery_source_list = ["arxiv", "openalex"]
    settings.discovery_per_source_max = 1
    settings.discovery_max_results = 5
    settings.min_external_relevance_score = 0.0

    arxiv = MagicMock()
    arxiv.search_arxiv.return_value = []

    openalex = MagicMock()
    openalex.search.return_value = []

    service = MultiSourceDiscoveryService(
        settings=settings,
        arxiv=arxiv,
        openalex=openalex,
    )

    assert service.search_by_source("unknown topic") == {}
