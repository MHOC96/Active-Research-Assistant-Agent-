"""Multi-source academic literature discovery."""

from __future__ import annotations

import logging

from research_assistant.config import Settings, get_settings
from research_assistant.discovery.arxiv import ArxivDiscoveryService
from research_assistant.discovery.openalex import OpenAlexDiscoveryService
from research_assistant.discovery.relevance import relevance_score, select_top_papers
from research_assistant.discovery.semantic_scholar import SemanticScholarDiscoveryService
from research_assistant.discovery.sources import source_label
from research_assistant.discovery.web import WebDiscoveryService
from research_assistant.models import ArxivPaper, DiscoveredPaper, ExternalCitation
from research_assistant.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class MultiSourceDiscoveryService:
    """Search multiple academic indexes and return top hits per source."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        arxiv: ArxivDiscoveryService | None = None,
        openalex: OpenAlexDiscoveryService | None = None,
        semantic_scholar: SemanticScholarDiscoveryService | None = None,
        web: WebDiscoveryService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._arxiv = arxiv or ArxivDiscoveryService(self.settings)
        self._openalex = openalex or OpenAlexDiscoveryService(self.settings)
        self._semantic_scholar = semantic_scholar or SemanticScholarDiscoveryService(self.settings)
        self._web = web or WebDiscoveryService(self.settings)

    @property
    def enabled_sources(self) -> list[str]:
        return self.settings.discovery_source_list

    def search_by_source(self, query: str) -> dict[str, list[DiscoveredPaper]]:
        per_source = self.settings.discovery_per_source_max
        results: dict[str, list[DiscoveredPaper]] = {}

        for source in self.enabled_sources:
            papers = self._search_source(source, query)
            top = select_top_papers(query, papers, max_select=per_source)
            if top:
                results[source] = top
                logger.info(
                    "discovery source=%s query=%s results=%s",
                    source,
                    query,
                    len(top),
                )
        return results

    def search_arxiv(self, query: str, *, max_results: int | None = None) -> list[ArxivPaper]:
        """Backward-compatible arXiv-only search."""
        papers = self._arxiv.search_arxiv(query, max_results=max_results)
        return papers

    def _search_source(self, source: str, query: str) -> list[DiscoveredPaper]:
        if source == "arxiv":
            return [_from_arxiv(paper) for paper in self._arxiv.search_arxiv(query)]
        if source == "openalex":
            return self._openalex.search(query, max_results=self.settings.discovery_max_results)
        if source == "semantic_scholar":
            return self._semantic_scholar.search(
                query, max_results=self.settings.discovery_max_results
            )
        if source == "web":
            return self._web.search(query, max_results=self.settings.discovery_max_results)
        return []


def deduplicate_papers(
    papers: list[DiscoveredPaper],
    metadata: MetadataStore,
) -> list[DiscoveredPaper]:
    """Remove papers already ingested or duplicated within the result set."""
    seen: set[str] = set()
    unique: list[DiscoveredPaper] = []

    for paper in papers:
        key = paper.document_id
        if key in seen:
            continue
        if paper.arxiv_id and metadata.is_ingested(paper.arxiv_id):
            continue
        seen.add(key)
        unique.append(paper)

    return unique


def select_papers_for_ingestion(
    query: str,
    papers: list[DiscoveredPaper],
    *,
    max_select: int,
) -> list[DiscoveredPaper]:
    """Select ingestible papers ranked by query relevance."""
    ingestible = [paper for paper in papers if paper.ingestible]
    return select_top_papers(query, ingestible, max_select=max_select)


def external_citations_from_sources(
    by_source: dict[str, list[DiscoveredPaper]],
    query: str,
) -> list[ExternalCitation]:
    citations: list[ExternalCitation] = []
    for _source, papers in by_source.items():
        for paper in papers:
            citations.append(
                ExternalCitation(
                    source=paper.source,
                    source_label=source_label(paper.source),
                    title=paper.title,
                    authors=paper.authors,
                    published_date=paper.published_date,
                    url=paper.landing_url or paper.pdf_url or "",
                    doi=paper.doi,
                    arxiv_id=paper.arxiv_id,
                    publisher=paper.publisher,
                    relevance_score=relevance_score(query, paper),
                )
            )
    return citations


def flatten_for_ingestion(by_source: dict[str, list[DiscoveredPaper]]) -> list[DiscoveredPaper]:
    return [paper for papers in by_source.values() for paper in papers]


def _from_arxiv(paper: ArxivPaper) -> DiscoveredPaper:
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
