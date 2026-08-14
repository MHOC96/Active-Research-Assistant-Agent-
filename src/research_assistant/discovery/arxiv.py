"""arXiv literature discovery."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import arxiv

from research_assistant.config import Settings, get_settings
from research_assistant.models import ArxivPaper
from research_assistant.security.urls import validate_https_url
from research_assistant.storage.metadata_store import MetadataStore


@runtime_checkable
class ArxivSearchClient(Protocol):
    def results(self, search: arxiv.Search) -> list[arxiv.Result]:
        """Return search results from arXiv."""


class ArxivClientAdapter:
    """Adapter for the arxiv.Client API."""

    def __init__(self, client: arxiv.Client | None = None) -> None:
        self._client = client or arxiv.Client()

    def results(self, search: arxiv.Search) -> list[arxiv.Result]:
        return list(self._client.results(search))


class ArxivDiscoveryService:
    """Search arXiv metadata for relevant academic papers."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: ArxivSearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or ArxivClientAdapter()

    def search_arxiv(self, query: str, *, max_results: int | None = None) -> list[ArxivPaper]:
        limit = min(max_results or self.settings.discovery_max_results, 10)
        normalized = _normalize_search_query(query)
        if not normalized:
            return []

        search = arxiv.Search(query=f"all:{normalized}", max_results=limit)
        try:
            results = self._client.results(search)
        except Exception as exc:
            raise RuntimeError(f"ARXIV_SEARCH_FAILED: {exc}") from exc

        papers: list[ArxivPaper] = []
        for result in results:
            paper = _result_to_paper(result, self.settings.allowed_domains)
            if paper is not None:
                papers.append(paper)
        return papers


def deduplicate_papers(
    papers: list[ArxivPaper],
    metadata: MetadataStore,
) -> list[ArxivPaper]:
    """Remove papers already ingested or duplicated within the result set."""
    seen: set[str] = set()
    unique: list[ArxivPaper] = []

    for paper in papers:
        arxiv_id = MetadataStore.normalize_arxiv_id(paper.arxiv_id)
        if arxiv_id in seen:
            continue
        if metadata.is_ingested(arxiv_id):
            continue
        seen.add(arxiv_id)
        unique.append(paper.model_copy(update={"arxiv_id": arxiv_id}))

    return unique


def select_papers_for_ingestion(
    query: str,
    papers: list[ArxivPaper],
    *,
    max_select: int,
) -> list[ArxivPaper]:
    """Select the most query-relevant papers for ingestion."""
    if not papers:
        return []

    scored = [(paper, _relevance_score(query, paper)) for paper in papers]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [paper for paper, score in scored[:max_select] if score > 0]


def _normalize_search_query(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", query.strip())
    return cleaned


def _extract_arxiv_id(entry_id: str) -> str:
    tail = entry_id.rstrip("/").split("/")[-1]
    return MetadataStore.normalize_arxiv_id(tail)


def _result_to_paper(result: arxiv.Result, allowed_domains: frozenset[str]) -> ArxivPaper | None:
    arxiv_id = _extract_arxiv_id(result.entry_id)
    pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        validate_https_url(pdf_url, allowed_domains)
    except Exception:
        return None

    authors = [author.name for author in result.authors]
    published = result.published.isoformat() if result.published else None
    updated = result.updated.isoformat() if result.updated else None

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=result.title.replace("\n", " ").strip(),
        authors=authors,
        abstract=result.summary.replace("\n", " ").strip(),
        published_date=published,
        updated_date=updated,
        pdf_url=pdf_url,
        categories=list(result.categories),
    )


def _relevance_score(query: str, paper: ArxivPaper) -> float:
    terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9_.-]+", query) if len(term) > 2}
    if not terms:
        return 0.0

    corpus = f"{paper.title} {paper.abstract}".lower()
    matches = sum(1 for term in terms if term in corpus)
    return matches / len(terms)
