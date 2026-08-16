"""Semantic Scholar literature discovery."""

from __future__ import annotations

import logging
import re

import httpx

from research_assistant.config import Settings, get_settings
from research_assistant.models import DiscoveredPaper
from research_assistant.security.urls import validate_https_url

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarDiscoveryService:
    """Search Semantic Scholar for academic papers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, *, max_results: int | None = None) -> list[DiscoveredPaper]:
        limit = min(max_results or self.settings.discovery_max_results, 10)
        normalized = re.sub(r"\s+", " ", query.strip())
        if not normalized:
            return []

        params = {
            "query": normalized,
            "limit": limit,
            "fields": "paperId,title,authors,year,abstract,externalIds,openAccessPdf,url",
        }
        headers = {"User-Agent": "research-assistant/0.1"}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key

        try:
            with httpx.Client(timeout=30.0, headers=headers) as client:
                response = client.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []

        papers: list[DiscoveredPaper] = []
        for item in payload.get("data", []):
            paper = _item_to_paper(item, self.settings.allowed_domains)
            if paper is not None:
                papers.append(paper)
        return papers


def _item_to_paper(item: dict, allowed_domains: frozenset[str]) -> DiscoveredPaper | None:
    title = (item.get("title") or "").strip()
    paper_id = item.get("paperId")
    if not title or not paper_id:
        return None

    authors = [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
    year = item.get("year")
    published_date = str(year) if year else None

    external_ids = item.get("externalIds") or {}
    arxiv_raw = external_ids.get("ArXiv") or external_ids.get("arXiv")
    arxiv_id = arxiv_raw.split("v")[0].strip() if arxiv_raw else None
    doi = external_ids.get("DOI")

    pdf_url = None
    open_access = item.get("openAccessPdf") or {}
    oa_url = open_access.get("url")
    if oa_url:
        try:
            validate_https_url(oa_url, allowed_domains)
            pdf_url = oa_url
        except Exception:
            pdf_url = None

    if arxiv_id and not pdf_url:
        candidate = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            validate_https_url(candidate, allowed_domains)
            pdf_url = candidate
        except Exception:
            pdf_url = None

    landing_url = item.get("url") or (
        f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None
    )

    return DiscoveredPaper(
        source="semantic_scholar",
        external_id=paper_id,
        title=title,
        authors=authors,
        abstract=(item.get("abstract") or "").strip(),
        published_date=published_date,
        pdf_url=pdf_url,
        landing_url=landing_url,
        doi=doi,
        arxiv_id=arxiv_id,
    )
