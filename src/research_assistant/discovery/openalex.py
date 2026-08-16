"""OpenAlex literature discovery."""

from __future__ import annotations

import logging
import re

import httpx

from research_assistant.config import Settings, get_settings
from research_assistant.models import DiscoveredPaper
from research_assistant.security.urls import validate_https_url

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexDiscoveryService:
    """Search OpenAlex for academic works."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, *, max_results: int | None = None) -> list[DiscoveredPaper]:
        limit = min(max_results or self.settings.discovery_max_results, 10)
        normalized = re.sub(r"\s+", " ", query.strip())
        if not normalized:
            return []

        params: dict[str, str | int] = {
            "search": normalized,
            "per_page": limit,
        }
        if self.settings.openalex_mailto:
            params["mailto"] = self.settings.openalex_mailto

        headers = {"User-Agent": "research-assistant/0.1 (mailto:research@local)"}
        try:
            with httpx.Client(timeout=30.0, headers=headers) as client:
                response = client.get(OPENALEX_WORKS_URL, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning("OpenAlex search failed: %s", exc)
            return []

        papers: list[DiscoveredPaper] = []
        for work in payload.get("results", []):
            paper = _work_to_paper(work, self.settings.allowed_domains)
            if paper is not None:
                papers.append(paper)
        return papers


def _work_to_paper(work: dict, allowed_domains: frozenset[str]) -> DiscoveredPaper | None:
    title = (work.get("display_name") or work.get("title") or "").strip()
    if not title:
        return None

    openalex_id = str(work.get("id", "")).rstrip("/").split("/")[-1]
    if not openalex_id:
        return None

    ids = work.get("ids") or {}
    doi = _strip_doi(ids.get("doi"))
    arxiv_raw = ids.get("arxiv") or ids.get("arxiv_id")
    arxiv_id = _normalize_arxiv_id(arxiv_raw) if arxiv_raw else None

    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in work.get("authorships", [])
        if authorship.get("author", {}).get("display_name")
    ]

    year = work.get("publication_year")
    published_date = str(year) if year else None

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    landing_url = work.get("doi") or work.get("id") or ""
    pdf_url = None
    open_access = work.get("open_access") or {}
    oa_url = open_access.get("oa_url")
    if oa_url and str(oa_url).lower().endswith(".pdf"):
        try:
            validate_https_url(oa_url, allowed_domains)
            pdf_url = oa_url
        except Exception:
            pdf_url = None

    if arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            validate_https_url(pdf_url, allowed_domains)
        except Exception:
            pdf_url = None

    primary = work.get("primary_location") or {}
    if not landing_url and primary.get("landing_page_url"):
        landing_url = primary["landing_page_url"]

    return DiscoveredPaper(
        source="openalex",
        external_id=openalex_id,
        title=title,
        authors=authors,
        abstract=abstract,
        published_date=published_date,
        pdf_url=pdf_url,
        landing_url=landing_url or None,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions.append((index, word))
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def _strip_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "")


def _normalize_arxiv_id(value: str) -> str:
    tail = value.rstrip("/").split("/")[-1]
    return tail.split("v")[0].strip()
