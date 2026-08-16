"""Web search discovery for online documentation and corporate sources."""

from __future__ import annotations

import html
import logging
import re

import httpx

from research_assistant.config import Settings, get_settings
from research_assistant.discovery.publisher import extract_year, publisher_from_url, unwrap_redirect_url
from research_assistant.models import DiscoveredPaper

logger = logging.getLogger(__name__)

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
RESULT_LINK_PATTERN = re.compile(
    r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
RESULT_SNIPPET_PATTERN = re.compile(
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|td|div)>',
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")


class WebDiscoveryService:
    """Search the public web and return citable page metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, *, max_results: int | None = None) -> list[DiscoveredPaper]:
        normalized = re.sub(r"\s+", " ", query.strip())
        if not normalized:
            return []

        limit = min(max_results or self.settings.discovery_max_results, 10)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/0.1)",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"q": normalized, "b": "", "kl": "wt-wt"}

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                response = client.post(DDG_HTML_URL, data=data)
                response.raise_for_status()
                return parse_ddg_html(response.text, limit=limit)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return []


def parse_ddg_html(page_html: str, *, limit: int = 5) -> list[DiscoveredPaper]:
    """Parse DuckDuckGo HTML results into discovered papers."""
    links = list(RESULT_LINK_PATTERN.finditer(page_html))
    snippets = [_clean_html(match.group("snippet")) for match in RESULT_SNIPPET_PATTERN.finditer(page_html)]

    papers: list[DiscoveredPaper] = []
    for index, match in enumerate(links[:limit]):
        title = _clean_html(match.group("title"))
        url = unwrap_redirect_url(html.unescape(match.group("href")))
        if not title or not url.startswith("http"):
            continue

        snippet = snippets[index] if index < len(snippets) else ""
        publisher = publisher_from_url(url)
        year = extract_year(title, snippet, url)

        papers.append(
            DiscoveredPaper(
                source="web",
                external_id=url,
                title=title,
                authors=[],
                abstract=snippet,
                published_date=year,
                landing_url=url,
                publisher=publisher,
            )
        )
    return papers


def _clean_html(value: str) -> str:
    text = TAG_PATTERN.sub("", value)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text.encode("utf-8", errors="replace").decode("utf-8")
