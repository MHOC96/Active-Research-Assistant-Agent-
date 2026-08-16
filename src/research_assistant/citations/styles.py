"""Citation style registry and formatters."""

from __future__ import annotations

import re
from enum import StrEnum

from research_assistant.citations.validator import CITATION_PATTERN
from research_assistant.models import ExternalCitation, RetrievalHit


class CitationStyle(StrEnum):
    INTERNAL = "internal"
    APA7 = "apa7"
    MLA9 = "mla9"
    CHICAGO17 = "chicago17"
    IEEE = "ieee"
    HARVARD = "harvard"


CITATION_STYLE_INFO: dict[CitationStyle, str] = {
    CitationStyle.INTERNAL: "Machine-verifiable provenance: [arXiv:ID | Chunk N]",
    CitationStyle.APA7: "APA 7th edition (American Psychological Association)",
    CitationStyle.MLA9: "MLA 9th edition (Modern Language Association)",
    CitationStyle.CHICAGO17: "Chicago 17th edition, author-date",
    CitationStyle.IEEE: "IEEE numeric citation style",
    CitationStyle.HARVARD: "Harvard author-date style",
}


def list_citation_styles() -> list[tuple[str, str]]:
    """Return (style_id, description) pairs for user selection."""
    return [(style.value, description) for style, description in CITATION_STYLE_INFO.items()]


def parse_citation_style(value: str) -> CitationStyle:
    normalized = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "internal": CitationStyle.INTERNAL,
        "provenance": CitationStyle.INTERNAL,
        "apa": CitationStyle.APA7,
        "apa7": CitationStyle.APA7,
        "apa7th": CitationStyle.APA7,
        "mla": CitationStyle.MLA9,
        "mla9": CitationStyle.MLA9,
        "chicago": CitationStyle.CHICAGO17,
        "chicago17": CitationStyle.CHICAGO17,
        "ieee": CitationStyle.IEEE,
        "harvard": CitationStyle.HARVARD,
    }
    if normalized not in aliases:
        supported = ", ".join(style.value for style in CitationStyle)
        raise ValueError(f"Unknown citation style '{value}'. Supported: {supported}")
    return aliases[normalized]


def format_references_output(hits: list[RetrievalHit], style: CitationStyle) -> str:
    """Return only a formatted reference list for the retrieved sources."""
    if not hits:
        return "INSUFFICIENT_EVIDENCE: no sources found for the query."

    references = build_reference_list(hits, style)
    if not references:
        return "INSUFFICIENT_EVIDENCE: no citable sources found."

    return f"References\n\n{references}"


def format_grouped_references_output(
    hits: list[RetrievalHit],
    external_citations: list[ExternalCitation],
    style: CitationStyle,
    *,
    source_order: list[str] | None = None,
) -> str:
    """Return references grouped by discovery source (indexed arXiv + external indexes)."""
    sections: list[tuple[str, str]] = []
    local_arxiv_ids = {hit.arxiv_id for hit in hits if hit.arxiv_id}

    if hits:
        indexed_refs = build_reference_list(hits, style)
        if indexed_refs:
            sections.append(("arXiv (indexed)", indexed_refs))

    by_source: dict[str, list[ExternalCitation]] = {}
    for citation in external_citations:
        by_source.setdefault(citation.source, []).append(citation)

    order = source_order or list(by_source.keys())
    for source in order:
        source_citations = by_source.get(source, [])
        if not source_citations:
            continue
        source_citations.sort(key=lambda item: item.relevance_score, reverse=True)
        top = source_citations[0]
        if source == "arxiv" and top.arxiv_id and top.arxiv_id in local_arxiv_ids:
            continue
        entry = format_external_reference(top, style)
        if entry:
            sections.append((top.source_label, entry))

    if not sections:
        return "INSUFFICIENT_EVIDENCE: no sources found for the query."

    body = "\n\n".join(f"From {label}\n{content}" for label, content in sections)
    return f"References\n\n{body}"


def format_answer_citations(
    answer: str,
    hits: list[RetrievalHit],
    style: CitationStyle,
) -> str:
    """Replace internal provenance tags with the selected citation style."""
    if style == CitationStyle.INTERNAL or answer.startswith("INSUFFICIENT_EVIDENCE:"):
        return answer

    hit_map = {(hit.arxiv_id, hit.chunk_index): hit for hit in hits}
    ieee_numbers: dict[str, int] = {}
    next_ieee = 1

    def _ieee_number(arxiv_id: str) -> int:
        nonlocal next_ieee
        if arxiv_id not in ieee_numbers:
            ieee_numbers[arxiv_id] = next_ieee
            next_ieee += 1
        return ieee_numbers[arxiv_id]

    def _replace(match: re.Match[str]) -> str:
        arxiv_id = match.group(1)
        chunk_index = int(match.group(2))
        hit = hit_map.get((arxiv_id, chunk_index))
        if hit is None:
            return match.group(0)
        return format_in_text(hit, style, ieee_number=_ieee_number(arxiv_id))

    formatted = CITATION_PATTERN.sub(_replace, answer)
    references = build_reference_list(hits, style, ieee_numbers)
    if references:
        formatted = f"{formatted.rstrip()}\n\nReferences\n\n{references}"
    return formatted


def format_in_text(
    hit: RetrievalHit,
    style: CitationStyle,
    *,
    ieee_number: int | None = None,
) -> str:
    authors = _author_label(hit.authors)
    year = _publication_year(hit.published_date)
    page = _page_label(hit.page)

    if style == CitationStyle.APA7:
        page_part = f", p. {page}" if page else ""
        return f"({authors}, {year}{page_part})"

    if style == CitationStyle.MLA9:
        page_part = f" {page}" if page else ""
        return f"({authors}{page_part})"

    if style == CitationStyle.CHICAGO17:
        page_part = f", {page}" if page else ""
        return f"({authors} {year}{page_part})"

    if style == CitationStyle.IEEE:
        number = ieee_number if ieee_number is not None else 1
        return f"[{number}]"

    if style == CitationStyle.HARVARD:
        page_part = f", p. {page}" if page else ""
        return f"({authors} {year}{page_part})"

    return hit.provenance


def build_reference_list(
    hits: list[RetrievalHit],
    style: CitationStyle,
    ieee_numbers: dict[str, int] | None = None,
) -> str:
    """Build a deduplicated reference list ordered by first appearance."""
    seen: set[str] = set()
    ordered: list[RetrievalHit] = []
    for hit in hits:
        if hit.arxiv_id in seen:
            continue
        seen.add(hit.arxiv_id)
        ordered.append(hit)

    if style == CitationStyle.IEEE and ieee_numbers:
        ordered.sort(key=lambda hit: ieee_numbers.get(hit.arxiv_id, 999))

    entries = [format_reference(hit, style, ieee_numbers) for hit in ordered]
    return "\n\n".join(entry for entry in entries if entry)


def format_reference(
    hit: RetrievalHit,
    style: CitationStyle,
    ieee_numbers: dict[str, int] | None = None,
) -> str:
    authors = _reference_authors(hit.authors, style)
    year = _publication_year(hit.published_date)
    title = hit.title.rstrip(".")
    url = f"https://arxiv.org/abs/{hit.arxiv_id}"

    if style == CitationStyle.APA7:
        return (
            f"{authors} ({year}). {title}. arXiv. {url}"
        )

    if style == CitationStyle.MLA9:
        return f'{authors}. "{title}." arXiv, {year}, {url}.'

    if style == CitationStyle.CHICAGO17:
        return f'{authors}. {year}. "{title}." arXiv. {url}.'

    if style == CitationStyle.IEEE:
        number = (ieee_numbers or {}).get(hit.arxiv_id, 1)
        short_authors = _ieee_authors(hit.authors)
        return f'[{number}] {short_authors}, "{title}," arXiv:{hit.arxiv_id}, {year}. [Online]. Available: {url}'

    if style == CitationStyle.HARVARD:
        return f"{authors} ({year}) '{title}', arXiv preprint. Available at: {url} (Accessed: {year})."

    return ""


def format_external_reference(citation: ExternalCitation, style: CitationStyle) -> str:
    """Format a bibliographic entry from an external discovery source."""
    authors = _reference_authors(citation.authors, style)
    year = _publication_year(citation.published_date)
    title = citation.title.rstrip(".")
    url = citation.url
    if not url and citation.arxiv_id:
        url = f"https://arxiv.org/abs/{citation.arxiv_id}"
    elif not url and citation.doi:
        url = f"https://doi.org/{citation.doi}"

    venue = "arXiv" if citation.arxiv_id else citation.source_label

    if style == CitationStyle.APA7:
        return f"{authors} ({year}). {title}. {venue}. {url}" if url else f"{authors} ({year}). {title}. {venue}."

    if style == CitationStyle.MLA9:
        url_part = f", {url}" if url else ""
        return f'{authors}. "{title}." {venue}, {year}{url_part}.'

    if style == CitationStyle.CHICAGO17:
        url_part = f" {url}" if url else ""
        return f'{authors}. {year}. "{title}." {venue}.{url_part}'

    if style == CitationStyle.IEEE:
        short_authors = _ieee_authors(citation.authors)
        url_part = f" [Online]. Available: {url}" if url else ""
        return f'{short_authors}, "{title}," {venue}, {year}.{url_part}'

    if style == CitationStyle.HARVARD:
        url_part = f" Available at: {url}" if url else ""
        return f"{authors} ({year}) '{title}', {venue}.{url_part}"

    return ""


def _author_label(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return _surname(authors[0])
    if len(authors) == 2:
        return f"{_surname(authors[0])} & {_surname(authors[1])}"
    return f"{_surname(authors[0])} et al."


def _reference_authors(authors: list[str], style: CitationStyle) -> str:
    if not authors:
        return "Unknown"

    if style == CitationStyle.MLA9:
        if len(authors) == 1:
            return _surname(authors[0])
        if len(authors) == 2:
            return f"{_surname(authors[0])}, and {_surname(authors[1])}"
        return f"{_surname(authors[0])}, et al"

    formatted = [_apa_author(name) for name in authors[:20]]
    if len(authors) > 20:
        formatted.append("...")
        formatted.append(_apa_author(authors[-1]))
    separator = ", & " if style in {CitationStyle.APA7, CitationStyle.HARVARD} else ", "
    if len(formatted) == 1:
        return formatted[0]
    if style in {CitationStyle.APA7, CitationStyle.HARVARD} and len(formatted) >= 2:
        return separator.join([", ".join(formatted[:-1]), formatted[-1]])
    return ", ".join(formatted)


def _ieee_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= 3:
        return ", ".join(_ieee_author(name) for name in authors)
    return f"{_ieee_author(authors[0])} et al."


def _apa_author(name: str) -> str:
    parts = name.strip().split()
    if not parts:
        return "Unknown"
    if len(parts) == 1:
        return f"{parts[0]}, A."
    initials = " ".join(f"{part[0]}." for part in parts[:-1])
    return f"{parts[-1]}, {initials}".strip()


def _ieee_author(name: str) -> str:
    parts = name.strip().split()
    if not parts:
        return "Unknown"
    if len(parts) == 1:
        return parts[0]
    initials = " ".join(f"{part[0]}." for part in parts[:-1])
    return f"{initials} {parts[-1]}".strip()


def _surname(name: str) -> str:
    parts = name.strip().split()
    return parts[-1] if parts else "Unknown"


def _publication_year(published_date: str | None) -> str:
    if not published_date:
        return "n.d."
    match = re.match(r"(\d{4})", published_date)
    return match.group(1) if match else "n.d."


def _page_label(page: int | None) -> str | None:
    if page is None or page < 1:
        return None
    return str(page)
