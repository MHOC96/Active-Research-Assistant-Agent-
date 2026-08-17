"""Map query text segments to their supporting citations."""

from __future__ import annotations

from research_assistant.citations.styles import (
    CitationStyle,
    format_external_in_text,
    format_external_reference,
    format_reference,
)
from research_assistant.models import (
    ActiveResearchResult,
    CitationSourceItem,
    CitationSpan,
    ExternalCitation,
    RetrievalHit,
)
from research_assistant.discovery.relevance import best_hit_relevance
from research_assistant.orchestrator.paste_to_cite import is_paste_to_cite, sentence_spans_in_text


def build_citation_spans(
    original_text: str,
    subqueries: list[str],
    subquery_results: list[ActiveResearchResult],
    style: CitationStyle,
    *,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    topic_queries: list[str] | None = None,
    min_indexed_topic_score: float = 0.0,
) -> list[CitationSpan]:
    """Align sub-query results with text spans and attach formatted citations."""
    if not subquery_results:
        return []

    relevance_queries = topic_queries or [original_text, *subqueries]
    filter_kwargs = {
        "min_rerank_score": min_rerank_score,
        "min_external_relevance_score": min_external_relevance_score,
        "topic_queries": relevance_queries,
        "min_indexed_topic_score": min_indexed_topic_score,
    }

    text_spans = _text_spans_for_query(original_text)
    if len(text_spans) == 1 and len(subquery_results) > 1:
        sentence, start, end = text_spans[0]
        citations = _merge_citations(subquery_results, style, **filter_kwargs)
        return [
            CitationSpan(
                segment_id="seg-0",
                text=sentence.strip(),
                start=start,
                end=end,
                search_query=subqueries[0] if subqueries else sentence,
                citations=citations,
            )
        ]

    spans: list[CitationSpan] = []
    for index, result in enumerate(subquery_results):
        if index < len(text_spans):
            sentence, start, end = text_spans[index]
        elif text_spans:
            sentence, start, end = text_spans[-1]
        else:
            sentence, start, end = original_text.strip(), 0, len(original_text)

        search_query = subqueries[index] if index < len(subqueries) else result.query
        citations = _citations_for_result(result, style, **filter_kwargs)
        spans.append(
            CitationSpan(
                segment_id=f"seg-{index}",
                text=sentence.strip(),
                start=start,
                end=end,
                search_query=search_query,
                citations=citations,
            )
        )

    return spans


def _text_spans_for_query(text: str) -> list[tuple[str, int, int]]:
    if is_paste_to_cite(text):
        spans = sentence_spans_in_text(text)
        if spans:
            return spans

    stripped = text.strip()
    if not stripped:
        return []

    start = text.find(stripped)
    if start < 0:
        start = 0
    end = start + len(stripped)
    return [(stripped, start, end)]


def _merge_citations(
    results: list[ActiveResearchResult],
    style: CitationStyle,
    *,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    topic_queries: list[str] | None = None,
    min_indexed_topic_score: float = 0.0,
) -> list[CitationSourceItem]:
    merged: list[CitationSourceItem] = []
    seen: set[str] = set()
    for result in results:
        for item in _citations_for_result(
            result,
            style,
            min_rerank_score=min_rerank_score,
            min_external_relevance_score=min_external_relevance_score,
            topic_queries=topic_queries,
            min_indexed_topic_score=min_indexed_topic_score,
        ):
            key = f"{item.source}:{item.url or item.title}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _citations_for_result(
    result: ActiveResearchResult,
    style: CitationStyle,
    *,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    topic_queries: list[str] | None = None,
    min_indexed_topic_score: float = 0.0,
) -> list[CitationSourceItem]:
    items: list[CitationSourceItem] = []
    seen: set[str] = set()
    counter = 0
    relevance_queries = topic_queries or [result.query]

    for hit in _unique_papers(result.retrieval.candidates):
        if (hit.rerank_score or 0) < min_rerank_score:
            continue
        if best_hit_relevance(hit, relevance_queries) < min_indexed_topic_score:
            continue
        key = f"arxiv:{hit.arxiv_id}"
        if key in seen:
            continue
        seen.add(key)
        counter += 1
        items.append(
            CitationSourceItem(
                id=f"cite-{counter}",
                source="arxiv",
                source_label="arXiv",
                title=hit.title,
                url=f"https://arxiv.org/abs/{hit.arxiv_id}",
                reference=format_reference(hit, style),
                arxiv_id=hit.arxiv_id,
            )
        )

    for external in result.external_citations:
        if external.relevance_score < min_external_relevance_score:
            continue
        key = (external.source, external.url or external.title)
        token = f"{key[0]}:{key[1]}"
        if token in seen:
            continue
        seen.add(token)
        counter += 1
        in_text = format_external_in_text(external, style) if external.source == "web" else None
        items.append(
            CitationSourceItem(
                id=f"cite-{counter}",
                source=external.source,
                source_label=external.source_label,
                title=external.title,
                url=external.url,
                reference=format_external_reference(external, style),
                in_text=in_text,
                arxiv_id=external.arxiv_id,
            )
        )

    return items


def _unique_papers(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    best_by_paper: dict[str, RetrievalHit] = {}
    for hit in hits:
        existing = best_by_paper.get(hit.arxiv_id)
        if existing is None or (hit.rerank_score or 0) > (existing.rerank_score or 0):
            best_by_paper[hit.arxiv_id] = hit
    ordered = list(best_by_paper.values())
    ordered.sort(key=lambda hit: hit.rerank_score or 0.0, reverse=True)
    return ordered
