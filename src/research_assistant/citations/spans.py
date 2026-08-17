"""Map query text segments to their supporting citations."""

from __future__ import annotations

from research_assistant.citations.styles import (
    CitationStyle,
    format_external_in_text,
    format_external_reference,
    format_reference,
)
from research_assistant.discovery.relevance import best_hit_relevance
from research_assistant.models import (
    ActiveResearchResult,
    CitationSourceItem,
    CitationSpan,
    ExternalCitation,
    RetrievalHit,
)


def build_citation_spans(
    original_text: str,
    subqueries: list[str],
    subquery_results: list[ActiveResearchResult],
    style: CitationStyle,
    *,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    min_indexed_topic_score: float = 0.0,
    source_hits: list[RetrievalHit] | None = None,
    global_external: list[ExternalCitation] | None = None,
) -> list[CitationSpan]:
    """Align sub-query results with text spans and attach formatted citations."""
    if not subquery_results:
        return []

    allowed_arxiv_ids = {hit.arxiv_id for hit in source_hits} if source_hits is not None else None
    allowed_external_keys = (
        {(citation.source, citation.url or citation.title) for citation in global_external}
        if global_external is not None
        else None
    )

    text_spans = _text_spans_for_query(original_text)
    if len(text_spans) == 1 and len(subquery_results) > 1:
        sentence, start, end = text_spans[0]
        citations = _merge_citations(
            subquery_results,
            subqueries,
            style,
            min_rerank_score=min_rerank_score,
            min_external_relevance_score=min_external_relevance_score,
            min_indexed_topic_score=min_indexed_topic_score,
            allowed_arxiv_ids=allowed_arxiv_ids,
            allowed_external_keys=allowed_external_keys,
        )
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
    segment_count = len(text_spans) if text_spans else max(len(subquery_results), 1)

    for index in range(segment_count):
        if index < len(text_spans):
            sentence, start, end = text_spans[index]
        elif text_spans:
            sentence, start, end = text_spans[-1]
        else:
            sentence, start, end = original_text.strip(), 0, len(original_text)

        result = subquery_results[index] if index < len(subquery_results) else None
        search_query = subqueries[index] if index < len(subqueries) else (
            result.query if result is not None else sentence
        )

        citations: list[CitationSourceItem] = []
        if result is not None:
            citations = _citations_for_result(
                result,
                style,
                segment_query=search_query,
                min_rerank_score=min_rerank_score,
                min_external_relevance_score=min_external_relevance_score,
                min_indexed_topic_score=min_indexed_topic_score,
                allowed_arxiv_ids=allowed_arxiv_ids,
                allowed_external_keys=allowed_external_keys,
            )

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
    from research_assistant.orchestrator.paste_to_cite import is_paste_to_cite, sentence_spans_in_text

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
    subqueries: list[str],
    style: CitationStyle,
    *,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    min_indexed_topic_score: float = 0.0,
    allowed_arxiv_ids: set[str] | None = None,
    allowed_external_keys: set[tuple[str, str]] | None = None,
) -> list[CitationSourceItem]:
    merged: list[CitationSourceItem] = []
    seen: set[str] = set()
    for index, result in enumerate(results):
        segment_query = subqueries[index] if index < len(subqueries) else result.query
        for item in _citations_for_result(
            result,
            style,
            segment_query=segment_query,
            min_rerank_score=min_rerank_score,
            min_external_relevance_score=min_external_relevance_score,
            min_indexed_topic_score=min_indexed_topic_score,
            allowed_arxiv_ids=allowed_arxiv_ids,
            allowed_external_keys=allowed_external_keys,
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
    segment_query: str,
    min_rerank_score: float,
    min_external_relevance_score: float = 0.0,
    min_indexed_topic_score: float = 0.0,
    allowed_arxiv_ids: set[str] | None = None,
    allowed_external_keys: set[tuple[str, str]] | None = None,
) -> list[CitationSourceItem]:
    items: list[CitationSourceItem] = []
    seen: set[str] = set()
    counter = 0

    for hit in _unique_papers(result.retrieval.candidates):
        if (hit.rerank_score or 0) < min_rerank_score:
            continue
        if best_hit_relevance(hit, [segment_query]) < min_indexed_topic_score:
            continue
        if allowed_arxiv_ids is not None and hit.arxiv_id not in allowed_arxiv_ids:
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
        external_key = (external.source, external.url or external.title)
        if allowed_external_keys is not None and external_key not in allowed_external_keys:
            continue
        token = f"{external_key[0]}:{external_key[1]}"
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
