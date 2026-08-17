"""Research orchestrator coordinating retrieval, discovery, and references."""

from __future__ import annotations

import logging
import uuid

from research_assistant.citations.spans import build_citation_spans
from research_assistant.citations.styles import (
    CitationStyle,
    format_grouped_references_output,
    parse_citation_style,
)
from research_assistant.discovery.relevance import best_hit_relevance, filter_external_citations
from research_assistant.models import ActiveResearchResult, ExternalCitation, ResearchResponse, RetrievalHit
from research_assistant.orchestrator.llm import LLMClient
from research_assistant.orchestrator.query_processor import QueryProcessor
from research_assistant.pipeline.active_loop import ActiveLiteraturePipeline
from research_assistant.sufficiency.gate import evaluate_sufficiency
from research_assistant.utils.cancellation import CancellationToken

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """End-to-end orchestrator: retrieve, discover, ingest, and format references."""

    def __init__(
        self,
        pipeline: ActiveLiteraturePipeline,
        llm: LLMClient,
        *,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.query_processor = query_processor or QueryProcessor(llm)

    def answer(
        self,
        user_query: str,
        *,
        citation_style: CitationStyle | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ResearchResponse:
        request_id = cancellation.request_id if cancellation is not None else str(uuid.uuid4())
        style = citation_style or parse_citation_style(self.pipeline.settings.citation_style)

        if cancellation is not None:
            cancellation.raise_if_cancelled("query_analysis")

        analysis = self.query_processor.analyze(user_query)
        logger.info(
            "request_id=%s normalized_query=%s subqueries=%s",
            request_id,
            analysis.normalized_query,
            analysis.subqueries,
        )

        if cancellation is not None:
            cancellation.raise_if_cancelled("subquery_execution")

        subquery_results: list[ActiveResearchResult] = []
        unsupported_aspects: list[str] = []
        merged_hits = _merge_hits([])

        for subquery in analysis.subqueries:
            if cancellation is not None:
                cancellation.raise_if_cancelled("subquery_execution")

            result = self.pipeline.run(
                subquery,
                cancellation=cancellation,
                parent_request_id=request_id,
            )
            subquery_results.append(result)
            merged_hits = _merge_hits(merged_hits + result.retrieval.candidates)
            if not result.sufficient:
                unsupported_aspects.append(
                    f"No sufficient local evidence for sub-query '{subquery}'"
                    + (
                        f" ({result.insufficient_message})"
                        if result.insufficient_message
                        else ""
                    )
                )

        overall_sufficiency = evaluate_sufficiency(
            merged_hits,
            min_candidates=self.pipeline.settings.min_candidates,
            min_rerank_score=self.pipeline.settings.min_rerank_score,
        )

        if cancellation is not None:
            cancellation.raise_if_cancelled("references")

        settings = self.pipeline.settings
        topic_queries = [analysis.original_query, *analysis.subqueries]

        source_hits = _filter_indexed_hits(
            _unique_papers(merged_hits),
            topic_queries,
            min_rerank_score=settings.min_rerank_score,
            min_topic_score=settings.min_indexed_topic_score,
        )
        external_citations = filter_external_citations(
            _collect_external_citations(subquery_results),
            min_score=settings.min_external_relevance_score,
        )

        if not source_hits and not external_citations:
            message = _references_insufficient_message(unsupported_aspects, overall_sufficiency)
            return ResearchResponse(
                request_id=request_id,
                query=analysis.original_query,
                normalized_query=analysis.normalized_query,
                query_type=analysis.query_type,
                subqueries=analysis.subqueries,
                answer=message,
                citations_valid=True,
                citation_style=style.value,
                subquery_results=subquery_results,
                evidence_hits=merged_hits,
                citation_spans=[],
                sufficient=False,
                insufficient_message=message,
            )

        citation_spans = build_citation_spans(
            analysis.original_query,
            analysis.subqueries,
            subquery_results,
            style,
            min_rerank_score=settings.min_rerank_score,
            min_external_relevance_score=settings.min_external_relevance_score,
            min_indexed_topic_score=settings.min_indexed_topic_score,
            source_hits=source_hits,
            global_external=external_citations,
        )

        answer = format_grouped_references_output(
            source_hits,
            external_citations,
            style,
            source_order=settings.discovery_source_list,
            min_external_relevance_score=settings.min_external_relevance_score,
        )
        if not overall_sufficiency.sufficient:
            note = _references_insufficient_message(unsupported_aspects, overall_sufficiency)
            answer = f"{answer}\n\nNote: {note}"

        return ResearchResponse(
            request_id=request_id,
            query=analysis.original_query,
            normalized_query=analysis.normalized_query,
            query_type=analysis.query_type,
            subqueries=analysis.subqueries,
            answer=answer,
            citations_valid=True,
            citation_style=style.value,
            subquery_results=subquery_results,
            evidence_hits=merged_hits,
            external_citations=external_citations,
            citation_spans=citation_spans,
            sufficient=overall_sufficiency.sufficient,
            insufficient_message=None
            if overall_sufficiency.sufficient
            else _references_insufficient_message(unsupported_aspects, overall_sufficiency),
        )


def _merge_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    by_id: dict[str, RetrievalHit] = {}
    for hit in hits:
        existing = by_id.get(hit.chunk_id)
        if existing is None or (hit.rerank_score or 0) > (existing.rerank_score or 0):
            by_id[hit.chunk_id] = hit
    merged = list(by_id.values())
    merged.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return merged


def _filter_indexed_hits(
    hits: list[RetrievalHit],
    topic_queries: list[str],
    *,
    min_rerank_score: float,
    min_topic_score: float,
) -> list[RetrievalHit]:
    filtered: list[RetrievalHit] = []
    for hit in hits:
        if (hit.rerank_score or 0) < min_rerank_score:
            continue
        if best_hit_relevance(hit, topic_queries) < min_topic_score:
            continue
        filtered.append(hit)
    return filtered


def _collect_external_citations(
    results: list[ActiveResearchResult],
) -> list[ExternalCitation]:
    seen: set[tuple[str, str]] = set()
    collected: list[ExternalCitation] = []
    for result in results:
        for citation in result.external_citations:
            key = (citation.source, citation.url or citation.title)
            if key in seen:
                continue
            seen.add(key)
            collected.append(citation)
    return collected


def _unique_papers(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Keep the highest-scoring chunk per paper, ordered by relevance."""
    best_by_paper: dict[str, RetrievalHit] = {}
    for hit in hits:
        existing = best_by_paper.get(hit.arxiv_id)
        if existing is None or (hit.rerank_score or 0) > (existing.rerank_score or 0):
            best_by_paper[hit.arxiv_id] = hit
    ordered = list(best_by_paper.values())
    ordered.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return ordered


def _references_insufficient_message(
    unsupported_aspects: list[str],
    sufficiency,
) -> str:
    if unsupported_aspects:
        return f"INSUFFICIENT_EVIDENCE: {'; '.join(unsupported_aspects)}"
    reason = sufficiency.reason or "retrieved evidence did not meet sufficiency thresholds"
    return f"INSUFFICIENT_EVIDENCE: {reason}"
