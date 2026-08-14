"""Research orchestrator coordinating retrieval, discovery, and synthesis."""

from __future__ import annotations

import logging
import uuid

from research_assistant.models import ActiveResearchResult, ResearchResponse, RetrievalHit
from research_assistant.orchestrator.llm import LLMClient
from research_assistant.orchestrator.query_processor import QueryAnalysis, QueryProcessor
from research_assistant.orchestrator.synthesis import GroundedSynthesizer
from research_assistant.pipeline.active_loop import ActiveLiteraturePipeline
from research_assistant.sufficiency.gate import evaluate_sufficiency

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """End-to-end orchestrator per AGENTS.md operational workflow."""

    def __init__(
        self,
        pipeline: ActiveLiteraturePipeline,
        llm: LLMClient,
        *,
        query_processor: QueryProcessor | None = None,
        synthesizer: GroundedSynthesizer | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.query_processor = query_processor or QueryProcessor(llm)
        self.synthesizer = synthesizer or GroundedSynthesizer(llm)

    def answer(self, user_query: str) -> ResearchResponse:
        request_id = str(uuid.uuid4())
        analysis = self.query_processor.analyze(user_query)
        logger.info(
            "request_id=%s normalized_query=%s subqueries=%s",
            request_id,
            analysis.normalized_query,
            analysis.subqueries,
        )

        subquery_results: list[ActiveResearchResult] = []
        unsupported_aspects: list[str] = []
        merged_hits = _merge_hits([])

        for subquery in analysis.subqueries:
            result = self.pipeline.run(subquery)
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

        answer, citations_valid, citation_errors = self.synthesizer.synthesize(
            analysis.original_query,
            merged_hits,
            sufficiency=overall_sufficiency,
            unsupported_aspects=unsupported_aspects or None,
        )

        return ResearchResponse(
            request_id=request_id,
            query=analysis.original_query,
            normalized_query=analysis.normalized_query,
            query_type=analysis.query_type,
            subqueries=analysis.subqueries,
            answer=answer,
            citations_valid=citations_valid,
            citation_errors=citation_errors,
            subquery_results=subquery_results,
            evidence_hits=merged_hits,
            sufficient=overall_sufficiency.sufficient and not unsupported_aspects,
            insufficient_message=answer if answer.startswith("INSUFFICIENT_EVIDENCE:") else None,
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
