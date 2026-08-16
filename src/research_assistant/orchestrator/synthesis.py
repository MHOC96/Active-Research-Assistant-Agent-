"""Citation-grounded answer synthesis."""

from __future__ import annotations

from research_assistant.citations.validator import validate_citations
from research_assistant.config import Settings, get_settings
from research_assistant.models import RetrievalHit, SufficiencyResult
from research_assistant.orchestrator.llm import LLMClient
from research_assistant.utils.token_efficiency import truncate_passage

SYNTHESIS_SYSTEM_PROMPT = """Academic research assistant. Use ONLY provided evidence.

Rules:
1. Evidence-dependent claims need citations: [arXiv:<ID> | Chunk <N>]
2. Never invent citations, metrics, or results
3. Preserve technical tokens exactly
4. If evidence is missing, write: INSUFFICIENT_EVIDENCE: <what is missing>
5. Report disagreements with both citations
"""


class GroundedSynthesizer:
    """Generate citation-grounded answers from retrieved evidence."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_regenerations: int = 1,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.max_regenerations = max_regenerations
        self.settings = settings or get_settings()

    def synthesize(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        sufficiency: SufficiencyResult,
        unsupported_aspects: list[str] | None = None,
    ) -> tuple[str, bool, list[str]]:
        if not sufficiency.sufficient or not hits:
            message = _insufficient_message(sufficiency, unsupported_aspects)
            return message, True, []

        prompt = _build_prompt(query, hits, unsupported_aspects, self.settings)
        answer = self.llm.complete(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=self.settings.groq_synthesis_max_output_tokens,
        )

        for attempt in range(self.max_regenerations + 1):
            valid, errors = validate_citations(answer, hits)
            if valid:
                return answer, True, []
            if attempt >= self.max_regenerations:
                return answer, False, errors
            answer = self.llm.complete(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=_build_retry_prompt(answer, hits, errors),
                max_tokens=self.settings.groq_synthesis_max_output_tokens,
            )

        return answer, False, errors


def _format_evidence_block(hit: RetrievalHit, max_passage_chars: int) -> str:
    page = hit.page if hit.page is not None else "-"
    section = hit.section or "-"
    passage = truncate_passage(hit.passage, max_passage_chars)
    return f"{hit.provenance} | {hit.title} | p.{page} | {section}\n{passage}"


def _build_prompt(
    query: str,
    hits: list[RetrievalHit],
    unsupported_aspects: list[str] | None,
    settings: Settings,
) -> str:
    evidence_blocks = [
        _format_evidence_block(hit, settings.synthesis_max_passage_chars) for hit in hits
    ]

    unsupported = ""
    if unsupported_aspects:
        unsupported = (
            "\nUnsupported aspects:\n" + "\n".join(f"- {item}" for item in unsupported_aspects)
        )

    return (
        f"Query:\n{query}\n\n"
        f"Evidence:\n\n"
        + "\n\n---\n\n".join(evidence_blocks)
        + unsupported
    )


def _build_retry_prompt(answer: str, hits: list[RetrievalHit], errors: list[str]) -> str:
    valid_ids = "\n".join(f"- {hit.provenance}" for hit in hits)
    return (
        "Fix invalid citations in the answer below.\n"
        f"Valid citation IDs:\n{valid_ids}\n\n"
        f"Citation errors:\n" + "\n".join(f"- {error}" for error in errors) + "\n\n"
        f"Prior answer:\n{answer}\n\n"
        "Return the corrected answer only."
    )


def _insufficient_message(
    sufficiency: SufficiencyResult,
    unsupported_aspects: list[str] | None,
) -> str:
    if unsupported_aspects:
        joined = "; ".join(unsupported_aspects)
        return f"INSUFFICIENT_EVIDENCE: {joined}"
    reason = sufficiency.reason or "retrieved evidence did not meet sufficiency thresholds"
    return f"INSUFFICIENT_EVIDENCE: {reason}"
