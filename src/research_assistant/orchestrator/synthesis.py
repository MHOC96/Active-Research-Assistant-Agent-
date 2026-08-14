"""Citation-grounded answer synthesis."""

from __future__ import annotations

from research_assistant.citations.validator import validate_citations
from research_assistant.models import RetrievalHit, SufficiencyResult
from research_assistant.orchestrator.llm import LLMClient

SYNTHESIS_SYSTEM_PROMPT = """You are an academic research assistant.

Rules:
1. Answer research claims ONLY using the provided retrieved passages.
2. Every evidence-dependent technical claim MUST include an inline citation using EXACTLY this format: [arXiv:<ID> | Chunk <N>].
3. Never invent citations, papers, authors, metrics, dates, or experimental results.
4. Preserve important technical tokens exactly (API names, registers, equations, identifiers, numbers from sources).
5. If evidence is insufficient for part of the question, state: INSUFFICIENT_EVIDENCE: <specific missing evidence>.
6. Do not claim a source was consulted unless it appears in the evidence block.
7. If sources disagree, report both claims with their citations.
"""


class GroundedSynthesizer:
    """Generate citation-grounded answers from retrieved evidence."""

    def __init__(self, llm: LLMClient, *, max_regenerations: int = 1) -> None:
        self.llm = llm
        self.max_regenerations = max_regenerations

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

        prompt = _build_prompt(query, hits, unsupported_aspects)
        answer = self.llm.complete(system=SYNTHESIS_SYSTEM_PROMPT, user=prompt)

        for attempt in range(self.max_regenerations + 1):
            valid, errors = validate_citations(answer, hits)
            if valid:
                return answer, True, []
            if attempt >= self.max_regenerations:
                return answer, False, errors
            answer = self.llm.complete(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=(
                    f"{prompt}\n\nYour previous answer had invalid citations:\n"
                    + "\n".join(errors)
                    + "\nRegenerate using ONLY valid citation identifiers from the evidence."
                ),
            )

        return answer, False, errors


def _build_prompt(
    query: str,
    hits: list[RetrievalHit],
    unsupported_aspects: list[str] | None,
) -> str:
    evidence_blocks = []
    for hit in hits:
        evidence_blocks.append(
            "\n".join(
                [
                    f"Provenance: {hit.provenance}",
                    f"Title: {hit.title}",
                    f"Section: {hit.section or 'N/A'}",
                    f"Page: {hit.page if hit.page is not None else 'N/A'}",
                    f"Passage: {hit.passage}",
                ]
            )
        )

    unsupported = ""
    if unsupported_aspects:
        unsupported = (
            "\nUnsupported aspects (must acknowledge explicitly):\n"
            + "\n".join(f"- {item}" for item in unsupported_aspects)
        )

    return (
        f"User query:\n{query}\n\n"
        f"Evidence sufficiency: sufficient\n\n"
        f"Retrieved evidence:\n\n"
        + "\n\n---\n\n".join(evidence_blocks)
        + unsupported
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
