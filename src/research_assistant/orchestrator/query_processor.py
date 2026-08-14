"""Query normalization, classification, and decomposition."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from research_assistant.orchestrator.llm import LLMClient

QUERY_SYSTEM_PROMPT = """You analyze academic research queries.

Return ONLY valid JSON with this schema:
{
  "normalized_query": "clean technical query without conversational filler",
  "query_type": "simple" or "complex",
  "subqueries": ["list of independent evidence requirements"]
}

Rules:
- For simple factual questions, query_type must be "simple" and subqueries must contain one item equal to normalized_query.
- For multi-part comparison or multi-aspect questions, query_type must be "complex" and subqueries must list independent evidence requirements.
- Preserve technical tokens, identifiers, model names, and constraints.
- Do not answer the question.
"""


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    query_type: str = "simple"
    subqueries: list[str] = Field(default_factory=list)


class QueryProcessor:
    """Normalize, classify, and decompose user queries via the orchestrator LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
        raw = self.llm.complete(
            system=QUERY_SYSTEM_PROMPT,
            user=f"Query:\n{query.strip()}",
            max_tokens=512,
        )
        payload = _parse_json_payload(raw)
        normalized = str(payload.get("normalized_query") or query).strip()
        query_type = str(payload.get("query_type") or "simple").strip().lower()
        subqueries = payload.get("subqueries") or [normalized]
        if not isinstance(subqueries, list):
            subqueries = [normalized]
        subqueries = [str(item).strip() for item in subqueries if str(item).strip()]
        if not subqueries:
            subqueries = [normalized]
        if query_type != "complex":
            query_type = "simple"
            subqueries = [normalized]
        return QueryAnalysis(
            original_query=query.strip(),
            normalized_query=normalized,
            query_type=query_type,
            subqueries=subqueries,
        )


def _parse_json_payload(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {
            "normalized_query": text,
            "query_type": "simple",
            "subqueries": [text],
        }
