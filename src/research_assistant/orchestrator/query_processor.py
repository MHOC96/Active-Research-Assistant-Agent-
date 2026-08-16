"""Query normalization, classification, and decomposition."""

from __future__ import annotations

import json
import re
from collections import OrderedDict

from pydantic import BaseModel, Field

from research_assistant.config import Settings, get_settings
from research_assistant.orchestrator.llm import LLMClient
from research_assistant.utils.token_efficiency import (
    cap_subqueries,
    heuristic_normalize_query,
    is_likely_simple_query,
)

QUERY_SYSTEM_PROMPT = """Return ONLY JSON:
{"normalized_query":"...","query_type":"simple"|"complex","subqueries":["..."]}

Rules:
- simple: one subquery equal to normalized_query
- complex: minimal subqueries that cover all required evidence aspects
- preserve technical tokens and identifiers
- do not answer the question
"""


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    query_type: str = "simple"
    subqueries: list[str] = Field(default_factory=list)


class QueryProcessor:
    """Normalize, classify, and decompose user queries via the orchestrator LLM."""

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self._analysis_cache: OrderedDict[str, QueryAnalysis] = OrderedDict()

    def analyze(self, query: str) -> QueryAnalysis:
        stripped = query.strip()
        if self.settings.skip_query_llm_for_simple and is_likely_simple_query(stripped):
            normalized = heuristic_normalize_query(stripped)
            return QueryAnalysis(
                original_query=stripped,
                normalized_query=normalized,
                query_type="simple",
                subqueries=[normalized],
            )

        cache_key = stripped.casefold()
        cached = self._analysis_cache_get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"original_query": stripped})

        raw = self.llm.complete(
            system=QUERY_SYSTEM_PROMPT,
            user=stripped,
            max_tokens=self.settings.groq_query_max_output_tokens,
        )
        analysis = _build_analysis(stripped, raw, self.settings.max_subqueries)
        self._analysis_cache_set(cache_key, analysis)
        return analysis

    def _analysis_cache_get(self, key: str) -> QueryAnalysis | None:
        if self.settings.query_analysis_cache_size <= 0:
            return None
        cached = self._analysis_cache.get(key)
        if cached is None:
            return None
        self._analysis_cache.move_to_end(key)
        return cached

    def _analysis_cache_set(self, key: str, analysis: QueryAnalysis) -> None:
        if self.settings.query_analysis_cache_size <= 0:
            return
        self._analysis_cache[key] = analysis
        self._analysis_cache.move_to_end(key)
        while len(self._analysis_cache) > self.settings.query_analysis_cache_size:
            self._analysis_cache.popitem(last=False)


def _build_analysis(query: str, raw: str, max_subqueries: int) -> QueryAnalysis:
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
    else:
        subqueries = cap_subqueries(subqueries, max_subqueries, normalized)
    return QueryAnalysis(
        original_query=query,
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
