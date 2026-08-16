"""Shared relevance scoring for discovered papers."""

from __future__ import annotations

import re

from research_assistant.models import DiscoveredPaper


def relevance_score(query: str, paper: DiscoveredPaper) -> float:
    terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9_.-]+", query) if len(term) > 2}
    if not terms:
        return 0.0

    corpus = f"{paper.title} {paper.abstract}".lower()
    matches = sum(1 for term in terms if term in corpus)
    return matches / len(terms)


def select_top_papers(
    query: str,
    papers: list[DiscoveredPaper],
    *,
    max_select: int,
) -> list[DiscoveredPaper]:
    if not papers or max_select <= 0:
        return []

    scored = [(paper, relevance_score(query, paper)) for paper in papers]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [paper for paper, score in scored[:max_select] if score > 0]
