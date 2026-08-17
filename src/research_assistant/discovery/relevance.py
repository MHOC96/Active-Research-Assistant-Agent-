"""Shared relevance scoring for discovered papers and indexed hits."""

from __future__ import annotations

import re

from research_assistant.models import DiscoveredPaper, ExternalCitation, RetrievalHit

# Terms that match too many unrelated academic / business papers on their own.
_GENERIC_TERMS = frozenset(
    {
        "management",
        "manager",
        "managers",
        "managerial",
        "managing",
        "business",
        "owner",
        "owners",
        "simple",
        "task",
        "tasks",
        "experience",
        "knowledge",
        "problem",
        "problems",
        "organizational",
        "organization",
        "structure",
        "structures",
        "development",
        "digital",
        "transformation",
        "framework",
        "frameworks",
        "model",
        "models",
        "study",
        "studies",
        "research",
        "analysis",
        "system",
        "systems",
        "approach",
        "approaches",
        "process",
        "processes",
        "effective",
        "efficient",
        "need",
        "needs",
        "become",
        "feel",
        "frustrated",
        "deal",
        "dealing",
        "skills",
        "success",
        "successful",
        "decision",
        "decisions",
        "career",
        "until",
        "because",
        "when",
        "they",
        "will",
        "like",
        "adapt",
        "adaptation",
        "complex",
        "product",
        "architecture",
    }
)


def relevance_score(query: str, paper: DiscoveredPaper) -> float:
    """Score query-to-paper overlap with down-weighted generic academic terms."""
    terms = _content_terms(query)
    if not terms:
        return 0.0

    corpus = f"{paper.title} {paper.abstract}".lower()
    distinctive = [term for term in terms if term not in _GENERIC_TERMS]

    if distinctive:
        distinctive_matches = sum(1 for term in distinctive if term in corpus)
        generic_terms = [term for term in terms if term in _GENERIC_TERMS]
        generic_matches = sum(1 for term in generic_terms if term in corpus)

        bigram_bonus = sum(
            0.15 for bigram in _bigrams(distinctive) if bigram in corpus
        )

        if distinctive_matches == 0 and bigram_bonus == 0:
            if generic_matches > 0:
                return min(
                    0.25,
                    (generic_matches / max(len(generic_terms), 1)) * 0.25,
                )
            return 0.0

        distinctive_score = distinctive_matches / len(distinctive)
        generic_score = generic_matches / len(generic_terms) if generic_terms else 0.0
        score = 0.75 * distinctive_score + 0.25 * min(generic_score, 1.0) + bigram_bonus
        return min(1.0, score)

    matches = sum(1 for term in terms if term in corpus)
    return (matches / len(terms)) * 0.35


def hit_relevance_score(query: str, hit: RetrievalHit) -> float:
    """Score an indexed retrieval hit against a query using title and passage text."""
    paper = DiscoveredPaper(
        source="arxiv",
        external_id=hit.arxiv_id,
        title=hit.title,
        authors=hit.authors,
        abstract=hit.passage[:600],
    )
    return relevance_score(query, paper)


def best_hit_relevance(hit: RetrievalHit, queries: list[str]) -> float:
    """Return the highest relevance score across one or more query strings."""
    if not queries:
        return 0.0
    return max(hit_relevance_score(query, hit) for query in queries if query.strip())


def select_top_papers(
    query: str,
    papers: list[DiscoveredPaper],
    *,
    max_select: int,
    min_score: float = 0.0,
) -> list[DiscoveredPaper]:
    if not papers or max_select <= 0:
        return []

    scored = [(paper, relevance_score(query, paper)) for paper in papers]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [paper for paper, score in scored[:max_select] if score >= min_score]


def filter_external_citations(
    citations: list[ExternalCitation],
    *,
    min_score: float,
) -> list[ExternalCitation]:
    """Keep external citations that meet the minimum relevance threshold."""
    return [citation for citation in citations if citation.relevance_score >= min_score]


def _content_terms(query: str) -> list[str]:
    return [
        term.lower()
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", query)
        if len(term) > 2
    ]


def _bigrams(terms: list[str]) -> list[str]:
    return [f"{terms[index]} {terms[index + 1]}" for index in range(len(terms) - 1)]
