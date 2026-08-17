"""Tests for discovery and output relevance filtering."""

from research_assistant.discovery.relevance import (
    best_hit_relevance,
    filter_external_citations,
    hit_relevance_score,
    relevance_score,
    select_top_papers,
)
from research_assistant.models import DiscoveredPaper, ExternalCitation, RetrievalHit


def _paper(title: str, abstract: str = "") -> DiscoveredPaper:
    return DiscoveredPaper(
        source="arxiv",
        external_id="test",
        title=title,
        authors=[],
        abstract=abstract or title,
    )


def test_generic_management_query_scores_low_for_unrelated_papers():
    query = "management simple task"
    digital = _paper(
        "Building organizational competence for digital transformation in SMEs",
        "organizational competence management SMEs digital transformation",
    )
    career = _paper(
        "The Three Axes of Success: Career Decision-Making",
        "career decision making success experience knowledge",
    )

    assert relevance_score(query, digital) < 0.35
    assert relevance_score(query, career) < 0.35


def test_entrepreneur_transition_query_prefers_matching_paper():
    query = "entrepreneur manager transition small business"
    match = _paper(
        "From entrepreneur to manager in family firms",
        "entrepreneur manager transition leadership small business owners",
    )
    unrelated = _paper(
        "Product architecture and organizational structure",
        "organizational structure product development complex systems",
    )

    assert relevance_score(query, match) >= 0.35
    assert relevance_score(query, unrelated) < relevance_score(query, match)


def test_filter_external_citations_drops_low_scores():
    citations = [
        ExternalCitation(
            source="web",
            source_label="Web",
            title="Weak match",
            url="https://example.com/weak",
            relevance_score=0.2,
        ),
        ExternalCitation(
            source="openalex",
            source_label="OpenAlex",
            title="Strong match",
            url="https://openalex.org/W1",
            relevance_score=0.8,
        ),
    ]

    filtered = filter_external_citations(citations, min_score=0.35)
    assert len(filtered) == 1
    assert filtered[0].title == "Strong match"


def test_select_top_papers_respects_minimum_score():
    papers = [
        _paper("Weak", "generic management overview"),
        _paper(
            "Entrepreneur manager transition",
            "entrepreneur manager transition small business leadership",
        ),
    ]
    selected = select_top_papers(
        "entrepreneur manager transition",
        papers,
        max_select=2,
        min_score=0.35,
    )
    assert len(selected) == 1
    assert "Entrepreneur" in selected[0].title


def test_hit_relevance_score_uses_passage_text():
    hit = RetrievalHit(
        chunk_id="2406.01615:0",
        passage="Digital transformation and organizational competence in SMEs.",
        document_id="2406.01615",
        arxiv_id="2406.01615",
        title="Building organizational competence for digital transformation in SMEs",
        chunk_index=0,
        rerank_score=0.82,
    )
    entrepreneur_query = (
        "Business owners are entrepreneurs until they become managers. "
        "Management needs knowledge and experience."
    )

    assert hit_relevance_score("management simple task", hit) < 0.35
    assert best_hit_relevance(hit, [entrepreneur_query, "entrepreneur manager transition"]) < 0.35
