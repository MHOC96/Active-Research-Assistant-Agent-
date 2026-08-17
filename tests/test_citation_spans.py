"""Tests for citation span mapping."""

from research_assistant.citations.spans import build_citation_spans
from research_assistant.citations.styles import CitationStyle
from research_assistant.models import (
    ActiveResearchResult,
    ExternalCitation,
    HybridRetrieveResult,
    RetrievalHit,
    SufficiencyResult,
)


def _result(query: str, *, external: list[ExternalCitation] | None = None) -> ActiveResearchResult:
    return ActiveResearchResult(
        query=query,
        request_id="req-1",
        retrieval=HybridRetrieveResult(
            query=query,
            candidates=[],
            sufficiency=SufficiencyResult(
                sufficient=False,
                candidate_count=0,
                top_score=0.2,
            ),
        ),
        external_citations=external or [],
    )


def test_build_citation_spans_for_pasted_paragraph():
    paragraph = (
        "Cloud computing relies on containerization for deployment. "
        "Container engines enable horizontal scaling. "
        "Orchestration frameworks handle rolling updates."
    )
    subqueries = [
        "cloud computing containerization deployment",
        "container engines horizontal scaling",
        "container orchestration rolling updates",
    ]
    results = [
        _result(
            subqueries[0],
            external=[
                ExternalCitation(
                    source="web",
                    source_label="Web",
                    title="Cloud Containers",
                    url="https://example.com/cloud",
                    publisher="Example",
                    relevance_score=0.8,
                )
            ],
        ),
        _result(
            subqueries[1],
            external=[
                ExternalCitation(
                    source="openalex",
                    source_label="OpenAlex",
                    title="Container Scaling Study",
                    url="https://openalex.org/W1",
                    relevance_score=0.75,
                )
            ],
        ),
        _result(subqueries[2]),
    ]

    spans = build_citation_spans(
        paragraph,
        subqueries,
        results,
        CitationStyle.APA7,
        min_rerank_score=0.7,
    )

    assert len(spans) == 3
    assert spans[0].text.startswith("Cloud computing")
    assert spans[0].citations[0].source == "web"
    assert spans[1].citations[0].source == "openalex"
    assert spans[0].start < spans[1].start < spans[2].start


def test_build_citation_spans_merges_single_question_with_multiple_subqueries():
    query = "Compare RAG and GraphRAG accuracy"
    subqueries = ["RAG accuracy", "GraphRAG accuracy"]
    hit = RetrievalHit(
        chunk_id="2407.08608:0",
        passage="Graph RAG accuracy results.",
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Graph RAG Survey",
        chunk_index=0,
        rerank_score=0.91,
    )
    results = [
        ActiveResearchResult(
            query=subqueries[0],
            request_id="req-1",
            retrieval=HybridRetrieveResult(
                query=subqueries[0],
                candidates=[hit],
                sufficiency=SufficiencyResult(sufficient=True, candidate_count=1, top_score=0.91),
            ),
        ),
        _result(subqueries[1]),
    ]

    spans = build_citation_spans(
        query,
        subqueries,
        results,
        CitationStyle.APA7,
        min_rerank_score=0.7,
    )

    assert len(spans) == 1
    assert spans[0].text == query
    assert len(spans[0].citations) == 1
