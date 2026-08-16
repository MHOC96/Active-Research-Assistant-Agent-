"""Tests for citation style formatting."""

import pytest

from research_assistant.citations.styles import (
    CitationStyle,
    format_answer_citations,
    format_in_text,
    format_reference,
    format_grouped_references_output,
    format_references_output,
    list_citation_styles,
    parse_citation_style,
)
from research_assistant.models import RetrievalHit


def _hit(**kwargs) -> RetrievalHit:
    defaults = {
        "chunk_id": "1809.04281:6",
        "passage": "attention text",
        "document_id": "1809.04281",
        "arxiv_id": "1809.04281",
        "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
        "chunk_index": 6,
        "authors": ["Zihang Dai", "Zhilin Yang", "Yiming Yang"],
        "published_date": "2019-01-09",
        "page": 4,
        "rerank_score": 0.9,
    }
    defaults.update(kwargs)
    return RetrievalHit(**defaults)


def test_list_citation_styles_includes_common_formats():
    styles = dict(list_citation_styles())
    assert "apa7" in styles
    assert "mla9" in styles
    assert "ieee" in styles


def test_parse_citation_style_aliases():
    assert parse_citation_style("apa") == CitationStyle.APA7
    assert parse_citation_style("APA-7") == CitationStyle.APA7
    assert parse_citation_style("chicago") == CitationStyle.CHICAGO17


def test_parse_unknown_style_raises():
    with pytest.raises(ValueError, match="Unknown citation style"):
        parse_citation_style("vancouver")


def test_format_in_text_apa7():
    citation = format_in_text(_hit(), CitationStyle.APA7)
    assert citation == "(Dai et al., 2019, p. 4)"


def test_format_in_text_ieee():
    citation = format_in_text(_hit(), CitationStyle.IEEE, ieee_number=2)
    assert citation == "[2]"


def test_format_reference_apa7():
    reference = format_reference(_hit(), CitationStyle.APA7)
    assert "Dai, Z." in reference
    assert "(2019)" in reference
    assert "https://arxiv.org/abs/1809.04281" in reference


def test_format_answer_citations_replaces_and_appends_references():
    answer = "Attention uses relative positions [arXiv:1809.04281 | Chunk 6]."
    formatted = format_answer_citations(answer, [_hit()], CitationStyle.APA7)
    assert "(Dai et al., 2019, p. 4)" in formatted
    assert "References" in formatted
    assert "Transformer-XL" in formatted


def test_format_references_output_mla9():
    hit = _hit(
        arxiv_id="2502.11371",
        title="RAG vs. GraphRAG: A Systematic Evaluation and Key Insights",
        authors=["Yujian Han", "Other Author"],
        published_date="2025-02-16",
    )
    output = format_references_output([hit], CitationStyle.MLA9)
    assert output.startswith("References\n\n")
    assert 'Han, and Author. "RAG vs. GraphRAG' in output
    assert "https://arxiv.org/abs/2502.11371" in output
    assert ".." not in output


def test_format_references_output_insufficient():
    assert "INSUFFICIENT_EVIDENCE" in format_references_output([], CitationStyle.MLA9)


def test_format_grouped_references_output_groups_by_source():
    from research_assistant.models import ExternalCitation

    hit = _hit()
    external = [
        ExternalCitation(
            source="openalex",
            source_label="OpenAlex",
            title="Knowledge Graph RAG",
            authors=["Alice Smith"],
            published_date="2024",
            url="https://openalex.org/W123",
            relevance_score=0.8,
        )
    ]
    output = format_grouped_references_output(
        [hit],
        external,
        CitationStyle.MLA9,
        source_order=["arxiv", "openalex"],
    )
    assert "From arXiv (indexed)" in output
    assert "From OpenAlex" in output
    assert "Knowledge Graph RAG" in output


def test_format_external_reference_web_corporate_apa():
    from research_assistant.citations.styles import format_external_in_text, format_external_reference
    from research_assistant.models import ExternalCitation

    citation = ExternalCitation(
        source="web",
        source_label="Web",
        title="IT Service Management",
        publisher="ServiceNow",
        published_date="2023",
        url="https://www.servicenow.com/products/itsm.html",
    )
    reference = format_external_reference(citation, CitationStyle.APA7)
    assert "ServiceNow. (2023)." in reference
    assert "IT Service Management" in reference
    assert "In-text: (ServiceNow, 2023)" in reference
    assert format_external_in_text(citation, CitationStyle.APA7) == "(ServiceNow, 2023)"


def test_internal_style_keeps_provenance_tags():
    answer = "Claim [arXiv:1809.04281 | Chunk 6]."
    formatted = format_answer_citations(answer, [_hit()], CitationStyle.INTERNAL)
    assert formatted == answer
