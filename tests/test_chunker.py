"""Tests for section-aware chunking."""

import pytest

from research_assistant.config import Settings
from research_assistant.ingestion.chunker import SectionAwareChunker
from research_assistant.models import ContentType, ParsedElement


def test_chunker_splits_long_section():
    settings = Settings(
        CHUNK_TARGET_TOKENS=20,
        CHUNK_MAX_TOKENS=30,
        CHUNK_OVERLAP_TOKENS=5,
        MIN_CHUNK_CHARACTERS=20,
    )
    chunker = SectionAwareChunker(settings)
    elements = [
        ParsedElement(
            text=" ".join(["transformer"] * 80),
            content_type=ContentType.PARAGRAPH,
            section="Introduction",
        )
    ]

    chunks = chunker.chunk(
        elements,
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Paper",
    )

    assert len(chunks) >= 2
    assert all(c.section == "Introduction" for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunker_preserves_section_metadata():
    settings = Settings(MIN_CHUNK_CHARACTERS=10)
    chunker = SectionAwareChunker(settings)
    elements = [
        ParsedElement(
            text="Methods overview for the experiment.",
            content_type=ContentType.PARAGRAPH,
            section="Methods",
            subsection="Setup",
            page=3,
        )
    ]

    chunks = chunker.chunk(
        elements,
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Paper",
    )

    assert len(chunks) == 1
    assert chunks[0].section == "Methods"
    assert chunks[0].subsection == "Setup"
    assert chunks[0].page == 3


def test_chunker_rejects_empty_elements():
    chunker = SectionAwareChunker(Settings(MIN_CHUNK_CHARACTERS=80))
    with pytest.raises(RuntimeError, match="CHUNKING_FAILED"):
        chunker.chunk([], document_id="2407.08608", arxiv_id="2407.08608", title="Paper")
