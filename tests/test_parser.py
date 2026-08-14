"""Tests for Docling parser."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_assistant.ingestion.parser import DoclingParser
from research_assistant.models import ContentType, ParsedElement


def test_parse_delegates_to_extract_elements(tmp_path: Path):
    parser = DoclingParser(converter=MagicMock())
    document = MagicMock()
    parser._converter.convert.return_value = MagicMock(document=document)
    expected = [
        ParsedElement(
            text="Parsed paragraph.",
            content_type=ContentType.PARAGRAPH,
            section="Intro",
        )
    ]
    parser.extract_elements = MagicMock(return_value=expected)

    result = parser.parse(tmp_path / "paper.pdf")

    parser._converter.convert.assert_called_once()
    parser.extract_elements.assert_called_once_with(document)
    assert result == expected


def test_parse_raises_on_converter_failure(tmp_path: Path):
    parser = DoclingParser(converter=MagicMock())
    parser._converter.convert.side_effect = ValueError("broken pdf")

    with pytest.raises(RuntimeError, match="PDF_PARSE_FAILED"):
        parser.parse(tmp_path / "paper.pdf")


def test_parse_raises_on_empty_document(tmp_path: Path):
    parser = DoclingParser(converter=MagicMock())
    parser._converter.convert.return_value = MagicMock(document=None)

    with pytest.raises(RuntimeError, match="PDF_PARSE_FAILED"):
        parser.parse(tmp_path / "paper.pdf")
