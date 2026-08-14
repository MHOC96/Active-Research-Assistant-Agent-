"""Docling-based PDF parser."""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import (
    FormulaItem,
    ListItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import DocItemLabel

from research_assistant.models import ContentType, ParsedElement


class DoclingParser:
    """Parse PDFs into structured elements preserving sections and content types."""

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        self._converter = converter or DocumentConverter()

    def parse(self, pdf_path: Path) -> list[ParsedElement]:
        try:
            result = self._converter.convert(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"PDF_PARSE_FAILED: {exc}") from exc

        if result.document is None:
            raise RuntimeError("PDF_PARSE_FAILED: empty document")

        return self.extract_elements(result.document)

    def extract_elements(self, document: DoclingDocument) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        section: str | None = None
        subsection: str | None = None

        for item, _level in document.iterate_items():
            if isinstance(item, TitleItem):
                section = item.text.strip() or section
                subsection = None
                elements.append(
                    ParsedElement(
                        text=item.text.strip(),
                        content_type=ContentType.HEADING,
                        section=section,
                        subsection=subsection,
                        page=_page_no(item),
                    )
                )
                continue

            if isinstance(item, SectionHeaderItem):
                heading = item.text.strip()
                if item.level <= 1:
                    section = heading
                    subsection = None
                else:
                    subsection = heading
                elements.append(
                    ParsedElement(
                        text=heading,
                        content_type=ContentType.HEADING,
                        section=section,
                        subsection=subsection,
                        page=_page_no(item),
                    )
                )
                continue

            text, content_type = _item_text_and_type(item, document)
            if not text.strip():
                continue

            elements.append(
                ParsedElement(
                    text=text.strip(),
                    content_type=content_type,
                    section=section,
                    subsection=subsection,
                    page=_page_no(item),
                )
            )

        return elements


def _page_no(item: object) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return prov[0].page_no
    return None


def _item_text_and_type(item: object, document: DoclingDocument) -> tuple[str, ContentType]:
    if isinstance(item, TableItem):
        return item.export_to_markdown(document), ContentType.TABLE

    if isinstance(item, FormulaItem):
        return item.text.strip(), ContentType.EQUATION

    if isinstance(item, ListItem):
        return item.text.strip(), ContentType.LIST

    if isinstance(item, TextItem):
        label = item.label
        if label == DocItemLabel.CAPTION:
            return item.text.strip(), ContentType.FIGURE_CAPTION
        if label in {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}:
            return item.text.strip(), ContentType.PARAGRAPH
        return item.text.strip(), ContentType.PARAGRAPH

    text = getattr(item, "text", "") or ""
    return str(text).strip(), ContentType.PARAGRAPH
