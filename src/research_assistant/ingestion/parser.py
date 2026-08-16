"""Docling-based PDF parser."""

from __future__ import annotations

import os
from pathlib import Path

# Docling's layout models invoke torch.compile on Windows, which requires MSVC.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
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

from research_assistant.config import Settings, get_settings
from research_assistant.models import ContentType, ParsedElement

_CONVERTER_CACHE: dict[bool, DocumentConverter] = {}


class DoclingParser:
    """Parse PDFs into structured elements preserving sections and content types."""

    def __init__(
        self,
        converter: DocumentConverter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._converter = converter or _get_converter(fast=self.settings.fast_ingestion)

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


def _get_converter(*, fast: bool) -> DocumentConverter:
    if fast not in _CONVERTER_CACHE:
        _CONVERTER_CACHE[fast] = _default_converter(fast=fast)
    return _CONVERTER_CACHE[fast]


def _default_converter(*, fast: bool = False) -> DocumentConverter:
    """Build a converter tuned for text-native arXiv PDFs on CPU."""
    pdf_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=not fast,
        force_backend_text=True,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )


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
