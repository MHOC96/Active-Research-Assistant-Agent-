"""Section-aware token chunking."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from research_assistant.config import Settings, get_settings
from research_assistant.models import ChunkRecord, ContentType, ParsedElement


@dataclass
class _SectionGroup:
    section: str | None
    subsection: str | None
    elements: list[ParsedElement]


class SectionAwareChunker:
    """Chunk parsed elements by section with token size limits and overlap."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(
        self,
        elements: list[ParsedElement],
        *,
        document_id: str,
        arxiv_id: str,
        title: str,
        authors: list[str] | None = None,
        published_date: str | None = None,
    ) -> list[ChunkRecord]:
        if not elements:
            raise RuntimeError("CHUNKING_FAILED: no parsed elements")

        groups = self._group_by_section(elements)
        chunks: list[ChunkRecord] = []
        chunk_index = 0

        for group in groups:
            passages = self._chunk_group(group.elements)
            for passage, content_type, page in passages:
                if len(passage) < self.settings.min_chunk_characters:
                    continue
                chunks.append(
                    ChunkRecord(
                        chunk_id=ChunkRecord.make_chunk_id(document_id, chunk_index),
                        document_id=document_id,
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors or [],
                        published_date=published_date,
                        section=group.section,
                        subsection=group.subsection,
                        page=page,
                        chunk_index=chunk_index,
                        content_type=content_type,
                        passage=passage,
                        embedding_model=self.settings.gemini_embedding_model,
                        embedding_dimension=self.settings.embedding_dimension,
                    )
                )
                chunk_index += 1

        if not chunks:
            raise RuntimeError("CHUNKING_FAILED: no chunks met minimum size requirements")

        return chunks

    def _group_by_section(self, elements: list[ParsedElement]) -> list[_SectionGroup]:
        groups: list[_SectionGroup] = []
        current: _SectionGroup | None = None

        for element in elements:
            key = (element.section, element.subsection)
            if current is None or (current.section, current.subsection) != key:
                current = _SectionGroup(
                    section=element.section,
                    subsection=element.subsection,
                    elements=[],
                )
                groups.append(current)
            current.elements.append(element)

        return groups

    def _chunk_group(
        self, elements: list[ParsedElement]
    ) -> list[tuple[str, ContentType, int | None]]:
        paragraphs = [e.text for e in elements if e.text.strip()]
        if not paragraphs:
            return []

        default_type = elements[0].content_type
        default_page = next((e.page for e in elements if e.page is not None), None)

        tokens: list[int] = []
        for idx, paragraph in enumerate(paragraphs):
            tokens.extend(self._encoding.encode(paragraph))
            if idx < len(paragraphs) - 1:
                tokens.extend(self._encoding.encode("\n\n"))

        target = self.settings.chunk_target_tokens
        max_tokens = self.settings.chunk_max_tokens
        overlap = self.settings.chunk_overlap_tokens

        passages: list[tuple[str, ContentType, int | None]] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            if end - start > max_tokens:
                end = start + max_tokens

            if end < len(tokens) and end - start < target:
                end = min(start + target, len(tokens))

            chunk_tokens = tokens[start:end]
            passage = self._encoding.decode(chunk_tokens).strip()
            if passage:
                passages.append((passage, default_type, default_page))

            if end >= len(tokens):
                break
            start = max(end - overlap, start + 1)

        return passages
