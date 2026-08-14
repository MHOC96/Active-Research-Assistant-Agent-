"""Embedding service protocol and Gemini implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingService(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages for indexing."""
