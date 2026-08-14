"""ChromaDB dense vector index."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from research_assistant.config import Settings, get_settings
from research_assistant.models import ChunkRecord


class DenseIndex:
    """Dense retrieval via ChromaDB with cosine distance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.settings.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dimension": self.settings.embedding_dimension,
                "embedding_version": self.settings.embedding_version,
            },
        )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        for emb in embeddings:
            if len(emb) != self.settings.embedding_dimension:
                raise ValueError(
                    f"embedding dimension {len(emb)} != configured "
                    f"{self.settings.embedding_dimension}"
                )

        try:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.passage for c in chunks],
                metadatas=[self._chunk_metadata(c) for c in chunks],
            )
        except Exception as exc:
            raise RuntimeError(f"CHROMA_WRITE_FAILED: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise RuntimeError(f"CHROMA_WRITE_FAILED: {exc}") from exc

    def search(
        self, query_embedding: list[float], *, limit: int = 15
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (chunk_id, similarity, metadata) sorted by similarity desc."""
        if len(query_embedding) != self.settings.embedding_dimension:
            raise ValueError("query embedding dimension mismatch")

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "documents", "distances"],
        )

        hits: list[tuple[str, float, dict[str, Any]]] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for chunk_id, distance, metadata in zip(ids, distances, metadatas, strict=True):
            similarity = 1.0 - float(distance)  # cosine distance -> similarity
            hits.append((chunk_id, similarity, metadata or {}))

        hits.sort(key=lambda h: h[1], reverse=True)
        return hits

    @staticmethod
    def _chunk_metadata(chunk: ChunkRecord) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "arxiv_id": chunk.arxiv_id,
            "title": chunk.title,
            "authors": ",".join(chunk.authors),
            "published_date": chunk.published_date or "",
            "section": chunk.section or "",
            "subsection": chunk.subsection or "",
            "page": chunk.page if chunk.page is not None else -1,
            "chunk_index": chunk.chunk_index,
            "content_type": chunk.content_type.value,
            "source": chunk.source,
            "embedding_model": chunk.embedding_model,
            "embedding_dimension": chunk.embedding_dimension,
        }
