"""Transactional dual-index writes with rollback on failure."""

from __future__ import annotations

from research_assistant.models import ChunkRecord
from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.sparse_index import SparseIndex


class IndexTransaction:
    """Commit chunks to both ChromaDB and SQLite FTS5 atomically."""

    def __init__(self, dense: DenseIndex, sparse: SparseIndex) -> None:
        self.dense = dense
        self.sparse = sparse

    def commit_chunks(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]], *, document_id: str
    ) -> None:
        if not chunks:
            return

        try:
            self.sparse.upsert_chunks(chunks)
            self.dense.upsert_chunks(chunks, embeddings)
        except Exception:
            self.rollback_document(document_id)
            raise RuntimeError("INDEX_TRANSACTION_FAILED")

    def rollback_document(self, document_id: str) -> None:
        try:
            self.sparse.delete_document(document_id)
        except Exception:
            pass
        try:
            self.dense.delete_document(document_id)
        except Exception:
            pass

    def delete_document(self, document_id: str) -> None:
        self.rollback_document(document_id)
