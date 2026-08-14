"""SQLite FTS5 sparse index with BM25 ranking."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from research_assistant.models import ChunkRecord


class SparseIndex:
    """BM25 sparse retrieval via SQLite FTS5."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                arxiv_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_metadata_document
                ON chunk_metadata(document_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                passage,
                document_id UNINDEXED,
                arxiv_id UNINDEXED,
                title UNINDEXED,
                section UNINDEXED,
                tokenize='porter unicode61'
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Insert or replace chunks in metadata and FTS tables."""
        try:
            for chunk in chunks:
                metadata = {
                    "authors": chunk.authors,
                    "published_date": chunk.published_date,
                    "subsection": chunk.subsection,
                    "content_type": chunk.content_type.value,
                    "source": chunk.source,
                    "embedding_model": chunk.embedding_model,
                    "embedding_dimension": chunk.embedding_dimension,
                }
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO chunk_metadata
                    (chunk_id, document_id, arxiv_id, title, section, page, chunk_index, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.arxiv_id,
                        chunk.title,
                        chunk.section,
                        chunk.page,
                        chunk.chunk_index,
                        json.dumps(metadata),
                    ),
                )
                self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                self._conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, passage, document_id, arxiv_id, title, section)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.passage,
                        chunk.document_id,
                        chunk.arxiv_id,
                        chunk.title,
                        chunk.section or "",
                    ),
                )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise RuntimeError(f"SQLITE_WRITE_FAILED: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        chunk_ids = [
            row["chunk_id"]
            for row in self._conn.execute(
                "SELECT chunk_id FROM chunk_metadata WHERE document_id = ?", (document_id,)
            )
        ]
        for chunk_id in chunk_ids:
            self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute("DELETE FROM chunk_metadata WHERE document_id = ?", (document_id,))
        self._conn.commit()

    def search(self, query: str, *, limit: int = 15) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs ranked by BM25 relevance."""
        sanitized = self._sanitize_query(query)
        if not sanitized:
            return []

        rows = self._conn.execute(
            """
            SELECT chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (sanitized, limit),
        ).fetchall()

        # FTS5 bm25 returns negative values; lower is better. Invert for rank ordering.
        return [(row["chunk_id"], float(row["score"])) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        meta = json.loads(row["metadata_json"])
        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "arxiv_id": row["arxiv_id"],
            "title": row["title"],
            "section": row["section"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            **meta,
        }

    def get_passage(self, chunk_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT passage FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return row["passage"] if row else None

    def chunk_exists(self, chunk_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return row is not None

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Prepare a safe FTS5 query string."""
        tokens = []
        for token in query.split():
            cleaned = "".join(c for c in token if c.isalnum() or c in "._-:")
            if cleaned:
                tokens.append(f'"{cleaned}"')
        return " OR ".join(tokens) if tokens else ""
