"""Document-level metadata and ingestion state store."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from research_assistant.models import DocumentRecord, DocumentStatus


class MetadataStore:
    """SQLite store for document ingestion state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content_hash TEXT,
                source_url TEXT,
                status TEXT NOT NULL,
                authors_json TEXT,
                published_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def upsert_document(self, doc: DocumentRecord) -> None:
        now = self._now()
        row = self._conn.execute(
            "SELECT created_at FROM documents WHERE document_id = ?", (doc.document_id,)
        ).fetchone()
        created = row["created_at"] if row else now

        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, arxiv_id, title, content_hash, source_url, status,
             authors_json, published_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.document_id,
                doc.arxiv_id,
                doc.title,
                doc.content_hash,
                doc.source_url,
                doc.status.value,
                json.dumps(doc.authors),
                doc.published_date,
                created,
                now,
            ),
        )
        self._conn.commit()

    def update_status(self, document_id: str, status: DocumentStatus) -> None:
        self._conn.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?",
            (status.value, self._now(), document_id),
        )
        self._conn.commit()

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def get_by_arxiv_id(self, arxiv_id: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def get_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def is_ingested(self, arxiv_id: str) -> bool:
        doc = self.get_by_arxiv_id(arxiv_id)
        return doc is not None and doc.status == DocumentStatus.INGESTED

    def list_by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE status = ?", (status.value,)
        ).fetchall()
        return [self._row_to_document(row) for row in rows]

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            content_hash=row["content_hash"],
            source_url=row["source_url"],
            status=DocumentStatus(row["status"]),
            authors=json.loads(row["authors_json"] or "[]"),
            published_date=row["published_date"],
        )

    @staticmethod
    def normalize_arxiv_id(arxiv_id: str) -> str:
        """Normalize arXiv ID by removing version suffix."""
        return arxiv_id.split("v")[0].strip()
