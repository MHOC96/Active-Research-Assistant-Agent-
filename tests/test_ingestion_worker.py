"""Tests for ingestion worker."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_assistant.config import Settings
from research_assistant.ingestion.downloader import DownloadResult
from research_assistant.ingestion.worker import IngestionWorker
from research_assistant.models import ContentType, DocumentStatus, ParsedElement
from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.metadata_store import MetadataStore
from research_assistant.storage.sparse_index import SparseIndex


class MockEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 + i * 0.001 for i in range(768)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture
def ingestion_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(
        PERSIST_DIRECTORY=str(data / "chroma"),
        SQLITE_SPARSE_DB=str(data / "sparse.db"),
        METADATA_DB=str(data / "metadata.db"),
        DOWNLOAD_CACHE_DIR=str(data / "downloads"),
        EMBEDDING_DIMENSION=768,
        MIN_CHUNK_CHARACTERS=10,
    )
    monkeypatch.setattr("research_assistant.storage.dense_index.get_settings", lambda: settings)
    monkeypatch.setattr("research_assistant.ingestion.worker.get_settings", lambda: settings)

    metadata = MetadataStore(data / "metadata.db")
    sparse = SparseIndex(data / "sparse.db")
    dense = DenseIndex(settings)

    yield settings, metadata, sparse, dense

    metadata.close()
    sparse.close()


def test_ingestion_worker_success(ingestion_env, tmp_path: Path):
    settings, metadata, sparse, dense = ingestion_env

    pdf_path = settings.download_cache_dir / "2407.08608.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    downloader = MagicMock()
    downloader.download.return_value = DownloadResult(
        path=pdf_path,
        content_hash="hash123",
        source_url="https://arxiv.org/pdf/2407.08608.pdf",
        size_bytes=10,
    )

    parser = MagicMock()
    parser.parse.return_value = [
        ParsedElement(
            text="Transformer attention enables parallel sequence modeling.",
            content_type=ContentType.PARAGRAPH,
            section="Introduction",
            page=1,
        )
    ]

    worker = IngestionWorker(
        metadata,
        dense,
        sparse,
        MockEmbedder(),
        downloader=downloader,
        parser=parser,
        settings=settings,
    )

    result = worker.ingest_pdf_document(
        arxiv_id="2407.08608",
        pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
        title="Attention Paper",
        authors=["Author"],
    )

    assert result.status == DocumentStatus.INGESTED
    assert result.chunk_count == 1
    assert metadata.is_ingested("2407.08608")
    assert sparse.chunk_exists("2407.08608:0")


def test_ingestion_worker_skips_existing_document(ingestion_env):
    settings, metadata, sparse, dense = ingestion_env

    from research_assistant.models import DocumentRecord

    metadata.upsert_document(
        DocumentRecord(
            document_id="2407.08608",
            arxiv_id="2407.08608",
            title="Attention Paper",
            status=DocumentStatus.INGESTED,
        )
    )

    worker = IngestionWorker(
        metadata,
        dense,
        sparse,
        MockEmbedder(),
        downloader=MagicMock(),
        parser=MagicMock(),
        settings=settings,
    )

    result = worker.ingest_pdf_document(
        arxiv_id="2407.08608v2",
        pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
        title="Attention Paper",
    )

    assert result.skipped is True
    assert result.status == DocumentStatus.INGESTED


def test_ingestion_worker_marks_failed_on_parse_error(ingestion_env, tmp_path: Path):
    settings, metadata, sparse, dense = ingestion_env

    pdf_path = settings.download_cache_dir / "2407.08608.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")

    downloader = MagicMock()
    downloader.download.return_value = DownloadResult(
        path=pdf_path,
        content_hash="hash456",
        source_url="https://arxiv.org/pdf/2407.08608.pdf",
        size_bytes=10,
    )

    parser = MagicMock()
    parser.parse.side_effect = RuntimeError("PDF_PARSE_FAILED: corrupt")

    worker = IngestionWorker(
        metadata,
        dense,
        sparse,
        MockEmbedder(),
        downloader=downloader,
        parser=parser,
        settings=settings,
    )

    result = worker.ingest_pdf_document(
        arxiv_id="2407.08608",
        pdf_url="https://arxiv.org/pdf/2407.08608.pdf",
        title="Attention Paper",
    )

    assert result.status == DocumentStatus.FAILED
    doc = metadata.get_document("2407.08608")
    assert doc is not None
    assert doc.status == DocumentStatus.FAILED
