"""Tests for document metadata store."""

from research_assistant.models import DocumentRecord, DocumentStatus
from research_assistant.storage.metadata_store import MetadataStore


def test_upsert_and_get(metadata_store: MetadataStore):
    doc = DocumentRecord(
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Test Paper",
        status=DocumentStatus.INGESTED,
    )
    metadata_store.upsert_document(doc)
    loaded = metadata_store.get_document("2407.08608")
    assert loaded is not None
    assert loaded.title == "Test Paper"
    assert loaded.status == DocumentStatus.INGESTED


def test_is_ingested(metadata_store: MetadataStore):
    doc = DocumentRecord(
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Test",
        status=DocumentStatus.INGESTED,
    )
    metadata_store.upsert_document(doc)
    assert metadata_store.is_ingested("2407.08608") is True
    assert metadata_store.is_ingested("9999.99999") is False


def test_deduplication_by_content_hash(metadata_store: MetadataStore):
    doc = DocumentRecord(
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Test",
        content_hash="abc123",
        status=DocumentStatus.INGESTED,
    )
    metadata_store.upsert_document(doc)
    found = metadata_store.get_by_content_hash("abc123")
    assert found is not None
    assert found.arxiv_id == "2407.08608"


def test_normalize_arxiv_id():
    assert MetadataStore.normalize_arxiv_id("2407.08608v2") == "2407.08608"
