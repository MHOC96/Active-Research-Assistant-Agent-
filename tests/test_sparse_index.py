"""Tests for SQLite FTS5 sparse index."""

from research_assistant.models import ChunkRecord, ContentType
from research_assistant.storage.sparse_index import SparseIndex


def _sample_chunk(idx: int = 0, passage: str = "transformer attention mechanism") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=ChunkRecord.make_chunk_id("2407.08608", idx),
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        section="Introduction",
        page=1,
        chunk_index=idx,
        content_type=ContentType.PARAGRAPH,
        passage=passage,
    )


def test_upsert_and_search(sparse_index: SparseIndex):
    sparse_index.upsert_chunks([_sample_chunk()])
    results = sparse_index.search("transformer attention")
    assert len(results) >= 1
    assert results[0][0] == "2407.08608:0"


def test_arxiv_id_exact_match(sparse_index: SparseIndex):
    sparse_index.upsert_chunks(
        [_sample_chunk(passage="See arXiv paper 2407.08608 for details.")]
    )
    results = sparse_index.search("2407.08608")
    assert any(r[0] == "2407.08608:0" for r in results)


def test_chunk_exists(sparse_index: SparseIndex):
    chunk = _sample_chunk()
    sparse_index.upsert_chunks([chunk])
    assert sparse_index.chunk_exists(chunk.chunk_id) is True
    assert sparse_index.chunk_exists("missing:0") is False


def test_delete_document(sparse_index: SparseIndex):
    sparse_index.upsert_chunks([_sample_chunk(0), _sample_chunk(1)])
    sparse_index.delete_document("2407.08608")
    assert sparse_index.chunk_exists("2407.08608:0") is False
    assert sparse_index.search("transformer") == []
