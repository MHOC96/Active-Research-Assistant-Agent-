"""Tests for ChromaDB dense index."""

import pytest

from research_assistant.models import ChunkRecord, ContentType
from research_assistant.storage.dense_index import DenseIndex


def _chunk(idx: int = 0) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=ChunkRecord.make_chunk_id("2407.08608", idx),
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Dense Test",
        chunk_index=idx,
        content_type=ContentType.PARAGRAPH,
        passage=f"embedding test passage {idx}",
    )


def _embedding(seed: float = 0.1) -> list[float]:
    return [seed + i * 0.001 for i in range(768)]


def test_upsert_and_search(dense_index: DenseIndex):
    chunks = [_chunk(0), _chunk(1)]
    embeddings = [_embedding(0.1), _embedding(0.2)]
    dense_index.upsert_chunks(chunks, embeddings)

    results = dense_index.search(_embedding(0.1), limit=2)
    assert len(results) == 2
    assert results[0][0] in {"2407.08608:0", "2407.08608:1"}


def test_rejects_dimension_mismatch(dense_index: DenseIndex):
    with pytest.raises(ValueError, match="embedding dimension"):
        dense_index.upsert_chunks([_chunk()], [[0.1, 0.2]])


def test_delete_document(dense_index: DenseIndex):
    dense_index.upsert_chunks([_chunk()], [_embedding()])
    dense_index.delete_document("2407.08608")
    results = dense_index.search(_embedding(), limit=5)
    assert results == []
