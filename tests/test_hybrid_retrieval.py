"""Tests for hybrid retrieval pipeline."""

from __future__ import annotations

import pytest

from research_assistant.models import ChunkRecord, ContentType, RetrievalHit
from research_assistant.reranking.flashrank_reranker import FlashRankReranker
from research_assistant.retrieval.hybrid import HybridRetriever
from research_assistant.storage.index_transaction import IndexTransaction


class MockEmbedder:
    def embed_query(self, text: str) -> list[float]:
        seed = 0.15 if "attention" in text.lower() else 0.05
        return [seed + i * 0.001 for i in range(768)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class MockReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        top_k = top_k or 3
        query_terms = set(query.lower().split())
        scored = []
        for hit in candidates:
            overlap = sum(1 for term in query_terms if term in hit.passage.lower())
            score = 0.85 if overlap >= 2 else 0.35
            scored.append(hit.model_copy(update={"rerank_score": score}))
        scored.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        return scored[:top_k]


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=ChunkRecord.make_chunk_id("2407.08608", 0),
            document_id="2407.08608",
            arxiv_id="2407.08608",
            title="Attention Paper",
            section="Introduction",
            page=1,
            chunk_index=0,
            content_type=ContentType.PARAGRAPH,
            passage="The transformer attention mechanism enables parallel sequence modeling.",
        ),
        ChunkRecord(
            chunk_id=ChunkRecord.make_chunk_id("2407.08608", 1),
            document_id="2407.08608",
            arxiv_id="2407.08608",
            title="Attention Paper",
            section="Methods",
            page=2,
            chunk_index=1,
            content_type=ContentType.PARAGRAPH,
            passage="Optimization uses Adam with learning rate scheduling.",
        ),
    ]


@pytest.fixture
def indexed_corpus(dense_index, sparse_index, tmp_data_dir):
    from research_assistant.config import Settings

    settings = Settings(
        PERSIST_DIRECTORY=str(tmp_data_dir / "chroma"),
        SQLITE_SPARSE_DB=str(tmp_data_dir / "sparse.db"),
        EMBEDDING_DIMENSION=768,
        FINAL_TOP_K=2,
        RRF_CANDIDATE_K=15,
    )
    chunks = _chunks()
    embedder = MockEmbedder()
    embeddings = embedder.embed_documents([c.passage for c in chunks])
    IndexTransaction(dense_index, sparse_index).commit_chunks(
        chunks, embeddings, document_id="2407.08608"
    )
    return settings, embedder


def test_hybrid_retrieve_returns_reranked_hits(indexed_corpus, dense_index, sparse_index):
    settings, embedder = indexed_corpus
    retriever = HybridRetriever(
        dense_index,
        sparse_index,
        embedder,
        reranker=MockReranker(),
        settings=settings,
    )

    result = retriever.retrieve("transformer attention mechanism")

    assert len(result.candidates) == 2
    assert result.candidates[0].rerank_score == pytest.approx(0.85)
    assert result.candidates[0].rrf_score is not None
    assert result.candidates[0].dense_rank is not None or result.candidates[0].sparse_rank is not None
    assert result.sufficiency.sufficient is True


def test_hybrid_retrieve_insufficient_for_irrelevant_query(
    indexed_corpus, dense_index, sparse_index
):
    settings, embedder = indexed_corpus
    retriever = HybridRetriever(
        dense_index,
        sparse_index,
        embedder,
        reranker=MockReranker(),
        settings=settings,
    )

    result = retriever.retrieve("quantum chromatography spectroscopy")

    assert result.sufficiency.sufficient is False


def test_hybrid_retrieve_empty_index(dense_index, sparse_index, tmp_data_dir):
    from research_assistant.config import Settings

    settings = Settings(
        PERSIST_DIRECTORY=str(tmp_data_dir / "chroma"),
        EMBEDDING_DIMENSION=768,
    )
    retriever = HybridRetriever(
        dense_index,
        sparse_index,
        MockEmbedder(),
        reranker=MockReranker(),
        settings=settings,
    )
    result = retriever.retrieve("anything")
    assert result.candidates == []
    assert result.sufficiency.sufficient is False
