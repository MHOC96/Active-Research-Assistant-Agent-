"""Hybrid dense + sparse retrieval with RRF fusion and FlashRank reranking."""

from __future__ import annotations

from research_assistant.config import Settings, get_settings
from research_assistant.embeddings.base import EmbeddingService
from research_assistant.models import HybridRetrieveResult, RetrievalHit
from research_assistant.reranking.flashrank_reranker import FlashRankReranker
from research_assistant.retrieval.rrf import fuse_rankings
from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.sparse_index import SparseIndex
from research_assistant.sufficiency.gate import evaluate_sufficiency


class HybridRetriever:
    """Execute hybrid_retrieve workflow per AGENTS.md section 27."""

    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndex,
        embedder: EmbeddingService,
        reranker: FlashRankReranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.embedder = embedder
        self.reranker = reranker or FlashRankReranker(settings)
        self.settings = settings or get_settings()

    def retrieve(self, query: str, *, top_k: int | None = None) -> HybridRetrieveResult:
        top_k = top_k or self.settings.final_top_k

        try:
            query_embedding = self.embedder.embed_query(query)
            dense_results = self.dense.search(
                query_embedding, limit=self.settings.rrf_candidate_k
            )
            sparse_results = self.sparse.search(query, limit=self.settings.rrf_candidate_k)
        except Exception as exc:
            raise RuntimeError(f"RETRIEVAL_FAILED: {exc}") from exc

        dense_ranked = [chunk_id for chunk_id, _, _ in dense_results]
        sparse_ranked = [chunk_id for chunk_id, _ in sparse_results]

        fused = fuse_rankings(
            dense_ranked,
            sparse_ranked,
            w_dense=self.settings.rrf_dense_weight,
            w_sparse=self.settings.rrf_sparse_weight,
            k=self.settings.rrf_k_constant,
            candidate_k=self.settings.rrf_candidate_k,
        )

        pool: list[RetrievalHit] = []
        for chunk_id, rrf_score, dense_rank, sparse_rank in fused:
            hit = self._build_hit(chunk_id, dense_rank, sparse_rank, rrf_score, dense_results)
            if hit is not None:
                pool.append(hit)

        try:
            reranked = self.reranker.rerank(query, pool, top_k=top_k)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"RERANKING_FAILED: {exc}") from exc

        sufficiency = evaluate_sufficiency(
            reranked,
            min_candidates=self.settings.min_candidates,
            min_rerank_score=self.settings.min_rerank_score,
        )

        return HybridRetrieveResult(query=query, candidates=reranked, sufficiency=sufficiency)

    def _build_hit(
        self,
        chunk_id: str,
        dense_rank: int | None,
        sparse_rank: int | None,
        rrf_score: float,
        dense_results: list[tuple[str, float, dict]],
    ) -> RetrievalHit | None:
        chunk_meta = self.sparse.get_chunk(chunk_id)
        passage = self.sparse.get_passage(chunk_id)

        dense_meta = next((meta for cid, _, meta in dense_results if cid == chunk_id), None)

        if chunk_meta is None and dense_meta is None:
            return None

        meta = chunk_meta or {}
        chroma_meta = dense_meta or {}

        arxiv_id = meta.get("arxiv_id") or chroma_meta.get("arxiv_id", "")
        document_id = meta.get("document_id") or chroma_meta.get("document_id", "")
        title = meta.get("title") or chroma_meta.get("title", "")
        chunk_index = int(meta.get("chunk_index") or chroma_meta.get("chunk_index", 0))
        section = meta.get("section") or chroma_meta.get("section") or None
        page_val = meta.get("page")
        if page_val is None:
            page_val = chroma_meta.get("page")
        page = int(page_val) if page_val is not None and int(page_val) >= 0 else None

        if not passage:
            passage = self.dense.get_passage(chunk_id) or ""

        if not passage or not arxiv_id:
            return None

        return RetrievalHit(
            chunk_id=chunk_id,
            passage=passage,
            document_id=document_id,
            arxiv_id=arxiv_id,
            title=title,
            section=section or None,
            page=page,
            chunk_index=chunk_index,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            rrf_score=rrf_score,
        )
