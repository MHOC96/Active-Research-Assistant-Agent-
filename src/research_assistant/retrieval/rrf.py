"""Weighted Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    chunk_id: str
    rank: int


def compute_rrf_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    *,
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    k: int = 60,
) -> float:
    """Compute weighted RRF score for a single candidate.

    If a document is absent from one retrieval list, that channel contributes zero.
    """
    score = 0.0
    if dense_rank is not None:
        score += w_dense / (k + dense_rank)
    if sparse_rank is not None:
        score += w_sparse / (k + sparse_rank)
    return score


def fuse_rankings(
    dense_ranked: list[str],
    sparse_ranked: list[str],
    *,
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    k: int = 60,
    candidate_k: int = 15,
) -> list[tuple[str, float, int | None, int | None]]:
    """Fuse dense and sparse ranked chunk IDs using weighted RRF.

    Returns list of (chunk_id, rrf_score, dense_rank, sparse_rank) sorted by score desc.
    """
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ranked, start=1)}
    sparse_ranks = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ranked, start=1)}

    all_ids = set(dense_ranks) | set(sparse_ranks)
    fused: list[tuple[str, float, int | None, int | None]] = []

    for chunk_id in all_ids:
        d_rank = dense_ranks.get(chunk_id)
        s_rank = sparse_ranks.get(chunk_id)
        score = compute_rrf_score(d_rank, s_rank, w_dense=w_dense, w_sparse=w_sparse, k=k)
        fused.append((chunk_id, score, d_rank, s_rank))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused[:candidate_k]
