"""Tests for weighted Reciprocal Rank Fusion."""

import pytest

from research_assistant.retrieval.rrf import compute_rrf_score, fuse_rankings


def test_compute_rrf_dense_only():
    score = compute_rrf_score(dense_rank=1, sparse_rank=None, w_dense=0.6, w_sparse=0.4, k=60)
    assert score == pytest.approx(0.6 / 61)


def test_compute_rrf_both_channels():
    score = compute_rrf_score(dense_rank=1, sparse_rank=2, w_dense=0.6, w_sparse=0.4, k=60)
    expected = 0.6 / 61 + 0.4 / 62
    assert score == pytest.approx(expected)


def test_fuse_rankings_orders_by_rrf():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = fuse_rankings(dense, sparse, candidate_k=15)

    chunk_ids = [item[0] for item in fused]
    assert "a" in chunk_ids
    assert "b" in chunk_ids
    assert "d" in chunk_ids
    assert fused[0][1] >= fused[-1][1]


def test_fuse_rankings_respects_candidate_k():
    dense = [f"id{i}" for i in range(20)]
    sparse = [f"id{i}" for i in range(20, 0, -1)]
    fused = fuse_rankings(dense, sparse, candidate_k=5)
    assert len(fused) == 5


def test_absent_channel_contributes_zero():
    score_dense_only = compute_rrf_score(dense_rank=5, sparse_rank=None)
    score_sparse_only = compute_rrf_score(dense_rank=None, sparse_rank=5)
    score_both = compute_rrf_score(dense_rank=5, sparse_rank=5)
    assert score_both == score_dense_only + score_sparse_only
