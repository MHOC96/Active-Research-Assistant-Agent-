"""Tests for concurrent I/O helpers."""

from research_assistant.utils.concurrency import map_io_bound


def test_map_io_bound_preserves_order():
    values = [1, 2, 3, 4]
    result = map_io_bound(lambda value: value * 10, values, max_workers=2)
    assert result == [10, 20, 30, 40]


def test_map_io_bound_single_worker_matches_sequential():
    values = ["a", "b"]
    result = map_io_bound(lambda value: value.upper(), values, max_workers=1)
    assert result == ["A", "B"]
