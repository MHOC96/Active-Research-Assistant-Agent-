"""Thread-pool helpers for blocking I/O and CPU-bound work."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_io_bound(
    func: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int,
) -> list[R]:
    """Run a blocking function over items concurrently, preserving input order."""
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [func(item) for item in items]

    workers = min(max_workers, len(items))
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(func, item): index for index, item in enumerate(items)}
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()

    return [result for result in results if result is not None]
