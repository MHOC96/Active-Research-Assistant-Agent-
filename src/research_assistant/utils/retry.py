"""Bounded exponential backoff retry decorator."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from research_assistant.config import get_settings

P = ParamSpec("P")
T = TypeVar("T")

RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "temporarily unavailable",
    "rate limit",
)

NON_RETRYABLE_MARKERS = (
    "401",
    "403",
    "unauthenticated",
    "invalid api key",
    "permission denied",
    "access_token_type_unsupported",
)


def is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(marker in message for marker in NON_RETRYABLE_MARKERS):
        return False
    return any(marker in message for marker in RETRYABLE_MARKERS)


def with_retry(max_retries: int | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retries = max_retries if max_retries is not None else get_settings().max_retries
            delay = 0.5
            last_exc: Exception | None = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= retries or not is_retryable(exc):
                        raise
                    time.sleep(delay + random.uniform(0, 0.25))
                    delay *= 2

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
