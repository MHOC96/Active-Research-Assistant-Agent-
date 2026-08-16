"""Cooperative request cancellation for long-running pipeline work."""

from __future__ import annotations

import threading


class RequestCancelledError(Exception):
    """Raised when a pipeline stage detects cooperative cancellation."""

    def __init__(self, request_id: str, stage: str | None = None) -> None:
        self.request_id = request_id
        self.stage = stage
        detail = f"Request {request_id} was cancelled"
        if stage:
            detail = f"{detail} at stage: {stage}"
        super().__init__(detail)


class CancellationToken:
    """Thread-safe cancellation flag checked between pipeline stages."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._cancelled = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self, stage: str | None = None) -> None:
        if self.is_cancelled:
            raise RequestCancelledError(self.request_id, stage=stage)


class CancellationRegistry:
    """Tracks active request cancellation tokens."""

    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str) -> CancellationToken:
        token = CancellationToken(request_id)
        with self._lock:
            self._tokens[request_id] = token
        return token

    def get(self, request_id: str) -> CancellationToken | None:
        with self._lock:
            return self._tokens.get(request_id)

    def cancel(self, request_id: str) -> bool:
        token = self.get(request_id)
        if token is None:
            return False
        token.cancel()
        return True

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._tokens.pop(request_id, None)


cancellation_registry = CancellationRegistry()
