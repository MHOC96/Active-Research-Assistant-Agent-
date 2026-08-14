"""Round-robin Google API key rotation for Gemini rate limits."""

from __future__ import annotations

import threading

from google import genai


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
        )
    )


class GoogleApiKeyRotator:
    """Rotate through multiple Gemini API keys when rate limits are hit."""

    def __init__(self, keys: list[str]) -> None:
        cleaned = [key.strip() for key in keys if key.strip()]
        if not cleaned:
            raise ValueError("At least one Google API key is required")
        self._keys = cleaned
        self._index = 0
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def client(self) -> genai.Client:
        return genai.Client(api_key=self.current_key)

    def rotate(self) -> bool:
        """Advance to the next key. Returns False when all keys were tried."""
        with self._lock:
            if self._index + 1 >= len(self._keys):
                return False
            self._index += 1
            return True
