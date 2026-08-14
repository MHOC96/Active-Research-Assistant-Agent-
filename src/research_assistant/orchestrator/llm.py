"""Groq LLM client for orchestration and synthesis."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from groq import Groq

from research_assistant.config import Settings, get_settings
from research_assistant.utils.retry import with_retry


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return model completion text."""


class GroqLLMClient:
    """Groq chat completion client (temperature 0.0 per AGENTS.md)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for orchestration")
        self._client = Groq(api_key=self.settings.groq_api_key)

    @with_retry()
    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        tokens = max_tokens or self.settings.groq_max_output_tokens
        try:
            response = self._client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"SYNTHESIS_FAILED: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("SYNTHESIS_FAILED: empty model response")
        return content.strip()
