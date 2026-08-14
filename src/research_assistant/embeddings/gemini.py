"""Google Gemini embedding service."""

from __future__ import annotations

import google.generativeai as genai

from research_assistant.config import Settings, get_settings
from research_assistant.utils.retry import with_retry


class GeminiEmbeddingService:
    """Generates 768-dim embeddings via models/text-embedding-004."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings")
        genai.configure(api_key=self.settings.google_api_key)
        self._model = self.settings.gemini_embedding_model

    @with_retry()
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="retrieval_query")

    @with_retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, task_type="retrieval_document") for text in texts]

    def _embed(self, text: str, *, task_type: str) -> list[float]:
        try:
            result = genai.embed_content(
                model=self._model,
                content=text,
                task_type=task_type,
            )
        except Exception as exc:
            raise RuntimeError(f"EMBEDDING_FAILED: {exc}") from exc

        embedding = result.get("embedding")
        if not embedding:
            raise RuntimeError("EMBEDDING_FAILED: empty embedding response")

        if len(embedding) != self.settings.embedding_dimension:
            raise RuntimeError(
                f"EMBEDDING_FAILED: expected dimension "
                f"{self.settings.embedding_dimension}, got {len(embedding)}"
            )

        return [float(v) for v in embedding]
