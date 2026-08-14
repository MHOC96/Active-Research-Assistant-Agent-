"""Google Gemini embedding service."""

from __future__ import annotations

from google import genai
from google.genai import types

from research_assistant.config import Settings, get_settings
from research_assistant.utils.retry import with_retry

_TASK_TYPE_MAP = {
    "retrieval_query": "RETRIEVAL_QUERY",
    "retrieval_document": "RETRIEVAL_DOCUMENT",
}


class GeminiEmbeddingService:
    """Generates 768-dim embeddings via text-embedding-004."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings")
        self._client = genai.Client(api_key=self.settings.google_api_key)
        self._model = _normalize_embedding_model(self.settings.gemini_embedding_model)

    @with_retry()
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="retrieval_query")

    @with_retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, task_type="retrieval_document") for text in texts]

    def _embed(self, text: str, *, task_type: str) -> list[float]:
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=_TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT")
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"EMBEDDING_FAILED: {exc}") from exc

        if not response.embeddings:
            raise RuntimeError("EMBEDDING_FAILED: empty embedding response")

        embedding = [float(v) for v in response.embeddings[0].values]
        if len(embedding) != self.settings.embedding_dimension:
            raise RuntimeError(
                f"EMBEDDING_FAILED: expected dimension "
                f"{self.settings.embedding_dimension}, got {len(embedding)}"
            )
        return embedding


def _normalize_embedding_model(model: str) -> str:
    return model.removeprefix("models/")
