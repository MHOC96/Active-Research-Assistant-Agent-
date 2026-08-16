"""Google Gemini embedding service."""

from __future__ import annotations

from google.genai import types

from research_assistant.config import Settings, get_settings
from research_assistant.embeddings.key_rotator import GoogleApiKeyRotator, is_rate_limit_error
from research_assistant.utils.concurrency import map_io_bound
from research_assistant.utils.retry import with_retry

_TASK_TYPE_MAP = {
    "retrieval_query": "RETRIEVAL_QUERY",
    "retrieval_document": "RETRIEVAL_DOCUMENT",
}


class GeminiEmbeddingService:
    """Generates 768-dim embeddings via gemini-embedding-001."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        keys = self.settings.google_api_keys
        if not keys:
            raise ValueError(
                "At least one Google API key is required (GOOGLE_API_KEY or GOOGLE_API_KEYS)"
            )
        self._rotator = GoogleApiKeyRotator(keys)
        self._client = self._rotator.client()
        self._model = _normalize_embedding_model(self.settings.gemini_embedding_model)

    @with_retry()
    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], task_type="retrieval_query")[0]

    @with_retry()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = max(1, self.settings.embedding_batch_size)
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]

        if len(batches) == 1:
            return self._embed_batch(batches[0], task_type="retrieval_document")

        embedded_batches = map_io_bound(
            lambda batch: self._embed_batch(batch, task_type="retrieval_document"),
            batches,
            max_workers=self.settings.embedding_max_workers,
        )
        return [vector for batch_vectors in embedded_batches for vector in batch_vectors]

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        last_exc: Exception | None = None
        keys_tried = 0

        while keys_tried < self._rotator.key_count:
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=_TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT"),
                        output_dimensionality=self.settings.embedding_dimension,
                    ),
                )
            except Exception as exc:
                last_exc = exc
                if is_rate_limit_error(exc) and self._rotator.rotate():
                    keys_tried += 1
                    self._client = self._rotator.client()
                    continue
                raise RuntimeError(f"EMBEDDING_FAILED: {exc}") from exc

            if not response.embeddings or len(response.embeddings) != len(texts):
                raise RuntimeError("EMBEDDING_FAILED: empty or mismatched embedding response")

            batch_vectors: list[list[float]] = []
            for embedding in response.embeddings:
                vector = [float(v) for v in embedding.values]
                if len(vector) != self.settings.embedding_dimension:
                    raise RuntimeError(
                        f"EMBEDDING_FAILED: expected dimension "
                        f"{self.settings.embedding_dimension}, got {len(vector)}"
                    )
                batch_vectors.append(vector)
            return batch_vectors

        assert last_exc is not None
        raise RuntimeError(f"EMBEDDING_FAILED: all API keys rate limited: {last_exc}") from last_exc


def _normalize_embedding_model(model: str) -> str:
    return model.removeprefix("models/")
