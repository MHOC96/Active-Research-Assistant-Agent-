"""Tests for Gemini embedding service (mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from research_assistant.config import Settings
from research_assistant.embeddings.gemini import GeminiEmbeddingService


def test_gemini_embed_query():
    settings = Settings(GOOGLE_API_KEY="test-key", EMBEDDING_DIMENSION=768)
    service = GeminiEmbeddingService(settings)

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    with patch.object(service._client.models, "embed_content", return_value=mock_response) as mock_embed:
        vector = service.embed_query("test query")

    assert len(vector) == 768
    mock_embed.assert_called_once()
    assert mock_embed.call_args.kwargs["contents"] == ["test query"]


def test_gemini_embed_documents_batches():
    settings = Settings(
        GOOGLE_API_KEY="test-key",
        EMBEDDING_DIMENSION=768,
        EMBEDDING_BATCH_SIZE=2,
        EMBEDDING_MAX_WORKERS=1,
    )
    service = GeminiEmbeddingService(settings)

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_response_two = MagicMock()
    mock_response_two.embeddings = [mock_embedding, mock_embedding]
    mock_response_one = MagicMock()
    mock_response_one.embeddings = [mock_embedding]

    with patch.object(
        service._client.models,
        "embed_content",
        side_effect=[mock_response_two, mock_response_one],
    ) as mock_embed:
        vectors = service.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert mock_embed.call_count == 2
    assert mock_embed.call_args_list[0].kwargs["contents"] == ["a", "b"]
    assert mock_embed.call_args_list[1].kwargs["contents"] == ["c"]


def test_gemini_embed_documents_parallel_batches():
    settings = Settings(
        GOOGLE_API_KEY="test-key",
        EMBEDDING_DIMENSION=768,
        EMBEDDING_BATCH_SIZE=1,
        EMBEDDING_MAX_WORKERS=3,
    )
    service = GeminiEmbeddingService(settings)

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768

    def _embed_batch(**kwargs):
        contents = kwargs["contents"]
        response = MagicMock()
        response.embeddings = [mock_embedding for _ in contents]
        return response

    with patch.object(service._client.models, "embed_content", side_effect=_embed_batch) as mock_embed:
        vectors = service.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert mock_embed.call_count == 3


def test_gemini_requires_api_key():
    settings = Settings(GOOGLE_API_KEY="", GOOGLE_API_KEYS="")
    with pytest.raises(ValueError, match="Google API key"):
        GeminiEmbeddingService(settings)


def test_gemini_rotates_on_rate_limit():
    settings = Settings(
        GOOGLE_API_KEY="key-a",
        GOOGLE_API_KEYS="key-b",
        EMBEDDING_DIMENSION=768,
    )
    service = GeminiEmbeddingService(settings)

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    rate_limit = RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    with patch.object(
        service._client.models,
        "embed_content",
        side_effect=[rate_limit, mock_response],
    ) as mock_embed, patch.object(service._rotator, "client") as mock_client_factory:
        mock_client_factory.return_value = service._client
        vector = service.embed_query("test query")

    assert len(vector) == 768
    assert mock_embed.call_count == 2


def test_gemini_rejects_wrong_dimension():
    settings = Settings(GOOGLE_API_KEY="test-key", EMBEDDING_DIMENSION=768)
    service = GeminiEmbeddingService(settings)

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 512
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    with patch.object(service._client.models, "embed_content", return_value=mock_response):
        with pytest.raises(RuntimeError, match="EMBEDDING_FAILED"):
            service.embed_query("test")
