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


def test_gemini_requires_api_key():
    settings = Settings(GOOGLE_API_KEY="")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiEmbeddingService(settings)


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
