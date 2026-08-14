"""Tests for Gemini embedding service (mocked)."""

from unittest.mock import patch

import pytest

from research_assistant.config import Settings
from research_assistant.embeddings.gemini import GeminiEmbeddingService


def test_gemini_embed_query():
    settings = Settings(GOOGLE_API_KEY="test-key", EMBEDDING_DIMENSION=768)
    service = GeminiEmbeddingService(settings)

    with patch("research_assistant.embeddings.gemini.genai.embed_content") as mock_embed:
        mock_embed.return_value = {"embedding": [0.1] * 768}
        vector = service.embed_query("test query")

    assert len(vector) == 768
    mock_embed.assert_called_once()
    call_kwargs = mock_embed.call_args.kwargs
    assert call_kwargs["task_type"] == "retrieval_query"


def test_gemini_requires_api_key():
    settings = Settings(GOOGLE_API_KEY="")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiEmbeddingService(settings)


def test_gemini_rejects_wrong_dimension():
    settings = Settings(GOOGLE_API_KEY="test-key", EMBEDDING_DIMENSION=768)
    service = GeminiEmbeddingService(settings)

    with patch("research_assistant.embeddings.gemini.genai.embed_content") as mock_embed:
        mock_embed.return_value = {"embedding": [0.1] * 512}
        with pytest.raises(RuntimeError, match="EMBEDDING_FAILED"):
            service.embed_query("test")
