"""Tests for startup health checks."""

from unittest.mock import MagicMock, patch

from research_assistant.config import Settings
from research_assistant.health import validate_configuration, validate_external_services


def test_validate_configuration_detects_placeholders():
    settings = Settings(
        GROQ_API_KEY="gsk_your_groq_api_key_here",
        GOOGLE_API_KEY="your_google_api_key_here",
        GOOGLE_API_KEYS="",
    )
    errors = validate_configuration(settings)
    assert len(errors) == 2


def test_validate_external_services_success():
    settings = Settings(GROQ_API_KEY="groq-key", GOOGLE_API_KEY="google-key")

    mock_embed = MagicMock()
    mock_embed.embeddings = [MagicMock(values=[0.1] * 768)]

    with patch("groq.Groq") as groq_cls, patch("google.genai.Client") as genai_cls:
        groq_cls.return_value.chat.completions.create.return_value = MagicMock()
        genai_cls.return_value.models.embed_content.return_value = mock_embed
        errors = validate_external_services(settings)

    assert errors == []
