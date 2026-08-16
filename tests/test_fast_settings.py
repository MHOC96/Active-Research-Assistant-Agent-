"""Tests for fast performance settings."""

from research_assistant.config import Settings, apply_fast_settings


def test_apply_fast_settings_reduces_ingestion_scope():
    settings = apply_fast_settings(Settings())
    assert settings.fast_ingestion is True
    assert settings.max_new_documents_per_query == 1
    assert settings.max_discovery_rounds == 1
    assert settings.embedding_batch_size == 64
