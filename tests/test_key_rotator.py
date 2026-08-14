"""Tests for Google API key rotation."""

import pytest

from research_assistant.embeddings.key_rotator import GoogleApiKeyRotator, is_rate_limit_error


def test_is_rate_limit_error():
    assert is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED quota exceeded"))
    assert is_rate_limit_error(Exception("Rate limit exceeded"))
    assert not is_rate_limit_error(Exception("401 unauthenticated"))


def test_rotator_starts_on_first_key():
    rotator = GoogleApiKeyRotator(["alpha", "beta"])
    assert rotator.current_key == "alpha"
    assert rotator.key_count == 2


def test_rotator_cycles_through_keys():
    rotator = GoogleApiKeyRotator(["alpha", "beta", "gamma"])
    assert rotator.rotate() is True
    assert rotator.current_key == "beta"
    assert rotator.rotate() is True
    assert rotator.current_key == "gamma"
    assert rotator.rotate() is False
    assert rotator.current_key == "gamma"


def test_rotator_requires_keys():
    with pytest.raises(ValueError, match="At least one Google API key"):
        GoogleApiKeyRotator([])
