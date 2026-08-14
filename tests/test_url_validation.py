"""Tests for URL validation."""

import pytest

from research_assistant.security.urls import UrlValidationError, validate_https_url


def test_validate_https_url_accepts_arxiv():
    validate_https_url(
        "https://arxiv.org/pdf/2407.08608.pdf",
        frozenset({"arxiv.org", "export.arxiv.org"}),
    )


def test_validate_https_url_rejects_http():
    with pytest.raises(UrlValidationError, match="HTTPS-only"):
        validate_https_url("http://arxiv.org/pdf/2407.08608.pdf", frozenset({"arxiv.org"}))


def test_validate_https_url_rejects_unknown_domain():
    with pytest.raises(UrlValidationError, match="not in the allowed"):
        validate_https_url("https://evil.example/paper.pdf", frozenset({"arxiv.org"}))
