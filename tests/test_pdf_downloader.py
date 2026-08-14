"""Tests for secure PDF downloader."""

from pathlib import Path

import httpx
import pytest

from research_assistant.config import Settings
from research_assistant.ingestion.downloader import SecurePdfDownloader


def _pdf_bytes(extra: bytes = b" sample content") -> bytes:
    return b"%PDF-1.4\n" + extra


def test_download_valid_pdf(tmp_path: Path):
    settings = Settings(
        DOWNLOAD_CACHE_DIR=str(tmp_path / "downloads"),
        ALLOWED_DOWNLOAD_DOMAINS="arxiv.org",
        MAX_PDF_SIZE_MB=1,
        DOWNLOAD_TIMEOUT_SECONDS=5,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=_pdf_bytes())
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    downloader = SecurePdfDownloader(settings)
    downloader._fetch = lambda url: _download_via_transport(transport, url, settings)  # type: ignore[method-assign]

    result = downloader.download("https://arxiv.org/pdf/2407.08608.pdf", "2407.08608")
    assert result.path.exists()
    assert result.content_hash
    assert result.size_bytes > 0


def test_download_rejects_non_pdf(tmp_path: Path):
    settings = Settings(
        DOWNLOAD_CACHE_DIR=str(tmp_path / "downloads"),
        ALLOWED_DOWNLOAD_DOMAINS="arxiv.org",
    )
    downloader = SecurePdfDownloader(settings)

    def bad_fetch(url: str) -> bytes:
        return b"not-a-pdf"

    downloader._fetch = bad_fetch  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="PDF_VALIDATION_FAILED"):
        downloader.download("https://arxiv.org/pdf/2407.08608.pdf", "2407.08608")


def test_download_rejects_redirect_to_unallowed_domain(tmp_path: Path):
    settings = Settings(
        DOWNLOAD_CACHE_DIR=str(tmp_path / "downloads"),
        ALLOWED_DOWNLOAD_DOMAINS="arxiv.org",
        DOWNLOAD_TIMEOUT_SECONDS=5,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://arxiv.org/pdf/2407.08608.pdf":
            return httpx.Response(
                302,
                headers={"location": "https://evil.example/paper.pdf"},
            )
        return httpx.Response(200, content=_pdf_bytes())

    transport = httpx.MockTransport(handler)
    downloader = SecurePdfDownloader(settings)

    with pytest.raises(Exception, match="not in the allowed|PDF_DOWNLOAD_FAILED"):
        _download_via_transport(transport, "https://arxiv.org/pdf/2407.08608.pdf", settings)


def test_download_enforces_size_limit(tmp_path: Path):
    settings = Settings(
        DOWNLOAD_CACHE_DIR=str(tmp_path / "downloads"),
        ALLOWED_DOWNLOAD_DOMAINS="arxiv.org",
        MAX_PDF_SIZE_MB=0,
        DOWNLOAD_TIMEOUT_SECONDS=5,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_pdf_bytes(b"x" * 2048))

    transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError, match="PDF_DOWNLOAD_FAILED"):
        _download_via_transport(transport, "https://arxiv.org/pdf/2407.08608.pdf", settings)


def _download_via_transport(
    transport: httpx.MockTransport, url: str, settings: Settings
) -> bytes:
    timeout = httpx.Timeout(settings.download_timeout_seconds)
    with httpx.Client(transport=transport, follow_redirects=False, timeout=timeout) as client:
        downloader = SecurePdfDownloader(settings)
        final_url = downloader._resolve_redirects(client, url)
        return downloader._download_stream(client, final_url)
