"""Secure academic PDF downloader."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from research_assistant.config import Settings, get_settings
from research_assistant.security.paths import ensure_path_contained, safe_download_path
from research_assistant.security.urls import UrlValidationError, validate_https_url

PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    content_hash: str
    source_url: str
    size_bytes: int


class SecurePdfDownloader:
    """Download PDFs with HTTPS, domain, redirect, size, and signature validation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.download_cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, arxiv_id: str) -> DownloadResult:
        validate_https_url(url, self.settings.allowed_domains)
        target_path = safe_download_path(self.settings.download_cache_dir, arxiv_id)

        if target_path.exists():
            data = target_path.read_bytes()
            self._validate_pdf_bytes(data)
            return DownloadResult(
                path=target_path,
                content_hash=self._hash_bytes(data),
                source_url=url,
                size_bytes=len(data),
            )

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.settings.download_cache_dir,
                prefix=f".{arxiv_id}_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                ensure_path_contained(self.settings.download_cache_dir, tmp_path)

            data = self._fetch(url)
            self._validate_pdf_bytes(data)
            tmp_path.write_bytes(data)
            tmp_path.replace(target_path)
            ensure_path_contained(self.settings.download_cache_dir, target_path)

            return DownloadResult(
                path=target_path,
                content_hash=self._hash_bytes(data),
                source_url=url,
                size_bytes=len(data),
            )
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _fetch(self, url: str) -> bytes:
        timeout = httpx.Timeout(self.settings.download_timeout_seconds)
        try:
            with httpx.Client(follow_redirects=False, timeout=timeout) as client:
                final_url = self._resolve_redirects(client, url)
                return self._download_stream(client, final_url)
        except UrlValidationError:
            raise
        except httpx.HTTPError as exc:
            raise RuntimeError(f"PDF_DOWNLOAD_FAILED: {exc}") from exc

    def _resolve_redirects(self, client: httpx.Client, url: str, *, redirects: int = 0) -> str:
        if redirects > 5:
            raise RuntimeError("PDF_DOWNLOAD_FAILED: too many redirects")

        validate_https_url(url, self.settings.allowed_domains)
        response = client.get(url)

        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise RuntimeError("PDF_DOWNLOAD_FAILED: redirect without location header")
            next_url = str(response.url.join(location))
            return self._resolve_redirects(client, next_url, redirects=redirects + 1)

        if response.status_code >= 400:
            raise RuntimeError(f"PDF_DOWNLOAD_FAILED: HTTP {response.status_code}")

        final_url = str(response.url)
        validate_https_url(final_url, self.settings.allowed_domains)
        return final_url

    def _download_stream(self, client: httpx.Client, url: str) -> bytes:
        max_bytes = self.settings.max_pdf_size_bytes
        chunks: list[bytes] = []
        total = 0

        with client.stream("GET", url) as response:
            response.raise_for_status()
            validate_https_url(str(response.url), self.settings.allowed_domains)
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(
                        f"PDF_DOWNLOAD_FAILED: file exceeds {self.settings.max_pdf_size_mb} MB limit"
                    )
                chunks.append(chunk)

        return b"".join(chunks)

    @staticmethod
    def _validate_pdf_bytes(data: bytes) -> None:
        if not data.startswith(PDF_MAGIC):
            raise RuntimeError("PDF_VALIDATION_FAILED: content is not a valid PDF")

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
