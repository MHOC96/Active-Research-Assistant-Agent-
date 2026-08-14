"""Tests for download path security."""

from pathlib import Path

import pytest

from research_assistant.security.paths import (
    ensure_path_contained,
    safe_download_path,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename("2407.08608.pdf") == "2407.08608.pdf"
    assert sanitize_filename("../../../etc/passwd") == "......etcpasswd"
    assert "/" not in sanitize_filename("bad/name.pdf")


def test_ensure_path_contained(tmp_path: Path):
    base = tmp_path / "downloads"
    base.mkdir()
    safe = base / "paper.pdf"
    assert ensure_path_contained(base, safe) == safe.resolve()


def test_path_traversal_rejected(tmp_path: Path):
    base = tmp_path / "downloads"
    base.mkdir()
    evil = base / ".." / "outside.pdf"
    with pytest.raises(ValueError, match="path validation failure"):
        ensure_path_contained(base, evil)


def test_safe_download_path(tmp_path: Path):
    base = tmp_path / "downloads"
    base.mkdir()
    path = safe_download_path(base, "2407.08608")
    assert path.name == "2407.08608.pdf"
    assert path.parent.resolve() == base.resolve()
