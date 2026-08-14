"""Path traversal protection for download directory."""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Strict filename sanitization per AGENTS.md section 21.1."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "", name)


def ensure_path_contained(base_dir: Path, target_path: Path) -> Path:
    """Verify resolved target path remains inside base_dir."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            f"path validation failure: {resolved_target} is outside {resolved_base}"
        ) from exc
    return resolved_target


def safe_download_path(base_dir: Path, arxiv_id: str) -> Path:
    """Build a safe download path for an arXiv PDF."""
    safe_name = sanitize_filename(f"{arxiv_id}.pdf")
    if not safe_name:
        raise ValueError("invalid arxiv_id for filename generation")
    target = base_dir / safe_name
    return ensure_path_contained(base_dir, target)
