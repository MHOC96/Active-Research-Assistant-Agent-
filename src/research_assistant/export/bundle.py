"""Export a query session as a downloadable bundle."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from research_assistant.models import CitationSpan
from research_assistant.security.paths import safe_download_path


def slugify_query(query: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", query.strip().lower()).strip("-")
    return slug[:max_length] or "query"


def extract_arxiv_id(url: str) -> str | None:
    if not url:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url, re.IGNORECASE)
    return match.group(1) if match else None


def collect_arxiv_ids(citation_spans: list[CitationSpan]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for span in citation_spans:
        for citation in span.citations:
            arxiv_id = citation.arxiv_id or extract_arxiv_id(citation.url)
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            ordered.append(arxiv_id)
    return ordered


def build_query_bundle_zip(
    *,
    query: str,
    answer: str,
    citation_spans: list[CitationSpan],
    citation_style: str,
    download_cache_dir: Path,
    exports_dir: Path | None = None,
) -> Path:
    """Create a zip bundle with query text, references, manifest, and cached PDFs."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = slugify_query(query)
    bundle_name = f"{slug}-{timestamp}"

    if exports_dir is not None:
        bundle_dir = exports_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        papers_dir = bundle_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        persist_root = bundle_dir
    else:
        persist_root = None
        papers_dir = None

    arxiv_ids = collect_arxiv_ids(citation_spans)
    copied_papers: dict[str, str] = {}
    missing_papers: list[str] = []

    if papers_dir is not None:
        for arxiv_id in arxiv_ids:
            source = safe_download_path(download_cache_dir, arxiv_id)
            destination = papers_dir / f"{arxiv_id}.pdf"
            if not source.exists():
                missing_papers.append(arxiv_id)
                continue
            _copy_or_link(source, destination)
            copied_papers[arxiv_id] = f"papers/{arxiv_id}.pdf"

    manifest = {
        "bundle_name": bundle_name,
        "exported_at": datetime.now(UTC).isoformat(),
        "citation_style": citation_style,
        "query": query,
        "segments": [span.model_dump() for span in citation_spans],
        "papers": copied_papers,
        "missing_papers": missing_papers,
    }

    if persist_root is not None:
        (persist_root / "query.txt").write_text(query, encoding="utf-8")
        (persist_root / "references.txt").write_text(answer, encoding="utf-8")
        (persist_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        zip_path = persist_root.with_suffix(".zip")
    else:
        zip_path = Path(NamedTemporaryFile(prefix=f"{bundle_name}-", suffix=".zip", delete=False).name)

    _write_zip(
        zip_path,
        query=query,
        answer=answer,
        manifest=manifest,
        download_cache_dir=download_cache_dir,
        arxiv_ids=arxiv_ids,
        copied_papers=copied_papers,
        missing_papers=missing_papers,
        persist_root=persist_root,
    )
    return zip_path


def _copy_or_link(source: Path, destination: Path) -> None:
    try:
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_zip(
    zip_path: Path,
    *,
    query: str,
    answer: str,
    manifest: dict,
    download_cache_dir: Path,
    arxiv_ids: list[str],
    copied_papers: dict[str, str],
    missing_papers: list[str],
    persist_root: Path | None,
) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("query.txt", query)
        archive.writestr("references.txt", answer)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))

        for arxiv_id in arxiv_ids:
            relative = copied_papers.get(arxiv_id)
            if persist_root is not None and relative:
                paper_path = persist_root / relative
                if paper_path.exists():
                    archive.write(paper_path, arcname=relative)
                    continue

            source = safe_download_path(download_cache_dir, arxiv_id)
            if source.exists():
                archive.write(source, arcname=f"papers/{arxiv_id}.pdf")
            elif arxiv_id not in missing_papers:
                missing_papers.append(arxiv_id)
