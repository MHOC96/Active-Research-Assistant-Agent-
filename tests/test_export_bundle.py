"""Tests for query bundle export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_assistant.export.bundle import (
    build_query_bundle_zip,
    collect_arxiv_ids,
    extract_arxiv_id,
    slugify_query,
)
from research_assistant.models import CitationSourceItem, CitationSpan
from research_assistant.web.app import create_app


def test_slugify_query():
    assert slugify_query("Compare RAG and GraphRAG!") == "compare-rag-and-graphrag"


def test_extract_arxiv_id_from_url():
    assert extract_arxiv_id("https://arxiv.org/abs/2407.08608") == "2407.08608"
    assert extract_arxiv_id("https://arxiv.org/pdf/2407.08608.pdf") == "2407.08608"
    assert extract_arxiv_id("https://example.com") is None


def test_collect_arxiv_ids_deduplicates():
    spans = [
        CitationSpan(
            segment_id="seg-0",
            text="RAG improves grounding.",
            start=0,
            end=22,
            search_query="RAG grounding",
            citations=[
                CitationSourceItem(
                    id="cite-1",
                    source="arxiv",
                    source_label="arXiv",
                    title="Paper A",
                    url="https://arxiv.org/abs/2407.08608",
                    reference="Ref A",
                    arxiv_id="2407.08608",
                ),
                CitationSourceItem(
                    id="cite-2",
                    source="arxiv",
                    source_label="arXiv",
                    title="Paper A duplicate",
                    url="https://arxiv.org/abs/2407.08608",
                    reference="Ref A",
                    arxiv_id="2407.08608",
                ),
            ],
        )
    ]
    assert collect_arxiv_ids(spans) == ["2407.08608"]


def test_build_query_bundle_zip_includes_cached_pdf(tmp_path: Path):
    download_cache = tmp_path / "downloads"
    exports_dir = tmp_path / "exports"
    download_cache.mkdir()
    exports_dir.mkdir()

    arxiv_id = "2407.08608"
    pdf_path = download_cache / f"{arxiv_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    spans = [
        CitationSpan(
            segment_id="seg-0",
            text="RAG improves grounding.",
            start=0,
            end=22,
            search_query="RAG grounding",
            citations=[
                CitationSourceItem(
                    id="cite-1",
                    source="arxiv",
                    source_label="arXiv",
                    title="Paper A",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    reference="Ref A",
                    arxiv_id=arxiv_id,
                )
            ],
        )
    ]

    zip_path = build_query_bundle_zip(
        query="What is RAG?",
        answer="References\n\nRef A",
        citation_spans=spans,
        citation_style="mla9",
        download_cache_dir=download_cache,
        exports_dir=exports_dir,
    )

    assert zip_path.exists()
    bundle_dirs = list(exports_dir.iterdir())
    assert len(bundle_dirs) == 2  # folder + zip

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "query.txt" in names
        assert "references.txt" in names
        assert "manifest.json" in names
        assert f"papers/{arxiv_id}.pdf" in names

        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["query"] == "What is RAG?"
        assert arxiv_id in manifest["papers"]


def test_export_bundle_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    download_cache = tmp_path / "downloads"
    exports_dir = tmp_path / "exports"
    download_cache.mkdir()
    exports_dir.mkdir()

    arxiv_id = "1234.56789"
    (download_cache / f"{arxiv_id}.pdf").write_bytes(b"%PDF-1.4 test")

    from research_assistant.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "download_cache_dir", download_cache)
    monkeypatch.setattr(settings, "exports_dir", exports_dir)

    payload = {
        "query": "transformer attention",
        "answer": "References\n\nExample reference",
        "citation_style": "mla9",
        "citation_spans": [
            {
                "segment_id": "seg-0",
                "text": "Attention is all you need.",
                "start": 0,
                "end": 26,
                "search_query": "transformer attention",
                "citations": [
                    {
                        "id": "cite-1",
                        "source": "arxiv",
                        "source_label": "arXiv",
                        "title": "Attention",
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "reference": "Vaswani et al.",
                        "arxiv_id": arxiv_id,
                    }
                ],
            }
        ],
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/export/bundle", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers.get("content-disposition", "")
