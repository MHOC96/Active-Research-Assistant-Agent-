"""Pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.metadata_store import MetadataStore
from research_assistant.storage.sparse_index import SparseIndex


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    return data


@pytest.fixture
def sparse_index(tmp_data_dir: Path) -> SparseIndex:
    index = SparseIndex(tmp_data_dir / "sparse.db")
    yield index
    index.close()


@pytest.fixture
def metadata_store(tmp_data_dir: Path) -> MetadataStore:
    store = MetadataStore(tmp_data_dir / "metadata.db")
    yield store
    store.close()


@pytest.fixture
def dense_index(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> DenseIndex:
    from research_assistant.config import Settings, get_settings

    settings = Settings(
        PERSIST_DIRECTORY=str(tmp_data_dir / "chroma"),
        EMBEDDING_DIMENSION=768,
    )
    monkeypatch.setattr("research_assistant.storage.dense_index.get_settings", lambda: settings)
    get_settings.cache_clear()
    return DenseIndex(settings)
