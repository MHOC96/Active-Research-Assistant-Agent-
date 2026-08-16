"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration aligned with AGENTS.md section 29."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_max_output_tokens: int = Field(default=2048, alias="GROQ_MAX_OUTPUT_TOKENS")
    groq_query_max_output_tokens: int = Field(default=256, alias="GROQ_QUERY_MAX_OUTPUT_TOKENS")
    groq_synthesis_max_output_tokens: int = Field(
        default=1024, alias="GROQ_SYNTHESIS_MAX_OUTPUT_TOKENS"
    )

    # Gemini embeddings
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_api_keys_env: str = Field(default="", alias="GOOGLE_API_KEYS")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_max_workers: int = Field(default=4, alias="EMBEDDING_MAX_WORKERS")
    io_max_workers: int = Field(default=4, alias="IO_MAX_WORKERS")

    # Storage
    persist_directory: Path = Field(default=Path("./data/chroma_db"), alias="PERSIST_DIRECTORY")
    download_cache_dir: Path = Field(default=Path("./data/downloads"), alias="DOWNLOAD_CACHE_DIR")
    sqlite_sparse_db: Path = Field(default=Path("./data/sparse_index.db"), alias="SQLITE_SPARSE_DB")
    metadata_db: Path = Field(default=Path("./data/metadata.db"), alias="METADATA_DB")

    # Retrieval / RRF
    rrf_dense_weight: float = Field(default=0.6, alias="RRF_DENSE_WEIGHT")
    rrf_sparse_weight: float = Field(default=0.4, alias="RRF_SPARSE_WEIGHT")
    rrf_k_constant: int = Field(default=60, alias="RRF_K_CONSTANT")
    rrf_candidate_k: int = Field(default=15, alias="RRF_CANDIDATE_K")
    final_top_k: int = Field(default=3, alias="FINAL_TOP_K")
    max_subqueries: int = Field(default=5, alias="MAX_SUBQUERIES")
    synthesis_max_passage_chars: int = Field(default=2400, alias="SYNTHESIS_MAX_PASSAGE_CHARS")
    skip_query_llm_for_simple: bool = Field(default=True, alias="SKIP_QUERY_LLM_FOR_SIMPLE")
    query_analysis_cache_size: int = Field(default=64, alias="QUERY_ANALYSIS_CACHE_SIZE")

    # Reranking
    reranker_model: str = Field(default="ms-marco-MiniLM-L-12-v2", alias="RERANKER_MODEL")

    # Sufficiency
    min_candidates: int = Field(default=1, alias="MIN_CANDIDATES")
    min_rerank_score: float = Field(default=0.70, alias="MIN_RERANK_SCORE")

    # Chunking
    chunk_target_tokens: int = Field(default=700, alias="CHUNK_TARGET_TOKENS")
    chunk_max_tokens: int = Field(default=1000, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=100, alias="CHUNK_OVERLAP_TOKENS")
    min_chunk_characters: int = Field(default=80, alias="MIN_CHUNK_CHARACTERS")

    # Discovery
    discovery_sources: str = Field(
        default="arxiv,openalex,semantic_scholar",
        alias="DISCOVERY_SOURCES",
    )
    discovery_per_source_max: int = Field(default=1, alias="DISCOVERY_PER_SOURCE_MAX")
    discovery_max_results: int = Field(default=5, alias="DISCOVERY_MAX_RESULTS")
    max_new_documents_per_query: int = Field(default=3, alias="MAX_NEW_DOCUMENTS_PER_QUERY")
    max_discovery_rounds: int = Field(default=2, alias="MAX_DISCOVERY_ROUNDS")
    openalex_mailto: str = Field(default="", alias="OPENALEX_MAILTO")
    semantic_scholar_api_key: str = Field(default="", alias="SEMANTIC_SCHOLAR_API_KEY")

    # Download security
    max_pdf_size_mb: int = Field(default=50, alias="MAX_PDF_SIZE_MB")
    download_timeout_seconds: int = Field(default=30, alias="DOWNLOAD_TIMEOUT_SECONDS")
    allowed_download_domains: str = Field(
        default="arxiv.org,export.arxiv.org", alias="ALLOWED_DOWNLOAD_DOMAINS"
    )

    # Versioning
    index_version: str = Field(default="v1", alias="INDEX_VERSION")
    embedding_version: str = Field(default="gemini-embedding-001-768", alias="EMBEDDING_VERSION")
    reranker_version: str = Field(default="ms-marco-MiniLM-L-12-v2", alias="RERANKER_VERSION")

    # Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    # Citations
    citation_style: str = Field(default="mla9", alias="CITATION_STYLE")

    # Performance
    fast_ingestion: bool = Field(default=False, alias="FAST_INGESTION")

    @field_validator("persist_directory", "download_cache_dir", "sqlite_sparse_db", "metadata_db")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def google_api_keys(self) -> list[str]:
        """Primary GOOGLE_API_KEY first, then unique keys from GOOGLE_API_KEYS."""
        placeholders = {"your_google_api_key_here", ""}
        keys: list[str] = []
        primary = self.google_api_key.strip()
        if primary and primary not in placeholders:
            keys.append(primary)
        raw = self.google_api_keys_env.strip()
        if raw and raw not in placeholders:
            for token in raw.replace("\n", ",").split(","):
                key = token.strip().strip("'").strip('"')
                if key and key not in placeholders and key not in keys:
                    keys.append(key)
        return keys

    @property
    def chroma_collection_name(self) -> str:
        return f"research_chunks_{self.index_version}"

    @property
    def allowed_domains(self) -> frozenset[str]:
        return frozenset(d.strip().lower() for d in self.allowed_download_domains.split(",") if d.strip())

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def discovery_source_list(self) -> list[str]:
        from research_assistant.discovery.sources import parse_discovery_sources

        return parse_discovery_sources(self.discovery_sources)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def apply_fast_settings(settings: Settings) -> Settings:
    """Tune discovery, chunking, and ingestion for ~1-minute cold-start targets."""
    return settings.model_copy(
        update={
            "fast_ingestion": True,
            "max_new_documents_per_query": 1,
            "max_discovery_rounds": 1,
            "discovery_max_results": 3,
            "chunk_target_tokens": 1200,
            "chunk_max_tokens": 1500,
            "chunk_overlap_tokens": 50,
            "embedding_batch_size": 64,
            "embedding_max_workers": 6,
            "io_max_workers": 4,
            "synthesis_max_passage_chars": 1800,
        }
    )
