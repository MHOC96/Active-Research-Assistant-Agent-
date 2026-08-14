"""Shared domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    EQUATION = "equation"
    FIGURE_CAPTION = "figure_caption"
    LIST = "list"


class DocumentStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    PARSING = "PARSING"
    PARSED = "PARSED"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    INGESTED = "INGESTED"
    FAILED = "FAILED"


class ErrorType(StrEnum):
    ARXIV_SEARCH_FAILED = "ARXIV_SEARCH_FAILED"
    PDF_DOWNLOAD_FAILED = "PDF_DOWNLOAD_FAILED"
    PDF_VALIDATION_FAILED = "PDF_VALIDATION_FAILED"
    PDF_PARSE_FAILED = "PDF_PARSE_FAILED"
    CHUNKING_FAILED = "CHUNKING_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    CHROMA_WRITE_FAILED = "CHROMA_WRITE_FAILED"
    SQLITE_WRITE_FAILED = "SQLITE_WRITE_FAILED"
    INDEX_TRANSACTION_FAILED = "INDEX_TRANSACTION_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    RERANKING_FAILED = "RERANKING_FAILED"
    SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ChunkRecord(BaseModel):
    """A indexed document chunk with full provenance metadata."""

    chunk_id: str
    document_id: str
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None
    section: str | None = None
    subsection: str | None = None
    page: int | None = None
    chunk_index: int
    content_type: ContentType = ContentType.PARAGRAPH
    passage: str
    source: str = "arxiv"
    embedding_model: str = "models/text-embedding-004"
    embedding_dimension: int = 768

    @classmethod
    def make_chunk_id(cls, document_id: str, chunk_index: int) -> str:
        return f"{document_id}:{chunk_index}"

    @property
    def provenance(self) -> str:
        return f"[arXiv:{self.arxiv_id} | Chunk {self.chunk_index}]"


class DocumentRecord(BaseModel):
    document_id: str
    arxiv_id: str
    title: str
    content_hash: str | None = None
    source_url: str | None = None
    status: DocumentStatus = DocumentStatus.DISCOVERED
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None


class RetrievalHit(BaseModel):
    """Single retrieval result before or after reranking."""

    chunk_id: str
    passage: str
    document_id: str
    arxiv_id: str
    title: str
    section: str | None = None
    page: int | None = None
    chunk_index: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def provenance(self) -> str:
        return f"[arXiv:{self.arxiv_id} | Chunk {self.chunk_index}]"


class SufficiencyResult(BaseModel):
    sufficient: bool
    candidate_count: int
    top_score: float | None
    reason: str | None = None


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published_date: str | None = None
    updated_date: str | None = None
    pdf_url: str
    categories: list[str] = Field(default_factory=list)


class HybridRetrieveResult(BaseModel):
    query: str
    candidates: list[RetrievalHit]
    sufficiency: SufficiencyResult


class ParsedElement(BaseModel):
    """Structured document element extracted before chunking."""

    text: str
    content_type: ContentType
    section: str | None = None
    subsection: str | None = None
    page: int | None = None


class IngestionResult(BaseModel):
    document_id: str
    arxiv_id: str
    status: DocumentStatus
    chunk_count: int = 0
    skipped: bool = False
    error_type: ErrorType | None = None
    error_message: str | None = None


class DiscoveryRoundResult(BaseModel):
    round_number: int
    papers_discovered: int
    papers_selected: int
    papers_ingested: int
    ingestion_results: list[IngestionResult] = Field(default_factory=list)


class ActiveResearchResult(BaseModel):
    query: str
    request_id: str
    retrieval: HybridRetrieveResult
    discovery_rounds: list[DiscoveryRoundResult] = Field(default_factory=list)
    papers_discovered: int = 0
    papers_ingested: int = 0
    insufficient_message: str | None = None

    @property
    def sufficient(self) -> bool:
        return self.retrieval.sufficiency.sufficient


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    normalized_query: str
    query_type: str
    subqueries: list[str] = Field(default_factory=list)
    answer: str
    citations_valid: bool = True
    citation_errors: list[str] = Field(default_factory=list)
    subquery_results: list[ActiveResearchResult] = Field(default_factory=list)
    evidence_hits: list[RetrievalHit] = Field(default_factory=list)
    sufficient: bool = False
    insufficient_message: str | None = None
