"""Document ingestion worker."""

from __future__ import annotations

import logging

from research_assistant.config import Settings, get_settings
from research_assistant.embeddings.base import EmbeddingService
from research_assistant.ingestion.chunker import SectionAwareChunker
from research_assistant.ingestion.downloader import SecurePdfDownloader
from research_assistant.ingestion.parser import DoclingParser
from research_assistant.models import DocumentRecord, DocumentStatus, ErrorType, IngestionResult
from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.index_transaction import IndexTransaction
from research_assistant.storage.metadata_store import MetadataStore
from research_assistant.storage.sparse_index import SparseIndex

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Download, parse, chunk, embed, and index academic PDFs."""

    def __init__(
        self,
        metadata: MetadataStore,
        dense: DenseIndex,
        sparse: SparseIndex,
        embedder: EmbeddingService,
        *,
        downloader: SecurePdfDownloader | None = None,
        parser: DoclingParser | None = None,
        chunker: SectionAwareChunker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.metadata = metadata
        self.index_tx = IndexTransaction(dense, sparse)
        self.embedder = embedder
        self.downloader = downloader or SecurePdfDownloader(self.settings)
        self.parser = parser or DoclingParser(settings=self.settings)
        self.chunker = chunker or SectionAwareChunker(self.settings)

    def ingest_pdf_document(
        self,
        *,
        arxiv_id: str,
        pdf_url: str,
        title: str,
        authors: list[str] | None = None,
        published_date: str | None = None,
    ) -> IngestionResult:
        document_id = MetadataStore.normalize_arxiv_id(arxiv_id)

        if self.metadata.is_ingested(document_id):
            logger.info("Skipping already ingested document %s", document_id)
            return IngestionResult(
                document_id=document_id,
                arxiv_id=document_id,
                status=DocumentStatus.INGESTED,
                skipped=True,
            )

        doc = DocumentRecord(
            document_id=document_id,
            arxiv_id=document_id,
            title=title,
            source_url=pdf_url,
            authors=authors or [],
            published_date=published_date,
            status=DocumentStatus.DISCOVERED,
        )
        self.metadata.upsert_document(doc)

        try:
            self._set_status(document_id, DocumentStatus.DOWNLOADING)
            download = self.downloader.download(pdf_url, document_id)

            existing = self.metadata.get_by_content_hash(download.content_hash)
            if existing and existing.status == DocumentStatus.INGESTED:
                self._set_status(document_id, DocumentStatus.INGESTED)
                return IngestionResult(
                    document_id=document_id,
                    arxiv_id=document_id,
                    status=DocumentStatus.INGESTED,
                    skipped=True,
                )

            doc.content_hash = download.content_hash
            doc.source_url = download.source_url
            doc.status = DocumentStatus.DOWNLOADED
            self.metadata.upsert_document(doc)

            self._set_status(document_id, DocumentStatus.PARSING)
            logger.info("Parsing PDF for %s ...", document_id)
            elements = self.parser.parse(download.path)
            self._set_status(document_id, DocumentStatus.PARSED)

            chunks = self.chunker.chunk(
                elements,
                document_id=document_id,
                arxiv_id=document_id,
                title=title,
                authors=authors,
                published_date=published_date,
            )

            self._set_status(document_id, DocumentStatus.EMBEDDING)
            logger.info("Embedding %s chunks for %s ...", len(chunks), document_id)
            embeddings = self.embedder.embed_documents([c.passage for c in chunks])
            logger.info("Indexed %s chunks for %s", len(chunks), document_id)

            self._set_status(document_id, DocumentStatus.INDEXING)
            self.index_tx.commit_chunks(chunks, embeddings, document_id=document_id)

            self._set_status(document_id, DocumentStatus.INGESTED)
            return IngestionResult(
                document_id=document_id,
                arxiv_id=document_id,
                status=DocumentStatus.INGESTED,
                chunk_count=len(chunks),
            )
        except Exception as exc:
            error_type = _classify_error(exc)
            self.index_tx.rollback_document(document_id)
            self._set_status(document_id, DocumentStatus.FAILED)
            logger.error(
                "Ingestion failed for %s operation=%s error=%s",
                document_id,
                error_type.value,
                exc,
            )
            return IngestionResult(
                document_id=document_id,
                arxiv_id=document_id,
                status=DocumentStatus.FAILED,
                error_type=error_type,
                error_message=str(exc),
            )

    def _set_status(self, document_id: str, status: DocumentStatus) -> None:
        self.metadata.update_status(document_id, status)


def _classify_error(exc: Exception) -> ErrorType:
    message = str(exc)
    for error_type in ErrorType:
        if error_type.value in message:
            return error_type
    if "path validation failure" in message.lower():
        return ErrorType.PDF_DOWNLOAD_FAILED
    return ErrorType.INDEX_TRANSACTION_FAILED
