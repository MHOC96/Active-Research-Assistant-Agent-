"""Application bootstrap and dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass

from research_assistant.config import Settings, get_settings
from research_assistant.discovery.arxiv import ArxivDiscoveryService
from research_assistant.embeddings.gemini import GeminiEmbeddingService
from research_assistant.ingestion.worker import IngestionWorker
from research_assistant.orchestrator.agent import ResearchOrchestrator
from research_assistant.orchestrator.llm import GroqLLMClient
from research_assistant.pipeline.active_loop import ActiveLiteraturePipeline
from research_assistant.reranking.flashrank_reranker import FlashRankReranker
from research_assistant.retrieval.hybrid import HybridRetriever
from research_assistant.storage.dense_index import DenseIndex
from research_assistant.storage.metadata_store import MetadataStore
from research_assistant.storage.sparse_index import SparseIndex


@dataclass
class ApplicationContext:
    settings: Settings
    metadata: MetadataStore
    sparse: SparseIndex
    orchestrator: ResearchOrchestrator

    def close(self) -> None:
        self.metadata.close()
        self.sparse.close()


def build_application(settings: Settings | None = None) -> ApplicationContext:
    """Construct a fully wired research assistant application."""
    settings = settings or get_settings()
    metadata = MetadataStore(settings.metadata_db)
    sparse = SparseIndex(settings.sqlite_sparse_db)
    dense = DenseIndex(settings)
    embedder = GeminiEmbeddingService(settings)
    retriever = HybridRetriever(
        dense,
        sparse,
        embedder,
        reranker=FlashRankReranker(settings),
        settings=settings,
    )
    ingestion_worker = IngestionWorker(metadata, dense, sparse, embedder, settings=settings)
    pipeline = ActiveLiteraturePipeline(
        retriever=retriever,
        discovery=ArxivDiscoveryService(settings),
        ingestion_worker=ingestion_worker,
        metadata=metadata,
        settings=settings,
    )
    llm = GroqLLMClient(settings)
    orchestrator = ResearchOrchestrator(pipeline, llm)
    return ApplicationContext(
        settings=settings,
        metadata=metadata,
        sparse=sparse,
        orchestrator=orchestrator,
    )
