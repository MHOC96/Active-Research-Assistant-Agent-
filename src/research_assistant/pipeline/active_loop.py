"""Active literature discovery loop."""

from __future__ import annotations

import logging
import uuid

from research_assistant.config import Settings, get_settings
from research_assistant.discovery.arxiv import (
    ArxivDiscoveryService,
    deduplicate_papers,
    select_papers_for_ingestion,
)
from research_assistant.ingestion.worker import IngestionWorker
from research_assistant.models import (
    ActiveResearchResult,
    DiscoveryRoundResult,
    DocumentStatus,
    HybridRetrieveResult,
)
from research_assistant.retrieval.hybrid import HybridRetriever
from research_assistant.storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


class ActiveLiteraturePipeline:
    """Retrieve locally, discover on arXiv, ingest, and re-retrieve until sufficient."""

    def __init__(
        self,
        retriever: HybridRetriever,
        discovery: ArxivDiscoveryService,
        ingestion_worker: IngestionWorker,
        metadata: MetadataStore,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.discovery = discovery
        self.ingestion_worker = ingestion_worker
        self.metadata = metadata
        self.settings = settings or get_settings()

    def run(self, query: str, *, top_k: int | None = None) -> ActiveResearchResult:
        request_id = str(uuid.uuid4())
        logger.info("request_id=%s query=%s status=started", request_id, query)

        retrieval = self.retriever.retrieve(query, top_k=top_k)
        if retrieval.sufficiency.sufficient:
            logger.info("request_id=%s sufficiency=sufficient", request_id)
            return ActiveResearchResult(
                query=query,
                request_id=request_id,
                retrieval=retrieval,
            )

        discovery_rounds: list[DiscoveryRoundResult] = []
        papers_discovered = 0
        papers_ingested = 0

        for round_number in range(1, self.settings.max_discovery_rounds + 1):
            round_result = self._discovery_round(query, round_number)
            discovery_rounds.append(round_result)
            papers_discovered += round_result.papers_discovered
            papers_ingested += round_result.papers_ingested

            retrieval = self.retriever.retrieve(query, top_k=top_k)
            if retrieval.sufficiency.sufficient:
                logger.info(
                    "request_id=%s sufficiency=sufficient after round=%s",
                    request_id,
                    round_number,
                )
                return ActiveResearchResult(
                    query=query,
                    request_id=request_id,
                    retrieval=retrieval,
                    discovery_rounds=discovery_rounds,
                    papers_discovered=papers_discovered,
                    papers_ingested=papers_ingested,
                )

        insufficient_message = _insufficient_message(retrieval)
        logger.info("request_id=%s sufficiency=insufficient", request_id)
        return ActiveResearchResult(
            query=query,
            request_id=request_id,
            retrieval=retrieval,
            discovery_rounds=discovery_rounds,
            papers_discovered=papers_discovered,
            papers_ingested=papers_ingested,
            insufficient_message=insufficient_message,
        )

    def _discovery_round(self, query: str, round_number: int) -> DiscoveryRoundResult:
        discovered = self.discovery.search_arxiv(
            query,
            max_results=self.settings.discovery_max_results,
        )
        deduped = deduplicate_papers(discovered, self.metadata)
        selected = select_papers_for_ingestion(
            query,
            deduped,
            max_select=self.settings.max_new_documents_per_query,
        )

        ingestion_results = []
        ingested_count = 0
        for paper in selected:
            result = self.ingestion_worker.ingest_pdf_document(
                arxiv_id=paper.arxiv_id,
                pdf_url=paper.pdf_url,
                title=paper.title,
                authors=paper.authors,
                published_date=paper.published_date,
            )
            ingestion_results.append(result)
            if result.status == DocumentStatus.INGESTED and not result.skipped:
                ingested_count += 1

        return DiscoveryRoundResult(
            round_number=round_number,
            papers_discovered=len(discovered),
            papers_selected=len(selected),
            papers_ingested=ingested_count,
            ingestion_results=ingestion_results,
        )


def _insufficient_message(retrieval: HybridRetrieveResult) -> str:
    reason = retrieval.sufficiency.reason or "evidence remained below configured thresholds"
    return f"INSUFFICIENT_EVIDENCE: {reason}"
