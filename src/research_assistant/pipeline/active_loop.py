"""Active literature discovery loop."""

from __future__ import annotations

import logging
import uuid

from research_assistant.config import Settings, get_settings
from research_assistant.discovery.multi import (
    MultiSourceDiscoveryService,
    deduplicate_papers,
    external_citations_from_sources,
    flatten_for_ingestion,
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
from research_assistant.utils.cancellation import CancellationToken
from research_assistant.utils.concurrency import map_io_bound

logger = logging.getLogger(__name__)


class ActiveLiteraturePipeline:
    """Retrieve locally, discover across sources, ingest, and re-retrieve until sufficient."""

    def __init__(
        self,
        retriever: HybridRetriever,
        discovery: MultiSourceDiscoveryService,
        ingestion_worker: IngestionWorker,
        metadata: MetadataStore,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.discovery = discovery
        self.ingestion_worker = ingestion_worker
        self.metadata = metadata
        self.settings = settings or get_settings()

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        cancellation: CancellationToken | None = None,
        parent_request_id: str | None = None,
    ) -> ActiveResearchResult:
        request_id = parent_request_id or str(uuid.uuid4())
        logger.info("request_id=%s query=%s status=started", request_id, query)

        if cancellation is not None:
            cancellation.raise_if_cancelled("retrieval")

        retrieval = self.retriever.retrieve(query, top_k=top_k)
        if retrieval.sufficiency.sufficient:
            logger.info("request_id=%s sufficiency=sufficient", request_id)
            return ActiveResearchResult(
                query=query,
                request_id=request_id,
                retrieval=retrieval,
            )

        discovery_rounds: list[DiscoveryRoundResult] = []
        external_citations = []
        papers_discovered = 0
        papers_ingested = 0

        for round_number in range(1, self.settings.max_discovery_rounds + 1):
            if cancellation is not None:
                cancellation.raise_if_cancelled("discovery")

            round_result = self._discovery_round(
                query,
                round_number,
                cancellation=cancellation,
            )
            discovery_rounds.append(round_result)
            papers_discovered += round_result.papers_discovered
            papers_ingested += round_result.papers_ingested
            external_citations.extend(round_result.external_citations)

            if cancellation is not None:
                cancellation.raise_if_cancelled("retrieval")

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
                    external_citations=_unique_external_citations(external_citations),
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
            external_citations=_unique_external_citations(external_citations),
            insufficient_message=insufficient_message,
        )

    def _discovery_round(
        self,
        query: str,
        round_number: int,
        *,
        cancellation: CancellationToken | None = None,
    ) -> DiscoveryRoundResult:
        if cancellation is not None:
            cancellation.raise_if_cancelled("multi_source_search")

        by_source = self.discovery.search_by_source(query)
        discovered_total = sum(len(papers) for papers in by_source.values())
        external = external_citations_from_sources(by_source, query)

        all_discovered = flatten_for_ingestion(by_source)
        deduped = deduplicate_papers(all_discovered, self.metadata)
        selected = select_papers_for_ingestion(
            query,
            deduped,
            max_select=self.settings.max_new_documents_per_query,
        )

        if cancellation is not None:
            cancellation.raise_if_cancelled("pdf_download")

        self._prefetch_downloads(selected, cancellation=cancellation)

        ingestion_results = []
        ingested_count = 0
        for paper in selected:
            if cancellation is not None:
                cancellation.raise_if_cancelled("ingestion")

            arxiv_paper = paper.to_arxiv_paper()
            if arxiv_paper is None:
                continue

            result = self.ingestion_worker.ingest_pdf_document(
                arxiv_id=arxiv_paper.arxiv_id,
                pdf_url=arxiv_paper.pdf_url,
                title=arxiv_paper.title,
                authors=arxiv_paper.authors,
                published_date=arxiv_paper.published_date,
            )
            ingestion_results.append(result)
            if result.status == DocumentStatus.INGESTED and not result.skipped:
                ingested_count += 1

        return DiscoveryRoundResult(
            round_number=round_number,
            papers_discovered=discovered_total,
            papers_selected=len(selected),
            papers_ingested=ingested_count,
            ingestion_results=ingestion_results,
            external_citations=external,
            discovered_by_source={source: len(papers) for source, papers in by_source.items()},
        )

    def _prefetch_downloads(
        self,
        papers,
        *,
        cancellation: CancellationToken | None = None,
    ) -> None:
        ingestible = [paper for paper in papers if paper.ingestible]
        if not ingestible or self.settings.io_max_workers <= 1:
            return

        def _download(paper) -> None:
            if cancellation is not None:
                cancellation.raise_if_cancelled("pdf_download")
            arxiv_paper = paper.to_arxiv_paper()
            if arxiv_paper is None:
                return
            try:
                self.ingestion_worker.downloader.download(
                    arxiv_paper.pdf_url,
                    arxiv_paper.arxiv_id,
                )
            except Exception as exc:
                logger.warning("Prefetch download failed for %s: %s", arxiv_paper.arxiv_id, exc)

        logger.info("Prefetching %s PDFs in parallel ...", len(ingestible))
        map_io_bound(_download, ingestible, max_workers=self.settings.io_max_workers)


def _insufficient_message(retrieval: HybridRetrieveResult) -> str:
    reason = retrieval.sufficiency.reason or "evidence remained below configured thresholds"
    return f"INSUFFICIENT_EVIDENCE: {reason}"


def _unique_external_citations(citations):
    from research_assistant.models import ExternalCitation

    seen: set[tuple[str, str]] = set()
    unique: list[ExternalCitation] = []
    for citation in citations:
        key = (citation.source, citation.url or citation.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique
