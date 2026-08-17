"""FastAPI application for local research assistant UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from research_assistant.bootstrap import ApplicationContext, build_application
from research_assistant.citations.styles import list_citation_styles, parse_citation_style
from research_assistant.config import apply_fast_settings, get_settings
from research_assistant.export.bundle import build_query_bundle_zip, slugify_query
from research_assistant.health import validate_configuration, validate_external_services
from research_assistant.models import CitationSpan, ResearchResponse
from research_assistant.utils.cancellation import RequestCancelledError, cancellation_registry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=8000)
    request_id: str | None = Field(default=None, max_length=64)
    citation_style: str = "mla9"
    fast: bool = True


class QueryResponse(BaseModel):
    request_id: str
    query: str
    normalized_query: str
    query_type: str
    subqueries: list[str]
    answer: str
    citation_style: str
    citations_valid: bool
    citation_errors: list[str]
    sufficient: bool
    insufficient_message: str | None = None
    papers_ingested: int = 0
    papers_discovered: int = 0
    source_count: int = 0
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    elapsed_seconds: float


class CancelResponse(BaseModel):
    cancelled: bool
    request_id: str


class ExportBundleRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1)
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    citation_style: str = "mla9"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting research assistant UI contexts ...")
    app.state.normal_context = build_application(settings)
    app.state.fast_context = build_application(apply_fast_settings(settings))
    try:
        yield
    finally:
        app.state.normal_context.close()
        app.state.fast_context.close()


def _build_query_response(response: ResearchResponse, elapsed_seconds: float) -> QueryResponse:
    papers_ingested = sum(result.papers_ingested for result in response.subquery_results)
    papers_discovered = sum(result.papers_discovered for result in response.subquery_results)
    indexed_ids = {hit.arxiv_id for hit in response.evidence_hits}
    source_count = len(indexed_ids)
    seen_external: set[tuple[str, str]] = set()
    for citation in response.external_citations:
        if citation.arxiv_id and citation.arxiv_id in indexed_ids:
            continue
        key = (citation.source, citation.url or citation.title)
        if key in seen_external:
            continue
        seen_external.add(key)
        source_count += 1

    return QueryResponse(
        request_id=response.request_id,
        query=response.query,
        normalized_query=response.normalized_query,
        query_type=response.query_type,
        subqueries=response.subqueries,
        answer=response.answer,
        citation_style=response.citation_style,
        citations_valid=response.citations_valid,
        citation_errors=response.citation_errors,
        sufficient=response.sufficient,
        insufficient_message=response.insufficient_message,
        papers_ingested=papers_ingested,
        papers_discovered=papers_discovered,
        source_count=source_count,
        citation_spans=response.citation_spans,
        elapsed_seconds=elapsed_seconds,
    )


async def _watch_client_disconnect(request: Request, request_id: str, token) -> None:
    try:
        while not token.is_cancelled:
            if await request.is_disconnected():
                logger.info("Client disconnected; cancelling request_id=%s", request_id)
                token.cancel()
                break
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Active Research Assistant",
        description="Local citation-grounded research UI",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        config_errors = validate_configuration(settings)
        service_errors = validate_external_services(settings) if not config_errors else []
        return {
            "ok": not config_errors and not service_errors,
            "config_errors": config_errors,
            "service_errors": service_errors,
            "citation_style": settings.citation_style,
            "fast_ingestion": settings.fast_ingestion,
        }

    @app.get("/api/citation-styles")
    async def citation_styles() -> list[dict[str, str]]:
        return [{"id": style_id, "description": description} for style_id, description in list_citation_styles()]

    @app.post("/api/query/cancel/{request_id}", response_model=CancelResponse)
    async def cancel_query(request_id: str) -> CancelResponse:
        if cancellation_registry.cancel(request_id):
            logger.info("Cancellation requested for request_id=%s", request_id)
            return CancelResponse(cancelled=True, request_id=request_id)
        raise HTTPException(status_code=404, detail="Request not found or already completed")

    @app.post("/api/query", response_model=QueryResponse)
    async def run_query(payload: QueryRequest, request: Request) -> QueryResponse:
        try:
            style = parse_citation_style(payload.citation_style)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        context: ApplicationContext = (
            app.state.fast_context if payload.fast else app.state.normal_context
        )

        request_id = payload.request_id or str(uuid.uuid4())
        token = cancellation_registry.register(request_id)
        watcher = asyncio.create_task(_watch_client_disconnect(request, request_id, token))
        started = time.perf_counter()

        def _execute() -> ResearchResponse:
            return context.orchestrator.answer(
                payload.query.strip(),
                citation_style=style,
                cancellation=token,
            )

        try:
            response = await asyncio.to_thread(_execute)
        except RequestCancelledError as exc:
            logger.info("Pipeline cancelled for request_id=%s stage=%s", request_id, exc.stage)
            raise HTTPException(status_code=499, detail="Request cancelled") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            cancellation_registry.unregister(request_id)

        return _build_query_response(response, round(time.perf_counter() - started, 2))

    @app.post("/api/export/bundle")
    async def export_bundle(payload: ExportBundleRequest) -> FileResponse:
        try:
            parse_citation_style(payload.citation_style)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        settings = get_settings()
        settings.exports_dir.mkdir(parents=True, exist_ok=True)
        zip_path = build_query_bundle_zip(
            query=payload.query.strip(),
            answer=payload.answer,
            citation_spans=payload.citation_spans,
            citation_style=payload.citation_style,
            download_cache_dir=settings.download_cache_dir,
            exports_dir=settings.exports_dir,
        )
        filename = f"{slugify_query(payload.query)}-bundle.zip"
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=filename,
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
