"""Active literature discovery."""

from research_assistant.discovery.arxiv import ArxivDiscoveryService
from research_assistant.discovery.multi import (
    MultiSourceDiscoveryService,
    deduplicate_papers,
    external_citations_from_sources,
    flatten_for_ingestion,
    select_papers_for_ingestion,
)
from research_assistant.discovery.openalex import OpenAlexDiscoveryService
from research_assistant.discovery.semantic_scholar import SemanticScholarDiscoveryService

__all__ = [
    "ArxivDiscoveryService",
    "MultiSourceDiscoveryService",
    "OpenAlexDiscoveryService",
    "SemanticScholarDiscoveryService",
    "deduplicate_papers",
    "external_citations_from_sources",
    "flatten_for_ingestion",
    "select_papers_for_ingestion",
]
