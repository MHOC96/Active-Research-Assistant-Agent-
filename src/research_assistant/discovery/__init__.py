"""Active literature discovery."""

from research_assistant.discovery.arxiv import (
    ArxivDiscoveryService,
    deduplicate_papers,
    select_papers_for_ingestion,
)

__all__ = [
    "ArxivDiscoveryService",
    "deduplicate_papers",
    "select_papers_for_ingestion",
]
