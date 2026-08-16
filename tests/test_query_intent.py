"""Tests for corporate query intent routing."""

from research_assistant.discovery.query_intent import (
    discovery_sources_for_query,
    is_corporate_query,
)


def test_servicenow_query_is_corporate():
    query = "ServiceNow architecture and its benefits for Fortune 500 enterprises"
    assert is_corporate_query(query) is True


def test_research_query_is_not_corporate():
    assert is_corporate_query("Compare RAG and GraphRAG accuracy in recent papers") is False
    assert is_corporate_query("Transformer attention mechanism in language models") is False


def test_corporate_query_prefers_web_sources():
    enabled = ["arxiv", "openalex", "semantic_scholar", "web"]
    sources = discovery_sources_for_query(
        "ServiceNow ITSM multi-instance architecture",
        enabled,
    )
    assert sources[0] == "web"
    assert "arxiv" not in sources
