"""Query intent helpers for discovery routing."""

from __future__ import annotations

import re

CORPORATE_VENDOR_TERMS = frozenset(
    {
        "servicenow",
        "salesforce",
        "microsoft",
        "sap",
        "oracle",
        "workday",
        "atlassian",
        "jira",
        "zendesk",
        "snowflake",
        "databricks",
        "aws",
        "azure",
        "google cloud",
        "ibm",
        "cisco",
        "vmware",
        "splunk",
        "datadog",
    }
)

ACADEMIC_SIGNAL_TERMS = frozenset(
    {
        "arxiv",
        "paper",
        "papers",
        "research",
        "study",
        "meta-analysis",
        "systematic review",
        "graphrag",
        "transformer",
        "rag",
        "benchmark",
        "dataset",
    }
)


def is_corporate_query(query: str) -> bool:
    """Return True when the query targets vendor docs or enterprise products, not papers."""
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    if not normalized:
        return False

    if any(vendor in normalized for vendor in CORPORATE_VENDOR_TERMS):
        return True

    if "fortune 500" in normalized and not any(term in normalized for term in ACADEMIC_SIGNAL_TERMS):
        return True

    return False


def discovery_sources_for_query(query: str, enabled: list[str]) -> list[str]:
    """Reorder or narrow discovery sources based on query intent."""
    if not is_corporate_query(query):
        return enabled

    preferred = [source for source in ("web", "openalex", "semantic_scholar") if source in enabled]
    return preferred or enabled
