"""Tests for paste-to-cite query extraction."""

from research_assistant.orchestrator.paste_to_cite import (
    heuristic_citation_queries,
    is_paste_to_cite,
)


CLOUD_PARAGRAPH = (
    "Cloud computing architectures rely on containerization to package applications "
    "with their complete runtime dependencies, ensuring consistent behavior across "
    "disparate deployment environments. By abstracting the underlying operating "
    "system and hardware layers, lightweight container engines enable rapid horizontal "
    "scaling and efficient resource utilization compared to traditional virtualization. "
    "To manage these ephemeral instances at scale, automated orchestration frameworks "
    "handle service discovery, rolling updates, and self-healing mechanisms without "
    "requiring manual cluster administration."
)


def test_is_paste_to_cite_detects_paragraph():
    assert is_paste_to_cite(CLOUD_PARAGRAPH) is True
    assert is_paste_to_cite("What is Kubernetes?") is False


def test_heuristic_citation_queries_splits_by_sentence():
    queries = heuristic_citation_queries(CLOUD_PARAGRAPH, max_queries=4)

    assert len(queries) == 3
    joined = " ".join(queries).lower()
    assert "containerization" in joined or "container" in joined
    assert "orchestration" in joined or "rolling updates" in joined
    assert "virtualization" in joined or "scaling" in joined
    assert len(queries[0]) < len(CLOUD_PARAGRAPH)
