"""Tests for local web UI."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from research_assistant.utils.cancellation import RequestCancelledError
from research_assistant.web.app import create_app


def test_health_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert "ok" in payload
    assert "config_errors" in payload


def test_citation_styles_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/api/citation-styles")
    assert response.status_code == 200
    styles = response.json()
    assert any(style["id"] == "mla9" for style in styles)


def test_index_page_served():
    with TestClient(create_app()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Active Research Assistant" in response.text


def test_cancel_unknown_request_returns_404():
    with TestClient(create_app()) as client:
        response = client.post("/api/query/cancel/does-not-exist")
    assert response.status_code == 404


def test_cancel_active_request():
    with TestClient(create_app()) as client:
        from research_assistant.utils.cancellation import cancellation_registry

        token = cancellation_registry.register("active-req")
        response = client.post("/api/query/cancel/active-req")
        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        assert token.is_cancelled is True
        cancellation_registry.unregister("active-req")


def test_query_returns_499_when_pipeline_cancelled():
    def _raise_cancel(*_args, **_kwargs):
        raise RequestCancelledError("req-499", stage="retrieval")

    with TestClient(create_app()) as client:
        client.app.state.fast_context.orchestrator.answer = _raise_cancel
        client.app.state.normal_context.orchestrator.answer = _raise_cancel
        response = client.post(
            "/api/query",
            json={
                "query": "transformer attention",
                "request_id": "req-499",
                "fast": True,
            },
        )

    assert response.status_code == 499
    assert response.json()["detail"] == "Request cancelled"
