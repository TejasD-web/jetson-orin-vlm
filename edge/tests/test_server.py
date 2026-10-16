"""Smoke tests for the edge inference service."""

import pytest
from fastapi.testclient import TestClient

from src.server import app


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_list_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_chat_completions_returns_501():
    """Until llama-server forwarding is wired, expect 501."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 501
