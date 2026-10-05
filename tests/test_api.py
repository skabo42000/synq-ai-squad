"""Tests for the web endpoint (api.py): password check, input limits, busy guard, error handling,
and the privacy/security boundaries (no public schema pages, no-store responses, no leaked error text).

The app runs for real (FastAPI's TestClient); only the AI behind it is a scripted fake.
No provider calls, no paid jobs.
"""

import logging

import pytest
from conftest import GOOD_DRAFT, ScriptedLLM, SpyIndex, make_plan, make_review
from fastapi.testclient import TestClient
from pydantic import SecretStr

from synq_ai_squad.api import create_app
from synq_ai_squad.squad import build_graph

PASSWORD = "test-password-0123456789"
BODY = {"request": "A LinkedIn post for dental clinics about missed calls"}


class NeverCalledGraph:
    """For requests that must be rejected before the squad runs."""

    def invoke(self, _):
        pytest.fail("a rejected request reached the squad")


@pytest.fixture
def client(settings):
    llm = ScriptedLLM(make_plan(), reviews=[make_review(9)], drafts=[GOOD_DRAFT])
    app = create_app(settings, graph=build_graph(llm, SpyIndex(), settings))
    with TestClient(app) as c:  # "with" runs the startup checks, like a real server start
        yield c


@pytest.fixture
def guarded_client(settings):
    with TestClient(create_app(settings, graph=NeverCalledGraph())) as c:
        yield c


def test_health_needs_no_password(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_the_page_is_served(client):
    assert "<title>Synq Content Squad</title>" in client.get("/").text


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_generate_rejects_a_missing_or_wrong_password(guarded_client, headers):
    assert guarded_client.post("/generate", json=BODY, headers=headers).status_code == 401


@pytest.mark.parametrize("text", ["hi", "x" * 501])
def test_requests_that_are_too_short_or_long_never_reach_the_squad(guarded_client, text):
    r = guarded_client.post("/generate", json={"request": text}, headers={"X-API-Key": PASSWORD})
    assert r.status_code == 422


def test_generate_returns_the_finished_piece(client):
    r = client.post("/generate", json=BODY, headers={"X-API-Key": PASSWORD})
    assert r.status_code == 200
    data = r.json()
    assert data["content"] == GOOD_DRAFT
    assert data["approved"] is True
    assert data["critic_scores"] == [9]


def test_a_second_request_while_busy_gets_429(client):
    client.app.state.busy.acquire()
    try:
        assert client.post("/generate", json=BODY, headers={"X-API-Key": PASSWORD}).status_code == 429
    finally:
        client.app.state.busy.release()


def test_a_crash_inside_the_squad_becomes_a_clean_502(settings, caplog):
    class BrokenGraph:
        def invoke(self, _):
            raise RuntimeError("SENSITIVE_PROVIDER_VALUE_SHOULD_NOT_APPEAR")

    with caplog.at_level(logging.ERROR), TestClient(create_app(settings, graph=BrokenGraph())) as c:
        r = c.post("/generate", json=BODY, headers={"X-API-Key": PASSWORD})
    assert r.status_code == 502
    assert "SENSITIVE_PROVIDER_VALUE" not in r.text  # not shown to the caller...
    assert "SENSITIVE_PROVIDER_VALUE" not in caplog.text  # ...and not written to the server logs
    assert "RuntimeError" in caplog.text  # the error type is still logged for debugging
    assert not c.app.state.busy.locked()  # the busy lock is always released


def test_responses_are_private_and_never_cached(guarded_client):
    for r in (guarded_client.get("/health"), guarded_client.post("/generate", json=BODY)):
        assert r.headers["Cache-Control"] == "no-store"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "no-referrer"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_public_schema_pages_are_disabled(guarded_client, path):
    assert guarded_client.get(path).status_code == 404


def test_the_server_refuses_to_start_with_a_weak_password(settings):
    weak = settings.model_copy(update={"squad_api_key": SecretStr("short")})
    with pytest.raises(RuntimeError, match="at least 20"), TestClient(create_app(weak, graph=NeverCalledGraph())):
        pass
