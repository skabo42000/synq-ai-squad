"""API boundaries tested with a fake graph: no provider calls or paid jobs."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("SQUAD_API_KEY", "offline-api-test-key-not-live")
    fake = ModuleType("synq_ai_squad.squad")
    fake.PASS_SCORE = 8
    fake.graph = SimpleNamespace(invoke=lambda _: pytest.fail("unauthorized request invoked graph"))
    monkeypatch.setitem(sys.modules, "synq_ai_squad.squad", fake)
    path = Path(__file__).parents[1] / "src/synq_ai_squad/api.py"
    spec = importlib.util.spec_from_file_location("offline_squad_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, TestClient(module.app)


def test_api_requires_key_and_never_caches_responses(api):
    _, client = api
    for headers in ({}, {"X-API-Key": "wrong"}):
        response = client.post("/generate", json={"request": "A synthetic request"}, headers=headers)
        assert response.status_code == 401
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/health").json() == {"status": "ok"}
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_invalid_input_is_rejected_before_graph(api):
    _, client = api
    response = client.post("/generate", headers={"X-API-Key": "offline-api-test-key-not-live"},
                           json={"request": "x" * 501})
    assert response.status_code == 422


def test_provider_errors_do_not_expose_sensitive_details_and_release_guard(api, capsys):
    module, client = api
    def fail(_):
        raise RuntimeError("SENSITIVE_PROVIDER_VALUE_SHOULD_NOT_APPEAR")
    module.graph.invoke = fail
    response = client.post("/generate", headers={"X-API-Key": "offline-api-test-key-not-live"},
                           json={"request": "A synthetic request"})
    assert response.status_code == 502
    assert "SENSITIVE_PROVIDER_VALUE" not in response.text
    assert "SENSITIVE_PROVIDER_VALUE" not in capsys.readouterr().out
    assert not module.busy.locked()
