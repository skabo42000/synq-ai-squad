"""Tests for the model gateway (models.py) and cost estimates (pricing.py). No network calls:
models are constructed with dummy keys, and outages are simulated with fake chat models."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import SecretStr

from synq_ai_squad.models import ModelGateway, error_label, is_transient, tier_for
from synq_ai_squad.pricing import PRICES, cost_usd


@pytest.fixture
def keys(settings):
    return settings.model_copy(
        update={"google_api_key": SecretStr("dummy-google"), "groq_api_key": SecretStr("dummy-groq")}
    )


# ---------- routing ----------


@pytest.mark.parametrize(
    "routing, final_round, tier",
    [
        ("all-cheap", False, "cheap"),
        ("all-cheap", True, "cheap"),
        ("all-strong", False, "strong"),
        ("cascade", False, "cheap"),  # cheap first...
        ("cascade", True, "strong"),  # ...strong only for the last chance
    ],
)
def test_tier_for_each_routing_strategy(routing, final_round, tier):
    assert tier_for(routing, final_round) == tier


def test_strong_routing_needs_a_groq_key(settings):
    no_groq = settings.model_copy(update={"google_api_key": SecretStr("dummy-google"), "groq_api_key": None})
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        ModelGateway(no_groq, routing="cascade")
    assert ModelGateway(no_groq, routing="all-cheap").routing == "all-cheap"  # cheap-only still works


# ---------- fallback during an outage ----------


class ProviderDown(BaseChatModel):
    """A chat model whose provider is having an outage."""

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        raise RuntimeError("503 UNAVAILABLE")

    @property
    def _llm_type(self) -> str:
        return "down"


def test_an_outage_on_one_provider_falls_back_to_the_other(keys):
    gw = ModelGateway(keys, routing="all-cheap")
    gw._models[("write", "cheap")] = ProviderDown()
    gw._models[("write", "strong")] = FakeListChatModel(responses=["draft from the backup provider"])
    assert gw.text("write", "Write a post") == "draft from the backup provider"


def test_without_a_backup_the_outage_is_raised(settings):
    gw = ModelGateway(settings.model_copy(update={"google_api_key": SecretStr("dummy-google")}), routing="all-cheap")
    gw._models[("write", "cheap")] = ProviderDown()
    with pytest.raises(RuntimeError, match="503"):
        gw.text("write", "Write a post")


def test_usage_starts_empty_and_resets(keys):
    gw = ModelGateway(keys)
    assert gw.usage()["est_cost_usd"] == 0
    gw.reset_usage()
    assert gw.usage()["tokens_by_role"] == {}


# ---------- error handling helpers ----------


class StatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_error_labels_never_include_the_provider_message():
    e = StatusError(503, "overloaded, account org_123 request text: secret plans")
    assert error_label(e) == "StatusError 503"
    assert error_label(ValueError("details")) == "ValueError"


@pytest.mark.parametrize(
    "error, transient",
    [
        (StatusError(503, "UNAVAILABLE"), True),
        (StatusError(429, "rate limit"), True),
        (TimeoutError("timeout"), True),
        (ValueError("bad schema"), False),
    ],
)
def test_transient_errors_are_recognised(error, transient):
    assert is_transient(error) is transient


# ---------- cost estimates ----------


def test_cost_uses_list_prices_per_million_tokens():
    usage = {"gemini-3.1-flash-lite": {"input_tokens": 1_000_000, "output_tokens": 100_000}}
    assert cost_usd(usage) == pytest.approx(0.25 + 0.15)


def test_unknown_models_cost_nothing_rather_than_crashing():
    assert cost_usd({"some-new-model": {"input_tokens": 5000, "output_tokens": 5000}}) == 0


def test_every_configured_model_has_a_price(settings):
    for model in (settings.agent_model, settings.strong_model, settings.judge_model):
        assert model in PRICES
