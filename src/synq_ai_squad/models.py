"""The model gateway: the one place where the squad talks to AI models.

The graph never creates a model itself. It gets an LLMClient with three abilities:
  plan(prompt)                          -> a Plan        (Manager)
  review(prompt, final_round=...)       -> a Review      (Critic)
  text(role, prompt, final_round=...)   -> plain text    (Researcher, Writer)
The graph only says WHAT it needs and whether this is the last chance (the final round).
The gateway decides WHICH model answers, and adds what production use needs:

  - Routing: two tiers, cheap (Gemini flash-lite) and strong (Qwen on Groq). The strategy
    (Settings.routing) picks the tier per call:
      all-cheap  every call on the cheap tier
      all-strong every call on the strong tier
      cascade    cheap tier first; the Writer and Critic move to the strong tier only for the final round,
                 after the cheap tier has already failed twice. You pay for the strong model only when needed.
  - Time limits and retries on every call (a stuck provider can no longer hang a request for minutes).
  - Fallback: if a provider fails (outage, rate limit, bad output), the same call goes to the other provider.
  - Usage records: tokens per role and per model, so every run has a cost.
"""

import threading
from collections import defaultdict
from typing import Any, Literal, Protocol, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

from synq_ai_squad.config import Settings
from synq_ai_squad.pricing import cost_usd
from synq_ai_squad.schemas import Plan, Review

TextRole = Literal["research", "write"]
Role = Literal["manager", "research", "write", "critic"]
Tier = Literal["cheap", "strong"]
TEMPERATURE: dict[Role, float] = {"manager": 0, "research": 0, "write": 0.7, "critic": 0}  # the Writer is creative

TRANSIENT_MARKERS = ("429", "503", "500", "resource_exhausted", "unavailable", "rate limit", "overloaded", "timeout")


def error_label(e: BaseException) -> str:
    """A log-safe description: the error type plus its numeric status code, never the provider's message
    (provider messages can echo request data or account details)."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    return f"{type(e).__name__} {code}" if isinstance(code, int) else type(e).__name__


def is_transient(e: BaseException) -> bool:
    """An outage, rate limit or timeout: worth retrying, unlike a bug or a bad request."""
    text = f"{type(e).__name__} {getattr(e, 'status_code', '')} {getattr(e, 'code', '')} {e}".lower()
    return any(m in text for m in TRANSIENT_MARKERS)


class LLMClient(Protocol):
    def plan(self, prompt: str) -> Plan: ...
    def review(self, prompt: str, *, final_round: bool = False) -> Review: ...
    def text(self, role: TextRole, prompt: str, *, final_round: bool = False) -> str: ...


def tier_for(routing: str, final_round: bool) -> Tier:
    if routing == "all-strong":
        return "strong"
    if routing == "cascade" and final_round:
        return "strong"
    return "cheap"


class ModelGateway:
    """Implements LLMClient with routing, time limits, cross-provider fallback and usage records.

    Usage records are kept per gateway; the API and the evals run one squad job at a time, so
    reset_usage() before a run and usage() after it describe exactly that run.
    """

    def __init__(self, settings: Settings, routing: str | None = None) -> None:
        self.routing = routing or settings.routing
        self._models: dict[tuple[Role, Tier], BaseChatModel] = {}
        for role, temperature in TEMPERATURE.items():
            self._models[(role, "cheap")] = ChatGoogleGenerativeAI(
                model=settings.agent_model,
                temperature=temperature,
                api_key=settings.require("google_api_key"),
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
            if settings.groq_api_key is not None:  # without a Groq key: cheap tier only, no fallback
                self._models[(role, "strong")] = ChatGroq(
                    model=settings.strong_model,
                    temperature=temperature,
                    api_key=settings.groq_api_key,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                )
        if self.routing != "all-cheap" and settings.groq_api_key is None:
            raise RuntimeError(f"Routing {self.routing!r} needs the strong tier: set GROQ_API_KEY.")
        self._lock = threading.Lock()
        self._usage: dict[str, UsageMetadataCallbackHandler] = defaultdict(UsageMetadataCallbackHandler)

    # ---------- the LLMClient interface ----------

    def plan(self, prompt: str) -> Plan:
        return cast(Plan, self._call("manager", prompt, final_round=False, schema=Plan))

    def review(self, prompt: str, *, final_round: bool = False) -> Review:
        return cast(Review, self._call("critic", prompt, final_round=final_round, schema=Review))

    def text(self, role: TextRole, prompt: str, *, final_round: bool = False) -> str:
        return str(self._call(role, prompt, final_round=final_round).text)

    # ---------- routing, fallback, usage ----------

    def _runnable(self, role: Role, tier: Tier, schema: type[BaseModel] | None) -> Runnable:
        def wrap(model: BaseChatModel) -> Runnable:
            return model.with_structured_output(schema) if schema else model

        primary = wrap(self._models[(role, tier)])
        backup = self._models.get((role, "strong" if tier == "cheap" else "cheap"))
        # with_fallbacks: if the primary provider raises (outage, rate limit, timeout, unusable output),
        # the same prompt goes to the other provider instead of failing the whole request.
        return primary.with_fallbacks([wrap(backup)]) if backup else primary

    def _call(self, role: Role, prompt: str, *, final_round: bool, schema: type[BaseModel] | None = None) -> Any:
        tier = tier_for(self.routing, final_round)
        config: Any = {
            "callbacks": [self._usage[role]],
            "tags": [f"role:{role}", f"tier:{tier}"],  # visible in LangSmith traces
            "metadata": {"role": role, "tier": tier, "routing": self.routing},
        }
        return self._runnable(role, tier, schema).invoke(prompt, config=config)

    def reset_usage(self) -> None:
        with self._lock:
            self._usage.clear()

    def usage(self) -> dict[str, Any]:
        """Tokens by role and by model for the calls since reset_usage(), with the estimated cost."""
        with self._lock:
            by_role = {role: dict(h.usage_metadata) for role, h in self._usage.items()}
        by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for models in by_role.values():
            for model, u in models.items():
                counts = cast(dict[str, int], u)
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    by_model[model][k] += int(counts.get(k, 0))
        return {
            "routing": self.routing,
            "tokens_by_role": {r: sum(int(u["total_tokens"]) for u in m.values()) for r, m in by_role.items()},
            "tokens_by_model": {m: dict(u) for m, u in by_model.items()},
            "est_cost_usd": round(cost_usd(by_model), 6),
        }
