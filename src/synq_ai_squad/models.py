"""The one place where the squad talks to AI models.

The graph never creates a model itself. It gets an LLMClient, which has exactly three abilities:
  plan(prompt)        -> a Plan        (Manager)
  review(prompt)      -> a Review      (Critic)
  text(role, prompt)  -> plain text    (Researcher, Writer)
That makes the graph testable with a fake client (no keys, no cost), and it gives model routing,
retries and fallbacks a single home (Phase 2 of the upgrade plan).
"""

from typing import Literal, Protocol, cast

from langchain_google_genai import ChatGoogleGenerativeAI

from synq_ai_squad.config import Settings
from synq_ai_squad.schemas import Plan, Review

TextRole = Literal["research", "write"]


class LLMClient(Protocol):
    def plan(self, prompt: str) -> Plan: ...
    def review(self, prompt: str) -> Review: ...
    def text(self, role: TextRole, prompt: str) -> str: ...


class GeminiClient:
    """Every role on one Gemini model, as in the original squad."""

    def __init__(self, settings: Settings) -> None:
        key = settings.require("google_api_key")

        def model(temperature: float) -> ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(model=settings.agent_model, temperature=temperature, api_key=key)

        self._plan = model(0).with_structured_output(Plan)
        self._review = model(0).with_structured_output(Review)  # strict and consistent
        self._text = {"research": model(0), "write": model(0.7)}  # the Writer is a bit creative

    def plan(self, prompt: str) -> Plan:
        return cast(Plan, self._plan.invoke(prompt))

    def review(self, prompt: str) -> Review:
        return cast(Review, self._review.invoke(prompt))

    def text(self, role: TextRole, prompt: str) -> str:
        return self._text[role].invoke(prompt).text
