"""Shared test helpers: a scripted fake AI and a fake search index.

They let the tests run the REAL graph and the REAL web app without any API key or cost.
The fake AI returns answers we script in advance and remembers every prompt it was sent,
so a test can check both what the squad decided and what it asked the models.
"""

import os
from collections.abc import Iterable

import pytest
from langchain_core.documents import Document

from synq_ai_squad.checks import BOOKING_URL
from synq_ai_squad.config import Settings
from synq_ai_squad.models import TextRole
from synq_ai_squad.schemas import CheckedClaim, Plan, Review

# Tests must never send traces to LangSmith, even if the developer's shell has tracing switched on.
os.environ["LANGSMITH_TRACING"] = "false"

GOOD_DRAFT = f"Every lead gets a friendly reply in seconds. Book a free strategy call: {BOOKING_URL}"


def make_plan(queries: Iterable[str] = ("lead capture", "booking", "faq")) -> Plan:
    return Plan(
        topic="Missed calls",
        content_format="LinkedIn post",
        audience="Dental clinic owners",
        angle="Answer every call",
        search_queries=list(queries),
    )


def make_review(score: int, issues: Iterable[str] = (), unsupported: Iterable[str] = ()) -> Review:
    claims = [CheckedClaim(claim=c, evidence="", supported=False) for c in unsupported]
    return Review(claims=claims, issues=list(issues), score=score)


class ScriptedLLM:
    """Implements the LLMClient interface with pre-written answers."""

    def __init__(self, plan: Plan, reviews: list[Review], drafts: list[str], notes: str = "- fact [1]") -> None:
        self._plan, self._reviews, self._drafts, self._notes = plan, iter(reviews), iter(drafts), notes
        self.prompts: dict[str, list[str]] = {"plan": [], "review": [], "research": [], "write": []}

    def plan(self, prompt: str) -> Plan:
        self.prompts["plan"].append(prompt)
        return self._plan

    def review(self, prompt: str) -> Review:
        self.prompts["review"].append(prompt)
        return next(self._reviews).model_copy(deep=True)  # the graph edits reviews; keep the script clean

    def text(self, role: TextRole, prompt: str) -> str:
        self.prompts[role].append(prompt)
        return self._notes if role == "research" else next(self._drafts)


class SpyIndex:
    """A search index that returns the same sections for every query and records the queries."""

    def __init__(self, docs: list[Document] | None = None) -> None:
        self.queries: list[str] = []
        self.docs = docs or [
            Document(page_content="# Services\n24/7 lead capture", metadata={"source": "services.md"}),
            Document(page_content="# FAQ\nWhat does it cost?", metadata={"source": "faq.md"}),
        ]

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.queries.append(query)
        return self.docs[:k]


@pytest.fixture
def settings() -> Settings:
    # Keys are set to None explicitly, so a real key in the environment can never leak into a test.
    return Settings(
        google_api_key=None,
        groq_api_key=None,
        squad_api_key="test-password-0123456789",
        pass_score=8,
        max_rounds=3,
        results_per_search=3,
    )
