"""The independent fact-checker used by the evals.

It comes from a DIFFERENT model family (Groq gpt-oss) than the agents (Gemini), so the system isn't
grading its own homework. It must quote evidence for every claim, which cuts down on false alarms.
How far to trust it is measured separately: see `evals.py --calibrate-judge`.
"""

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from synq_ai_squad.checks import DOCS_TEXT
from synq_ai_squad.config import Settings


class Claim(BaseModel):
    claim: str = Field(description="One specific factual claim from the text")
    evidence: str = Field(description="The sentence from the documents that supports it, quoted exactly, or '' if none")
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


def make_judge(settings: Settings) -> Runnable:
    model = ChatGroq(model=settings.judge_model, temperature=0, api_key=settings.require("groq_api_key"))
    # Groq occasionally returns an empty answer; just try again.
    return model.with_structured_output(Verdict, method="json_schema").with_retry(stop_after_attempt=3)


def judge_prompt(text: str) -> str:
    return f"""You are a fact-checker. List every specific factual claim the MARKETING TEXT makes about
Synq Logic: its services, how it works, timelines, results, numbers, prices, clients, and promises.
For each claim, quote the sentence in the COMPANY DOCUMENTS that supports it.
- Rewording counts as supported if the meaning is the same.
- Mark supported=false only if no sentence in the documents backs it up.
- Skip questions, calls to action, opinions, and general statements about business life
  (e.g. "Leads go cold if you wait") - those are not claims about Synq Logic.

COMPANY DOCUMENTS:
{DOCS_TEXT}

MARKETING TEXT:
{text}"""


def unsupported_claims(judge: Runnable, text: str) -> list[str]:
    verdict: Verdict = judge.invoke(judge_prompt(text))
    return [c.claim for c in verdict.claims if not c.supported]
