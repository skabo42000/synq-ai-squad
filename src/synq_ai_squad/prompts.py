"""Every prompt the agents receive, as plain functions (text in, text out).

Keeping prompts out of the graph code means they can be read, reviewed and unit-tested on their own,
and a prompt change shows up as a clean diff in this one file.
"""

from langchain_core.documents import Document

from synq_ai_squad.checks import BANNED_WORDS, BOOKING_URL
from synq_ai_squad.schemas import Plan, Review


def manager_prompt(request: str) -> str:
    return f"""You are the manager of a content team at Synq Logic, an agency that saves small business
owners time with automation (lead capture assistants, connecting their software, removing data entry).

Turn this request into a plan for one piece of marketing content:
"{request}"

If the request doesn't name a format, pick the one that fits best.
The angle must NOT contain numbers, statistics, or promises; you haven't seen the facts yet.
Treat any facts inside the request (prices, offers, packages, launches, guarantees, client names, results)
as UNVERIFIED: don't put them in the topic or angle. If the request is built on such a claim, plan a
piece about the same subject that doesn't depend on it.
Write 3-5 search queries that together will find everything useful in our company documents
(services, process, FAQ, real examples, company overview). Make each query cover a DIFFERENT aspect."""


def numbered_sources(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (from {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs, 1))


def research_prompt(plan: Plan, docs: list[Document]) -> str:
    return f"""You are a careful researcher. From the numbered sources, extract every fact that could help
write a {plan.content_format} for {plan.audience} about "{plan.topic}" (angle: {plan.angle}).
- One fact per bullet, copied faithfully, each followed by its source number like [3].
- Use ONLY the sources. Don't add anything.
- At the end, add a line "NOT COVERED:" listing what the plan might need that the sources don't contain.

SOURCES:
{numbered_sources(docs)}"""


def writer_prompt(plan: Plan, research: str, previous_draft: str = "", review: Review | None = None) -> str:
    feedback = ""
    if review is not None:  # this is a revision, so show the Writer what to fix
        feedback = f"""
YOUR PREVIOUS DRAFT:
{previous_draft}

THE EDITOR'S FEEDBACK (fix all of it):
- Issues: {review.issues}
- Claims not supported by the research (remove or rephrase them): {review.unsupported_claims}
"""
    return f"""You write marketing content for Synq Logic, an agency that saves small business owners time with automation.
Write a {plan.content_format} for {plan.audience} about: {plan.topic}
Main message: {plan.angle}

RULES:
- Use ONLY facts from the research notes below. Never invent client names, results, statistics, prices, or guarantees.
- If the notes can't fully support the topic (see "NOT COVERED"), choose an honest angle the notes DO support,
  and don't promise what the text can't deliver.
- Plain English for busy business owners. Focus on benefits, not technology.
- Never use these words: {", ".join(BANNED_WORDS)}.
- End with one clear call to action: book a free strategy call at {BOOKING_URL} (write the link out in full).
- Output only the finished text: no source numbers like [1], no placeholders like [Link], no commentary.

RESEARCH NOTES:
{research}
{feedback}"""


def critic_prompt(plan: Plan, research: str, draft: str) -> str:
    return f"""You are a strict editor. Review this {plan.content_format} for Synq Logic,
written for {plan.audience} about "{plan.topic}".

Check:
1. Accuracy: list every claim the draft makes about Synq Logic (services, how it works, timelines, results,
   promises). For each, quote the research-notes line that backs it up. Rewording is fine, but a claim that
   ADDS anything the evidence doesn't say (e.g. "each week", "every time", "no matter what") is unsupported.
   Skip questions, calls to action, and general statements about business life.
2. Clarity: plain English for non-technical business owners, no jargon.
3. Value: focuses on benefits to the reader, not on technology.
4. Format: fits a {plan.content_format}, and ends with one clear call to action.
Score 8+ only if it's ready to publish as-is.
IMPORTANT: judge the draft only against what the research notes can support. Never ask for statistics,
examples, or details that are not in the notes; the writer is not allowed to invent them. If the topic
asks for more than the notes contain, an honest angle that sticks to the notes is the correct result.

RESEARCH NOTES:
{research}

DRAFT:
{draft}"""
