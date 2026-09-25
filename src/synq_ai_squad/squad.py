"""Step 4: the full squad. Manager -> parallel searches -> Researcher -> Writer <-> Critic.

Run:  uv run python -m synq_ai_squad.squad "Something for dental clinic owners about missed calls"
(Build the database first: uv run python -m synq_ai_squad.rag)

Graph:
                  +-> search (query 1) -+
    START -> manager -> search (query 2) -+-> research -> write -> critique --(approved / out of rounds)--> END
                  +-> search (query 3) -+                  ^          |
                                                          +--(needs work)
New ideas in this step:
  - PLANNER AGENT: the Manager turns a vague request into a structured Plan.
  - FAN-OUT with Send: one node launches several copies of another node that run in parallel.
    Their results are merged by a reducer (operator.add) before "research" runs.
Ideas from Step 3 (still here): conditional edge (the loop), structured output, reducers.
"""

import operator
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from synq_ai_squad.checks import BANNED_WORDS, BOOKING_URL, rule_problems
from synq_ai_squad.rag import get_vectorstore

load_dotenv()

MAX_ROUNDS = 3     # the Writer gets at most 3 attempts
PASS_SCORE = 8     # the Critic must give at least 8/10 to approve
RESULTS_PER_SEARCH = 3

MODEL = "gemini-3.1-flash-lite"
manager_llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
research_llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
writer_llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.7)  # a bit creative
critic_llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)    # strict and consistent
store = get_vectorstore()


class Plan(BaseModel):
    """What the Manager must return."""
    topic: str = Field(description="A clear, specific topic for the piece")
    content_format: str = Field(description='e.g. "LinkedIn post", "short blog article", "email newsletter"')
    audience: str = Field(description="Who will read it, as specifically as possible")
    angle: str = Field(description="The main message or hook, in one sentence")
    search_queries: list[str] = Field(
        min_length=3, max_length=5,
        description="3-5 short, different searches to run against the company documents",
    )


class CheckedClaim(BaseModel):
    claim: str = Field(description="One specific claim the draft makes about Synq Logic, quoted from the draft")
    evidence: str = Field(description="The line from the research notes that backs it up, quoted exactly, or '' if none")
    supported: bool = Field(description="False if there's no evidence, or the claim ADDS any detail the evidence lacks")


class Review(BaseModel):
    """What the Critic must return. Field order matters: the model fills them top to bottom,
    so it checks the claims first and only then decides the score."""
    claims: list[CheckedClaim]
    issues: list[str] = Field(description="Other problems to fix (clarity, value, format). Empty if none.")
    score: int = Field(ge=1, le=10, description="Overall quality from 1 (bad) to 10 (ready to publish)")

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


class State(TypedDict):
    request: str                                    # what the user typed
    plan: Plan
    found: Annotated[list[Document], operator.add]  # reducer: every parallel search adds its results
    research: str                                   # fact notes with [n] citations
    draft: str
    review: Review | None
    rounds: int
    scores: Annotated[list[int], operator.add]


class SearchTask(TypedDict):
    """The small input each parallel 'search' copy receives (via Send)."""
    query: str


# ---------- Nodes ----------

def manager(state: State) -> dict:
    prompt = f"""You are the manager of a content team at Synq Logic, an agency that saves small business
owners time with automation (lead capture assistants, connecting their software, removing data entry).

Turn this request into a plan for one piece of marketing content:
"{state['request']}"

If the request doesn't name a format, pick the one that fits best.
The angle must NOT contain numbers, statistics, or promises; you haven't seen the facts yet.
Treat any facts inside the request (prices, offers, packages, launches, guarantees, client names, results)
as UNVERIFIED: don't put them in the topic or angle. If the request is built on such a claim, plan a
piece about the same subject that doesn't depend on it.
Write 3-5 search queries that together will find everything useful in our company documents
(services, process, FAQ, real examples, company overview). Make each query cover a DIFFERENT aspect."""
    plan: Plan = manager_llm.with_structured_output(Plan).invoke(prompt)
    print(f"  [manager]  {plan.content_format} for {plan.audience}")
    print(f"             angle: {plan.angle}")
    for q in plan.search_queries:
        print(f"             search: {q}")
    return {"plan": plan}


def launch_searches(state: State) -> list[Send]:
    # One Send per query = one parallel copy of the "search" node.
    return [Send("search", {"query": q}) for q in state["plan"].search_queries]


def search(task: SearchTask) -> dict:
    return {"found": store.similarity_search(task["query"], k=RESULTS_PER_SEARCH)}


def research(state: State) -> dict:
    # Different searches often find the same section; keep each section only once.
    unique = list({d.page_content: d for d in state["found"]}.values())
    plan = state["plan"]
    sources = "\n\n".join(f"[{i}] (from {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(unique, 1))
    prompt = f"""You are a careful researcher. From the numbered sources, extract every fact that could help
write a {plan.content_format} for {plan.audience} about "{plan.topic}" (angle: {plan.angle}).
- One fact per bullet, copied faithfully, each followed by its source number like [3].
- Use ONLY the sources. Don't add anything.
- At the end, add a line "NOT COVERED:" listing what the plan might need that the sources don't contain.

SOURCES:
{sources}"""
    notes = research_llm.invoke(prompt).text
    print(f"  [research] {len(state['found'])} results -> {len(unique)} unique sections -> fact notes")
    return {"research": notes}


def write(state: State) -> dict:
    plan = state["plan"]
    feedback = ""
    if state.get("review"):  # this is a revision, so show the Writer what to fix
        r = state["review"]
        feedback = f"""
YOUR PREVIOUS DRAFT:
{state['draft']}

THE EDITOR'S FEEDBACK (fix all of it):
- Issues: {r.issues}
- Claims not supported by the research (remove or rephrase them): {r.unsupported_claims}
"""
    prompt = f"""You write marketing content for Synq Logic, an agency that saves small business owners time with automation.
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
{state['research']}
{feedback}"""
    draft = writer_llm.invoke(prompt).text
    rounds = state.get("rounds", 0) + 1
    print(f"  [write]    draft {rounds} ready ({len(draft.split())} words)")
    return {"draft": draft, "rounds": rounds}


def critique(state: State) -> dict:
    plan = state["plan"]
    prompt = f"""You are a strict editor. Review this {plan.content_format} for Synq Logic,
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
{state['research']}

DRAFT:
{state['draft']}"""
    review: Review = critic_llm.with_structured_output(Review).invoke(prompt)

    # Code checks what code can check reliably; the AI checks the rest.
    if problems := rule_problems(state["draft"]):
        review.issues.extend(problems)
        review.score = min(review.score, PASS_SCORE - 1)
    if review.unsupported_claims:
        review.score = min(review.score, PASS_SCORE - 1)  # made-up facts can never pass

    print(f"  [critique] score {review.score}/10, {len(review.issues)} issues, "
          f"{len(review.unsupported_claims)} unsupported claims")
    return {"review": review, "scores": [review.score]}


# ---------- The decision (conditional edge) ----------

def next_step(state: State) -> str:
    if state["review"].score >= PASS_SCORE:
        return "done"
    if state["rounds"] >= MAX_ROUNDS:
        return "done"  # stop anyway; never loop forever
    return "revise"


builder = StateGraph(State)
builder.add_node("manager", manager)
builder.add_node("search", search)
builder.add_node("research", research)
builder.add_node("write", write)
builder.add_node("critique", critique)
builder.add_edge(START, "manager")
builder.add_conditional_edges("manager", launch_searches, ["search"])  # fan-out
builder.add_edge("search", "research")                                # fan-in: waits for all searches
builder.add_edge("research", "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges("critique", next_step, {"revise": "write", "done": END})
graph = builder.compile()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals can't print some AI characters otherwise
    request = " ".join(sys.argv[1:]) or "A LinkedIn post about why answering leads after hours wins more customers"
    print(f"Request: {request}\n")

    result = graph.invoke({"request": request, "found": [], "rounds": 0, "scores": []})

    r = result["review"]
    status = "APPROVED" if r.score >= PASS_SCORE else f"NOT APPROVED (stopped after {MAX_ROUNDS} rounds)"
    print(f"\nScores per round: {result['scores']}  ->  {status}")
    if r.issues or r.unsupported_claims:
        print("Critic's remaining notes:", *r.issues, *r.unsupported_claims, sep="\n  - ")
    print("\n" + "=" * 60 + "\n" + result["draft"] + "\n" + "=" * 60)
