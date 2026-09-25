"""Step 3: Researcher -> Writer -> Critic, with a revision loop.

Run:  uv run python -m synq_ai_squad.squad "Why after-hours lead capture matters" "LinkedIn post"
(Build the database first: uv run python -m synq_ai_squad.rag)

Graph:
    START -> research -> write -> critique --(approved, or out of rounds)--> END
                           ^          |
                           +--(needs work)

New ideas in this step:
  - CONDITIONAL EDGE: a function looks at the state and picks the next node. That's the loop.
  - STRUCTURED OUTPUT: the Critic returns a Python object (score, issues), not free text,
    so our code can make decisions with it.
  - REDUCER: `scores` uses operator.add, so each round's score is APPENDED instead of replaced.
  - SUBGRAPH: the whole Researcher from Step 2 is reused as a single step here.
"""

import operator
import re
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from synq_ai_squad.researcher import graph as researcher_graph

load_dotenv()

MAX_ROUNDS = 3     # the Writer gets at most 3 attempts
PASS_SCORE = 8     # the Critic must give at least 8/10 to approve

# Words our clients' readers shouldn't see. Checked by plain code, not by the AI.
BANNED_WORDS = ["n8n", "API", "webhook", "workflow", "LLM", "audit"]

writer_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)  # a bit creative
critic_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)    # strict and consistent


class Review(BaseModel):
    """What the Critic must return. Pydantic checks the shape for us."""
    score: int = Field(ge=1, le=10, description="Overall quality from 1 (bad) to 10 (ready to publish)")
    issues: list[str] = Field(description="Specific problems to fix. Empty if there are none.")
    unsupported_claims: list[str] = Field(
        description="Any fact, number, name, or promise in the draft that is NOT in the research notes."
    )


class State(TypedDict):
    topic: str
    content_format: str                          # e.g. "LinkedIn post", "short blog article"
    research: str                                # facts from the Researcher, with [n] citations
    draft: str
    review: Review | None
    rounds: int                                  # how many drafts the Writer has made
    scores: Annotated[list[int], operator.add]   # reducer: new scores are appended


# ---------- Nodes ----------

def research(state: State) -> dict:
    # Reuse the Step 2 Researcher graph as one step of this bigger graph.
    result = researcher_graph.invoke(
        {"question": f"List every fact from the documents that is useful for writing about: {state['topic']}"}
    )
    print(f"  [research] found {len(result['sources'])} relevant sections")
    return {"research": result["answer"]}


def write(state: State) -> dict:
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
Write a {state['content_format']} about: {state['topic']}

RULES:
- Use ONLY facts from the research notes below. Never invent client names, results, statistics, prices, or guarantees.
- If the notes can't fully support the topic (for example, it asks for numbers the notes don't have),
  choose an honest angle the notes DO support, and don't promise what the text can't deliver.
- Plain English for busy business owners. Focus on benefits, not technology.
- Never use these words: {", ".join(BANNED_WORDS)}.
- End with one clear call to action (for example, booking a free strategy call).
- Output only the finished text, with no source numbers like [1] and no commentary.

RESEARCH NOTES:
{state['research']}
{feedback}"""
    draft = writer_llm.invoke(prompt).text
    rounds = state.get("rounds", 0) + 1
    print(f"  [write]    draft {rounds} ready ({len(draft.split())} words)")
    return {"draft": draft, "rounds": rounds}


def critique(state: State) -> dict:
    prompt = f"""You are a strict editor. Review this {state['content_format']} for Synq Logic about "{state['topic']}".

Check:
1. Accuracy: every fact, number, and promise must appear in the research notes. List anything that doesn't.
2. Clarity: plain English for non-technical business owners, no jargon.
3. Value: focuses on benefits to the reader, not on technology.
4. Format: fits a {state['content_format']}, and ends with one clear call to action.
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
    found = [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", state["draft"], re.IGNORECASE)]
    if found:
        review.issues.append(f"Remove these banned words: {found}")
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
builder.add_node("research", research)
builder.add_node("write", write)
builder.add_node("critique", critique)
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges("critique", next_step, {"revise": "write", "done": END})
graph = builder.compile()


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "Why answering leads after hours wins more customers"
    content_format = sys.argv[2] if len(sys.argv) > 2 else "LinkedIn post"
    print(f"Topic: {topic}\nFormat: {content_format}\n")

    result = graph.invoke({"topic": topic, "content_format": content_format, "rounds": 0, "scores": []})

    r = result["review"]
    status = "APPROVED" if r.score >= PASS_SCORE else f"NOT APPROVED (stopped after {MAX_ROUNDS} rounds)"
    print(f"\nScores per round: {result['scores']}  ->  {status}")
    if r.issues or r.unsupported_claims:
        print("Critic's remaining notes:", *r.issues, *r.unsupported_claims, sep="\n  - ")
    print("\n" + "=" * 60 + "\n" + result["draft"] + "\n" + "=" * 60)
