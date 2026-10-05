"""The full squad: Manager -> parallel searches -> Researcher -> Writer <-> Critic.

Run:  uv run python -m synq_ai_squad.squad "Something for dental clinic owners about missed calls"
(Build the search index first: uv run python -m synq_ai_squad.rag)

Graph:
                  +-> search (query 1) -+
    START -> manager -> search (query 2) -+-> research -> write -> critique --(approved / out of rounds)--> END
                  +-> search (query 3) -+                  ^          |
                                                          +--(needs work)
Ideas used here:
  - PLANNER AGENT: the Manager turns a vague request into a structured Plan.
  - FAN-OUT with Send: one node launches several parallel copies of another node;
    a reducer (operator.add) merges their results before "research" runs.
  - CONDITIONAL EDGE: next_step() decides whether the draft goes back to the Writer.
  - DEPENDENCY INJECTION: build_graph() receives its AI client and search index instead of
    creating them, so tests can pass in fakes and run with no keys.
"""

import logging
import operator
import sys
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from synq_ai_squad import prompts
from synq_ai_squad.checks import rule_problems
from synq_ai_squad.config import Settings, configure_logging, get_settings
from synq_ai_squad.models import GeminiClient, LLMClient
from synq_ai_squad.schemas import Plan, Review

log = logging.getLogger(__name__)


class SearchIndex(Protocol):
    """Anything that can find relevant sections, e.g. LangChain's InMemoryVectorStore."""

    def similarity_search(self, query: str, k: int) -> list[Document]: ...


class State(TypedDict, total=False):
    request: str  # what the user typed
    plan: Plan
    found: Annotated[list[Document], operator.add]  # reducer: every parallel search adds its results
    research: str  # fact notes with [n] citations
    draft: str
    review: Review | None
    rounds: int
    scores: Annotated[list[int], operator.add]  # reducer: one score appended per round


class SearchTask(TypedDict):
    """The small input each parallel 'search' copy receives (via Send)."""

    query: str


def initial_state(request: str) -> State:
    return {"request": request, "found": [], "rounds": 0, "scores": []}


def build_graph(llm: LLMClient, index: SearchIndex, settings: Settings) -> CompiledStateGraph:
    def manager(state: State) -> dict[str, Any]:
        plan = llm.plan(prompts.manager_prompt(state["request"]))
        log.info("[manager]  %s for %s", plan.content_format, plan.audience)
        log.info("           angle: %s", plan.angle)
        for q in plan.search_queries:
            log.info("           search: %s", q)
        return {"plan": plan}

    def launch_searches(state: State) -> list[Send]:
        # One Send per query = one parallel copy of the "search" node.
        return [Send("search", {"query": q}) for q in state["plan"].search_queries]

    def search(state: SearchTask) -> dict[str, Any]:
        return {"found": index.similarity_search(state["query"], k=settings.results_per_search)}

    def research(state: State) -> dict[str, Any]:
        # Different searches often find the same section; keep each section only once.
        unique = list({d.page_content: d for d in state["found"]}.values())
        notes = llm.text("research", prompts.research_prompt(state["plan"], unique))
        log.info("[research] %d results -> %d unique sections -> fact notes", len(state["found"]), len(unique))
        return {"research": notes}

    def write(state: State) -> dict[str, Any]:
        prompt = prompts.writer_prompt(state["plan"], state["research"], state.get("draft", ""), state.get("review"))
        draft = llm.text("write", prompt)
        rounds = state.get("rounds", 0) + 1
        log.info("[write]    draft %d ready (%d words)", rounds, len(draft.split()))
        return {"draft": draft, "rounds": rounds}

    def critique(state: State) -> dict[str, Any]:
        review = llm.review(prompts.critic_prompt(state["plan"], state["research"], state["draft"]))

        # Code checks what code can check reliably; the AI checks the rest.
        if problems := rule_problems(state["draft"]):
            review.issues.extend(problems)
            review.score = min(review.score, settings.pass_score - 1)
        if review.unsupported_claims:
            review.score = min(review.score, settings.pass_score - 1)  # made-up facts can never pass

        log.info(
            "[critique] score %d/10, %d issues, %d unsupported claims",
            review.score,
            len(review.issues),
            len(review.unsupported_claims),
        )
        return {"review": review, "scores": [review.score]}

    def next_step(state: State) -> str:
        review = state["review"]
        assert review is not None  # critique always sets it before this runs
        if review.score >= settings.pass_score:
            return "done"
        if state["rounds"] >= settings.max_rounds:
            return "done"  # stop anyway; never loop forever
        return "revise"

    builder = StateGraph(State)
    builder.add_node("manager", manager)
    builder.add_node("search", search, input_schema=SearchTask)  # receives a SearchTask via Send, not the State
    builder.add_node("research", research)
    builder.add_node("write", write)
    builder.add_node("critique", critique)
    builder.add_edge(START, "manager")
    builder.add_conditional_edges("manager", launch_searches, ["search"])  # fan-out
    builder.add_edge("search", "research")  # fan-in: waits for all searches
    builder.add_edge("research", "write")
    builder.add_edge("write", "critique")
    builder.add_conditional_edges("critique", next_step, {"revise": "write", "done": END})
    return builder.compile()


def build_default_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """The real squad: Gemini for every agent, the saved search index from rag.py."""
    from synq_ai_squad.rag import get_vectorstore  # imported here so tests never touch the index file

    settings = settings or get_settings()
    return build_graph(GeminiClient(settings), get_vectorstore(settings), settings)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows terminals need this for AI text
    configure_logging()
    settings = get_settings()
    request = " ".join(sys.argv[1:]) or "A LinkedIn post about why answering leads after hours wins more customers"
    print(f"Request: {request}\n", flush=True)  # flush so it shows before the progress lines

    result = build_default_graph(settings).invoke(initial_state(request))

    r = result["review"]
    status = (
        "APPROVED" if r.score >= settings.pass_score else f"NOT APPROVED (stopped after {settings.max_rounds} rounds)"
    )
    print(f"\nScores per round: {result['scores']}  ->  {status}")
    if r.issues or r.unsupported_claims:
        print("Critic's remaining notes:", *r.issues, *r.unsupported_claims, sep="\n  - ")
    print("\n" + "=" * 60 + "\n" + result["draft"] + "\n" + "=" * 60)


if __name__ == "__main__":
    main()
