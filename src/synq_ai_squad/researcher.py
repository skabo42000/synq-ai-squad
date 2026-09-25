"""Step 2b: the RAG Researcher agent (two nodes).

Run:  uv run python -m synq_ai_squad.researcher "What services do you offer?"
(Run `uv run python -m synq_ai_squad.rag` first to build the database.)

Graph:  START -> retrieve -> answer -> END
  retrieve: search the docs for the chunks most related to the question
  answer:   Gemini answers using ONLY those chunks, and cites where each fact came from
"""

import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from synq_ai_squad.rag import get_vectorstore

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
store = get_vectorstore()

TOP_K = 4  # how many chunks to hand to the model


class State(TypedDict):
    question: str
    sources: list[Document]  # filled in by "retrieve"
    answer: str              # filled in by "answer"


def retrieve(state: State) -> dict:
    results = store.similarity_search_with_relevance_scores(state["question"], k=TOP_K)
    sources = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 2)  # 1.0 = perfect match, lower = less related
        sources.append(doc)
    return {"sources": sources}


def answer(state: State) -> dict:
    # Label each chunk [1], [2]... so the model can cite them.
    context = "\n\n".join(
        f"[{i}] (from {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(state["sources"], 1)
    )
    prompt = f"""You are a careful researcher for Synq Logic.
Answer the question using ONLY the numbered sources below.
- After each fact, cite the source number in brackets, like [2].
- If the sources do not contain the answer, say exactly: "The documents don't cover this."
- Never guess or add facts, prices, or numbers that are not in the sources.

SOURCES:
{context}

QUESTION: {state['question']}"""
    return {"answer": llm.invoke(prompt).text}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
graph = builder.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What services does Synq Logic offer?"
    result = graph.invoke({"question": question})
    print("QUESTION:", result["question"], "\n")
    print("ANSWER:\n" + result["answer"], "\n")
    print("SOURCES USED:")
    for i, d in enumerate(result["sources"], 1):
        print(f"  [{i}] {d.metadata['source']} > {d.metadata.get('section', '-')}  (match {d.metadata['score']})")
