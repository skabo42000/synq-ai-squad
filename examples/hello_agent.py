"""Step 1: your first LangGraph agent (one node).

Run:  uv run python examples/hello_agent.py "What is RAG?"

The three core LangGraph ideas, all in this file:
  1. STATE - a shared "notebook" every agent reads and writes.
  2. NODE  - one worker; just a normal Python function.
  3. EDGE  - an arrow saying who works next.
Later, the Manager, Researcher, Writer and Critic will each be a node.
"""

import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()  # puts GOOGLE_API_KEY from .env into the environment

# The AI model. Your current Gemini key only allows this one.
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)


# 1. STATE: what the notebook holds.
class State(TypedDict):
    question: str
    answer: str


# 2. NODE: reads the state, does work, returns only the fields it changed.
def answer_node(state: State) -> dict:
    reply = llm.invoke(f"Answer in 3 short sentences: {state['question']}")
    return {"answer": reply.text}


# 3. EDGES: wire the graph together.  START -> answer -> END
builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is a multi-agent AI system?"
    result = graph.invoke({"question": question})
    print("Question:", result["question"])
    print("Answer:  ", result["answer"])
