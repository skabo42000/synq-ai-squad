"""Tests for how the documents are split into searchable sections (rag.py).

Only the splitting is tested here; embedding needs an API key and is covered by the evals.
"""

from synq_ai_squad.config import KNOWLEDGE_DIR
from synq_ai_squad.rag import load_and_split


def test_every_document_is_split_into_sections():
    chunks = load_and_split()
    sources = {c.metadata["source"] for c in chunks}
    assert sources == {p.name for p in KNOWLEDGE_DIR.glob("*.md")}
    assert len(chunks) >= 20  # 28 today; a big drop means the splitting broke


def test_every_section_carries_its_document_title():
    # The "contextual chunk header" fix from Step 2: without it, the FAQ answer
    # "What does it cost?" never mentions Synq Logic and loses out in search.
    for c in load_and_split():
        if "section" in c.metadata:
            assert c.page_content.startswith(f"# {c.metadata['title']}\n")


def test_faq_answers_stay_whole():
    cost = [c for c in load_and_split() if c.metadata.get("section") == "What does it cost?"]
    assert len(cost) == 1
    assert "free strategy call" in cost[0].page_content
