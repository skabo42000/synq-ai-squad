"""Rule checks written in plain code: fast, free, and 100% consistent.

Used in two places:
  - the Critic (squad.py) runs them on every draft, so the loop fixes these problems
  - the evaluations (evals.py) run them to measure quality across many requests
"""

import re

from synq_ai_squad.rag import DOCS_DIR

BOOKING_URL = "https://calendly.com/synqlog/30min"
BANNED_WORDS = ["n8n", "API", "webhook", "workflow", "LLM", "audit"]

NUMBER = re.compile(r"\$?\d[\d,.]*%?")
URL = re.compile(r"https?://\S+")

# All the text in docs/ (the eval judge reads it to fact-check drafts).
DOCS_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.md")))


def numbers_in(text: str) -> set[str]:
    # Whole numbers only, ignoring digits inside links (e.g. "30min" in the booking URL).
    return {n.rstrip(".,") for n in NUMBER.findall(URL.sub("", text))}


DOCS_NUMBERS = numbers_in(DOCS_TEXT)


def banned_words(text: str) -> list[str]:
    # "s?" so plurals count too: "workflows", "webhooks", "APIs".
    return [w for w in BANNED_WORDS if re.search(rf"\b{re.escape(w)}s?\b", text, re.IGNORECASE)]


def placeholders(text: str) -> list[str]:
    # "[Link]" or "[Your Name]" style gaps. A real markdown link "[text](https://...)" is fine.
    return re.findall(r"\[[^\]]+\](?!\()", text)


def invented_numbers(text: str) -> list[str]:
    # Every number, price or percentage in the draft must appear in docs/ as a whole number.
    # (Matching any substring let "30 days" pass because of "30min" in the booking link.)
    return sorted(numbers_in(text) - DOCS_NUMBERS)


def rule_problems(text: str) -> list[str]:
    """Every rule broken by this text, in plain English. Empty list = all rules pass."""
    problems = []
    if found := banned_words(text):
        problems.append(f"Remove these banned words: {found}")
    if found := placeholders(text):
        problems.append(f"Replace these placeholders with real text: {found}")
    if BOOKING_URL not in text:
        problems.append(f"The call to action must include the booking link: {BOOKING_URL}")
    if found := invented_numbers(text):
        problems.append(f"These numbers are not in our documents, remove them: {found}")
    return problems
