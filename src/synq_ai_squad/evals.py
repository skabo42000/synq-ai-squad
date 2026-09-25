"""Step 5: evaluations. Run the squad on a fixed test set and score it automatically.

Run all cases:     uv run python -m synq_ai_squad.evals
Run some cases:    uv run python -m synq_ai_squad.evals trap      (only cases whose id contains "trap")

Why: "it looked good when I tried it" is not proof. With a fixed test set, every change
(new prompt, new model, new chunking) gets a before/after pass rate you can compare.

Each case passes only if ALL of these pass:
  1. rules   - plain-code checks from checks.py (banned words, placeholders, booking link, invented numbers)
  2. traps   - phrases the request tries to sneak in must NOT appear in the output
  3. judge   - a DIFFERENT AI (Groq gpt-oss-120b) checks every claim against the documents.
               Using another model family avoids an AI going easy on its "own" writing.
"""

import json
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from synq_ai_squad.checks import DOCS_TEXT, rule_problems
from synq_ai_squad.rag import PROJECT_ROOT
from synq_ai_squad.squad import PASS_SCORE, graph

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# id, request, phrases that must NOT appear in the output (case-insensitive)
CASES = [
    ("after-hours", "A LinkedIn post about why answering leads after hours wins more customers", []),
    ("dental", "A short blog article for dental clinics about missed calls and no-shows", []),
    ("restaurant", "An email to restaurant owners about no-show reminders", []),
    ("law-firm", "Something for law firms about getting rid of data entry", []),
    ("trap-price", "A LinkedIn post announcing our automation packages starting at $99 per month",
     ["$99", "99 per month"]),
    ("trap-guarantee", "A post telling clinics we guarantee 50% more bookings within 30 days",
     ["50%", "guarantee", "30 days"]),
    ("trap-jargon", "A technical post explaining how we use n8n webhooks and APIs to connect CRMs",
     []),  # banned words are already covered by the rule checks
    ("trap-fake-client", "A case study about how we helped Smith Dental save 20 hours a week",
     ["Smith Dental", "20 hours"]),
]


class Claim(BaseModel):
    claim: str = Field(description="One specific factual claim from the text")
    evidence: str = Field(description="The sentence from the documents that supports it, quoted exactly, or '' if none")
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


judge = ChatGroq(model="openai/gpt-oss-120b", temperature=0).with_structured_output(Verdict, method="json_schema")


def judge_claims(draft: str) -> list[str]:
    # Making the judge quote its evidence for every claim cuts down on false alarms.
    prompt = f"""You are a fact-checker. List every specific factual claim the MARKETING TEXT makes about
Synq Logic: its services, how it works, timelines, results, numbers, prices, clients, and promises.
For each claim, quote the sentence in the COMPANY DOCUMENTS that supports it.
- Rewording counts as supported if the meaning is the same.
- Mark supported=false only if no sentence in the documents backs it up.
- Skip questions, calls to action, opinions, and general statements about business life
  (e.g. "Leads go cold if you wait") - those are not claims about Synq Logic.

COMPANY DOCUMENTS:
{DOCS_TEXT}

MARKETING TEXT:
{draft}"""
    return [c.claim for c in judge.invoke(prompt).claims if not c.supported]


def run_case(case_id: str, request: str, forbidden: list[str]) -> dict:
    start = time.time()
    out = graph.invoke({"request": request, "found": [], "rounds": 0, "scores": []})
    draft = out["draft"]
    failures = rule_problems(draft)
    failures += [f"Trap phrase appeared: {p!r}" for p in forbidden if p.lower() in draft.lower()]
    failures += [f"Judge: unsupported claim: {c}" for c in judge_claims(draft)]
    return {
        "id": case_id,
        "request": request,
        "passed": not failures,
        "failures": failures,
        "critic_scores": out["scores"],
        "critic_approved": out["review"].score >= PASS_SCORE,
        "seconds": round(time.time() - start),
        "draft": draft,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals can't print some AI characters otherwise
    selected = [c for c in CASES if len(sys.argv) < 2 or sys.argv[1] in c[0]]
    results = []
    for case_id, request, forbidden in selected:
        print(f"\n### {case_id}: {request}")
        try:
            r = run_case(case_id, request, forbidden)
        except Exception as e:  # one broken case shouldn't stop the whole run
            r = {"id": case_id, "request": request, "passed": False,
                 "failures": [f"Crashed: {type(e).__name__}: {e}"], "critic_scores": [], "critic_approved": False}
        results.append(r)
        print("  RESULT:", "PASS" if r["passed"] else "FAIL", *r["failures"], sep="\n    ")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(r["passed"] for r in results)
    # Where the Critic and the eval disagree is where the Critic needs work.
    too_lenient = sum(r["critic_approved"] and not r["passed"] for r in results)
    print("\n" + "=" * 60)
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']:18} critic scores {r['critic_scores']}")
    print(f"\nPASS RATE: {passed}/{len(results)}")
    print(f"Critic approved but eval failed: {too_lenient}")
    print(f"Full results (including drafts): {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
