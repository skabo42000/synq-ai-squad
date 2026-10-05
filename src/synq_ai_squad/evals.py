"""Run the evaluation set and score the squad automatically.

    uv run python -m synq_ai_squad.evals                        # every case once (~15 min), report + quality gate
    uv run python -m synq_ai_squad.evals --only fabrication     # one category, or cases whose id contains the text
    uv run python -m synq_ai_squad.evals --repeats 3            # run each case 3 times: shows run-to-run spread
    uv run python -m synq_ai_squad.evals --compare A.json B.json   # what changed between two saved reports
    uv run python -m synq_ai_squad.evals --calibrate-judge      # how well the judge agrees with human labels

Why: "it looked good when I tried it" is not proof. The dataset is fixed and versioned (evals/dataset.yaml),
so every change (prompt, model, retrieval) gets a before/after number, and the quality gate
(evals/thresholds.yaml) makes the command exit with an error when quality drops, which fails CI.

Outputs:
  evals/results/<stamp>.json   every run including the full drafts (git-ignored; a CI artifact)
  evals/reports/<stamp>.md     readable report        } small enough to commit,
  evals/reports/<stamp>.json   summary for --compare  } so the history of results lives in git
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import Runnable
from langgraph.graph.state import CompiledStateGraph

from synq_ai_squad.config import Settings, configure_logging, get_settings
from synq_ai_squad.evaluation.dataset import EVALS_DIR, Case, load_dataset, load_labels, load_thresholds
from synq_ai_squad.evaluation.judge import make_judge, unsupported_claims
from synq_ai_squad.evaluation.scoring import (
    RunResult,
    Summary,
    calibration_metrics,
    check_gate,
    inconclusive_reason,
    is_pass,
    render_comparison,
    render_report,
    score_draft,
    summarize,
)
from synq_ai_squad.squad import build_default_graph, initial_state

log = logging.getLogger(__name__)

RETRY_WAITS = (30, 90)  # seconds to wait before retrying a run that hit a provider outage or rate limit
TRANSIENT_MARKERS = ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate limit", "overloaded", "timeout")
# A daily allowance won't refill in 90 seconds: retrying only burns time, so the run stops instead.
DAILY_QUOTA_MARKERS = ("per day", "perday", "tokens per day", "requests per day", "(TPD)", "(RPD)")
EXIT_PASS, EXIT_GATE_FAILED, EXIT_BAD_INPUT, EXIT_INCONCLUSIVE = 0, 1, 2, 3


class DailyQuotaExhausted(Exception):
    pass


def _error_text(e: Exception) -> str:
    return f"{type(e).__name__} {getattr(e, 'code', '')} {e}".lower()


def is_transient(e: Exception) -> bool:
    return any(m.lower() in _error_text(e) for m in TRANSIENT_MARKERS)


def is_daily_quota(e: Exception) -> bool:
    return any(m.lower() in _error_text(e) for m in DAILY_QUOTA_MARKERS)


def run_once(graph: CompiledStateGraph, judge: Runnable, settings: Settings, case: Case, repeat: int) -> RunResult:
    start = time.time()
    with get_usage_metadata_callback() as usage:
        out = graph.invoke(initial_state(case.request))
    seconds = time.time() - start
    failures = score_draft(case, out["draft"], unsupported_claims(judge, out["draft"]))
    return RunResult(
        case_id=case.id,
        category=case.category,
        repeat=repeat,
        passed=is_pass(failures),
        failures=failures,
        critic_scores=out["scores"],
        critic_approved=out["review"].score >= settings.pass_score,
        rounds=out["rounds"],
        seconds=round(seconds, 1),
        tokens=sum(u["total_tokens"] for u in usage.usage_metadata.values()),
        draft=out["draft"],
    )


def run_with_retries(
    graph: CompiledStateGraph, judge: Runnable, settings: Settings, case: Case, repeat: int
) -> RunResult:
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        try:
            return run_once(graph, judge, settings, case, repeat)
        except Exception as e:
            if is_daily_quota(e):
                raise DailyQuotaExhausted(type(e).__name__) from e
            if wait is None or not is_transient(e):
                # Record the error type only: provider messages can echo request data or account details.
                log.warning("  %s crashed: %s", case.id, type(e).__name__)
                return RunResult(
                    case_id=case.id,
                    category=case.category,
                    repeat=repeat,
                    passed=False,
                    failures={"crash": [type(e).__name__]},
                )
            log.warning("  %s: provider unavailable (%s), retry %d in %ds", case.id, type(e).__name__, attempt, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.getenv("GITHUB_SHA", "unknown")[:7]


def save_outputs(stamp: str, results: list[RunResult], summary: Summary, report: str) -> Path:
    (EVALS_DIR / "results").mkdir(parents=True, exist_ok=True)
    (EVALS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    full = EVALS_DIR / "results" / f"{stamp}.json"
    full.write_text(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    without_drafts = [r.model_dump(exclude={"draft"}) for r in results]
    report_json = EVALS_DIR / "reports" / f"{stamp}.json"
    report_json.write_text(
        json.dumps({"summary": summary.model_dump(), "runs": without_drafts}, indent=2), encoding="utf-8"
    )
    report_md = EVALS_DIR / "reports" / f"{stamp}.md"
    report_md.write_text(report, encoding="utf-8")
    if summary_file := os.getenv("GITHUB_STEP_SUMMARY"):  # shows the report on the GitHub Actions run page
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(report)
    return report_md


def evaluate(only: str | None, repeats: int) -> int:
    settings = get_settings()
    cases = load_dataset().select(only)
    if not cases:
        log.error("No cases match %r", only)
        return EXIT_BAD_INPUT
    graph, judge = build_default_graph(settings), make_judge(settings)

    results: list[RunResult] = []
    total = len(cases) * repeats
    try:
        for repeat in range(1, repeats + 1):
            for case in cases:
                log.info("[%d/%d] %s (repeat %d): %s", len(results) + 1, total, case.id, repeat, case.request[:70])
                r = run_with_retries(graph, judge, settings, case, repeat)
                results.append(r)
                reasons = [m for msgs in r.failures.values() for m in msgs]
                log.info("    -> %s %s", "PASS" if r.passed else "FAIL", "; ".join(reasons)[:300])
    except DailyQuotaExhausted as e:
        log.error("A provider's daily quota is used up (%s); stopping the run early.", e)

    summary = summarize(results)
    inconclusive = inconclusive_reason(summary, total)
    # The gate only applies to a complete, trustworthy run of the full dataset.
    gate = check_gate(summary, load_thresholds()) if not only and not inconclusive else []
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    meta = {
        "date": stamp,
        "commit": git_sha(),
        "agent model": settings.agent_model,
        "judge model": settings.judge_model,
        "cases": f"{len(cases)}" + (f" (filter: {only})" if only else ""),
        "repeats": str(repeats),
    }
    report_path = save_outputs(stamp, results, summary, render_report(summary, results, meta, gate, inconclusive))
    print("\n" + report_path.read_text(encoding="utf-8"))
    print(f"Report: {report_path.relative_to(EVALS_DIR.parent)}")
    if only:
        print("(quality gate skipped: it only applies to the full dataset)")
    if inconclusive:
        return EXIT_INCONCLUSIVE
    return EXIT_GATE_FAILED if gate else EXIT_PASS


def compare(a: Path, b: Path) -> int:
    def load(p: Path) -> Summary:
        return Summary.model_validate(json.loads(p.read_text(encoding="utf-8"))["summary"])

    print(render_comparison(load(a), load(b), a.stem, b.stem))
    return 0


def calibrate_judge() -> int:
    settings = get_settings()
    judge, labels = make_judge(settings), load_labels()
    flagged = []
    for i, label in enumerate(labels, 1):
        found = unsupported_claims(judge, label.claim)
        flagged.append(bool(found))
        human = "supported" if label.supported else "UNSUPPORTED"
        verdict = "UNSUPPORTED" if found else "supported"
        mark = "agree" if bool(found) != label.supported else "DISAGREE"
        log.info("[%d/%d] %-8s human=%s judge=%s  %s", i, len(labels), mark, human, verdict, label.claim)
    c = calibration_metrics(labels, flagged)
    print(f"\nJudge calibration on {c.total} human-labelled claims ({settings.judge_model}):")
    print(f"  accuracy  {c.accuracy:.0%}")
    print(f"  precision {c.precision:.0%}  (of claims the judge flagged, share that really were unsupported)")
    print(f"  recall    {c.recall:.0%}  (of unsupported claims, share the judge caught)")
    print("Disagreements:", *c.disagreements or ["none"], sep="\n  - ")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows terminals need this for AI text
    configure_logging()
    p = argparse.ArgumentParser(description="Evaluate the squad on the fixed dataset in evals/dataset.yaml.")
    p.add_argument("only_positional", nargs="?", help=argparse.SUPPRESS)  # old style: `evals trap`
    p.add_argument("--only", help="a category name, or text that case ids must contain")
    p.add_argument("--repeats", type=int, default=1, help="run every case this many times (default 1)")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE.json", "AFTER.json"))
    p.add_argument("--calibrate-judge", action="store_true")
    args = p.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))
    if args.calibrate_judge:
        sys.exit(calibrate_judge())
    sys.exit(evaluate(args.only or args.only_positional, max(1, args.repeats)))


if __name__ == "__main__":
    main()
