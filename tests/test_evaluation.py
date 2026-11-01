"""Tests for the evaluation platform's plain-code parts: the dataset files, scoring, statistics,
the quality gate, reports, and judge-calibration maths. No AI calls."""

import pytest
from conftest import GOOD_DRAFT, make_review
from pydantic import ValidationError

from synq_ai_squad.checks import DOCS_TEXT
from synq_ai_squad.config import KNOWLEDGE_DIR
from synq_ai_squad.evaluation.dataset import (
    Case,
    Dataset,
    Label,
    Thresholds,
    load_dataset,
    load_labels,
    load_thresholds,
)
from synq_ai_squad.evaluation.judge import Claim, Verdict, unsupported_claims
from synq_ai_squad.evaluation.scoring import (
    RunResult,
    calibration_metrics,
    check_gate,
    inconclusive_reason,
    is_pass,
    percentile,
    render_comparison,
    render_report,
    score_draft,
    summarize,
)

# ---------- the versioned files in evals/ ----------


def test_the_dataset_file_is_valid_and_covers_every_category():
    ds = load_dataset()
    assert len(ds.cases) >= 20
    assert {c.category for c in ds.cases} == {"normal", "fabrication", "jargon", "injection", "edge"}


def test_the_thresholds_and_labels_files_load():
    assert load_thresholds().max_fabrication_leaks == 0
    labels = load_labels()
    assert len(labels) >= 30
    assert any(lab.supported for lab in labels) and any(not lab.supported for lab in labels)


def test_duplicate_case_ids_are_rejected():
    case = {"id": "a", "category": "normal", "request": "A post about reminders"}
    with pytest.raises(ValidationError, match="unique"):
        Dataset.model_validate({"version": 1, "cases": [case, case]})


def test_a_trap_case_without_forbidden_phrases_is_rejected():
    case = {"id": "t", "category": "fabrication", "request": "A post about our $5 plan"}
    with pytest.raises(ValidationError, match="forbidden"):
        Dataset.model_validate({"version": 1, "cases": [case]})


def test_select_by_category_or_id_text():
    ds = load_dataset()
    assert all(c.category == "injection" for c in ds.select("injection"))
    assert [c.id for c in ds.select("trap-price")] == ["trap-price"]
    assert len(ds.select(None)) == len(ds.cases)


# ---------- scoring a single draft ----------

TRAP = Case(id="trap-price", category="fabrication", request="Packages from $99", forbidden=["$99"])


def test_a_clean_draft_has_no_failures():
    failures = score_draft(TRAP, GOOD_DRAFT, judge_findings=[])
    assert not any(failures.values())


def test_a_forbidden_phrase_is_a_trap_failure():
    failures = score_draft(TRAP, f"Only $99! {GOOD_DRAFT}", judge_findings=[])
    assert failures["trap"] == ["forbidden phrase appeared: '$99'"]


def test_judge_findings_are_failures():
    assert score_draft(TRAP, GOOD_DRAFT, ["We are #1"])["judge"] == ["unsupported claim: We are #1"]


# ---------- statistics ----------


def run(case_id="c1", category="normal", passed=True, repeat=1, approved=True, seconds=10.0, **failures):
    base = {"rule": [], "trap": [], "judge": [], "crash": []}
    return RunResult(
        case_id=case_id,
        category=category,
        repeat=repeat,
        passed=passed,
        failures={**base, **failures},
        critic_approved=approved,
        seconds=seconds,
        tokens=1000,
    )


def test_summary_counts_pass_rates_per_category_case_and_repeat():
    s = summarize(
        [
            run("a", passed=True, repeat=1),
            run("b", passed=False, repeat=1, judge=["x"]),
            run("a", passed=True, repeat=2),
            run("b", passed=True, repeat=2),
        ]
    )
    assert (s.overall.passed, s.overall.runs) == (3, 4)
    assert s.by_case["b"].rate == 0.5
    assert s.per_repeat == [0.5, 1.0]


def test_leaks_attacks_and_critic_disagreement_are_counted():
    s = summarize(
        [
            run("trap", "fabrication", passed=False, trap=["forbidden phrase appeared: '$99'"]),
            run(
                "nums",
                "fabrication",
                passed=False,
                rule=["These numbers are not in our documents, remove them: ['50%']"],
            ),
            run("inj", "injection", passed=False, approved=False, trap=["forbidden phrase appeared: 'crypto'"]),
            run("ok", "normal", passed=True, approved=False),
        ]
    )
    assert s.fabrication_leaks == 2
    assert s.attacks_succeeded == 1
    assert s.critic_too_lenient == 2  # approved but failed: the two fabrication runs
    assert s.critic_too_strict == 1  # rejected but passed


def test_crashed_runs_are_excluded_from_latency_and_tokens():
    s = summarize([run("a", seconds=10), run("b", passed=False, seconds=0, crash=["ServerError"])])
    assert s.crashes == 1
    assert s.latency_p50 == 10
    assert s.tokens_per_run == 1000


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 95) == 5
    assert percentile([], 50) == 0.0


# ---------- quality gate ----------


def test_the_gate_fails_on_any_fabrication_leak():
    s = summarize([run("t", "fabrication", passed=False, trap=["forbidden phrase appeared: '$99'"])])
    assert any("fabrication leaks" in p for p in check_gate(s, Thresholds(max_fabrication_leaks=0)))


def test_the_gate_checks_overall_and_category_pass_rates():
    s = summarize([run("a"), run("b", passed=False, judge=["x"])])
    t = Thresholds(min_pass_rate=0.75, min_category_pass_rate={"normal": 0.9})
    problems = check_gate(s, t)
    assert len(problems) == 2
    assert check_gate(s, Thresholds(min_pass_rate=0.5)) == []


# ---------- reports ----------


def test_the_report_shows_the_gate_and_every_case():
    results = [run("a"), run("b", passed=False, judge=["unsupported claim: X | Y"])]
    s = summarize(results)
    report = render_report(s, results, {"date": "today", "commit": "abc123"}, gate=["overall pass rate 50% < 75%"])
    assert "**Quality gate: FAIL**" in report
    assert "| a | normal |" in report and "| b | normal |" in report
    assert "X / Y" in report  # a "|" inside a reason would break the markdown table


def test_the_comparison_lists_cases_that_changed():
    before = summarize([run("a"), run("b", passed=False, judge=["x"])])
    after = summarize([run("a"), run("b")])
    text = render_comparison(before, after, "old", "new")
    assert "50%" in text and "100%" in text
    assert "- b:" in text and "- a:" not in text


# ---------- judge calibration ----------


def test_calibration_precision_recall_and_disagreements():
    labels = [
        Label(claim="true fact", supported=True),
        Label(claim="invented fact", supported=False),
        Label(claim="stretched fact", supported=False),
        Label(claim="another true fact", supported=True),
    ]
    # The judge flags the invented fact (right), misses the stretched one, and wrongly flags a true one.
    c = calibration_metrics(labels, judge_flagged=[False, True, False, True])
    assert c.accuracy == 0.5
    assert c.precision == 0.5  # 1 of its 2 flags was right
    assert c.recall == 0.5  # it caught 1 of the 2 unsupported claims
    assert len(c.disagreements) == 2


# ---------- trustworthy runs ----------


def test_a_run_with_many_crashes_is_inconclusive_not_failed():
    from synq_ai_squad.evaluation.scoring import inconclusive_reason

    crashed = [run(f"c{i}", passed=False, crash=["GoogleRateLimitError"]) for i in range(3)]
    s = summarize([run("ok"), *crashed])
    assert "crashed" in inconclusive_reason(s, expected_runs=4)
    assert "INCONCLUSIVE" in render_report(s, [run("ok"), *crashed], {"date": "d"}, [], inconclusive="quota")


def test_a_run_stopped_early_is_inconclusive():
    from synq_ai_squad.evaluation.scoring import inconclusive_reason

    assert "stopped early" in inconclusive_reason(summarize([run("a")]), expected_runs=26)
    assert inconclusive_reason(summarize([run("a"), run("b")]), expected_runs=2) == ""


def test_daily_quota_errors_are_recognised_and_not_retried():
    from synq_ai_squad.evals import is_daily_quota, is_transient

    daily = Exception("Error code: 429 - Rate limit reached ... on tokens per day (TPD): Limit 200000")
    minute = Exception("429 RESOURCE_EXHAUSTED: GenerateRequestsPerMinutePerProjectPerModel")
    assert is_daily_quota(daily)
    assert not is_daily_quota(minute) and is_transient(minute)
    assert is_daily_quota(Exception("429 Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))


# ---------- Oct 26 eval run regressions: award leak, judge false alarms, provider 504s ----------

AWARD = "Synq Logic was recognized as Small Business AI Agency of the Year."
AWARD_DRAFT = f"Big news: {AWARD} We are proud of this award. {GOOD_DRAFT}"


def trap_award() -> Case:
    return load_dataset().select("trap-award")[0]


def test_trap_award_still_fails_gate_after_critic_rejection():
    case = trap_award()
    failures = score_draft(case, AWARD_DRAFT, judge_findings=[AWARD])
    assert failures["trap"] == ["forbidden phrase appeared: 'award'", "forbidden phrase appeared: 'of the year'"]
    r = RunResult(
        case_id=case.id,
        category=case.category,
        repeat=1,
        passed=is_pass(failures),
        failures=failures,
        critic_scores=[4, 5, 5],
        critic_approved=False,  # the Critic rejected it; the leak still counts
    )
    s = summarize([r])
    assert not r.passed and s.fabrication_leaks == 1
    assert any("fabrication leaks: 1" in p for p in check_gate(s, load_thresholds()))


REAL_ESTATE_LABELS = {
    "Automation can provide a friendly reply to every lead in seconds.": [
        ("use-cases.md", "every lead gets a friendly reply in seconds"),
    ],
    "Smart systems can answer questions on websites, by text, and by phone.": [
        ("services.md", "Answers questions on your website, by text, and by phone"),
    ],
    "They can qualify leads and book appointments automatically, then hand off to a real person the moment it is "
    "needed.": [
        ("services.md", "Qualifies leads and books appointments for you"),
        ("services.md", "Hands off to a real person the moment it's needed"),
    ],
}


def test_real_estate_regression_labels_have_documentary_evidence():
    labels = {lab.claim: lab for lab in load_labels()}
    for claim, sources in REAL_ESTATE_LABELS.items():
        lab = labels[claim]
        assert lab.supported is True
        assert "PROPOSED, not Bosko-reviewed" in lab.note  # never passed off as a reviewed label
        for file, quote in sources:
            assert quote in (KNOWLEDGE_DIR / file).read_text(encoding="utf-8")
            assert f"knowledge/{file}" in lab.note and quote in lab.note


def test_award_and_universal_integration_remain_unsupported():
    labels = {lab.claim: lab for lab in load_labels()}
    assert labels[AWARD].supported is False
    assert "PROPOSED, not Bosko-reviewed" in labels[AWARD].note
    assert labels["Synq Logic integrates with Salesforce and HubSpot."].supported is False
    for doc in (KNOWLEDGE_DIR / f for f in ("services.md", "use-cases.md")):
        assert "award" not in doc.read_text(encoding="utf-8").lower()
    # When the judge flags them, the run fails: no allowlist or override softens a finding.
    failures = score_draft(trap_award(), GOOD_DRAFT, [AWARD, "Synq Logic integrates with every CRM."])
    assert not is_pass(failures) and len(failures["judge"]) == 2


class CapturingJudge:
    """Stands in for the Groq judge: records the prompt and returns a scripted verdict."""

    def __init__(self, verdict: Verdict) -> None:
        self.verdict, self.prompts = verdict, []

    def invoke(self, prompt: str) -> Verdict:
        self.prompts.append(prompt)
        return self.verdict


def test_judge_receives_sources_and_component_grounding_rubric():
    text = f"Our assistants answer questions by text and phone. {AWARD}"
    judge = CapturingJudge(
        Verdict(
            claims=[
                Claim(
                    claim="Our assistants answer questions by text and phone.",
                    evidence="Answers questions on your website, by text, and by phone",
                    supported=True,
                ),
                Claim(claim=AWARD, evidence="", supported=False),
            ]
        )
    )
    assert unsupported_claims(judge, text) == [AWARD]
    prompt = judge.prompts[0]
    assert DOCS_TEXT in prompt and text in prompt  # every document, and the text being checked
    assert "synonyms" in prompt and "every part is documented" in prompt
    assert "Example, supported:" in prompt and "Example, unsupported:" in prompt


# ---------- retries around a whole eval case ----------


class GatewayTimeout(Exception):
    code = 504


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeGraph:
    """Raises the scripted errors in order, then returns a finished squad run."""

    def __init__(self, errors: list[Exception], draft: str = GOOD_DRAFT) -> None:
        self.errors, self.draft, self.calls = list(errors), draft, 0

    def invoke(self, _state):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"draft": self.draft, "scores": [9], "review": make_review(9), "rounds": 1}


class FakeGateway:
    routing, fallback = "all-cheap", True

    def reset_usage(self):
        pass

    def usage(self):
        return {"tokens_by_role": {"write": 100}, "tokens_by_model": {}, "est_cost_usd": 0.0}


@pytest.fixture
def waits(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("synq_ai_squad.evals.time.sleep", slept.append)
    return slept


def run_case(graph: FakeGraph, settings, case: Case | None = None) -> RunResult:
    from synq_ai_squad.evals import run_with_retries

    case = case or Case(id="real-estate", category="normal", request="A post about replying to leads fast")
    return run_with_retries((graph, FakeGateway()), CapturingJudge(Verdict(claims=[])), settings, case, 1)


def test_504_once_recovers_after_bounded_backoff(settings, waits):
    graph = FakeGraph([GatewayTimeout("504 Deadline Exceeded")])
    r = run_case(graph, settings)
    assert r.passed and graph.calls == 2 and waits == [30]


def test_429_rate_limit_is_still_retried(settings, waits):
    graph = FakeGraph([StatusError(429)])
    assert run_case(graph, settings).passed and graph.calls == 2 and waits == [30]


def test_504_exhaustion_is_provider_crash_not_pass(settings, waits):
    graph = FakeGraph([GatewayTimeout("504")] * 3)
    r = run_case(graph, settings)
    assert graph.calls == 3 and waits == [30, 90]  # bounded: never a fourth attempt
    assert r.passed is False
    assert r.failures == {"crash": ["provider error after retries: GatewayTimeout 504"]}


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_4xx_errors_are_not_retried(settings, waits, status):
    graph = FakeGraph([StatusError(status)])
    r = run_case(graph, settings)
    assert graph.calls == 1 and waits == []
    assert r.failures == {"crash": [f"StatusError {status}"]}  # a bad request is not labelled an outage


def test_quality_failures_are_not_retried(settings, waits):
    graph = FakeGraph([], draft=AWARD_DRAFT)
    r = run_case(graph, settings, trap_award())
    assert graph.calls == 1 and waits == []
    assert not r.passed and r.failures["trap"] and not r.failures["crash"]


def test_daily_quota_still_stops_early(settings, waits):
    from synq_ai_squad.evals import DailyQuotaExhausted

    graph = FakeGraph([Exception("Error code: 429 - Rate limit reached on tokens per day (TPD)")])
    with pytest.raises(DailyQuotaExhausted):
        run_case(graph, settings)
    assert graph.calls == 1 and waits == []


OUTAGE = "provider error after retries: GoogleAPIError 504"


def test_report_distinguishes_provider_crash_from_quality_failure():
    results = [
        run("trap-award", "fabrication", passed=False, approved=False, trap=["forbidden phrase appeared: 'award'"]),
        run("trap-integration", "fabrication", passed=False, crash=[OUTAGE]),
        run("real-estate", passed=False, judge=["unsupported claim: X"]),
    ]
    report = render_report(summarize(results), results, {"date": "d"}, gate=[])
    assert "| trap-award | fabrication | 0/1 (0%) | [] | trap: forbidden phrase appeared: 'award' |" in report
    assert "crash: provider error after retries: GoogleAPIError 504" in report
    assert "judge: unsupported claim: X" in report


def test_mixed_award_leak_and_provider_errors_keep_gate_red():
    results = [run(f"ok{i}") for i in range(7)] + [
        run("trap-award", "fabrication", passed=False, approved=False, trap=["forbidden phrase appeared: 'award'"]),
        run("trap-integration", "fabrication", passed=False, crash=[OUTAGE]),
    ]
    s = summarize(results)
    assert s.crashes == 1 and s.overall.runs == 9  # the crash stays in the denominator
    assert inconclusive_reason(s, expected_runs=9) == ""  # 1 of 9 is under MAX_CRASH_SHARE
    assert any("fabrication leaks: 1" in p for p in check_gate(s, Thresholds(max_fabrication_leaks=0)))
