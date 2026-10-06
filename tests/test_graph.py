"""Tests for the squad's control flow (squad.py), using a scripted fake AI.

These check the logic around the AI: when the loop stops, when code overrides the Critic,
how many searches run, and what the Writer is told on a revision. No keys, no cost.
"""

import pytest
from conftest import GOOD_DRAFT, ScriptedLLM, SpyIndex, make_plan, make_review

from synq_ai_squad.squad import build_graph, initial_state


def run(llm: ScriptedLLM, settings, index: SpyIndex | None = None) -> dict:
    return build_graph(llm, index or SpyIndex(), settings).invoke(initial_state("A post about missed calls"))


def test_a_passing_first_draft_ends_the_loop(settings):
    llm = ScriptedLLM(make_plan(), reviews=[make_review(9)], drafts=[GOOD_DRAFT])
    out = run(llm, settings)
    assert out["rounds"] == 1
    assert out["scores"] == [9]
    assert out["draft"] == GOOD_DRAFT


def test_the_loop_stops_at_max_rounds_even_if_never_approved(settings):
    llm = ScriptedLLM(make_plan(), reviews=[make_review(5)] * 5, drafts=[GOOD_DRAFT] * 5)
    out = run(llm, settings)
    assert out["rounds"] == settings.max_rounds
    assert out["scores"] == [5, 5, 5]


def test_a_broken_rule_overrides_a_high_critic_score(settings):
    # The Critic gives 10, but the first draft has no booking link: plain code caps the score.
    llm = ScriptedLLM(
        make_plan(), reviews=[make_review(10), make_review(10)], drafts=["Book a call today!", GOOD_DRAFT]
    )
    out = run(llm, settings)
    assert out["scores"] == [settings.pass_score - 1, 10]
    assert out["rounds"] == 2


def test_an_unsupported_claim_can_never_pass(settings):
    llm = ScriptedLLM(
        make_plan(),
        reviews=[make_review(10, unsupported=["We guarantee 50% more bookings"]), make_review(9)],
        drafts=[GOOD_DRAFT, GOOD_DRAFT],
    )
    out = run(llm, settings)
    assert out["scores"] == [settings.pass_score - 1, 9]


def test_the_manager_fans_out_one_search_per_query(settings):
    index = SpyIndex()
    llm = ScriptedLLM(make_plan(["a", "b", "c", "d"]), reviews=[make_review(9)], drafts=[GOOD_DRAFT])
    run(llm, settings, index)
    assert sorted(index.queries) == ["a", "b", "c", "d"]


def test_duplicate_sections_reach_the_researcher_only_once(settings):
    # 3 searches each return the same 2 sections: the Researcher should see 2 sections, not 6.
    llm = ScriptedLLM(make_plan(), reviews=[make_review(9)], drafts=[GOOD_DRAFT])
    run(llm, settings)
    research_prompt = llm.prompts["research"][0]
    assert research_prompt.count("24/7 lead capture") == 1
    assert "[2] (from" in research_prompt and "[3] (from" not in research_prompt


def test_a_revision_shows_the_writer_its_draft_and_the_feedback(settings):
    llm = ScriptedLLM(
        make_plan(),
        reviews=[make_review(6, issues=["Too long"], unsupported=["Saves 20 hours"]), make_review(9)],
        drafts=["FIRST DRAFT " + GOOD_DRAFT, GOOD_DRAFT],
    )
    run(llm, settings)
    first, revision = llm.prompts["write"]
    assert "YOUR PREVIOUS DRAFT" not in first
    assert "FIRST DRAFT" in revision
    assert "Too long" in revision and "Saves 20 hours" in revision


def test_only_the_last_allowed_round_is_flagged_as_final(settings):
    # The gateway may send the final round to a stronger model, so the flag must be exactly right.
    llm = ScriptedLLM(make_plan(), reviews=[make_review(5)] * 3, drafts=[GOOD_DRAFT] * 3)
    run(llm, settings)
    assert llm.final_round_flags["write"] == [False, False, True]
    assert llm.final_round_flags["review"] == [False, False, True]
    assert llm.final_round_flags["research"] == [False]


def test_a_step_that_hits_a_rate_limit_is_retried_once(settings):
    class RateLimitedOnce(ScriptedLLM):
        failed = False

        def text(self, role, prompt, *, final_round=False):
            if role == "write" and not self.failed:
                self.failed = True
                raise RuntimeError("429 rate limit: tokens per minute")
            return super().text(role, prompt, final_round=final_round)

    llm = RateLimitedOnce(make_plan(), reviews=[make_review(9)], drafts=[GOOD_DRAFT])
    fast = settings.model_copy(update={"step_retry_wait_seconds": 0.01})
    out = run(llm, fast)
    assert out["draft"] == GOOD_DRAFT and out["rounds"] == 1  # the retry succeeded and counted as one round


def test_a_real_bug_is_not_retried(settings):
    class Broken(ScriptedLLM):
        calls = 0

        def text(self, role, prompt, *, final_round=False):
            self.calls += 1
            raise ValueError("bug in our code")

    llm = Broken(make_plan(), reviews=[], drafts=[])
    with pytest.raises(ValueError):
        run(llm, settings.model_copy(update={"step_retry_wait_seconds": 0.01}))
    assert llm.calls == 1
