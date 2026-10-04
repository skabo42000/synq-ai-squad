"""Tests for the plain-code rules in checks.py.

These rules guard every draft (inside the Critic) and every eval run, so they must be exactly right.
No AI calls here: the tests are free, take about a second, and run on GitHub after every push.
Run them:  uv run pytest
"""

import pytest

from synq_ai_squad.checks import BOOKING_URL, banned_words, invented_numbers, placeholders, rule_problems

CLEAN = f"Every lead gets a friendly reply in seconds, 24/7. Book a free strategy call: {BOOKING_URL}"


def test_clean_text_passes_every_rule():
    assert rule_problems(CLEAN) == []


# ---------- banned words ----------

@pytest.mark.parametrize("text, word", [
    ("We connect it with an API.", "API"),
    ("we use api calls", "API"),                       # any capitalisation
    ("Built on n8n.", "n8n"),
    ("A simple workflow does it.", "workflow"),
    ("Our workflows run themselves.", "workflow"),     # plurals count too
    ("We set up webhooks for you.", "webhook"),
    ("It talks to many APIs.", "API"),
    ("Start with a free audit.", "audit"),
])
def test_banned_words_are_caught(text, word):
    assert word in banned_words(text)


@pytest.mark.parametrize("text", [
    "A rapid reply wins customers.",                   # "api" inside a normal word
    "Your audience will love it.",                     # "audi" is not "audit"
    "We keep things running.",
])
def test_normal_words_are_not_flagged(text):
    assert banned_words(text) == []


# ---------- placeholders ----------

def test_placeholders_are_caught():
    assert placeholders("Book now: [Link]. Thanks, [Your Name]") == ["[Link]", "[Your Name]"]


def test_real_markdown_links_are_not_placeholders():
    assert placeholders(f"[Book a free call]({BOOKING_URL})") == []


# ---------- booking link ----------

def test_missing_booking_link_is_reported():
    problems = rule_problems("Book a free strategy call today!")
    assert any("booking link" in p for p in problems)


# ---------- numbers ----------

@pytest.mark.parametrize("text", [
    "Win back 10+ hours a week.",
    "Assistants that answer 24/7.",
    "Call (630) 581-1550.",
    "It takes 4 easy steps.",
])
def test_numbers_from_the_documents_are_allowed(text):
    assert invented_numbers(text) == []


@pytest.mark.parametrize("text, number", [
    ("Plans start at $99 per month.", "$99"),
    ("Get 50% more bookings.", "50%"),
    ("Smith Dental saved 20 hours a week.", "20"),
    ("Live within 30 days.", "30"),        # "30" only appears inside the booking URL ("30min")
    ("Answers in 15 seconds.", "15"),      # "15" only appears inside the phone number ("1550")
])
def test_invented_numbers_are_caught(text, number):
    assert number in invented_numbers(text)


def test_the_booking_url_itself_is_not_an_invented_number():
    assert invented_numbers(f"Book here: {BOOKING_URL}") == []
