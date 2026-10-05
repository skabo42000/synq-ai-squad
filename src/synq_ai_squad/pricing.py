"""List prices per million tokens, copied from the providers' official pages (never estimated from memory).

The squad runs on free tiers, so nothing is actually billed. Costs are reported at these paid-tier list prices,
so routing strategies can be compared on what they WOULD cost at scale.
Re-check the sources before relying on the numbers: prices change.
"""

from collections.abc import Mapping
from dataclasses import dataclass

PRICES_CHECKED = "2026-10-05"
GOOGLE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
GROQ_SOURCE = "https://console.groq.com/docs/models"


@dataclass(frozen=True)
class Price:
    input_per_m: float  # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens
    source: str


PRICES: dict[str, Price] = {
    "gemini-3.1-flash-lite": Price(0.25, 1.50, GOOGLE_SOURCE),  # standard paid tier, text input
    "qwen/qwen3.8-27b": Price(0.80, 4.00, GROQ_SOURCE),
    "openai/gpt-oss-120b": Price(0.15, 0.60, GROQ_SOURCE),
    "openai/gpt-oss-20b": Price(0.075, 0.30, GROQ_SOURCE),
    # gemini-embedding-001 is not listed on the pricing page (only its successor is), and embedding calls
    # don't report token usage, so search costs are not included in the estimates.
}


def cost_usd(usage_by_model: Mapping[str, Mapping[str, int]]) -> float:
    """Estimated cost of a run from {model: {"input_tokens": n, "output_tokens": m}}. Unknown models count as 0."""
    total = 0.0
    for model, usage in usage_by_model.items():
        price = PRICES.get(model)
        if price:
            total += (
                usage.get("input_tokens", 0) * price.input_per_m + usage.get("output_tokens", 0) * price.output_per_m
            )
    return total / 1_000_000
