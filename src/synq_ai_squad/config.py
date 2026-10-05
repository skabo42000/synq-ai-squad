"""All settings in one place, read from environment variables (and .env when running locally).

Why a settings class instead of os.getenv() everywhere:
  - every setting has a type and a default, and is checked when the app starts
  - keys are SecretStr, so they never show up in logs or error messages by accident
  - tests can build a Settings(...) object directly, with no .env file and no keys
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"  # the source facts the squad may use
STORE_FILE = PROJECT_ROOT / "vector_store.json"


class Settings(BaseSettings):
    # Keys (optional here; each entry point checks the ones it needs with require())
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    squad_api_key: SecretStr | None = None

    # Models: a cheap tier (Google) and a strong tier (Groq). They also back each other up during outages.
    agent_model: str = "gemini-3.1-flash-lite"  # cheap tier
    strong_model: str = "qwen/qwen3.8-27b"  # strong tier; a different family from the judge on purpose
    embedding_model: str = "gemini-embedding-001"
    judge_model: str = "openai/gpt-oss-120b"
    judge_reasoning_effort: Literal["low", "medium", "high"] | None = (
        "low"  # calibrated: 32/32 on 2026-10-05, ~35% fewer tokens
    )

    # Routing: which tier each call uses (see models.py). Chosen from eval data, not by guess.
    routing: Literal["all-cheap", "all-strong", "cascade"] = "all-cheap"
    llm_timeout_seconds: float = 45  # one model call may take at most this long...
    llm_max_retries: int = 1  # ...and is retried this often before falling back to the other provider

    # Squad behaviour
    pass_score: int = 8  # the Critic must give at least this to approve
    max_rounds: int = 3  # the Writer gets at most this many drafts
    results_per_search: int = 3  # sections returned by each parallel search

    def require(self, name: str) -> SecretStr:
        """Return a key (still wrapped as a secret), or stop with a clear message if it isn't set."""
        value = getattr(self, name)
        if not isinstance(value, SecretStr) or not value.get_secret_value():
            raise RuntimeError(f"{name.upper()} is not set. Add it to .env (locally) or the host's env vars.")
        return value


def configure_logging(fmt: str = "  %(message)s") -> None:
    """Show the squad's own progress messages; keep other libraries quiet unless something goes wrong."""
    logging.basicConfig(level=logging.WARNING, format=fmt)
    # "__main__" too: a module started with `python -m` logs under that name instead of its own.
    for name in ("synq_ai_squad", "__main__"):
        logging.getLogger(name).setLevel(logging.INFO)
    # Google's library warns about how LangChain calls it on every request; it isn't actionable here.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


@lru_cache
def get_settings() -> Settings:
    # load_dotenv also makes .env visible to libraries that read os.environ themselves (LangSmith tracing).
    # It never overrides real environment variables, so on Render the dashboard values win.
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()
