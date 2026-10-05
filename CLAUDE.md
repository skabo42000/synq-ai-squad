# Synq AI Squad

A learning project: a multi-agent content team (Manager, Researcher, Writer, Critic) built with Python + LangGraph. It writes marketing content for Synq Logic using only facts from `knowledge/`. Owner is learning AI engineering: explain changes in plain English and teach the concept behind each one.

Live: https://synq-ai-squad.onrender.com (Render free plan, auto-deploys from private GitHub `skabo42000/synq-ai-squad`, branch `main`). Page password = `SQUAD_API_KEY`.

## Layout
- `knowledge/` - source facts (copied word for word from the Synq Logic website). Never add invented claims here.
- `src/synq_ai_squad/rag.py` - split, embed, save `vector_store.json` (in-memory store, rebuilt at deploy).
- `squad.py` - `build_graph(llm, index, settings)`: manager -> parallel `search` (Send) -> research -> write <-> critique. No work at import time.
- `prompts.py` (all prompts as pure functions), `models.py` (`LLMClient` protocol + `ModelGateway`: cheap tier Gemini flash-lite, strong tier Groq qwen3.8-27b; routing all-cheap|all-strong|cascade via `ROUTING`; timeouts, cross-provider `with_fallbacks`, usage per role/model), `pricing.py` (list prices + sources), `schemas.py`, `config.py` (pydantic-settings; `Settings.require()` returns SecretStr).
- `examples/` - early learning steps; `scripts/check_keys.py` - list models a key can use.
- `checks.py` - plain-code rules (banned words, placeholders, booking link, numbers not in docs).
- `evals.py` (CLI) + `evaluation/` (dataset, scoring, judge): 26-case `evals/dataset.yaml`, gate `evals/thresholds.yaml`, judge labels `evals/judge_labels.yaml`; reports committed in `evals/reports/`, full drafts in `evals/results/` (git-ignored). CI workflow `evals.yml`: manual + weekly, secrets GOOGLE_API_KEY/GROQ_API_KEY.
- `api.py` + `static/index.html` - `create_app(settings, graph)` factory; the real graph is built in the lifespan.
- `tests/` - pytest with a scripted fake LLM (`conftest.py`): rules, splitting, graph control flow, API. CI (`tests.yml`) runs ruff, mypy, pytest+coverage, Docker build.

## Commands
```
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest   # what CI runs (no keys)
uv run python -m synq_ai_squad.rag                      # rebuild search index after editing knowledge/
uv run python -m synq_ai_squad.squad "your request"     # run the squad in the terminal
uv run python -m synq_ai_squad.evals [--only X] [--repeats N] [--compare A B] [--calibrate-judge]   # full run ~20 min
uv run uvicorn synq_ai_squad.api:app --port 8000        # local server, open http://localhost:8000
```

## Rules for changes
- Run the CI command above before every commit; add a test when changing a rule or graph behaviour. Prompt edits are behaviour changes: measure with evals.
- Upgrade plan (Senior portfolio): `~/.claude/plans/i-want-to-upgrade-transient-matsumoto.md` (Phase 0+1 done; Phase 2 code merged, routing comparison pending: run `evals --routing all-cheap|cascade|all-strong` on fresh quotas, pick default from data, add README table).
- Free-tier daily limits: Gemini flash-lite resets at midnight Pacific; Groq gpt-oss-120b judge 200k tokens/day (judge uses reasoning_effort=low). A full eval is ~26 judge calls; plan at most ~2-3 full runs/day.
- Measure with `evals` before and after any prompt/model/retrieval change; rerun the full set, report the pass rate honestly.
- After changing dependencies: `uv export --frozen --no-dev --no-hashes -o requirements.txt`, then commit and push (Render redeploys). Keep `numpy` as a direct dependency.
- `main` is protected (ruleset "Protect main - tests and security"): no direct pushes. Work on a branch, open a PR with `gh pr create`, wait for the required checks `pytest` and `secrets` (keep those job names), then `gh pr merge --rebase --delete-branch`. gh lives at `C:\Program Files\GitHub CLI\gh.exe` (not on PATH). Dependabot opens weekly update PRs.
- Secrets live only in `.env` (git-ignored) and Render env vars. Before every commit check that `.env`, `vector_store.json`, and `evals/results/` are not staged.
- Render status/logs: use the Render API with `RENDER_API_KEY` from `.env` (service `srv-darblimgekts738vog20`), never ask for dashboard click-lists.
- See the global skill `langgraph-python-agent` for Windows/OneDrive and deploy gotchas.
