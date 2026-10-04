# Synq AI Squad

A learning project: a multi-agent content team (Manager, Researcher, Writer, Critic) built with Python + LangGraph. It writes marketing content for Synq Logic using only facts from `docs/`. Owner is learning AI engineering: explain changes in plain English and teach the concept behind each one.

Live: https://synq-ai-squad.onrender.com (Render free plan, auto-deploys from private GitHub `skabo42000/synq-ai-squad`, branch `main`). Page password = `SQUAD_API_KEY`.

## Layout
- `docs/` - source facts (copied word for word from the Synq Logic website). Never add invented claims here.
- `src/synq_ai_squad/rag.py` - split docs, embed, save `vector_store.json` (in-memory store, rebuilt at deploy).
- `researcher.py` - Step 2 single Researcher demo. `hello_agent.py`, `check_keys.py` - Step 1.
- `squad.py` - the full graph: manager -> parallel `search` (Send) -> research -> write <-> critique.
- `checks.py` - plain-code rules (banned words, placeholders, booking link, numbers not in docs).
- `evals.py` - 8 test requests incl. traps; Groq judge; results in `evals/results/` (git-ignored).
- `api.py` + `static/index.html` - FastAPI endpoint and the simple password page.
- `tests/` - pytest unit tests for checks.py and rag splitting; `.github/workflows/tests.yml` runs them (README badge).

## Commands
```
uv run pytest                                           # unit tests (~1 s, no keys); also run by GitHub Actions on every push
uv run python -m synq_ai_squad.rag                      # rebuild search index after editing docs/
uv run python -m synq_ai_squad.squad "your request"     # run the squad in the terminal
uv run python -m synq_ai_squad.evals [trap]             # evaluate (all 8 cases ~8 min)
uv run uvicorn synq_ai_squad.api:app --port 8000        # local server, open http://localhost:8000
```

## Rules for changes
- Run `uv run pytest` before every commit; add a test when changing a rule in checks.py.
- Measure with `evals` before and after any prompt/model/retrieval change; rerun the full set, report the pass rate honestly.
- After changing dependencies: `uv export --frozen --no-dev --no-hashes -o requirements.txt`, then commit and push (Render redeploys). Keep `numpy` as a direct dependency.
- Secrets live only in `.env` (git-ignored) and Render env vars. Before every commit check that `.env`, `vector_store.json`, and `evals/results/` are not staged.
- Render status/logs: use the Render API with `RENDER_API_KEY` from `.env` (service `srv-darblimgekts738vog20`), never ask for dashboard click-lists.
- See the global skill `langgraph-python-agent` for Windows/OneDrive and deploy gotchas.
