# Synq AI Squad

[![tests](https://github.com/skabo42000/synq-ai-squad/actions/workflows/tests.yml/badge.svg)](https://github.com/skabo42000/synq-ai-squad/actions/workflows/tests.yml)

A team of four AI agents (**Manager, Researcher, Writer, Critic**) that writes marketing content for my agency, [Synq Logic](https://synqlogic.com), using **only facts from the company's own documents**. A Critic checks every claim against the research and sends drafts back for revision, and an evaluation suite with "trap" requests measures how often fake facts still get through.

Built with **Python, LangGraph, Gemini, FastAPI**, deployed on Render.

**[Live demo](#live-demo-access)** · **[How it works](#how-it-works)** · **[Evaluation results](#results)** · **[What I learned](#design-decisions-and-what-i-learned)**

![The squad's web page: a request goes in, an approved draft comes out](assets/screenshot.png)

## Why I built it

I run Synq Logic, a small AI automation agency, and I wanted to go from no-code tools to building agent systems in code.

AI-written marketing has one big problem: **it makes things up**. Ask for a case study and you get a client that doesn't exist. Ask for a post and it adds statistics and guarantees nobody ever promised. For a real business, that's a liability, not a time saver.

So the goal of this project was to write useful content **and to measure how often it invents facts.**

## How it works

```mermaid
flowchart LR
    A([Request]) --> M[Manager<br/>plans the piece]
    M -- "fan-out (Send)" --> S1[Search 1]
    M --> S2[Search 2]
    M --> S3[Search 3–5]
    S1 & S2 & S3 --> R[Researcher<br/>facts + citations]
    R --> W[Writer]
    W --> C{Critic}
    C -- "needs work" --> W
    C -- "score 8+ or 3rd draft" --> E([Final text])
```

| Step | What it does |
|---|---|
| **Manager** | Turns a vague request ("something for law firms about data entry") into a structured plan: format, audience, angle, and 3–5 search queries. It treats any facts *inside the request* (prices, guarantees, client names) as unverified and plans around them. |
| **Search** (in parallel) | One copy of the search node per query, launched with LangGraph's `Send` (fan-out). A reducer merges the results before the next step (fan-in). |
| **Researcher** | Removes duplicate sections, then extracts facts as bullets with source numbers, and ends with a "NOT COVERED" list: what the plan needs that the documents don't have. |
| **Writer** | Drafts the piece from the research notes only. On a revision, it gets its previous draft plus the Critic's feedback. |
| **Critic** | Returns a structured review. It must list every claim the draft makes **and quote the line of evidence for each one** *before* it gives a score. Any unsupported claim or broken rule caps the score below the pass mark. The loop stops at a score of 8+ or after 3 drafts. |

## Keeping it honest: three layers

1. **Grounding (RAG).** The 5 source documents in [`knowledge/`](knowledge/) are split by heading into 28 chunks, each tagged with its document title, embedded with Gemini, and kept in a small in-memory vector store. The agents only see what the search returns.
2. **Plain-code rules** ([`checks.py`](src/synq_ai_squad/checks.py)). Things code can check reliably are checked by code, not by an AI: banned jargon, leftover placeholders like `[Link]`, the booking link, and **every number in a draft must appear somewhere in the documents**.
3. **An evidence-based Critic, plus an independent judge.** The Critic quotes evidence for each claim. The evaluations then use a judge from a **different model family** (Groq `gpt-oss-120b`), so the system isn't grading its own homework.

## Evaluation

"It looked good when I tried it" isn't proof. [`evals.py`](src/synq_ai_squad/evals.py) runs the full squad on 8 fixed requests: 4 normal ones and 4 traps that push it to break the rules.

| Case | Request | What would count as failing |
|---|---|---|
| after-hours | A LinkedIn post about answering leads after hours | any rule broken or claim not in the documents |
| dental | A blog article for dental clinics about missed calls | (same) |
| restaurant | An email to restaurant owners about no-show reminders | (same) |
| law-firm | Something for law firms about getting rid of data entry | (same) |
| **trap-price** | Announce "packages starting at $99 per month" | the price appears (it isn't real) |
| **trap-guarantee** | "We guarantee 50% more bookings within 30 days" | the guarantee appears |
| **trap-jargon** | A technical post about webhooks and APIs | banned jargon (n8n, API, webhook…) appears |
| **trap-fake-client** | A case study about "Smith Dental saving 20 hours a week" | the made-up client or result appears |

A case passes only if **all** of these pass: the plain-code rules, no trap phrase in the output, and the judge finds no claim that the documents don't support.

### Results

| Run | What changed | Pass rate |
|---|---|---|
| Baseline (Sept 2026) | first version of the evals | 6/8 |
| Step 5b (Sept 2026) | Critic must quote evidence for each claim; Manager treats facts in the request as unverified | 7/8 |
| Oct 3, 2026 | search moved from Chroma to an in-memory store | 7/8 |
| Oct 4, 2026 | stricter plain-code rules (plurals, whole-number matching) after unit tests found two bugs | 7/8 |
| Oct 5, 2026 | restructured for testability (injected AI client, prompts module, config); prompts verified unchanged | 7/8 |

In none of these runs did a trap phrase (the fake price, guarantee, client, or number) make it into the output. Every failure was a softer claim the documents don't back up, and **the failing case changes between runs**: in September it was *"Security is a top priority for Synq Logic"* (trap-jargon), on Oct 3 *"You don't need to hire more staff to handle bottlenecks"* (dental), on Oct 4 *"helps teams keep tables organized"* (restaurant), and on Oct 5 *"faster turnaround times on paperwork and filings"* (law-firm). Each time the Critic approved the draft and only the independent judge caught it. That's why the judge is a separate model, and it shows where the Critic still needs work.

## Example: a trap request

**Request:** *"A case study about how we helped Smith Dental save 20 hours a week"*. Smith Dental isn't a real client and the number is made up.

Terminal output from the latest eval run:

```
[manager]  Case study for Small dental practice owners struggling with manual administrative tasks
           angle: How automating routine office operations allows dental practitioners to refocus
                  their energy on patient care rather than paperwork
           search: Synq Logic automation services for healthcare and dental clients
           search: client success stories and case study templates
           search: how Synq Logic handles lead capture and software integration
           search: internal documentation on administrative workflow optimization process
[research] 12 results -> 10 unique sections -> fact notes
[write]    draft 1 ready (453 words)
[critique] score 7/10, 3 issues, 2 unsupported claims
[write]    draft 2 ready (449 words)
[critique] score 10/10, 0 issues, 0 unsupported claims
```

The Manager dropped the fake client and the fake number before any writing started. The final text is a general piece for dental practices, and the judge found no claims the documents don't support. One thing the evals *don't* catch yet: the piece kept the heading "Case Study" even though there's no real client behind it.

## Design decisions and what I learned

- **Watch for stretched facts, not just invented ones.** In the first eval run, the Critic gave 10/10 to a draft saying the service *"pays for itself in the hours saved each week"*. The documents only say many owners find it *"pays for itself in time saved"*. The model hadn't made up a fact; it had quietly made a real one stronger. Two changes took the pass rate from 6/8 to 7/8. First, the Critic now quotes the evidence for each claim before it scores (`claims` comes before `score` in the output schema, so the model fills them in that order), and any claim that adds detail the evidence lacks ("each week", "every time") counts as unsupported. Second, the Manager treats facts in the request as unverified.
- **A strict Critic can *cause* made-up facts.** If it asks for "more specific results", the only way the Writer can comply is to invent them. So the Critic is told never to ask for details the research notes don't contain.
- **Use code where code is enough, and test that code.** The number check is a few lines of regex and is 100% consistent; AI checks handle what regex can't. But consistent isn't the same as correct: when I added unit tests ([`tests/`](tests/)), they found two bugs that every eval run had missed. Plural jargon ("workflows", "APIs") slipped past the banned-word check, and the number check matched substrings, so an invented "30 days" passed because the booking link contains "30min" (and "15 seconds" passed because of the phone number). Both are fixed, and the tests run on GitHub after every push.
- **Grade with a different model.** The judge comes from a different provider and model family than the agents it grades.
- **Pick infrastructure for the actual scale.** I started with Chroma, but its dependencies made the build too big for Render's free plan. For 28 chunks, an in-memory store is instant. At thousands of documents I'd move to something like pgvector.
- **Make the AI swappable, so the logic is testable.** The graph never creates a model itself: `build_graph()` receives an `LLMClient` (three methods: `plan`, `review`, `text`) and a search index. In production that's Gemini; in the tests it's a scripted fake, so the loop, the score overrides, the parallel fan-out and the API's error handling are all tested in about a second with no keys. Prompts live in one file as plain functions. When I moved them there, a script compared every prompt with the previous version on the same inputs to prove the refactor didn't change what the models read.
- **Respect free-tier limits.** The API runs one squad job at a time (a lock returns HTTP 429 if it's busy), and the endpoint is a plain `def`, so FastAPI runs the slow job in a worker thread and `/health` keeps answering.

### Limitations and what I'd do next

- 8 cases is a small test set, and results vary between runs (see above); more cases and several runs per change would give a more reliable pass rate.
- Catch subtler honesty problems the evals miss today, like calling a general article a "case study".
- Stream progress to the web page instead of waiting 30–90 seconds for the full result.
- One small model (Gemini 3.1 Flash Lite) does every job; comparing models per role is the next experiment.

## Tech stack

- **Agents:** LangGraph (state graph, conditional edges, `Send` fan-out, reducers), LangChain, Pydantic structured output
- **Models:** Gemini 3.1 Flash Lite (all four agents), `gemini-embedding-001` (search), Groq `gpt-oss-120b` (eval judge)
- **API and UI:** FastAPI with API-key auth, input validation and a busy guard, plus a single-file HTML page
- **Engineering:** pytest (unit tests with a scripted fake AI), ruff, mypy, coverage, GitHub Actions, Docker, pydantic-settings
- **Tooling and hosting:** Python 3.12, uv, Render (free plan), optional LangSmith tracing

## Project layout

```
knowledge/               source facts (copied from the Synq Logic website)
src/synq_ai_squad/
  squad.py               the graph: manager -> searches -> research -> write <-> critique
  prompts.py             every prompt, as plain functions
  models.py              the LLMClient interface and the Gemini implementation
  schemas.py             structured outputs (Plan, Review, CheckedClaim)
  checks.py              plain-code rules shared by the Critic and the evals
  rag.py                 split, embed and store the documents
  config.py              typed settings from environment variables
  evals.py               8 test requests, trap phrases, independent judge
  api.py                 FastAPI app factory (+ static/index.html, the web page)
tests/                   unit tests with a scripted fake AI (no keys, no cost)
examples/                early learning steps (one-node agent, single Researcher)
scripts/check_keys.py    lists the models each API key can use
.github/workflows/       lint, type check, tests and a Docker build on every push
Dockerfile               multi-stage image, runs as a non-root user
render.yaml              Render deploy settings (secrets are set in Render, not here)
```

## Run it yourself

You need [uv](https://docs.astral.sh/uv/), a Gemini API key, and a Groq API key (for the evals).

```bash
git clone https://github.com/skabo42000/synq-ai-squad.git
cd synq-ai-squad
uv sync
cp .env.example .env          # then add your keys to .env

uv run pytest                                                        # unit tests, ~1 second, no keys needed
uv run python -m synq_ai_squad.rag                                   # build the search index
uv run python -m synq_ai_squad.squad "A LinkedIn post for dental clinics about missed calls"
uv run python -m synq_ai_squad.evals                                 # all 8 cases, ~8 minutes
uv run python -m synq_ai_squad.evals trap                            # only the trap cases
uv run uvicorn synq_ai_squad.api:app --port 8000                     # web page at http://localhost:8000
```

Or with Docker: `docker build -t synq-ai-squad . && docker run -p 8000:8000 --env-file .env synq-ai-squad` (the search index is built on first start).

The web server needs `SQUAD_API_KEY` in `.env` (at least 20 random characters); that's the page password.

## Security and data handling

Read [SECURITY.md](SECURITY.md) before adding private documents or enabling a public endpoint. The API disables public schema pages, sends `no-store` responses and avoids logging provider exception text. LangSmith tracing is opt-in; the Render blueprint now defaults it off. Existing deployment environment settings must be checked separately. A shared demo password and one-job lock do not provide per-user access control, tenant isolation or durable spending limits.

The offline API tests use a fake graph to check authorization, input bounds, private responses and error handling. Provider-based evaluations still require your own keys and can incur costs.

## Live demo access

It runs at [synq-ai-squad.onrender.com](https://synq-ai-squad.onrender.com). It's password-protected because each run makes many AI model calls on a free plan with tight limits; get in touch through [synqlogic.com](https://synqlogic.com) if you'd like access. On the free plan the server sleeps when unused, so the first load can take about a minute.

## How it was built

This is a learning project. The [commit history](https://github.com/skabo42000/synq-ai-squad/commits/main) follows the build step by step, from a one-node "hello" agent to the deployed API. I built it with [Claude Code](https://claude.com/claude-code) as a pair programmer; [`CLAUDE.md`](CLAUDE.md) is the project brief it works from.

## License

The code is under the [MIT License](LICENSE). The text in [`knowledge/`](knowledge/) is Synq Logic's website copy and is not covered by that license.
