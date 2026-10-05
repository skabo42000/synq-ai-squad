# The squad as a container, so it runs the same way on any host.
#   Build:  docker build -t synq-ai-squad .
#   Run:    docker run -p 8000:8000 --env-file .env synq-ai-squad      -> http://localhost:8000
# On first start the search index is built from knowledge/ (needs GOOGLE_API_KEY).

# ---- Stage 1: install dependencies with uv (build tools stay in this stage) ----
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- Stage 2: small runtime image, no uv, runs as a non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app /app/.venv ./.venv
COPY --from=build --chown=app /app/src ./src
COPY --chown=app knowledge ./knowledge
RUN chown app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "synq_ai_squad.api:app", "--host", "0.0.0.0", "--port", "8000"]
