"""The squad as a web endpoint, so a website, a form, or n8n can use it.

Start the server:   uv run uvicorn synq_ai_squad.api:app --port 8000
Use it:             http://localhost:8000        (the simple page; password = your SQUAD_API_KEY)
API documentation is disabled on the shared demo.

Endpoints:
  GET  /          -> the simple page with a text box
  GET  /health    -> {"status": "ok"}  (no key needed; used to check the server is up)
  POST /generate  -> runs the whole squad. Send header X-API-Key and body {"request": "..."}.
"""

import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from synq_ai_squad.config import Settings, configure_logging, get_settings
from synq_ai_squad.models import error_label
from synq_ai_squad.squad import build_graph, initial_state

log = logging.getLogger(__name__)

PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
MIN_PASSWORD_LENGTH = 20
SECURITY_HEADERS = {
    "Cache-Control": "no-store",  # drafts and errors are never stored by browsers or proxies
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class GenerateRequest(BaseModel):
    request: str = Field(
        min_length=5, max_length=500, examples=["A LinkedIn post for dental clinics about missed calls"]
    )


class GenerateResponse(BaseModel):
    content: str
    approved: bool = Field(description="True if the Critic passed it. If False, review it carefully before using.")
    content_format: str
    audience: str
    angle: str
    critic_scores: list[int]
    critic_notes: list[str] = Field(description="Problems the Critic still saw in the final version")
    seconds: int
    usage: dict[str, Any] | None = Field(
        default=None, description="Routing used, tokens by role and model, estimated cost at list prices"
    )


def check_key(request: Request, key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> None:
    expected = request.app.state.settings.require("squad_api_key").get_secret_value()
    # compare_digest takes the same time whether the guess is close or not, so it leaks nothing.
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


def create_app(settings: Settings | None = None, graph: Any = None, llm: Any = None) -> FastAPI:
    """Build the web app. Tests pass in their own settings and a graph that uses fake models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once when the server starts: check the config and build the real squad.
        # Failing here stops the deploy with a clear message instead of failing on the first visitor.
        configure_logging("%(asctime)s %(levelname)s %(name)s: %(message)s")
        if app.state.settings is None:
            app.state.settings = get_settings()  # read .env / env vars only now, never at import
        if len(app.state.settings.require("squad_api_key").get_secret_value()) < MIN_PASSWORD_LENGTH:
            raise RuntimeError(f"SQUAD_API_KEY must be at least {MIN_PASSWORD_LENGTH} random characters.")
        if app.state.graph is None:
            from synq_ai_squad.models import ModelGateway
            from synq_ai_squad.rag import ensure_vectorstore

            s = app.state.settings
            app.state.llm = ModelGateway(s)
            app.state.graph = build_graph(app.state.llm, ensure_vectorstore(s), s)
        yield

    app = FastAPI(
        title="Synq AI Squad",
        description="Manager, Researcher, Writer and Critic, as one endpoint.",
        lifespan=lifespan,
        docs_url=None,  # no public schema pages on the shared demo
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.graph = graph
    app.state.llm = llm  # the model gateway, read after each run for token usage and cost
    # Free AI plans allow only a few requests per minute, so run one squad job at a time.
    app.state.busy = threading.Lock()

    @app.middleware("http")
    async def private_responses(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        # The simple page with a text box. It calls /generate for you, with the password you enter.
        return PAGE

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain "def" (not "async def") makes FastAPI run this in a worker thread,
    # so the slow squad job doesn't freeze the server for /health checks.
    @app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(check_key)])
    def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        state = request.app.state
        if not state.busy.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="The squad is busy with another request. Try again in a minute."
            )
        try:
            start = time.time()
            if hasattr(state.llm, "reset_usage"):
                state.llm.reset_usage()
            out = state.graph.invoke(initial_state(body.request))
        except Exception as e:
            # Log only the error type: provider error text can echo request data or account details.
            log.error("squad failed: %s", error_label(e))  # type and status code only, never the message
            raise HTTPException(status_code=502, detail="The squad failed to finish. Please try again.") from e
        finally:
            state.busy.release()

        plan, review = out["plan"], out["review"]
        return GenerateResponse(
            content=out["draft"],
            approved=review.score >= state.settings.pass_score,
            content_format=plan.content_format,
            audience=plan.audience,
            angle=plan.angle,
            critic_scores=out["scores"],
            critic_notes=review.issues + review.unsupported_claims,
            seconds=round(time.time() - start),
            usage=state.llm.usage() if hasattr(state.llm, "usage") else None,
        )

    return app


app = create_app()
