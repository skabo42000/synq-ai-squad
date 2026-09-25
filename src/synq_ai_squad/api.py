"""Step 6: the squad as a web endpoint, so a website, a form, or n8n can use it.

Start the server:   uv run uvicorn synq_ai_squad.api:app --port 8000
Use it:             http://localhost:8000        (the simple page; password = your SQUAD_API_KEY)
Developer test page: http://localhost:8000/docs

Endpoints:
  GET  /          -> the simple page with a text box
  GET  /health    -> {"status": "ok"}  (no key needed; used to check the server is up)
  POST /generate  -> runs the whole squad. Send header X-API-Key and body {"request": "..."}.
"""

import os
import secrets
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from synq_ai_squad.squad import PASS_SCORE, graph

load_dotenv()

API_KEY = os.getenv("SQUAD_API_KEY", "")
if len(API_KEY) < 20:
    raise RuntimeError("Set SQUAD_API_KEY in .env (at least 20 random characters) before starting the server.")

app = FastAPI(title="Synq AI Squad", description="Manager, Researcher, Writer and Critic, as one endpoint.")

# Free AI plans allow only a few requests per minute, so run one squad job at a time.
busy = threading.Lock()


def check_key(key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> None:
    # compare_digest takes the same time whether the guess is close or not, so it leaks nothing.
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


class GenerateRequest(BaseModel):
    request: str = Field(min_length=5, max_length=500,
                         examples=["A LinkedIn post for dental clinics about missed calls"])


class GenerateResponse(BaseModel):
    content: str
    approved: bool = Field(description="True if the Critic passed it. If False, review it carefully before using.")
    content_format: str
    audience: str
    angle: str
    critic_scores: list[int]
    critic_notes: list[str] = Field(description="Problems the Critic still saw in the final version")
    seconds: int


PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    # The simple page with a text box. It calls /generate for you, with the password you enter.
    return PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A plain "def" (not "async def") makes FastAPI run this in a background thread,
# so the slow squad job doesn't freeze the server for /health checks.
@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(check_key)])
def generate(body: GenerateRequest) -> GenerateResponse:
    if not busy.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The squad is busy with another request. Try again in a minute.")
    try:
        start = time.time()
        out = graph.invoke({"request": body.request, "found": [], "rounds": 0, "scores": []})
    except Exception as e:
        # Log the details on the server, but don't leak internals to the caller.
        print(f"  [api] squad failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="The squad failed to finish. Please try again.")
    finally:
        busy.release()

    plan, review = out["plan"], out["review"]
    return GenerateResponse(
        content=out["draft"],
        approved=review.score >= PASS_SCORE,
        content_format=plan.content_format,
        audience=plan.audience,
        angle=plan.angle,
        critic_scores=out["scores"],
        critic_notes=review.issues + review.unsupported_claims,
        seconds=round(time.time() - start),
    )
