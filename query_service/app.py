"""
Minimal question-asking service: a client types a question into this web
page, and it runs the full PIKE-RAG decomposition loop
(orchestrator.DecompositionOrchestrator.answer()) against whatever
mcp_server/ already has loaded, in the background, with live round-by-round
progress served to the page.

Until this existed, asking pikerag_integration a question meant running
cli.py by hand in a terminal. This is the first client-facing entry point
for it, mirroring upload_service/app.py's job/polling pattern.

Run:
    cd query_service
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8090
Then open http://localhost:8090/

Needs pikerag_integration/requirements.txt installed in the same venv (this
imports its modules directly, not as an installed package - same sys.path
trick as orchestrator.py/cli.py use) and pikerag_integration/.env filled in
with a provider's API key (see that project's README for the Anthropic vs.
Gemini choice). mcp_server/'s Docker stack must be running too, since
answering a question means querying Qdrant.
"""

import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PIKERAG = ROOT / "pikerag_integration"
for path in (PIKERAG, PIKERAG / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=PIKERAG / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator import DecompositionOrchestrator, DEFAULT_MAX_SUB_QUESTIONS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [query] %(levelname)s: %(message)s")
logger = logging.getLogger("query_service")

PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}


def _build_client(provider: str):
    # Imported lazily per-provider - same reasoning as cli.py's own
    # _build_client: picking one provider shouldn't require the other
    # provider's SDK or API key to be present too.
    if provider == "anthropic":
        from llm_clients.anthropic_client import AnthropicClient
        return AnthropicClient()
    elif provider == "gemini":
        from llm_clients.gemini_client import GeminiClient
        return GeminiClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}', expected 'anthropic' or 'gemini'.")


app = FastAPI(title="InrBonds AI - Ask a Question")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

# In-memory job tracking - fine for a single-process demo service, same
# caveat as upload_service/app.py: a real deployment would move this to a
# shared store so status survives a restart and works across workers.
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        _JOBS.setdefault(job_id, {}).update(fields)


class AskRequest(BaseModel):
    question: str
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    job_id = str(uuid.uuid4())
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    _set_job(
        job_id,
        status="queued",
        question=question,
        provider=provider,
        round=0,
        phase=None,
        error=None,
        result=None,
    )

    thread = threading.Thread(
        target=_run_question,
        args=(job_id, question, provider, request.max_sub_questions),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "provider": provider}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return {"job_id": job_id, **job}


def _run_question(job_id: str, question: str, provider: str, max_sub_questions: int) -> None:
    """
    Runs in a background thread - a full decomposition answer can take
    minutes across several LLM round-trips (each a blocking network call),
    which would stall the async event loop for the whole request if done
    inline (same reasoning as upload_service/app.py's _run_pipeline).
    """
    model = os.getenv("ANTHROPIC_MODEL" if provider == "anthropic" else "GEMINI_MODEL") or PROVIDER_DEFAULTS[provider]
    client = None
    try:
        _set_job(job_id, status="running")
        client = _build_client(provider)
        orchestrator = DecompositionOrchestrator(
            llm_client=client,
            llm_config={"model": model, "max_tokens": int(os.getenv("MAX_TOKENS", "4096")), "temperature": 0},
            max_sub_questions=max_sub_questions,
        )

        def on_round(round_number: int, phase: str) -> None:
            logger.info("[%s] round %d: %s", job_id, round_number, phase)
            _set_job(job_id, round=round_number, phase=phase)

        result = orchestrator.answer(question, on_round=on_round)
        logger.info("[%s] Done", job_id)
        _set_job(job_id, status="done", result=result)

    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of dying silently in a thread
        logger.exception("[%s] Question failed", job_id)
        _set_job(job_id, status="failed", error=str(exc))

    finally:
        if client is not None:
            client.close()
