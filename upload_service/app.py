"""
Minimal document upload service: a client drops a bond/NCD prospectus
PDF into this web page, and it runs the full pipeline end to end -
extraction (src/ingestion/canonical_extractor.py) followed by all
three database loaders (mcp_server/scripts/load_canonical_to_*.py) -
in the background, with live progress served to the page.

Nothing upstream of this exists yet: up to this point, getting a new
document into the system meant someone with terminal access running
four commands by hand. This is the first client-facing entry point.

Run:
    cd upload_service
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8080
Then open http://localhost:8080/
"""

import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MCP_SERVER = ROOT / "mcp_server"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

for path in (SRC, MCP_SERVER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# -----------------------------------------------------------------------
# This service WRITES to all three databases (extraction + load), unlike
# the MCP server (mcp_server/server.py), which only ever READS from them
# via the now-read-only mcp_readonly user. Setting SQL_USER/SQL_PASSWORD
# here, BEFORE importing anything that imports mcp_server/config.py,
# means config.py's load_dotenv() (which never overwrites an
# already-set env var) picks these up instead of .env's mcp_readonly -
# so this service gets write access without weakening the MCP server's
# read-only grant. Override via INGEST_SQL_USER/INGEST_SQL_PASSWORD env
# vars for a real deployment instead of reusing root.
# -----------------------------------------------------------------------
os.environ.setdefault("SQL_USER", os.getenv("INGEST_SQL_USER", "root"))
os.environ.setdefault("SQL_PASSWORD", os.getenv("INGEST_SQL_PASSWORD", "rootpass_change_me"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ingestion.canonical_extractor import extract_canonical_document, save_canonical_document  # noqa: E402
from ingestion.chunker import make_document_id  # noqa: E402

from scripts.load_canonical_to_mysql import load_canonical_document as load_into_mysql  # noqa: E402
from scripts.load_canonical_to_qdrant import load_canonical_chunks as load_into_qdrant  # noqa: E402
from scripts.load_canonical_to_neo4j import load_canonical_document as load_into_neo4j  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [upload] %(levelname)s: %(message)s")
logger = logging.getLogger("upload_service")

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="InrBonds AI - Document Upload")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

# In-memory job tracking - fine for a single-process demo service; a real
# deployment would move this to a shared store so status survives a
# restart and works across multiple worker processes.
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        _JOBS.setdefault(job_id, {}).update(fields)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
    document_type: str | None = Form(None),
) -> dict[str, Any]:
    original_name = Path(file.filename or "document.pdf").name  # strip any path components
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    resolved_document_id = (document_id or "").strip() or make_document_id(original_name)

    with _JOBS_LOCK:
        existing = _JOBS.get(resolved_document_id)
        if existing and existing["status"] not in ("done", "failed"):
            raise HTTPException(
                status_code=409,
                detail=f"A job for document_id '{resolved_document_id}' is already {existing['status']}.",
            )

    pdf_path = DATA_RAW / f"{resolved_document_id}.pdf"
    contents = await file.read()
    pdf_path.write_bytes(contents)

    _set_job(
        resolved_document_id,
        status="queued",
        original_filename=original_name,
        document_type=(document_type or "").strip() or None,
        error=None,
        summary=None,
    )

    thread = threading.Thread(
        target=_run_pipeline,
        args=(resolved_document_id, pdf_path, (document_type or "").strip() or None),
        daemon=True,
    )
    thread.start()

    return {"job_id": resolved_document_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return {"job_id": job_id, **job}


def _run_pipeline(document_id: str, pdf_path: Path, document_type_override: str | None) -> None:
    """
    Runs in a background thread (extraction is CPU-bound, synchronous
    code - running it directly inside an async request handler would
    block the whole event loop for the ~60-90s a full document takes).
    """
    json_path = DATA_PROCESSED / f"{document_id}.json"

    try:
        logger.info("[%s] Extracting canonical document from %s", document_id, pdf_path.name)
        _set_job(document_id, status="extracting")
        document = extract_canonical_document(pdf_path=str(pdf_path), document_id=document_id)
        if document_type_override:
            document.document.document_type = document_type_override
        save_canonical_document(document=document, output_path=str(json_path))

        logger.info("[%s] Loading into MySQL", document_id)
        _set_job(document_id, status="loading_sql")
        load_into_mysql(json_path)

        logger.info("[%s] Loading into Qdrant", document_id)
        _set_job(document_id, status="loading_qdrant")
        load_into_qdrant(str(json_path), issuer=None, document_type=document_type_override)

        logger.info("[%s] Loading into Neo4j", document_id)
        _set_job(document_id, status="loading_neo4j")
        load_into_neo4j(str(json_path))

        issue = document.issuer.issues[0] if document.issuer.issues else None
        summary = {
            "issuer": document.issuer.name,
            "document_name": document.document.document_name,
            "series_count": len(issue.series) if issue else 0,
            "ratings_count": len(issue.ratings) if issue else 0,
            "lead_managers_count": len(issue.lead_managers) if issue else 0,
            "debenture_trustee": (issue.debenture_trustee.value if issue and issue.debenture_trustee else None),
            "chunk_count": len(document.chunks),
        }
        logger.info("[%s] Done: %s", document_id, summary)
        _set_job(document_id, status="done", summary=summary)

    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of dying silently in a thread
        logger.exception("[%s] Pipeline failed", document_id)
        _set_job(document_id, status="failed", error=str(exc))
