# InrBonds AI — Document Upload Service

The first client-facing piece of the project: a minimal web page where
someone drops in a new bond/NCD prospectus PDF, and it runs the full
pipeline end to end in the background — extraction
(`src/ingestion/canonical_extractor.py`) followed by all three
database loaders (`mcp_server/scripts/load_canonical_to_*.py`) — with
live progress shown on the page.

Before this existed, getting a new document into the system meant
someone with terminal access running the extractor and three loader
scripts by hand, one at a time.

## Setup

This service imports directly from both `../src` (the extraction
pipeline) and `../mcp_server` (the database loaders), so it needs
**both** of those requirements files installed, plus its own:

```bash
cd upload_service
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows; macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -r ../requirements.txt           # pymupdf, pdfplumber, pydantic
pip install -r ../mcp_server/requirements.txt # mysql-connector-python, qdrant-client, neo4j, sentence-transformers, ...
```

Or, simpler: reuse the venv you already set up for `mcp_server/` (it
already has everything mcp_server needs) and just add this service's
two extra packages and the root pipeline's:
```bash
cd mcp_server
.venv\Scripts\Activate.ps1
pip install -r ..\upload_service\requirements.txt
pip install -r ..\requirements.txt
cd ..\upload_service
uvicorn app:app --reload --port 8080
```

## Prerequisites

- Docker containers (MySQL, Qdrant, Neo4j) running: `docker compose up -d` from `mcp_server/`.
- `mcp_server/.env` in place (`cp .env.example .env` if you haven't already).

This service writes to the databases, so it needs write-capable
credentials — it defaults to `root`/`rootpass_change_me` (matching
`docker-compose.yml`), independent of whatever `.env`'s `SQL_USER` is
set to for the (now read-only) MCP server. Override with
`INGEST_SQL_USER`/`INGEST_SQL_PASSWORD` environment variables instead
of reusing root for anything beyond local testing.

## Run

```bash
uvicorn app:app --reload --port 8080
```

Open `http://localhost:8080/`, pick a PDF (e.g.
`data/raw/capri_global_ncd_2026.pdf`), optionally set a document ID,
and click **Upload & Process**. A 200-page document takes roughly
60-90 seconds for extraction plus another 10-20 seconds to load into
all three databases — the page polls `/api/status/{job_id}` every
1.5s and shows each stage as it completes.

## What this is (and isn't)

- **In-memory job tracking** — fine for a single-process local demo;
  restarting the service loses in-flight job status (the underlying
  data is unaffected, since it's already committed to the databases
  by the time a job reaches `done`).
- **No auth, no upload size limits, no rate limiting** — this is a
  local development/demo tool, not something to expose on the open
  internet as-is.
- **Synchronous per-document processing** — one document is extracted
  and loaded at a time per upload; uploading two different documents
  concurrently runs two background threads over the same database
  connections, which works but isn't optimized for throughput.
