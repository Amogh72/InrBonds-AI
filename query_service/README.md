# InrBonds AI — Query Service

The first client-facing piece for `pikerag_integration/`: a minimal web
page where someone types a question about a loaded bond/NCD document,
and it runs the full PIKE-RAG decomposition loop
(`orchestrator.DecompositionOrchestrator.answer()`) in the background,
with live round-by-round progress shown on the page.

Before this existed, asking a question meant running
`pikerag_integration/cli.py` by hand in a terminal. This mirrors
`upload_service/`'s job/polling pattern — same shape, different
pipeline.

## Setup

This service imports directly from `../pikerag_integration` (not as an
installed package — same `sys.path` trick `cli.py`/`orchestrator.py`
use), so it needs that project's requirements installed too, plus its
own:

```bash
cd query_service
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows; macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -r ../pikerag_integration/requirements.txt
```

Or, simpler: reuse the venv you already set up for
`pikerag_integration/` and just add this service's two extra packages:

```bash
cd pikerag_integration
.venv\Scripts\Activate.ps1
pip install -r ..\query_service\requirements.txt
cd ..\query_service
uvicorn app:app --reload --port 8090
```

## Prerequisites

- Docker containers (MySQL, Qdrant, Neo4j) running: `docker compose up -d` from `mcp_server/`, with at least one document loaded.
- `pikerag_integration/.env` filled in with one provider's API key (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`, matching `LLM_PROVIDER` — see that project's README for the free-tier-vs-paid tradeoff).

This service only *reads* from the databases (via `vector_search`), it
never writes — unlike `upload_service/`, there's no separate
write-capable credential to configure here.

## Run

```bash
uvicorn app:app --reload --port 8090
```

Open `http://localhost:8090/`, type a question (e.g. "What is PFC's
Series III coupon?"), and click **Ask**. The page polls
`/api/status/{job_id}` every 1.5s and shows which decomposition round
is running and what phase it's in (proposing sub-questions → retrieving
candidates → selecting the best one → ... → composing the final
answer) — a single question can take anywhere from several seconds to
a few minutes depending on how many rounds it takes and which provider
answers it, so this exists specifically so the page shows *something*
moving rather than a blank spinner the whole time.

The final answer shows the rationale and citations by default; the
full decomposition trace (every round's proposed sub-questions,
retrieved candidates, and what was selected and why) is available
behind a "Show full decomposition trace" toggle for anyone who wants
to see the reasoning, not just take the answer on faith.

## What this is (and isn't)

- **In-memory job tracking** — fine for a single-process local demo;
  restarting the service loses in-flight job status.
- **No auth, no rate limiting, no per-user quota tracking** — this is a
  local development/demo tool, not something to expose on the open
  internet as-is. If running on Gemini's free tier, be aware the
  *provider's own* daily quota (not anything this service tracks) is
  shared across every question asked through it.
- **One LLM provider for the whole service**, chosen server-side via
  `LLM_PROVIDER` in `.env` — there's no per-question provider picker in
  the UI, so end users never need their own API key.
- **Same scope limitations as `pikerag_integration/` itself** — see
  that project's README's "Current scope and honest limitations"
  section (single retrieval tool, no deliberate multi-document
  routing). This service is a UI on top of that loop, not a fix for
  its gaps.
