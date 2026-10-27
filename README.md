# InrBonds AI

Question answering over Indian bond / NCD (non-convertible debenture) offer
documents. A prospectus PDF goes in; you can then ask questions such as
*"What is PFC's Series III coupon?"* or *"Compare PFC's and IIFL's debenture
trustee"* and get an answer with citations to the source chunks.

The project has three parts:

1. **Ingestion**: extracts a structured, validated representation of each PDF.
2. **Retrieval**: stores it in three databases behind an MCP server.
3. **Question answering**: an LLM answers using a decomposition loop adapted
   from Microsoft's [PIKE-RAG](https://github.com/microsoft/PIKE-RAG).

## How it fits together

```
 PDF prospectus
      |
      v
 src/ingestion/canonical_extractor.py   -> data/processed/<doc>.json
      |                                    (one canonical BondDocument)
      v
 mcp_server/scripts/load_canonical_to_*.py
      |
      +--> MySQL   exact facts (series terms, ratings, trustee, ...)
      +--> Qdrant  vector search over document chunks
      +--> Neo4j   issuer / participant relationships
      |
      v
 mcp_server/        tools over the three databases (read-only)
      |
      v
 pikerag_integration/orchestrator.py    propose sub-questions -> retrieve ->
      |                                  select -> repeat -> answer
      v
 query_service/     web UI with live progress and a decomposition trace
```

`upload_service/` is a second web UI that runs the extraction and database
loading for a new PDF in the background, so no terminal is needed.

## Repository layout

| Folder | What it does |
|---|---|
| `src/ingestion/` | PDF parsing, structure detection, table extraction, chunking, canonical extraction |
| `src/schema/` | The canonical `BondDocument` schema (pydantic) |
| `mcp_server/` | MCP server and loaders for MySQL, Qdrant and Neo4j, plus `docker-compose.yml` |
| `pikerag_integration/` | PIKE-RAG decomposition loop, LLM clients (Claude, Gemini), retrieval adapters |
| `query_service/` | Web UI for asking questions (port 8090) |
| `upload_service/` | Web UI for uploading and loading a PDF (port 8080) |
| `data/raw/` | Sample prospectuses (PFC, IIFL Wealth Prime, Capri Global) |
| `tests/` | Unit tests for the ingestion pipeline |

Each service folder has its own README with details.

## Quick start

Requirements: Python 3.12, Docker, and one LLM API key (Anthropic or Gemini).
The commands below are PowerShell; on macOS/Linux use `source .venv/bin/activate`.

**1. Start the databases**

```powershell
cd mcp_server
copy .env.example .env
docker compose up -d
```

If port 3306 is already in use on your machine, set `MYSQL_HOST_PORT=3307`
and `SQL_PORT=3307` in `mcp_server/.env`.

**2. Extract a document** (generated files in `data/processed/` are gitignored)

```powershell
cd ..
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.ingestion.canonical_extractor data/raw/pfc_ncd_2026.pdf --document-id pfc_ncd_2026 --output data/processed/pfc_ncd_2026.json
```

**3. Load it into the databases**

```powershell
pip install -r mcp_server/requirements.txt
cd mcp_server
python scripts/setup_qdrant_collection.py
python scripts/load_canonical_to_mysql.py  ../data/processed/pfc_ncd_2026.json
python scripts/load_canonical_to_qdrant.py ../data/processed/pfc_ncd_2026.json
python scripts/load_canonical_to_neo4j.py  ../data/processed/pfc_ncd_2026.json
```

Alternatively, run `upload_service` and upload the PDF from the browser (see
`upload_service/README.md`).

**4. Ask questions**

```powershell
cd ..\pikerag_integration
copy .env.example .env      # set one API key and LLM_PROVIDER
pip install -r requirements.txt
python cli.py "What is PFC's Series III coupon?" --show-trace
```

Or use the web UI:

```powershell
pip install -r ..\query_service\requirements.txt
cd ..\query_service
uvicorn app:app --reload --port 8090
```

Then open http://localhost:8090. The first run downloads the embedding model
(`all-mpnet-base-v2`), so allow some time.

## How PIKE-RAG is used

We adapted PIKE-RAG's knowledge-aware task decomposition rather than using its
codebase directly. The details are in `pikerag_integration/README.md` and
`pikerag_integration/vendor/pikerag/NOTICE.md`.

- **Taken from PIKE-RAG (MIT licensed, vendored with attribution):** the four
  decomposition prompts and the LLM client interface.
- **Written for this project:** the decompose, retrieve, select, answer loop
  (`orchestrator.py`); retrieval over our own Qdrant, MySQL and Neo4j stack;
  Claude and Gemini clients; the web UI.
- **Not used:** PIKE-RAG's own retriever and its offline atomic-question
  preprocessing, which our data has not been through.

The LLM proposes sub-questions and picks the best retrieved candidate. The
searching itself is done by code: vector search in Qdrant, plus rule-based SQL
and graph lookups when a sub-question names a known issuer, participant or
series.

## Limitations

- Structured SQL/graph retrieval only triggers for known entities (loaded
  issuers, participants, "Series X Category Y" patterns). Other questions use
  semantic search only.
- Each question takes several LLM round trips, so latency and cost are higher
  than single-shot RAG. We have not benchmarked decomposition against plain
  vector search.
- Not load-tested. Job tracking in the web services is in-memory, and there is
  no authentication. These are local demo tools.
- Default database passwords in `docker-compose.yml` are for local use only.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest tests
```

`mcp_server/tests/` and `pikerag_integration/tests/` contain further tests;
see each folder's README for how to run them.
