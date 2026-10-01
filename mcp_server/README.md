# inrbonds MCP Server — Bond Document Assistant

The communication layer between an LLM and three backing databases
(MySQL, Qdrant, Neo4j), exposed over the Model Context Protocol. The
LLM never touches the databases directly — every tool call returns a
`ToolSuccess`/`ToolError` with citation metadata attached.

## Where this came from

This was originally built by a teammate (Ankit) against the pipeline's
earlier, pre-`canonical_schema.py` intermediate outputs (a standalone
`series_terms.json`, a standalone `chunks.json`, and a standalone
`domain_knowledge.json` entity-graph export). Since then the pipeline
was consolidated into one canonical `BondDocument` schema
(`src/schema/canonical_schema.py`) with ratings, debenture trustee,
lead managers, and all issue-level terms (shelf limit, green shoe
option, security cover, ...) hoisted onto `BondIssue`/`IssueTerms`,
and `structured_fact` + `prose` chunks merged into one `chunks` list
with `chunk_type`/`series_id`/`category_id` tags.

This version reconciles the original MCP server/database design
against that current schema: the three databases, the MCP tool
surface, and the overall "LLM never touches a DB directly, every
answer carries a citation" design are all unchanged — the schemas and
loader scripts are rewritten to read `data/processed/<document_id>.json`
(the real canonical pipeline output) directly, and to cover the facts
(ratings, trustee, lead managers, issue-level terms) that didn't exist
in the old intermediate files at all.

## Project layout

```
mcp_server/
├── server.py            # FastMCP server, defines the tools
├── config.py             # env-driven config for all 3 databases
├── models.py              # ToolSuccess / ToolError / Citation schemas
├── db/
│   ├── sql_db.py          # MySQL — sql_query / sql_search_bond_terms / sql_get_issue_overview
│   ├── vector_db.py        # Qdrant — vector_search
│   ├── graph_db.py         # Neo4j — graph_query / graph_find_issues_by_participant
│   └── graph_utils.py      # label/rel-type sanitizing helpers
├── sql/schema.sql         # MySQL schema
├── graph/schema.cypher    # Neo4j constraints/indexes
├── scripts/
│   ├── setup_qdrant_collection.py
│   ├── load_canonical_to_qdrant.py   # chunks -> Qdrant
│   ├── load_canonical_to_mysql.py    # structured facts -> MySQL
│   ├── load_canonical_to_neo4j.py    # relationships -> Neo4j
│   └── test_tools.py                 # manual end-to-end smoke test
├── tests/test_loaders.py  # fast, DB-free unit tests for the loaders' pure logic
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

## Setup

```bash
cd mcp_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in real credentials for non-local use

docker compose up -d    # starts MySQL, Qdrant, Neo4j locally
```

### Load real data

This reads directly from the pipeline's canonical JSON output — run
the main pipeline first (see the repo root README) to produce
`data/processed/pfc_ncd_2026.json` (or any other document), then:

```bash
python scripts/setup_qdrant_collection.py
python scripts/load_canonical_to_mysql.py  ../data/processed/pfc_ncd_2026.json
python scripts/load_canonical_to_qdrant.py ../data/processed/pfc_ncd_2026.json
python scripts/load_canonical_to_neo4j.py  ../data/processed/pfc_ncd_2026.json
```

Each loader is idempotent — re-running it against the same (or a
corrected) canonical JSON upserts rather than duplicates. Load more
than one document's JSON to populate the databases with multiple
issuers — this is specifically what makes `graph_find_issues_by_participant`
and cross-document SQL queries interesting (see "Current limitations"
below).

### Wire an embedding provider

`server.py`'s embedding function uses a local `sentence-transformers`
model (`all-mpnet-base-v2` by default, 768-dim, runs offline) — change
`EMBEDDING_MODEL`/`EMBEDDING_DIM` in `.env` to swap models, and re-run
`setup_qdrant_collection.py` against a fresh collection if the
dimension changes.

## Running

```bash
python server.py                                   # stdio (for Claude Desktop, local MCP clients)
MCP_TRANSPORT=streamable-http python server.py       # HTTP, for remote clients
```

### Registering with an MCP client (e.g. Claude Desktop)
```json
{
  "mcpServers": {
    "inrbonds-document-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

## Tools exposed

| Tool | Backend | Purpose |
|---|---|---|
| `sql_query` | MySQL | Raw read-only SQL fallback |
| `sql_search_bond_terms` | MySQL | Exact series/category terms: coupon, yield, tenor, face value, ... |
| `sql_get_issue_overview` | MySQL | Issue-level terms, debenture trustee, ratings, lead managers |
| `vector_search` | Qdrant | Semantic search over prose and/or structured_fact chunk text |
| `graph_query` | Neo4j | Issuer → issue → series/ratings/trustee/lead-manager neighborhood (or raw Cypher) |
| `graph_find_issues_by_participant` | Neo4j | Cross-document: every issue a named trustee/lead manager/rating agency touched |
| `health_check` | all three | Connectivity diagnostic |

Every tool returns:
```json
{
  "ok": true,
  "tool": "sql_search_bond_terms",
  "data": [ { "...": "..." } ],
  "citations": [ { "source_type": "sql", "source_id": "series:1042", "...": "..." } ],
  "row_count": 1,
  "truncated": false
}
```

## How the three databases connect to each other

They don't talk to each other directly — they're linked only by
shared identity strings, set consistently by the loaders from the same
canonical JSON:

- **issuer name** — the exact `BondDocument.issuer.name` string, present
  natively in all three (SQL's `issuers.issuer_name`, Qdrant's `issuer`
  payload field, Neo4j's `Issuer.name`).
- **document_name** — the source PDF filename, consistent across all three.
- **page numbers / chunk IDs** — every tool's `Citation` carries
  `document_name` + `page_number` + a section label, so an answer
  sourced from MySQL and one sourced from Qdrant both point back to
  the same underlying document comparably.

There is still no automated join between the databases in one call —
that's left to whatever orchestration layer decides which tool(s) to
invoke for a given question.

## Security notes

- `sql_query` and `graph_query`'s raw-query modes both reject
  non-read-only statements in code. **Do not rely on this alone** —
  create dedicated read-only DB users/roles for this service in MySQL
  and Neo4j (the `mcp_readonly` user in `.env.example` is a reminder,
  not an enforced grant — you must create it in your actual MySQL
  instance).
- Query timeouts and a max-rows cap (`MAX_ROWS_RETURNED`) protect
  against runaway queries or accidental full-table dumps being handed
  to the LLM.
- Results are truncated with a `truncated: true` flag rather than
  silently dropped.

## Current limitations

1. **No cross-database join logic.** The three tools are independently
   correct, but nothing automatically combines a SQL fact with its
   Qdrant source passage or its graph relationships in a single call —
   that's for an orchestration/answer-generation layer sitting above
   this MCP server.
2. **Tested against one issuer's data at a time so far.** The
   generic-design decisions (atomic category splitting, idempotent
   upserts, reusable RatingAgency/DebentureTrustee/LeadManager nodes)
   are built with multiple documents/issuers in mind, but the real
   payoff of `graph_find_issues_by_participant` (and multi-issuer SQL
   queries) only shows once more than one document has actually been
   loaded.
3. **No production security review.** Read-only DB users are a setup
   step for local development; credentials, network exposure, and
   access control haven't been reviewed for any real deployment target.
4. **`IssueTerms.additional_terms`** (a free-form `Dict[str, Fact]` for
   less-common facts) is intentionally not modeled in SQL — it stays
   reachable via Qdrant/Neo4j instead of an open-ended EAV table.
