"""
MCP server for the inrbonds Bond Document Assistant.

Exposes retrieval tools to the LLM over the Model Context Protocol,
backed by three databases loaded from the real pipeline's canonical
BondDocument JSON (src/schema/canonical_schema.py):
  - sql_query / sql_search_bond_terms / sql_get_issue_overview
        -> structured bond data (MySQL)
  - vector_search  -> semantic search over chunk text (Qdrant)
  - graph_query / graph_find_issues_by_participant
        -> entity relationships (Neo4j)

The LLM never touches the databases directly. Every tool response is a
ToolSuccess/ToolError with citation metadata attached, so downstream
answer generation can cite its sources.

Run:
    python server.py                # stdio transport (for local MCP clients)
    MCP_TRANSPORT=streamable-http python server.py   # HTTP transport
"""

import logging

from mcp.server.fastmcp import FastMCP

from config import server_config, qdrant_config
from db.sql_db import sql_db
from db.vector_db import vector_db
from db.graph_db import graph_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mcp.server")

mcp = FastMCP(server_config.name)


# ---------------------------------------------------------------------------
# Embedding provider wiring
# Local sentence-transformers model - runs offline, no API key/cost.
# Loaded lazily and cached so it's only pulled into memory once per process,
# not once per call (model load is the expensive part; encode() is fast).
# Must match config.qdrant_config.embedding_model / embedding_dim exactly,
# since the same function is used at both ingestion time (loader script)
# and query time (vector_search) - if they ever diverge, similarity scores
# become meaningless.
# ---------------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s' (first call only, this can take a moment)...",
                     qdrant_config.embedding_model)
        _embedding_model = SentenceTransformer(qdrant_config.embedding_model)
        actual_dim = _embedding_model.get_sentence_embedding_dimension()
        if actual_dim != qdrant_config.embedding_dim:
            raise RuntimeError(
                f"EMBEDDING_DIM in .env is {qdrant_config.embedding_dim} but "
                f"'{qdrant_config.embedding_model}' actually produces {actual_dim}-dim vectors. "
                f"Fix EMBEDDING_DIM in .env and recreate the Qdrant collection."
            )
    return _embedding_model


def _default_embed_fn(text: str) -> list[float]:
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


vector_db.set_embed_fn(_default_embed_fn)


# ---------------------------------------------------------------------------
# SQL tools
# ---------------------------------------------------------------------------
@mcp.tool()
def sql_query(
    sql: str,
    citation_table: str | None = None,
    citation_pk_field: str | None = None,
) -> dict:
    """
    Run a read-only SQL query against the structured bond database (MySQL).

    Use for: anything the typed tools (sql_search_bond_terms,
    sql_get_issue_overview) don't cover - custom aggregations, joins
    across issuers/documents, etc.

    Args:
        sql: A SELECT (or WITH ... SELECT) statement. Any DML/DDL is rejected.
        citation_table: Name of the primary table queried, used to build citations.
        citation_pk_field: Name of the primary-key column in the result set,
            used to build row-level citations like "series:1042".

    Returns a JSON object with `data` (rows), `citations`, `row_count`, and `truncated`.
    """
    result = sql_db.run_query(sql, citation_table=citation_table, citation_pk_field=citation_pk_field)
    return result.model_dump()


@mcp.tool()
def sql_search_bond_terms(
    issuer: str | None = None,
    document_name: str | None = None,
    series_code: str | None = None,
    category_code: str | None = None,
    tenor: str | None = None,
    nature_of_indebtedness: str | None = None,
) -> dict:
    """
    Typed, filter-based search over exact bond/NCD series terms (coupon,
    effective yield, issue price, maturity amount, tenor, face value, etc.)

    Prefer this over sql_query whenever the question maps to specific known
    fields.

    Use for: "what is the coupon for Series II Category III", "which series
    have a 10-year tenor", "what is the face value of Series III", "which
    bonds are secured", "what is the maturity amount of Series V Category IV".

    Args:
        issuer: Exact issuer name, e.g. "Power Finance Corporation Limited".
        document_name: Exact source filename, e.g. "pfc_ncd_2026.pdf".
        series_code: Series identifier as detected by the extractor, e.g. "II".
            Not limited to I-V - whatever series exist in the data.
        category_code: Investor category code, always atomic - "I", "II",
            "III", or "IV". If the source document merges categories (e.g.
            one combined row for I, II, III & IV), that row was split into
            separate rows at load time, so query them individually here.
        tenor: Partial match, e.g. "10 years".
        nature_of_indebtedness: e.g. "Secured".

    Returns a JSON object with `data` (series + category-term rows),
    `citations` (document_name + source page + series/category section),
    `row_count`, `truncated`.
    """
    result = sql_db.search_bond_terms(
        issuer=issuer,
        document_name=document_name,
        series_code=series_code,
        category_code=category_code,
        tenor=tenor,
        nature_of_indebtedness=nature_of_indebtedness,
    )
    return result.model_dump()


@mcp.tool()
def sql_get_issue_overview(
    issuer: str | None = None,
    document_name: str | None = None,
) -> dict:
    """
    Typed search over issue-level facts that don't belong to any one
    series: issue size, shelf limit, green shoe option, security
    cover/type, listing/exchange, open/close/allotment dates,
    debenture trustee, ratings, and lead managers.

    Use for: "what is the shelf limit", "who is the debenture trustee",
    "what is the green shoe option", "what ratings has this issue
    received", "who are the lead managers", "when does the issue
    open/close".

    Returns a JSON object with `data` (issue rows, each including
    nested `ratings` and `lead_managers`), `citations`, `row_count`,
    `truncated`.
    """
    result = sql_db.get_issue_overview(issuer=issuer, document_name=document_name)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Vector tool
# ---------------------------------------------------------------------------
@mcp.tool()
def vector_search(
    query_text: str,
    top_k: int = 8,
    issuer_filter: str | None = None,
    document_type_filter: str | None = None,
    chunk_type_filter: str | None = None,
    series_id_filter: str | None = None,
    category_id_filter: str | None = None,
    score_threshold: float | None = None,
) -> dict:
    """
    Semantic search over bond document chunk text stored in Qdrant -
    both "prose" chunks (definitions, risk factors, procedural text)
    and "structured_fact" chunks (one series/category's terms as a
    short natural-language statement).

    Use for: open-ended or conceptual questions that don't map to a
    structured column, e.g. "what does the document say about early
    redemption", "explain the green shoe option", "what does ASBA mean".

    Args:
        query_text: Natural language query to embed and search with.
        top_k: Number of chunks to return.
        issuer_filter: Optional exact-match filter on issuer name.
        document_type_filter: Optional exact-match filter, e.g. "Bond/NCD Prospectus".
        chunk_type_filter: Optional exact-match filter, "structured_fact" or "prose".
            Prefer sql_search_bond_terms over leaving this unset when you
            specifically want a structured_fact chunk - it's an exact
            lookup instead of a similarity search.
        series_id_filter: Optional exact-match filter on the internal series
            id (e.g. "pfc_ncd_2026_series_III"), to scope a search to one series.
        category_id_filter: Optional exact-match filter on investor category
            (e.g. "III").
        score_threshold: Optional minimum similarity score (0-1).

    Returns a JSON object with `data` (matching chunks + text + page/section),
    `citations`, `row_count`.
    """
    result = vector_db.search(
        query_text=query_text,
        top_k=top_k,
        issuer_filter=issuer_filter,
        document_type_filter=document_type_filter,
        chunk_type_filter=chunk_type_filter,
        series_id_filter=series_id_filter,
        category_id_filter=category_id_filter,
        score_threshold=score_threshold,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------
@mcp.tool()
def graph_query(
    cypher: str | None = None,
    issuer_name: str | None = None,
    depth: int = 2,
) -> dict:
    """
    Query entity relationships (issuer -> issue -> series/ratings/
    trustee/lead managers) stored in Neo4j.

    Two modes:
      1. Provide `cypher` for a free-form read-only Cypher query (MATCH/RETURN only).
      2. Provide `issuer_name` (and optional `depth`) to use the pre-built
         "relationship neighborhood" recipe - safer and usually sufficient.

    Use for: "what is this issuer's rating history across agencies",
    "what series does this issue have", "who is the debenture trustee
    and who are the lead managers, all at once".

    Returns a JSON object with `data` (nodes/relationships), `citations`, `row_count`.
    """
    if cypher:
        result = graph_db.run_cypher(cypher)
    elif issuer_name:
        result = graph_db.issuer_relationship_neighborhood(issuer_name, depth=depth)
    else:
        from models import ToolError
        result = ToolError(error="missing_argument", detail="Provide either `cypher` or `issuer_name`.")
    return result.model_dump()


@mcp.tool()
def graph_find_issues_by_participant(
    participant_name: str,
    role: str = "any",
) -> dict:
    """
    Cross-document lookup: every issue a named debenture trustee, lead
    manager, or rating agency has been involved with. This is the kind
    of multi-hop, cross-document question a graph answers directly
    that a single SQL table doesn't - and it only gets interesting
    once more than one document has been loaded.

    Use for: "which issuers has Beacon Trusteeship Limited served as
    trustee for", "which issues has CRISIL rated", "which issuers has
    A.K. Capital Services Limited lead-managed".

    Args:
        participant_name: Case-insensitive substring match, e.g. "Beacon" or "CRISIL".
        role: "trustee", "lead_manager", "rating_agency", or "any" (default,
            searches across all three).

    Returns a JSON object with `data` (issuer/issue/participant rows), `citations`, `row_count`.
    """
    result = graph_db.find_issues_by_participant(participant_name, role=role)
    return result.model_dump()


@mcp.tool()
def health_check() -> dict:
    """Check connectivity to all three backing databases. Useful for diagnostics."""
    return {
        "sql": sql_db.health_check(),
        "vector": vector_db.health_check(),
        "graph": graph_db.health_check(),
    }


if __name__ == "__main__":
    logger.info("Starting MCP server '%s' on transport=%s", server_config.name, server_config.transport)
    if server_config.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=server_config.http_host, port=server_config.http_port)
    else:
        mcp.run(transport="stdio")
