"""
SQL database module (MySQL) for structured bond information:
issuers, issues, ratings, debenture trustee, lead managers, series,
and series x investor-category terms. Schema: sql/schema.sql, loaded
from a canonical BondDocument JSON by
scripts/load_canonical_to_mysql.py.

Design notes:
- Uses SQLAlchemy's engine + text() so the MCP tool layer never builds
  raw f-string SQL (avoids injection from LLM-generated input).
- Read-only by convention: the DB user configured for this service should
  only have SELECT grants. Enforce that at the MySQL grant level too,
  not just in application code.
- Every row returned carries enough identifying info (table + primary key)
  to construct a Citation without a second round trip.
"""

import logging
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import sql_config, server_config
from models import Citation, ToolError, ToolSuccess

logger = logging.getLogger("mcp.sql")

# Statements the LLM-facing tool must never be allowed to execute.
_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "replace", "merge", "call", "exec",
)


class SQLDatabase:
    def __init__(self) -> None:
        self._engine: Engine = self._build_engine()

    def _build_engine(self) -> Engine:
        # mysql+mysqlconnector://user:password@host:port/database
        url = (
            f"mysql+mysqlconnector://{sql_config.user}:{sql_config.password}"
            f"@{sql_config.host}:{sql_config.port}/{sql_config.database}"
        )
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connection_timeout": server_config.query_timeout_seconds},
        )

    @contextmanager
    def _connect(self):
        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _is_read_only(sql: str) -> bool:
        lowered = sql.strip().lower()
        if not lowered.startswith(("select", "with")):
            return False
        return not any(f" {kw} " in f" {lowered} " or lowered.startswith(kw) for kw in _FORBIDDEN_KEYWORDS)

    def run_query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        citation_table: str | None = None,
        citation_pk_field: str | None = None,
    ) -> ToolSuccess | ToolError:
        """
        Execute a read-only SQL query and return rows + citation metadata.

        citation_table / citation_pk_field let the caller tell us which
        column in the result set is the primary key, so we can build a
        Citation like source_id="series:1042" per row. If omitted,
        citations are coarser (query-level only).
        """
        if not self._is_read_only(sql):
            return ToolError(error="rejected_query", detail="Only read-only SELECT/WITH statements are permitted.")

        try:
            with self._connect() as conn:
                result = conn.execute(text(sql).execution_options(timeout=server_config.query_timeout_seconds), params or {})
                columns = result.keys()
                rows = result.fetchmany(server_config.max_rows_returned + 1)
        except SQLAlchemyError as exc:
            logger.exception("SQL query failed")
            return ToolError(error="query_failed", detail=str(exc.__cause__ or exc))

        truncated = len(rows) > server_config.max_rows_returned
        rows = rows[: server_config.max_rows_returned]

        data = [dict(zip(columns, row)) for row in rows]

        citations: list[Citation] = []
        for row in data:
            source_id = (
                f"{citation_table}:{row.get(citation_pk_field)}"
                if citation_table and citation_pk_field and citation_pk_field in row
                else f"query_result:{citation_table or sql_config.database}"
            )
            citations.append(
                Citation(
                    source_type="sql",
                    source_id=source_id,
                    document_name=citation_table,
                )
            )

        return ToolSuccess(
            tool="sql_query",
            data=data,
            citations=citations,
            row_count=len(data),
            truncated=truncated,
        )

    def search_bond_terms(
        self,
        issuer: str | None = None,
        document_name: str | None = None,
        series_code: str | None = None,
        category_code: str | None = None,
        tenor: str | None = None,
        nature_of_indebtedness: str | None = None,
    ) -> ToolSuccess | ToolError:
        """
        Typed, filter-based search over series + series-category terms
        (tenor, coupon, effective yield, issue price, maturity amount,
        face value) - the SQL counterpart to vector_search. Unlike
        run_query, the caller never writes SQL; they pass filters and
        get back rows with citations.

        category_code is always an atomic code (e.g. "I", "II", "III",
        "IV") as defined in investor_categories - never a compound code
        like "I_II_III_IV". The loader splits any compound
        category_id into one row per atomic category at load time (see
        load_canonical_to_mysql.py's split_category_id()).

        All filters are optional and AND-ed together; omit a filter to not
        constrain on it.
        """
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if issuer:
            conditions.append("i.issuer_name = :issuer")
            params["issuer"] = issuer
        if document_name:
            conditions.append("d.document_name = :document_name")
            params["document_name"] = document_name
        if series_code:
            conditions.append("s.series_code = :series_code")
            params["series_code"] = series_code
        if category_code:
            conditions.append("ic.category_code = :category_code")
            params["category_code"] = category_code
        if tenor:
            conditions.append("s.tenor LIKE :tenor")
            params["tenor"] = f"%{tenor}%"
        if nature_of_indebtedness:
            conditions.append("s.nature_of_indebtedness = :nature")
            params["nature"] = nature_of_indebtedness

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params["limit"] = server_config.max_rows_returned

        sql = f"""
            SELECT
                s.series_id, i.issuer_name AS issuer, d.document_name, iss.issue_name,
                s.series_code, s.tenor, s.frequency, s.coupon_type, s.face_value_inr,
                s.mode_of_interest_payment, s.maturity_redemption,
                s.nature_of_indebtedness, s.put_call_option,
                ic.category_code, sct.issue_price_inr, sct.coupon_percent,
                sct.effective_yield_percent, sct.maturity_amount_inr
            FROM series s
            JOIN issues iss ON s.issue_id = iss.issue_id
            JOIN documents d ON iss.document_id = d.document_id
            JOIN issuers i ON d.issuer_id = i.issuer_id
            LEFT JOIN series_category_terms sct ON sct.series_id = s.series_id
            LEFT JOIN investor_categories ic ON sct.category_id = ic.category_id
            {where_clause}
            ORDER BY s.series_code, ic.category_code
            LIMIT :limit
        """

        try:
            with self._connect() as conn:
                result = conn.execute(text(sql), params)
                columns = result.keys()
                rows = result.fetchmany(server_config.max_rows_returned + 1)
        except SQLAlchemyError as exc:
            logger.exception("search_bond_terms failed")
            return ToolError(error="query_failed", detail=str(exc.__cause__ or exc))

        truncated = len(rows) > server_config.max_rows_returned
        rows = rows[: server_config.max_rows_returned]
        data = [dict(zip(columns, row)) for row in rows]

        # Fetch page provenance for the series in this result set (one extra
        # query instead of N+1). series_provenance has one row per
        # (series_id, page_number, source_field) - we collapse source_field
        # here since citations only need page numbers.
        series_ids = sorted({row["series_id"] for row in data if row.get("series_id") is not None})
        pages_by_series: dict[int, list[int]] = {}
        if series_ids:
            try:
                with self._connect() as conn:
                    prov_stmt = text(
                        "SELECT DISTINCT series_id, page_number FROM series_provenance WHERE series_id IN :ids"
                    ).bindparams(bindparam("ids", expanding=True))
                    for sid, page in conn.execute(prov_stmt, {"ids": series_ids}):
                        pages_by_series.setdefault(sid, []).append(page)
            except SQLAlchemyError:
                logger.exception("Provenance lookup failed, continuing without page numbers")

        citations: list[Citation] = []
        for row in data:
            sid = row.get("series_id")
            pages = sorted(pages_by_series.get(sid, []))
            section = f"Series {row.get('series_code')}"
            if row.get("category_code"):
                section += f" / Category {row['category_code']}"
            citations.append(
                Citation(
                    source_type="sql",
                    source_id=f"series:{sid}",
                    document_name=row.get("document_name"),
                    page_number=pages[0] if pages else None,
                    section=section,
                )
            )

        return ToolSuccess(
            tool="sql_search_bond_terms",
            data=data,
            citations=citations,
            row_count=len(data),
            truncated=truncated,
        )

    def get_issue_overview(
        self,
        issuer: str | None = None,
        document_name: str | None = None,
    ) -> ToolSuccess | ToolError:
        """
        Typed search over issue-level facts that don't belong to any
        one series: IssueTerms (issue_size, shelf_limit,
        green_shoe_option, security_cover, security_type, listing,
        exchange, open/close/allotment dates), debenture_trustee, and
        ratings. The SQL counterpart to a Rating/IssueTerms-focused
        vector_search, but exact rather than semantic.

        Use for: "what is the shelf limit", "who is the debenture
        trustee", "what is the green shoe option", "what ratings has
        this issue received", "when does the issue open/close".
        """
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if issuer:
            conditions.append("i.issuer_name = :issuer")
            params["issuer"] = issuer
        if document_name:
            conditions.append("d.document_name = :document_name")
            params["document_name"] = document_name

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params["limit"] = server_config.max_rows_returned

        sql = f"""
            SELECT
                iss.issue_id, i.issuer_name AS issuer, d.document_name, iss.issue_name,
                iss.issue_size, iss.issue_open_date, iss.issue_close_date, iss.allotment_date,
                iss.listing, iss.exchange, iss.security_type, iss.shelf_limit,
                iss.green_shoe_option, iss.security_cover, iss.debenture_trustee
            FROM issues iss
            JOIN documents d ON iss.document_id = d.document_id
            JOIN issuers i ON d.issuer_id = i.issuer_id
            {where_clause}
            LIMIT :limit
        """

        try:
            with self._connect() as conn:
                result = conn.execute(text(sql), params)
                columns = result.keys()
                rows = result.fetchmany(server_config.max_rows_returned + 1)
        except SQLAlchemyError as exc:
            logger.exception("get_issue_overview failed")
            return ToolError(error="query_failed", detail=str(exc.__cause__ or exc))

        truncated = len(rows) > server_config.max_rows_returned
        rows = rows[: server_config.max_rows_returned]
        data = [dict(zip(columns, row)) for row in rows]

        issue_ids = sorted({row["issue_id"] for row in data if row.get("issue_id") is not None})
        ratings_by_issue: dict[int, list[dict[str, Any]]] = {}
        lead_managers_by_issue: dict[int, list[str]] = {}
        if issue_ids:
            try:
                with self._connect() as conn:
                    rating_stmt = text(
                        "SELECT issue_id, agency, rating, outlook, instrument, rating_date "
                        "FROM ratings WHERE issue_id IN :ids"
                    ).bindparams(bindparam("ids", expanding=True))
                    for iid, agency, rating, outlook, instrument, rating_date in conn.execute(rating_stmt, {"ids": issue_ids}):
                        ratings_by_issue.setdefault(iid, []).append(
                            {"agency": agency, "rating": rating, "outlook": outlook,
                             "instrument": instrument, "rating_date": rating_date}
                        )

                    lm_stmt = text(
                        "SELECT issue_id, name FROM lead_managers WHERE issue_id IN :ids"
                    ).bindparams(bindparam("ids", expanding=True))
                    for iid, name in conn.execute(lm_stmt, {"ids": issue_ids}):
                        lead_managers_by_issue.setdefault(iid, []).append(name)
            except SQLAlchemyError:
                logger.exception("Ratings/lead-manager lookup failed, continuing without them")

        citations: list[Citation] = []
        for row in data:
            iid = row["issue_id"]
            row["ratings"] = ratings_by_issue.get(iid, [])
            row["lead_managers"] = lead_managers_by_issue.get(iid, [])
            citations.append(
                Citation(
                    source_type="sql",
                    source_id=f"issue:{iid}",
                    document_name=row.get("document_name"),
                    section=row.get("issue_name"),
                )
            )

        return ToolSuccess(
            tool="sql_get_issue_overview",
            data=data,
            citations=citations,
            row_count=len(data),
            truncated=truncated,
        )

    def health_check(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False


sql_db = SQLDatabase()
