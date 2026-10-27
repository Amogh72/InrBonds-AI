"""
Graph database module (Neo4j) for the relationship-shaped facts on a
BondIssue: which agency rated it, who its debenture trustee and lead
managers are, and which series it has. Schema: graph/schema.cypher,
loaded from a canonical BondDocument JSON by
scripts/load_canonical_to_neo4j.py.

Design notes:
- Exposes a safe, parameterized `run_cypher` (read-only enforced by
  convention + DB user privileges) plus typed recipes for the
  questions this graph is actually good at: "what's connected to this
  issuer" and "which issues has this trustee/lead manager/rating
  agency been involved with" - the latter is exactly the kind of
  cross-document, multi-hop question a relational JOIN makes awkward
  but a graph traversal answers directly, and only becomes interesting
  once more than one document has been loaded.
- No APOC dependency (see load_canonical_to_neo4j.py's docstring) -
  labels and relationship types are a small fixed set now, not
  arbitrary runtime strings.
"""

import logging
from typing import Any

from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import Neo4jError

from config import neo4j_config, server_config
from models import Citation, ToolError, ToolSuccess

logger = logging.getLogger("mcp.graph")

_FORBIDDEN_KEYWORDS = ("create", "merge", "delete", "set", "remove", "drop", "call apoc.", "call db.")


class GraphDatabaseClient:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=basic_auth(neo4j_config.user, neo4j_config.password),
        )

    def close(self) -> None:
        self._driver.close()

    @staticmethod
    def _is_read_only(cypher: str) -> bool:
        lowered = cypher.lower()
        return not any(kw in lowered for kw in _FORBIDDEN_KEYWORDS)

    def run_cypher(self, cypher: str, params: dict[str, Any] | None = None) -> ToolSuccess | ToolError:
        if not self._is_read_only(cypher):
            return ToolError(error="rejected_query", detail="Only read-only Cypher (MATCH/RETURN) is permitted.")

        try:
            with self._driver.session(database=neo4j_config.database) as session:
                result = session.run(cypher, params or {}, timeout=server_config.query_timeout_seconds)
                records = [record.data() for record in result]
        except Neo4jError as exc:
            logger.exception("Cypher query failed")
            return ToolError(error="query_failed", detail=str(exc))

        truncated = len(records) > server_config.max_rows_returned
        records = records[: server_config.max_rows_returned]

        citations = [
            Citation(source_type="graph", source_id=f"cypher_result:{i}")
            for i in range(len(records))
        ]

        return ToolSuccess(
            tool="graph_query",
            data=records,
            citations=citations,
            row_count=len(records),
            truncated=truncated,
        )

    def issuer_relationship_neighborhood(self, issuer_name: str, depth: int = 2) -> ToolSuccess | ToolError:
        """
        Pre-built recipe: every issue from this issuer, plus each
        issue's series, ratings, debenture trustee, and lead managers.
        `depth` is accepted for forward compatibility (e.g. a future
        Guarantor/parent-subsidiary hop) but the current schema is
        exactly two hops deep (Issuer -> Issue -> {Series, RatingAgency,
        DebentureTrustee, LeadManager}), so it's not used to bound the
        query yet.
        """
        cypher = """
        MATCH (i:Issuer {name: $issuer_name})-[:ISSUED]->(iss:Issue)
        OPTIONAL MATCH (iss)-[:HAS_SERIES]->(s:Series)
        OPTIONAL MATCH (iss)-[r:RATED_BY]->(a:RatingAgency)
        OPTIONAL MATCH (iss)-[:HAS_DEBENTURE_TRUSTEE]->(t:DebentureTrustee)
        OPTIONAL MATCH (iss)-[:HAS_LEAD_MANAGER]->(lm:LeadManager)
        RETURN i.name AS issuer,
               iss.issue_id AS issue_id, iss.issue_name AS issue_name,
               iss.document_name AS document_name,
               collect(DISTINCT s.series_number) AS series_numbers,
               collect(DISTINCT {agency: a.name, rating: r.rating, outlook: r.outlook}) AS ratings,
               collect(DISTINCT t.name) AS debenture_trustees,
               collect(DISTINCT lm.name) AS lead_managers
        LIMIT $limit
        """
        params = {"issuer_name": issuer_name, "limit": server_config.max_rows_returned}
        return self.run_cypher(cypher, params)

    def find_issues_by_participant(
        self,
        participant_name: str,
        role: str = "any",
        limit: int | None = None,
    ) -> ToolSuccess | ToolError:
        """
        Cross-document lookup: every issue a named trustee, lead
        manager, or rating agency has been involved with. Answers
        "which issuers has Beacon Trusteeship Limited served as
        trustee for", "which issues has CRISIL rated", "which issuers
        has A.K. Capital Services lead-managed" - a case-insensitive
        substring match on the participant's name across whichever
        role(s) you ask for.

        Args:
            participant_name: substring to match (case-insensitive)
                against the trustee/lead-manager/rating-agency name.
            role: "trustee", "lead_manager", "rating_agency", or "any"
                (default) to search across all three.
        """
        role_to_pattern = {
            "trustee": "(iss)-[:HAS_DEBENTURE_TRUSTEE]->(p:DebentureTrustee)",
            "lead_manager": "(iss)-[:HAS_LEAD_MANAGER]->(p:LeadManager)",
            "rating_agency": "(iss)-[:RATED_BY]->(p:RatingAgency)",
        }
        patterns = [role_to_pattern[role]] if role in role_to_pattern else list(role_to_pattern.values())

        params = {
            "participant_name": participant_name,
            "limit": limit or server_config.max_rows_returned,
        }

        union_parts = []
        for i, pattern in enumerate(patterns):
            union_parts.append(
                f"""
                MATCH (i:Issuer)-[:ISSUED]->(iss:Issue), {pattern}
                WHERE toLower(p.name) CONTAINS toLower($participant_name)
                RETURN i.name AS issuer, iss.issue_id AS issue_id, iss.issue_name AS issue_name,
                       iss.document_name AS document_name, p.name AS participant_name,
                       labels(p)[0] AS participant_role
                """
            )
        cypher = " UNION ".join(union_parts) + " LIMIT $limit"

        return self.run_cypher(cypher, params)

    def health_check(self) -> bool:
        try:
            with self._driver.session(database=neo4j_config.database) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False


graph_db = GraphDatabaseClient()
