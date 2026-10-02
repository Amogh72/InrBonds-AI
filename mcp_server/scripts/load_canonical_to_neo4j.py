"""
Loads a canonical BondDocument JSON directly into Neo4j, modeling the
real relationship-shaped facts on BondIssue: ratings (by agency),
debenture trustee, lead managers, and the issue's series.

This replaces load_domain_knowledge_to_neo4j.py, which read a
standalone domain_knowledge.json (a generic entity_id/entity_type/
predicate graph export produced by an earlier version of
domain_extractor.py). That export format doesn't exist in the current
pipeline - domain_extractor.py now feeds targeted fields directly into
canonical_schema.py's typed BondIssue/Rating/Fact models instead of a
separate generic entity graph, so this loader walks BondDocument
itself rather than a side-channel export.

A consequence worth calling out: the old loader needed Neo4j's APOC
plugin (apoc.merge.node/apoc.merge.relationship) because labels and
relationship types came from arbitrary extractor-produced strings at
runtime, which plain Cypher can't parameterize. Our label/relationship-
type set is now small and fixed (Issuer, Issue, Series, RatingAgency,
DebentureTrustee, LeadManager / ISSUED, HAS_SERIES, RATED_BY,
HAS_DEBENTURE_TRUSTEE, HAS_LEAD_MANAGER), so plain MERGE is enough -
no APOC dependency.

Usage:
    python scripts/load_canonical_to_neo4j.py data/processed/pfc_ncd_2026.json
"""

import sys
import os
import json
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neo4j import GraphDatabase, basic_auth

from config import neo4j_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("load_canonical_to_neo4j")


def _fact_value(fact):
    if not fact:
        return None
    return fact.get("value")


def load_canonical_document(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_meta = data["document"]
    issuer_data = data["issuer"]
    document_name = document_meta["document_name"]

    driver = GraphDatabase.driver(neo4j_config.uri, auth=basic_auth(neo4j_config.user, neo4j_config.password))
    try:
        with driver.session(database=neo4j_config.database) as session:
            session.execute_write(_write_issuer, issuer_data, document_name)
    finally:
        driver.close()

    logger.info("Done loading %s into Neo4j.", document_name)


def _write_issuer(tx, issuer_data: dict, document_name: str) -> None:
    issuer_name = issuer_data["name"]

    tx.run("MERGE (i:Issuer {name: $name})", name=issuer_name)
    logger.info("Issuer: %s", issuer_name)

    for issue in issuer_data.get("issues", []):
        _write_issue(tx, issuer_name, issue, document_name)


def _write_issue(tx, issuer_name: str, issue: dict, document_name: str) -> None:
    terms = issue.get("terms") or {}

    props = {
        "issue_id": issue["issue_id"],
        "issue_name": issue.get("issue_name"),
        "issue_type": issue.get("issue_type"),
        "document_name": document_name,
        "issue_size": _fact_value(terms.get("issue_size")),
        "issue_open_date": _fact_value(terms.get("issue_open_date")),
        "issue_close_date": _fact_value(terms.get("issue_close_date")),
        "shelf_limit": _fact_value(terms.get("shelf_limit")),
        "green_shoe_option": _fact_value(terms.get("green_shoe_option")),
        "security_cover": _fact_value(terms.get("security_cover")),
        "security_type": _fact_value(terms.get("security_type")),
    }

    tx.run(
        """
        MATCH (i:Issuer {name: $issuer_name})
        MERGE (iss:Issue {issue_id: $issue_id})
        SET iss += $props
        MERGE (i)-[:ISSUED]->(iss)
        """,
        issuer_name=issuer_name, issue_id=issue["issue_id"], props=props,
    )

    trustee_name = _fact_value(issue.get("debenture_trustee"))
    if trustee_name:
        tx.run(
            """
            MATCH (iss:Issue {issue_id: $issue_id})
            MERGE (t:DebentureTrustee {name: $name})
            MERGE (iss)-[:HAS_DEBENTURE_TRUSTEE]->(t)
            """,
            issue_id=issue["issue_id"], name=trustee_name,
        )

    for fact in issue.get("lead_managers") or []:
        name = _fact_value(fact)
        if not name:
            continue
        tx.run(
            """
            MATCH (iss:Issue {issue_id: $issue_id})
            MERGE (lm:LeadManager {name: $name})
            MERGE (iss)-[:HAS_LEAD_MANAGER]->(lm)
            """,
            issue_id=issue["issue_id"], name=name,
        )

    for rating in issue.get("ratings") or []:
        tx.run(
            """
            MATCH (iss:Issue {issue_id: $issue_id})
            MERGE (a:RatingAgency {name: $agency})
            MERGE (iss)-[r:RATED_BY]->(a)
            SET r.rating = $rating, r.outlook = $outlook,
                r.instrument = $instrument, r.rating_date = $rating_date
            """,
            issue_id=issue["issue_id"],
            agency=rating["agency"],
            rating=rating["rating"],
            outlook=rating.get("outlook"),
            instrument=rating.get("instrument"),
            rating_date=rating.get("rating_date"),
        )

    for series in issue.get("series", []):
        _write_series(tx, issue["issue_id"], series)

    written_series = len(issue.get("series") or [])
    written_ratings = len(issue.get("ratings") or [])
    written_lms = len(issue.get("lead_managers") or [])
    logger.info(
        "  Issue %s: %d series, %d ratings, %d lead managers, trustee=%s",
        issue["issue_id"], written_series, written_ratings, written_lms, trustee_name or "(none)",
    )


def _write_series(tx, issue_id: str, series: dict) -> None:
    terms = series.get("terms") or {}

    props = {
        "series_number": series.get("series_number"),
        "series_name": series.get("series_name"),
        "tenor": _fact_value(terms.get("tenor")),
        "frequency": _fact_value(terms.get("frequency")),
        "coupon_type": _fact_value(terms.get("coupon_type")),
        "nature_of_indebtedness": _fact_value(terms.get("nature_of_indebtedness")),
        "put_call_option": _fact_value(terms.get("put_call_option")),
    }

    tx.run(
        """
        MATCH (iss:Issue {issue_id: $issue_id})
        MERGE (s:Series {series_id: $series_id})
        SET s += $props
        MERGE (iss)-[:HAS_SERIES]->(s)
        """,
        issue_id=issue_id, series_id=series["series_id"], props=props,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a canonical BondDocument JSON into Neo4j.")
    parser.add_argument("json_path", help="Path to a canonical BondDocument JSON (e.g. data/processed/pfc_ncd_2026.json)")
    args = parser.parse_args()

    load_canonical_document(args.json_path)
