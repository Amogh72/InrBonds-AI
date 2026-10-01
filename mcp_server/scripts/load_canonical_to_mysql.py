"""
Load a canonical BondDocument JSON (the real pipeline output, e.g.
data/processed/pfc_ncd_2026.json - see src/schema/canonical_schema.py)
into the structured MySQL schema (sql/schema.sql).

This replaces the original load_series_to_mysql.py, which read a
standalone series_terms.json (table_extractor.py's raw pre-hoist
output) and therefore never saw ratings, debenture_trustee,
lead_managers, or any IssueTerms field - all of that only exists on
BondIssue AFTER canonical_extractor.py's hoist_shared_series_terms()
and domain_extractor.py passes run, so a loader reading the older
file had nothing to load it from.

Input: a BondDocument JSON, as written by
    python -m ingestion.canonical_extractor ... (or test fixtures) and
    saved under data/processed/<document_id>.json

Target tables:
    issuers, documents, issues, ratings, lead_managers,
    series, investor_categories, series_category_terms, series_provenance

This script does NOT touch Qdrant or Neo4j - see
load_canonical_to_qdrant.py / load_canonical_to_neo4j.py for those.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mysql.connector
from mysql.connector import Error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse config.py's sql_config (which calls load_dotenv()) instead of
# reading os.getenv() directly, like the original version of this
# script did - that meant editing .env had zero effect on this loader
# specifically, even though the other two loaders (which import
# qdrant_config/neo4j_config from the same config.py) already picked
# up .env correctly. A real credential mismatch this caused: the
# loader silently fell back to its own root/"" defaults instead of
# .env's mcp_readonly/changeme, so it either connected as the wrong
# user or failed outright depending on what root's password happened
# to be on a given machine.
from config import sql_config  # noqa: E402

DB_HOST = sql_config.host
DB_PORT = sql_config.port
DB_USER = sql_config.user
DB_PASSWORD = sql_config.password
DB_NAME = sql_config.database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("inrbonds.sql_loader")


# =========================================================
# FACT HELPERS
#
# Every leaf value in a BondDocument is either a plain dict shaped
# like canonical_schema.Fact ({"value": ..., "unit": ..., "raw_value":
# ..., "provenance": [...]}) or None. These helpers pull just the
# parts the SQL schema stores (SQL is not provenance-source-text-aware
# the way Qdrant/Neo4j are - page numbers go into series_provenance,
# full source_text does not get a column here).
# =========================================================

def fact_value(fact):
    """Return a Fact's scalar value (or None), tolerating a missing Fact."""
    if not fact:
        return None
    return fact.get("value")


def fact_pages(fact):
    """Return the list of page numbers a Fact's provenance points to."""
    if not fact:
        return []
    return [p["page"] for p in fact.get("provenance", []) if p.get("page") is not None]


def split_category_id(category_id):
    """
    A BondSeries.InvestorCategory.category_id is sometimes a compound
    code like "I_II_III_IV" (one source table row that applies to
    several categories at once - see table_extractor.py's
    extract_category() and its Capri Global regression test). SQL
    stores one row per atomic category, so split on "_" generically -
    no fixed combination list, unlike the original loader's
    normalize_category_key().
    """
    if not category_id:
        return []
    return [part for part in str(category_id).split("_") if part]


# =========================================================
# HELPERS
# =========================================================

def get_db_connection():
    logger.info("Connecting to MySQL at %s:%s as user '%s'", DB_HOST, DB_PORT, DB_USER)
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


def load_canonical_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "document" not in data or "issuer" not in data:
        raise ValueError("Not a canonical BondDocument JSON (missing 'document'/'issuer').")

    return data


# =========================================================
# ISSUER / DOCUMENT
# =========================================================

def get_or_create_issuer(cursor, name):
    cursor.execute("SELECT issuer_id FROM issuers WHERE issuer_name = %s", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute("INSERT INTO issuers (issuer_name) VALUES (%s)", (name,))
    return cursor.lastrowid


def get_or_create_document(cursor, issuer_id, document_meta):
    document_name = document_meta["document_name"]

    cursor.execute(
        "SELECT document_id FROM documents WHERE issuer_id = %s AND document_name = %s",
        (issuer_id, document_name),
    )
    row = cursor.fetchone()

    values = (
        document_meta.get("document_type") or "Unknown",
        document_meta.get("document_id"),
        document_meta.get("document_date"),
        document_meta.get("page_count"),
    )

    if row:
        document_id = row[0]
        cursor.execute(
            """
            UPDATE documents
            SET document_type = %s, document_identifier = %s,
                document_date = %s, page_count = %s
            WHERE document_id = %s
            """,
            values + (document_id,),
        )
        return document_id

    cursor.execute(
        """
        INSERT INTO documents
            (issuer_id, document_name, document_type, document_identifier, document_date, page_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (issuer_id, document_name) + values,
    )
    return cursor.lastrowid


# =========================================================
# ISSUE (terms, debenture_trustee are columns here - see schema.sql)
# =========================================================

def upsert_issue(cursor, document_id, issue):
    terms = issue.get("terms") or {}
    issue_identifier = issue.get("issue_id")

    row_values = (
        issue.get("issue_name"),
        issue.get("issue_type"),
        fact_value(terms.get("issue_size")),
        fact_value(terms.get("issue_open_date")),
        fact_value(terms.get("issue_close_date")),
        fact_value(terms.get("allotment_date")),
        fact_value(terms.get("listing")),
        fact_value(terms.get("exchange")),
        fact_value(terms.get("security_type")),
        fact_value(terms.get("shelf_limit")),
        fact_value(terms.get("green_shoe_option")),
        fact_value(terms.get("security_cover")),
        fact_value(issue.get("debenture_trustee")),
    )

    cursor.execute(
        "SELECT issue_id FROM issues WHERE document_id = %s AND issue_identifier = %s",
        (document_id, issue_identifier),
    )
    row = cursor.fetchone()

    if row:
        issue_db_id = row[0]
        cursor.execute(
            """
            UPDATE issues
            SET issue_name = %s, issue_type = %s,
                issue_size = %s, issue_open_date = %s, issue_close_date = %s,
                allotment_date = %s, listing = %s, exchange = %s,
                security_type = %s, shelf_limit = %s, green_shoe_option = %s,
                security_cover = %s, debenture_trustee = %s
            WHERE issue_id = %s
            """,
            row_values + (issue_db_id,),
        )
        return issue_db_id

    cursor.execute(
        """
        INSERT INTO issues
            (document_id, issue_identifier, issue_name, issue_type,
             issue_size, issue_open_date, issue_close_date, allotment_date,
             listing, exchange, security_type, shelf_limit, green_shoe_option,
             security_cover, debenture_trustee)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (document_id, issue_identifier) + row_values,
    )
    return cursor.lastrowid


# =========================================================
# RATINGS
# =========================================================

def upsert_ratings(cursor, issue_id, ratings):
    for rating in ratings or []:
        cursor.execute(
            "SELECT rating_id FROM ratings WHERE issue_id = %s AND agency = %s AND rating = %s",
            (issue_id, rating["agency"], rating["rating"]),
        )
        if cursor.fetchone():
            continue

        cursor.execute(
            """
            INSERT INTO ratings (issue_id, agency, rating, outlook, instrument, rating_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                issue_id,
                rating["agency"],
                rating["rating"],
                rating.get("outlook"),
                rating.get("instrument"),
                rating.get("rating_date"),
            ),
        )


# =========================================================
# LEAD MANAGERS
# =========================================================

def upsert_lead_managers(cursor, issue_id, lead_managers):
    for fact in lead_managers or []:
        name = fact_value(fact)
        if not name:
            continue

        cursor.execute(
            "SELECT lead_manager_id FROM lead_managers WHERE issue_id = %s AND name = %s",
            (issue_id, name),
        )
        if cursor.fetchone():
            continue

        cursor.execute(
            "INSERT INTO lead_managers (issue_id, name) VALUES (%s, %s)",
            (issue_id, name),
        )


# =========================================================
# CATEGORY LOOKUP
# =========================================================

def get_or_create_category(cursor, category_code):
    cursor.execute(
        "SELECT category_id FROM investor_categories WHERE category_code = %s",
        (category_code,),
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute(
        "INSERT INTO investor_categories (category_code, category_name) VALUES (%s, %s)",
        (category_code, f"Category {category_code}"),
    )
    return cursor.lastrowid


# =========================================================
# SERIES
# =========================================================

def upsert_series(cursor, issue_id, series):
    terms = series.get("terms") or {}
    series_code = series["series_number"] or series["series_id"]

    row_values = (
        fact_value(terms.get("tenor")),
        fact_value(terms.get("frequency")),
        fact_value(terms.get("coupon_type")),
        fact_value(terms.get("face_value")),
        fact_value(terms.get("mode_of_interest_payment")),
        fact_value(terms.get("maturity_redemption")),
        fact_value(terms.get("nature_of_indebtedness")),
        fact_value(terms.get("put_call_option")),
        fact_value(terms.get("minimum_application")),
    )

    cursor.execute(
        "SELECT series_id FROM series WHERE issue_id = %s AND series_code = %s",
        (issue_id, series_code),
    )
    row = cursor.fetchone()

    if row:
        series_db_id = row[0]
        cursor.execute(
            """
            UPDATE series
            SET series_identifier = %s, tenor = %s, frequency = %s, coupon_type = %s,
                face_value_inr = %s, mode_of_interest_payment = %s,
                maturity_redemption = %s, nature_of_indebtedness = %s,
                put_call_option = %s, minimum_application = %s
            WHERE series_id = %s
            """,
            (series["series_id"],) + row_values + (series_db_id,),
        )
        return series_db_id

    cursor.execute(
        """
        INSERT INTO series
            (issue_id, series_identifier, series_code, tenor, frequency, coupon_type,
             face_value_inr, mode_of_interest_payment, maturity_redemption,
             nature_of_indebtedness, put_call_option, minimum_application)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (issue_id, series["series_id"], series_code) + row_values,
    )
    return cursor.lastrowid


def insert_category_terms(cursor, series_db_id, investor_categories):
    for category in investor_categories or []:
        terms = category.get("terms") or {}
        values = (
            fact_value(terms.get("issue_price")),
            fact_value(terms.get("coupon")),
            fact_value(terms.get("effective_yield")),
            fact_value(terms.get("maturity_amount")),
        )

        for atomic_code in split_category_id(category.get("category_id")):
            category_db_id = get_or_create_category(cursor, atomic_code)

            cursor.execute(
                "SELECT series_category_term_id FROM series_category_terms "
                "WHERE series_id = %s AND category_id = %s",
                (series_db_id, category_db_id),
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE series_category_terms
                    SET issue_price_inr = %s, coupon_percent = %s,
                        effective_yield_percent = %s, maturity_amount_inr = %s
                    WHERE series_category_term_id = %s
                    """,
                    values + (existing[0],),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO series_category_terms
                        (series_id, category_id, issue_price_inr, coupon_percent,
                         effective_yield_percent, maturity_amount_inr)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (series_db_id, category_db_id) + values,
                )


def insert_series_provenance(cursor, series_db_id, series):
    """
    One row per (page, source_field). source_field is the SeriesTerms
    field name the page supports (e.g. "tenor") so a series with terms
    drawn from several pages keeps that distinction, plus one row with
    source_field="series" for the series-level provenance list itself.
    """
    pages_by_field = {"series": [p["page"] for p in series.get("provenance", []) if p.get("page") is not None]}

    for field_name, fact in (series.get("terms") or {}).items():
        if isinstance(fact, dict):
            pages = fact_pages(fact)
            if pages:
                pages_by_field[field_name] = pages

    for source_field, pages in pages_by_field.items():
        for page in sorted(set(pages)):
            cursor.execute(
                "SELECT provenance_id FROM series_provenance "
                "WHERE series_id = %s AND page_number = %s AND source_field = %s",
                (series_db_id, page, source_field),
            )
            if cursor.fetchone():
                continue

            cursor.execute(
                "INSERT INTO series_provenance (series_id, page_number, source_field) VALUES (%s, %s, %s)",
                (series_db_id, page, source_field),
            )


# =========================================================
# MAIN LOAD FUNCTION
# =========================================================

def load_canonical_document(json_path):
    data = load_canonical_json(json_path)

    document_meta = data["document"]
    issuer_data = data["issuer"]

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        logger.info("Connected to MySQL database '%s'", DB_NAME)

        issuer_id = get_or_create_issuer(cursor, issuer_data["name"])
        logger.info("Issuer '%s' -> issuer_id=%s", issuer_data["name"], issuer_id)

        document_id = get_or_create_document(cursor, issuer_id, document_meta)
        logger.info("Document '%s' -> document_id=%s", document_meta["document_name"], document_id)

        loaded_issues = loaded_series = loaded_ratings = loaded_lead_managers = 0

        for issue in issuer_data.get("issues", []):
            issue_db_id = upsert_issue(cursor, document_id, issue)
            loaded_issues += 1

            upsert_ratings(cursor, issue_db_id, issue.get("ratings"))
            loaded_ratings += len(issue.get("ratings") or [])

            upsert_lead_managers(cursor, issue_db_id, issue.get("lead_managers"))
            loaded_lead_managers += len(issue.get("lead_managers") or [])

            for series in issue.get("series", []):
                series_db_id = upsert_series(cursor, issue_db_id, series)
                insert_category_terms(cursor, series_db_id, series.get("investor_categories"))
                insert_series_provenance(cursor, series_db_id, series)
                loaded_series += 1

        connection.commit()

        logger.info("=" * 60)
        logger.info("SQL LOAD COMPLETE")
        logger.info("Document      : %s", document_meta["document_name"])
        logger.info("Issuer        : %s", issuer_data["name"])
        logger.info("Issues loaded : %d", loaded_issues)
        logger.info("Series loaded : %d", loaded_series)
        logger.info("Ratings loaded: %d", loaded_ratings)
        logger.info("Lead managers : %d", loaded_lead_managers)
        logger.info("=" * 60)

    except Error:
        if connection:
            connection.rollback()
        logger.exception("MySQL error. Transaction rolled back.")
        raise
    except Exception:
        if connection:
            connection.rollback()
        logger.exception("SQL loading failed. Transaction rolled back.")
        raise
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
        logger.info("MySQL connection closed.")


def main():
    parser = argparse.ArgumentParser(description="Load a canonical BondDocument JSON into MySQL.")
    parser.add_argument("json_path", help="Path to a canonical BondDocument JSON (e.g. data/processed/pfc_ncd_2026.json)")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        logger.error("JSON file does not exist: %s", json_path)
        sys.exit(1)

    try:
        load_canonical_document(json_path)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
