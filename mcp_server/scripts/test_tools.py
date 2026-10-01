"""
Exercises every MCP tool directly (bypassing the MCP protocol layer, calling
the underlying Python functions in server.py) so you can sanity-check the
whole stack in one run without opening the MCP inspector UI.

Requires all three databases running and loaded - see mcp_server/README.md
for the load order (setup_qdrant_collection.py, then the three
load_canonical_to_*.py scripts against data/processed/pfc_ncd_2026.json).

Run:
    python scripts/test_tools.py

Each check prints PASS/FAIL against a value read directly from
data/processed/pfc_ncd_2026.json at the time this script was written - a
FAIL can legitimately mean the loaders haven't been run yet, not that
anything is broken.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402  (imports mcp tools + triggers embedding model setup)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main():
    # -------------------------------------------------------------------
    # 1. health_check
    # -------------------------------------------------------------------
    section("health_check")
    health = server.health_check()
    print(json.dumps(health, indent=2))
    check("MySQL reachable", health["sql"] is True)
    check("Qdrant reachable", health["vector"] is True)
    check("Neo4j reachable", health["graph"] is True)

    # -------------------------------------------------------------------
    # 2. sql_search_bond_terms — exact structured facts.
    # Expected values read directly from data/processed/pfc_ncd_2026.json.
    # -------------------------------------------------------------------
    section("sql_search_bond_terms: Series II, Category III coupon")
    result = server.sql_search_bond_terms(series_code="II", category_code="III")
    print(json.dumps(result, indent=2, default=str))
    if result["ok"] and result["data"]:
        coupon = result["data"][0].get("coupon_percent")
        check("Series II / Cat III coupon == 7.1", coupon is not None and abs(float(coupon) - 7.1) < 0.001,
              f"got {coupon}")
    else:
        check("Series II / Cat III coupon == 7.1", False, "no rows returned — has load_canonical_to_mysql.py been run?")

    section("sql_search_bond_terms: Series V, Category IV maturity amount")
    result = server.sql_search_bond_terms(series_code="V", category_code="IV")
    print(json.dumps(result, indent=2, default=str))
    if result["ok"] and result["data"]:
        maturity = result["data"][0].get("maturity_amount_inr")
        check("Series V / Cat IV maturity == 2879.58", maturity is not None and abs(float(maturity) - 2879.58) < 0.01,
              f"got {maturity}")
    else:
        check("Series V / Cat IV maturity == 2879.58", False, "no rows returned")

    section("sql_search_bond_terms: 15-year tenor series")
    result = server.sql_search_bond_terms(tenor="15 years")
    print(json.dumps(result, indent=2, default=str))
    series_codes = {row["series_code"] for row in result["data"]} if result["ok"] else set()
    check("Series IV and V both have 15-year tenor", {"IV", "V"}.issubset(series_codes), f"got {series_codes}")

    section("sql_search_bond_terms: Series III face value (zero coupon)")
    result = server.sql_search_bond_terms(series_code="III")
    print(json.dumps(result, indent=2, default=str))
    if result["ok"] and result["data"]:
        face_value = result["data"][0].get("face_value_inr")
        check("Series III face value == 100000.00", face_value is not None and abs(float(face_value) - 100000.00) < 0.01,
              f"got {face_value}")

    # Category I_II is a compound row in the source table - the loader
    # must have split it into separate "I" and "II" rows (see
    # load_canonical_to_mysql.py's split_category_id()).
    section("sql_search_bond_terms: Series I, Category I and II both present atomically")
    result_i = server.sql_search_bond_terms(series_code="I", category_code="I")
    result_ii = server.sql_search_bond_terms(series_code="I", category_code="II")
    check(
        "Category I and II both resolve individually (not 'I_II')",
        result_i["ok"] and result_i["row_count"] > 0 and result_ii["ok"] and result_ii["row_count"] > 0,
    )

    # -------------------------------------------------------------------
    # 3. sql_get_issue_overview — hoisted issue-level terms, ratings,
    # trustee, lead managers (none of this existed in the original
    # schema/loader at all).
    # -------------------------------------------------------------------
    section("sql_get_issue_overview: PFC")
    result = server.sql_get_issue_overview(issuer="Power Finance Corporation Limited")
    print(json.dumps(result, indent=2, default=str))
    if result["ok"] and result["data"]:
        row = result["data"][0]
        check("issue_size == '₹500 crore'", row.get("issue_size") == "₹500 crore", f"got {row.get('issue_size')}")
        check("debenture_trustee == 'Beacon Trusteeship Limited'",
              row.get("debenture_trustee") == "Beacon Trusteeship Limited", f"got {row.get('debenture_trustee')}")
        check("3 ratings present", len(row.get("ratings", [])) == 3, f"got {len(row.get('ratings', []))}")
        check("4 lead managers present", len(row.get("lead_managers", [])) == 4, f"got {len(row.get('lead_managers', []))}")
    else:
        check("sql_get_issue_overview returned data", False, "no rows — has load_canonical_to_mysql.py been run?")

    # -------------------------------------------------------------------
    # 4. sql_query — raw SQL fallback
    # -------------------------------------------------------------------
    section("sql_query: raw SELECT against series table")
    result = server.sql_query("SELECT series_code, tenor FROM series ORDER BY series_code", citation_table="series")
    print(json.dumps(result, indent=2, default=str))
    check("sql_query returns 5 series rows", result["ok"] and result["row_count"] == 5, f"got {result.get('row_count')}")

    # -------------------------------------------------------------------
    # 5. vector_search — semantic search over chunk content, now
    # distinguishing structured_fact vs prose chunk_type.
    # -------------------------------------------------------------------
    section("vector_search: coupon rate question (prose only)")
    result = server.vector_search(
        query_text="What is the coupon rate for Series II NCDs?",
        top_k=3,
        issuer_filter="Power Finance Corporation Limited",
        chunk_type_filter="prose",
    )
    if result["ok"]:
        print(f"Returned {result['row_count']} chunks")
        for chunk in result["data"]:
            print(f"  - {chunk.get('document_name')} p.{chunk.get('start_page')} "
                  f"[{chunk.get('chunk_type')}] score={chunk.get('score'):.3f}")
        check("All results are prose chunks", all(c.get("chunk_type") == "prose" for c in result["data"]))
    else:
        check("vector_search succeeded", False, str(result))

    section("vector_search: scoped to Series III, Category III (structured_fact)")
    result = server.vector_search(
        query_text="effective yield",
        top_k=3,
        series_id_filter="pfc_ncd_2026_series_III",
        category_id_filter="III",
    )
    if result["ok"] and result["data"]:
        check("6.85 percent appears in the scoped chunk", "6.85 percent" in result["data"][0]["text"],
              result["data"][0]["text"][:200])
    else:
        check("scoped vector_search returned a chunk", False, str(result))

    # -------------------------------------------------------------------
    # 6. graph tools — relationship-shaped facts.
    # -------------------------------------------------------------------
    section("graph_query: issuer_name recipe for PFC")
    result = server.graph_query(issuer_name="Power Finance Corporation Limited")
    print(json.dumps(result, indent=2, default=str))
    if result["ok"] and result["data"]:
        row = result["data"][0]
        check("3 ratings in graph neighborhood", len(row.get("ratings", [])) == 3, f"got {len(row.get('ratings', []))}")
        check("5 series in graph neighborhood", len(row.get("series_numbers", [])) == 5,
              f"got {len(row.get('series_numbers', []))}")
    else:
        check("graph_query returned data", False, "no rows — has load_canonical_to_neo4j.py been run?")

    section("graph_find_issues_by_participant: Beacon Trusteeship (trustee)")
    result = server.graph_find_issues_by_participant("Beacon Trusteeship", role="trustee")
    print(json.dumps(result, indent=2, default=str))
    check("At least one issue found for Beacon Trusteeship as trustee", result["ok"] and result["row_count"] > 0)

    print("\nDone.")


if __name__ == "__main__":
    main()
