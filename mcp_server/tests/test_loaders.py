"""
Fast, DB-free unit tests for the pure-logic parts of the loader
scripts (the pieces that reconcile canonical_schema.py's shapes into
each database's shape). No MySQL/Qdrant/Neo4j connection required -
these exist to catch a mapping bug (wrong field name, a compound
category_id not being split, a None Fact not being handled) without
needing the full docker-compose stack running.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.load_canonical_to_mysql import fact_value, fact_pages, split_category_id  # noqa: E402
from scripts.load_canonical_to_qdrant import _normalize_chunk  # noqa: E402


class TestFactHelpers:

    def test_fact_value_extracts_value(self):
        assert fact_value({"value": "₹500 crore", "raw_value": "₹500 crore", "provenance": []}) == "₹500 crore"

    def test_fact_value_handles_missing_fact(self):
        assert fact_value(None) is None

    def test_fact_pages_collects_all_provenance_pages(self):
        fact = {
            "value": 7.1,
            "provenance": [
                {"document_id": "pfc_ncd_2026", "page": 78, "source_text": "7.1%"},
                {"document_id": "pfc_ncd_2026", "page": 79, "source_text": "7.1%"},
            ],
        }
        assert fact_pages(fact) == [78, 79]

    def test_fact_pages_handles_missing_fact(self):
        assert fact_pages(None) == []


class TestSplitCategoryId:
    """
    extract_category() in the main pipeline (src/ingestion/table_extractor.py)
    can return a compound category_id like "I_II_III_IV" when one source
    table row applies to several investor categories at once (see its
    Capri Global regression test) - split_category_id() is what lets the
    SQL loader turn that into one row per atomic category instead of a
    single unqueryable compound code.
    """

    @pytest.mark.parametrize(
        "category_id,expected",
        [
            ("I", ["I"]),
            ("III", ["III"]),
            ("I_II", ["I", "II"]),
            ("I_II_III_IV", ["I", "II", "III", "IV"]),
            ("default", ["default"]),
            (None, []),
            ("", []),
        ],
    )
    def test_split(self, category_id, expected):
        assert split_category_id(category_id) == expected


class TestNormalizeChunkForQdrant:
    """
    _normalize_chunk() maps one canonical_schema.Chunk record to the
    Qdrant payload shape db/vector_db.py's search() expects -
    specifically, that chunk_type/series_id/category_id survive the
    mapping (the whole point of this reconciliation - the original
    loader never carried these fields at all).
    """

    def test_structured_fact_chunk_carries_series_and_category(self):
        chunk = {
            "chunk_id": "pfc_ncd_2026_chunk_series_III_cat_III",
            "chunk_type": "structured_fact",
            "text": "Coupon Type: Zero Coupon",
            "series_id": "pfc_ncd_2026_series_III",
            "category_id": "III",
            "section_path": None,
            "start_page": 78,
            "end_page": 78,
        }
        payload = _normalize_chunk(chunk, "pfc_ncd_2026.pdf", "Power Finance Corporation Limited", "Bond/NCD Prospectus")

        assert payload["chunk_type"] == "structured_fact"
        assert payload["series_id"] == "pfc_ncd_2026_series_III"
        assert payload["category_id"] == "III"
        assert payload["issuer"] == "Power Finance Corporation Limited"
        assert payload["section"] is None

    def test_prose_chunk_has_no_series_or_category_but_has_section_path(self):
        chunk = {
            "chunk_id": "pfc_ncd_2026_0001",
            "chunk_type": "prose",
            "text": "DEFINITIONS AND ABBREVIATIONS...",
            "series_id": None,
            "category_id": None,
            "section_path": ["SECTION I – GENERAL", "DEFINITIONS AND ABBREVIATIONS"],
            "start_page": 3,
            "end_page": 3,
        }
        payload = _normalize_chunk(chunk, "pfc_ncd_2026.pdf", "Power Finance Corporation Limited", "Bond/NCD Prospectus")

        assert payload["chunk_type"] == "prose"
        assert payload["series_id"] is None
        assert payload["category_id"] is None
        assert payload["section"] == "SECTION I – GENERAL"
        assert payload["subsection"] == "DEFINITIONS AND ABBREVIATIONS"
        assert payload["section_path"] == "SECTION I – GENERAL > DEFINITIONS AND ABBREVIATIONS"
