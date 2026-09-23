"""
Regression tests for structure_detector.py, run against both real
fixture prospectuses.

test_no_duplicate_title_collapses_onto_same_page exists specifically
because it failed before a fix landed: IIFL's TOC lists "DECLARATION"
four times (one per signing director, each genuinely on its own
physical page 194-197) and the resolver was collapsing all four onto
the single first-matching candidate page instead of spreading across
the four distinct real headings.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestion import structure_detector as sd  # noqa: E402

from conftest import PFC_PDF, IIFL_PDF, requires_pfc_pdf, requires_iifl_pdf  # noqa: E402


@pytest.fixture(scope="session")
def pfc_structure(pfc_layout):
    """Reuses conftest.py's shared pfc_layout - see its docstring."""
    return sd.detect_structure(PFC_PDF, pfc_layout, verbose=False)


@pytest.fixture(scope="session")
def iifl_structure(iifl_layout):
    return sd.detect_structure(IIFL_PDF, iifl_layout, verbose=False)


@requires_pfc_pdf
class TestPFCStructure:

    def test_uses_visible_toc_strategy(self, pfc_structure):
        """
        PFC's embedded PDF outline is present but unreliable (589
        entries, only 7% confirmed against the visible TOC/layout) -
        detect_structure must reject it and fall back to the visible
        TOC, not silently trust 589 mostly-unconfirmed bookmarks.
        """
        assert pfc_structure["strategy"] == "visible_toc_plus_layout"

    def test_section_count_and_all_confirmed(self, pfc_structure):
        sections = pfc_structure["sections"]
        assert len(sections) == 20
        assert all(s["confirmed"] for s in sections)

    def test_no_overlapping_or_backwards_boundaries(self, pfc_structure):
        for section in pfc_structure["sections"]:
            assert section["start_page"] <= section["end_page"]

    def test_key_sections_at_expected_pages(self, pfc_structure):
        by_title = {s["title"]: s for s in pfc_structure["sections"]}
        assert by_title["ISSUE STRUCTURE"]["start_page"] == 74
        assert by_title["TERMS OF THE ISSUE"]["start_page"] == 82


@requires_iifl_pdf
class TestIIFLStructure:

    def test_uses_visible_toc_strategy(self, iifl_structure):
        assert iifl_structure["strategy"] == "visible_toc_plus_layout"

    def test_no_duplicate_title_collapses_onto_same_page(self, iifl_structure):
        declarations = [
            s for s in iifl_structure["sections"]
            if s["title"].strip().upper() == "DECLARATION"
        ]

        assert len(declarations) == 4

        resolved_pages = [s["start_page"] for s in declarations]
        assert resolved_pages == sorted(resolved_pages)
        assert len(set(resolved_pages)) == 4, (
            "identically-titled TOC entries collapsed onto the same "
            f"page instead of resolving to distinct pages: {resolved_pages}"
        )
        assert all(s["confirmed"] for s in declarations)

    def test_no_overlapping_or_backwards_boundaries(self, iifl_structure):
        for section in iifl_structure["sections"]:
            assert section["start_page"] <= section["end_page"]

    def test_issue_structure_section_at_expected_page(self, iifl_structure):
        by_title = {s["title"]: s for s in iifl_structure["sections"]}
        assert by_title["ISSUE STRUCTURE"]["start_page"] == 127
