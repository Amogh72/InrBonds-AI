import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestion import canonical_extractor as ce  # noqa: E402
from ingestion import table_extractor  # noqa: E402

PFC_PDF = ROOT / "data" / "raw" / "pfc_ncd_2026.pdf"
IIFL_PDF = ROOT / "data" / "raw" / "iifl_wealth_prime_2023.pdf"

# Real PDFs are large (multi-MB) and only meaningfully present once a
# contributor has pulled/placed them locally (see .gitignore's explicit
# per-file exceptions under data/raw/). Skip rather than fail when a
# fixture PDF isn't available, so the rest of the suite still runs.
requires_pfc_pdf = pytest.mark.skipif(
    not PFC_PDF.exists(), reason=f"fixture PDF not found: {PFC_PDF}"
)
requires_iifl_pdf = pytest.mark.skipif(
    not IIFL_PDF.exists(), reason=f"fixture PDF not found: {IIFL_PDF}"
)


@pytest.fixture(scope="session")
def pfc_series_data():
    """Table-extractor output only (fast: skips the text/domain pass)."""
    series_data, tables_found = table_extractor.extract_series_terms(
        PFC_PDF, verbose=False
    )
    return series_data, tables_found


@pytest.fixture(scope="session")
def iifl_series_data():
    series_data, tables_found = table_extractor.extract_series_terms(
        IIFL_PDF, verbose=False
    )
    return series_data, tables_found


@pytest.fixture(scope="session")
def pfc_document():
    """
    Full canonical_extractor.py pipeline (table + text/domain
    extraction) against the real PFC prospectus. Session-scoped and
    slow (~70-80s) since it scans all 274 pages for domain facts -
    every test that needs it shares this single run.
    """
    return ce.extract_canonical_document(
        pdf_path=PFC_PDF,
        document_id="pfc_ncd_2026",
    )


@pytest.fixture(scope="session")
def iifl_document():
    """Same as pfc_document, for the second (235-page) test document."""
    return ce.extract_canonical_document(
        pdf_path=IIFL_PDF,
        document_id="iifl_wealth_prime_2023",
    )
