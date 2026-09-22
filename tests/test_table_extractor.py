"""
Regression tests for table_extractor.py's geometry-aware series
extraction, locking in values already validated by hand against two
real prospectuses with structurally different tables (PFC: 5 series,
fixed-width columns; IIFL: 8 series, different column geometry).

These are deliberately about the table-derived NUMBERS only (no PDF
text/domain extraction), so they stay fast.
"""

import pytest

from conftest import requires_pfc_pdf, requires_iifl_pdf


@requires_pfc_pdf
class TestPFCSeriesExtraction:

    def test_finds_five_series(self, pfc_series_data):
        series_data, _tables_found = pfc_series_data
        assert set(series_data.keys()) == {"I", "II", "III", "IV", "V"}

    def test_series_i_terms(self, pfc_series_data):
        series_data, _ = pfc_series_data
        s1 = series_data["I"]
        assert s1["tenor"] == "5 years"
        assert s1["frequency"] == "Annual"
        assert s1["face_value_inr"] == 1000.0
        assert s1["coupon_by_category"] == {
            "I_II": 6.85, "III": 6.9, "IV": 7.0,
        }
        assert s1["effective_yield_by_category"] == {
            "I_II": 6.85, "III": 6.9, "IV": 7.0,
        }

    def test_series_iii_is_zero_coupon_with_discounted_issue_price(
        self, pfc_series_data
    ):
        series_data, _ = pfc_series_data
        s3 = series_data["III"]
        assert s3["frequency"] == "Zero Coupon NCD"
        assert s3["face_value_inr"] == 100000.0
        # Zero-coupon NCDs have no periodic coupon at all.
        assert s3["coupon_by_category"] == {}
        assert s3["issue_price_by_category"] == {
            "I_II": 51502.0, "III": 51263.0, "IV": 50780.0,
        }
        assert s3["effective_yield_by_category"] == {
            "I_II": 6.8, "III": 6.85, "IV": 6.95,
        }

    def test_series_v_is_cumulative_with_compounded_maturity(
        self, pfc_series_data
    ):
        series_data, _ = pfc_series_data
        s5 = series_data["V"]
        assert s5["frequency"] == "Cumulative"
        assert s5["coupon_by_category"] == {}
        assert s5["maturity_amount_by_category"] == {
            "I_II": 2780.50, "III": 2839.56, "IV": 2879.58,
        }

    def test_merged_cells_shared_across_all_series(self, pfc_series_data):
        """
        Fields drawn from a single table cell merged across all 5
        series columns should carry the identical value on every
        series (this is what hoist_shared_series_terms later
        collapses to the issue level in canonical_extractor.py).
        """
        series_data, _ = pfc_series_data
        values = {s["nature_of_indebtedness"] for s in series_data.values()}
        assert values == {"Secured"}


@requires_iifl_pdf
class TestIIFLSeriesExtraction:

    def test_finds_eight_series(self, iifl_series_data):
        series_data, _tables_found = iifl_series_data
        assert set(series_data.keys()) == {
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
        }

    @pytest.mark.parametrize(
        "series_id,tenor,frequency,coupon",
        [
            ("I", "18 Months", "Monthly", 8.91),
            ("II", "18 Months", "Annual", 9.22),
            ("VII", "60 Months", "Monthly", 9.26),
            ("VIII", "60 Months", "Annual", 9.66),
        ],
    )
    def test_selected_series_terms(
        self, iifl_series_data, series_id, tenor, frequency, coupon
    ):
        series_data, _ = iifl_series_data
        series = series_data[series_id]
        assert series["tenor"] == tenor
        assert series["frequency"] == frequency
        assert series["coupon_by_category"]["default"] == coupon
