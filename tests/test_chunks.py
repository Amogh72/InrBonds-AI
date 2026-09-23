"""
Regression tests for structured_fact chunk generation
(canonical_extractor.generate_structured_fact_chunks and friends).

test_no_doubled_unit_suffix and test_derived_value_not_raw_source_text
exist specifically because they failed before a fix landed: the first
version of the chunk-text formatter preferred Fact.raw_value over
Fact.value, which is backwards - raw_value is documented as an audit
trail back to the original source text (sometimes deliberately
different from value, e.g. coupon_type), not a display preference.
"""

from conftest import requires_pfc_pdf, requires_iifl_pdf


def _chunk_by_id(document, chunk_id):
    return next(c for c in document.chunks if c.chunk_id == chunk_id)


@requires_pfc_pdf
class TestPFCChunks:

    def test_chunk_count_matches_series_categories_plus_overview_plus_ratings(
        self, pfc_document
    ):
        issue = pfc_document.issuer.issues[0]
        expected = 1 + len(issue.ratings) + sum(
            len(s.investor_categories) for s in issue.series
        )
        assert len(pfc_document.chunks) == expected == 19

    def test_all_chunks_are_structured_fact_type(self, pfc_document):
        assert all(c.chunk_type == "structured_fact" for c in pfc_document.chunks)

    def test_derived_value_not_raw_source_text(self, pfc_document):
        """
        Series III's frequency raw text is "Zero Coupon NCD"; its
        derived coupon_type is "Zero Coupon". The chunk must show the
        derived value, not silently reprint the raw source text
        raw_value carries for audit purposes.
        """
        chunk = _chunk_by_id(
            pfc_document, "pfc_ncd_2026_chunk_series_III_cat_III"
        )
        assert "Coupon Type: Zero Coupon" in chunk.text
        assert "Coupon Type: Zero Coupon NCD" not in chunk.text
        # The raw source text is still preserved separately, unchanged.
        assert "Frequency of Interest Payment: Zero Coupon NCD" in chunk.text

    def test_no_doubled_unit_suffix(self, pfc_document):
        """
        effective_yield's raw_value already carries a literal "%"
        (e.g. "6.85%"); the formatter must not ALSO append the unit
        string "percent" on top of that.
        """
        chunk = _chunk_by_id(
            pfc_document, "pfc_ncd_2026_chunk_series_III_cat_III"
        )
        assert "6.85 percent" in chunk.text
        assert "%" not in chunk.text
        assert "percent percent" not in chunk.text

    def test_issue_overview_chunk_has_key_facts(self, pfc_document):
        chunk = _chunk_by_id(pfc_document, "pfc_ncd_2026_chunk_issue_overview")
        assert "Power Finance Corporation Limited" in chunk.text
        assert "Beacon Trusteeship Limited" in chunk.text
        assert "A.K. Capital Services Limited" in chunk.text
        assert chunk.text.count("Lead Manager:") == 4

    def test_rating_chunks_not_merged(self, pfc_document):
        rating_chunks = [
            c for c in pfc_document.chunks if "rating" in c.chunk_id
        ]
        assert len(rating_chunks) == 3
        agencies = {
            line.split(": ", 1)[1]
            for c in rating_chunks
            for line in c.text.splitlines()
            if line.startswith("Credit Rating Agency:")
        }
        assert agencies == {"CARE", "ICRA", "CRISIL Ratings Limited"}

    def test_hoisted_terms_appear_once_in_overview_not_per_series(
        self, pfc_document
    ):
        """
        nature_of_indebtedness ("Secured") was hoisted to issue level
        - it should appear in the overview chunk, and NOT be
        duplicated into all 5 series' chunks (there is nowhere left
        for it to come from on the series, since
        hoist_shared_series_terms already cleared it there).
        """
        overview = _chunk_by_id(
            pfc_document, "pfc_ncd_2026_chunk_issue_overview"
        )
        assert "Security Type: Secured" in overview.text

        series_chunks = [
            c for c in pfc_document.chunks if c.series_id is not None
        ]
        assert all("Secured" not in c.text for c in series_chunks)


@requires_iifl_pdf
class TestIIFLChunks:

    def test_chunk_count(self, iifl_document):
        issue = iifl_document.issuer.issues[0]
        expected = 1 + len(issue.ratings) + sum(
            len(s.investor_categories) for s in issue.series
        )
        assert len(iifl_document.chunks) == expected == 11

    def test_series_with_single_default_category_still_chunked(
        self, iifl_document
    ):
        """
        IIFL's series don't have per-category rows in the source
        table (extract_category found no explicit "Category N" text),
        so each series has exactly one "default" category - that
        must still produce one real chunk per series, not be dropped.
        """
        chunk = _chunk_by_id(
            iifl_document, "iifl_wealth_prime_2023_chunk_series_I_cat_default"
        )
        assert "360 ONE Prime Limited" in chunk.text
        assert "Coupon: 8.91 percent" in chunk.text
