"""
End-to-end regression tests for the full pipeline (table extraction +
text/domain extraction merged into one canonical BondDocument), run
against real fixture prospectuses. These are slow (each document
fixture takes ~60-80s, once per test session - see conftest.py) but
they're what actually proves the pipeline works, not just its parts
in isolation.
"""

from conftest import requires_pfc_pdf, requires_iifl_pdf, requires_capri_pdf


@requires_pfc_pdf
class TestPFCCanonicalDocument:

    def test_issuer_resolved_correctly(self, pfc_document):
        assert pfc_document.issuer.name == "Power Finance Corporation Limited"
        assert pfc_document.issuer.provenance

    def test_five_series_present(self, pfc_document):
        issue = pfc_document.issuer.issues[0]
        assert [s.series_number for s in issue.series] == [
            "I", "II", "III", "IV", "V",
        ]

    def test_three_distinct_ratings_not_merged(self, pfc_document):
        issue = pfc_document.issuer.issues[0]
        agencies = {r.agency for r in issue.ratings}
        assert len(issue.ratings) == 3
        assert agencies == {"CARE", "ICRA", "CRISIL Ratings Limited"}
        assert all(r.outlook == "Stable" for r in issue.ratings)

    def test_shelf_and_green_shoe_sizing(self, pfc_document):
        terms = pfc_document.issuer.issues[0].terms
        assert terms.issue_size.value == "₹500 crore"
        assert terms.shelf_limit.value == "₹10,000 CRORE"
        assert terms.green_shoe_option.value == "₹4,500 CRORE"
        assert terms.security_cover.value == "100%"

    def test_debenture_trustee_and_lead_managers(self, pfc_document):
        issue = pfc_document.issuer.issues[0]
        assert issue.debenture_trustee.value == "Beacon Trusteeship Limited"
        assert len(issue.lead_managers) == 4

    def test_shared_terms_hoisted_not_duplicated_per_series(self, pfc_document):
        """
        nature_of_indebtedness ("Secured") is identical across all 5
        series in the source table - hoist_shared_series_terms should
        have moved it to issue.terms.security_type exactly once,
        clearing it from every individual series.
        """
        issue = pfc_document.issuer.issues[0]
        assert issue.terms.security_type.value == "Secured"
        assert all(
            series.terms.nature_of_indebtedness is None
            for series in issue.series
        )

    def test_zero_coupon_series_keeps_raw_value_and_gets_classified(
        self, pfc_document
    ):
        issue = pfc_document.issuer.issues[0]
        series_iii = next(
            s for s in issue.series if s.series_number == "III"
        )
        # Raw extracted text must survive unchanged.
        assert series_iii.terms.frequency.value == "Zero Coupon NCD"
        # New normalized classification, derived from the same value.
        assert series_iii.terms.coupon_type.value == "Zero Coupon"

    def test_every_fact_has_provenance(self, pfc_document):
        issue = pfc_document.issuer.issues[0]
        for series in issue.series:
            for fact in (
                series.terms.tenor,
                series.terms.frequency,
                series.terms.face_value,
            ):
                if fact is not None:
                    assert fact.provenance, (
                        f"{series.series_name} fact missing provenance"
                    )


@requires_iifl_pdf
class TestIIFLCanonicalDocument:
    """
    Second, unrelated issuer - these tests exist to catch anything
    that only ever worked for PFC's specific document. See
    domain_extractor.py's history for the 3 real bugs this document
    already found once.
    """

    def test_issuer_resolved_with_leading_digit_in_name(self, iifl_document):
        assert iifl_document.issuer.name == "360 ONE Prime Limited"

    def test_eight_series_present(self, iifl_document):
        issue = iifl_document.issuer.issues[0]
        assert [s.series_number for s in issue.series] == [
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
        ]

    def test_two_distinct_ratings_not_dropped(self, iifl_document):
        issue = iifl_document.issuer.issues[0]
        agencies = {r.agency for r in issue.ratings}
        assert len(issue.ratings) == 2
        assert agencies == {"CRISIL", "ICRA Limited"}

    def test_green_shoe_option_correct_direction_and_value(self, iifl_document):
        """
        This document states the amount BEFORE the "Green Shoe
        Option" label, and has an unrelated ₹82 crore figure (from a
        past, different issuance) elsewhere that a naive scan could
        pick up instead.
        """
        terms = iifl_document.issuer.issues[0].terms
        assert terms.green_shoe_option.value == "₹ 800 CRORE"
        assert terms.green_shoe_option.provenance[0].page == 1

    def test_debenture_trustee_and_lead_managers(self, iifl_document):
        issue = iifl_document.issuer.issues[0]
        assert issue.debenture_trustee.value == "Beacon Trusteeship Limited"
        assert len(issue.lead_managers) == 3


@requires_capri_pdf
class TestCapriCanonicalDocument:
    """
    A third real, unrelated prospectus, added specifically to keep
    testing genericity rather than trusting the first two documents
    generalize. It exercises three conventions the first two never
    did: a differently-worded rating-letter phrasing, a footnote-
    marked "V**" series header, and a combined "Category I, II, III
    & IV" row label - each caught a real bug (see conftest.py's
    capri_document docstring).
    """

    def test_issuer_resolved_correctly(self, capri_document):
        assert capri_document.issuer.name == "Capri Global Capital Limited"
        assert capri_document.issuer.provenance

    def test_six_series_present_including_footnoted_v(self, capri_document):
        issue = capri_document.issuer.issues[0]
        assert [s.series_number for s in issue.series] == [
            "I", "II", "III", "IV", "V", "VI",
        ]

    def test_two_distinct_ratings_different_phrasing_convention(
        self, capri_document
    ):
        """
        This document's rating disclosure has no "credit rating
        letter dated ... by <agency> ... assigning a rating of"
        phrasing at all - it uses "rated <grade> for an amount of
        ... by <agency> vide its rating letter dated <date>"
        instead, and only the FIRST of several items keeps the word
        "rated". Both agency names also join words with a bare "&"
        or "and" rather than every word being capitalized.
        """
        issue = capri_document.issuer.issues[0]
        by_agency = {r.agency: r for r in issue.ratings}
        assert len(issue.ratings) == 2
        assert set(by_agency) == {
            "Acuite Ratings & Research Limited",
            "Infomerics Valuation and Rating Limited",
        }
        assert by_agency["Acuite Ratings & Research Limited"].rating == (
            "ACUITE AA | Stable"
        )
        assert by_agency["Acuite Ratings & Research Limited"].outlook == "Stable"
        assert by_agency["Infomerics Valuation and Rating Limited"].rating == (
            "IVR AA/ Positive"
        )
        assert (
            by_agency["Infomerics Valuation and Rating Limited"].outlook
            == "Positive"
        )

    def test_rating_grade_not_polluted_by_earlier_unrelated_quote(
        self, capri_document
    ):
        """
        Page 1 has an unrelated quoted cross-reference ("Issue
        Structure") sitting right before the real rating grade. A
        first version of the ALT rating pattern let its own trailing
        "for an amount of" requirement pull the match back to start
        at that earlier quote, swallowing everything in between into
        the grade.
        """
        issue = capri_document.issuer.issues[0]
        for rating in issue.ratings:
            assert "Issue Structure" not in rating.rating

    def test_debenture_trustee_and_lead_managers(self, capri_document):
        issue = capri_document.issuer.issues[0]
        assert issue.debenture_trustee.value == "IDBI Trusteeship Services Limited"
        assert len(issue.lead_managers) == 1
        assert issue.lead_managers[0].value == "Nuvama Wealth Management Limited"

    def test_shelf_and_green_shoe_sizing(self, capri_document):
        terms = capri_document.issuer.issues[0].terms
        assert terms.shelf_limit.value == "₹20,000 MILLION"
        assert terms.green_shoe_option.value == "₹4,000 MILLION"
        assert terms.security_cover.value == "110%"

    def test_combined_four_category_row_not_collapsed_to_two(self, capri_document):
        """
        The source table's coupon/yield/maturity rows are labeled
        "... for NCD Holders in Category I, II, III & IV" - one row,
        one value, applying to all four categories together. A first
        version of extract_category() matched "category i, ii" as a
        substring and returned just "I_II", silently dropping
        Category III and IV from that value's coverage entirely.
        """
        issue = capri_document.issuer.issues[0]
        series_v = next(
            s for s in issue.series if s.series_number == "V"
        )
        assert len(series_v.investor_categories) == 1
        category = series_v.investor_categories[0]
        assert category.category_id == "I_II_III_IV"
        assert category.terms["coupon"].value == 9.3
        assert category.terms["effective_yield"].value == 9.29

    def test_every_series_has_a_tenor(self, capri_document):
        issue = capri_document.issuer.issues[0]
        for series in issue.series:
            assert series.terms.tenor is not None
            assert series.terms.tenor.provenance
