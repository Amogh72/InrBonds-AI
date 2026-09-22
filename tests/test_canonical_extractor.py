"""
End-to-end regression tests for the full pipeline (table extraction +
text/domain extraction merged into one canonical BondDocument), run
against both real fixture prospectuses. These are slow (each document
fixture takes ~60-80s, once per test session - see conftest.py) but
they're what actually proves the pipeline works, not just its parts
in isolation.
"""

from conftest import requires_pfc_pdf, requires_iifl_pdf


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
