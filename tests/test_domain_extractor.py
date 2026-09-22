"""
Fast, PDF-free unit tests for domain_extractor.py's structural text
extractors. Text snippets here are copied verbatim from the two real
prospectuses this project has been validated against (PFC NCD 2026,
IIFL/360 ONE Wealth Prime 2023) - not invented examples - so a
regression here means real extracted output would regress too.

Several of these tests exist specifically because they failed before
a fix landed:
  - test_rating_letter_items_* : the "for the Issue" vs "in respect
    of" terminal-phrase bug that silently dropped a whole rating.
  - test_company_name_with_leading_digit : the "360 ONE Prime
    Limited" -> "ONE Prime Limited" truncation bug.
  - test_green_shoe_option_* : the amount-before-term vs
    amount-after-term direction bug.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestion import domain_extractor as de  # noqa: E402


class TestRatingLetterItems:

    def test_pfc_three_agencies_all_captured(self):
        text = (
            '11. Credit rating letter dated March 28, 2025, revalidated as on '
            'December 31, 2025, read with press release October 8, 2025, by '
            'CARE assigning a rating of “CARE AAA; Stable/ CARE A1+’ '
            '(pronounced as CARE Triple A; Outlook: Stable/ A One Plus)” '
            'in respect of the NCDs. '
            '12. Credit rating letter dated March 26, 2025, revalidated as on '
            'January 2, 2026, read with press release March 26, 2025, by ICRA '
            'assigning a rating of “[ICRA AAA] (Stable) (pronounced ICRA '
            'triple A: Outlook: Stable)” in respect of the NCDs. '
            '13. Credit rating letter dated March 28, 2025, revalidated as on '
            'January 6, 2026, read with press release and credit bulletin dated '
            'March 27, 2025 and July 29, 2025, by Crisil assigning a rating of '
            '“Crisil AAA/Stable” (pronounced as “Crisil triple A '
            'rating” with stable outlook) in respect of the NCDs.'
        )

        items = de.extract_rating_letter_items(text)

        assert [item["agency"] for item in items] == ["CARE", "ICRA", "Crisil"]
        assert items[0]["rating"] == "CARE AAA; Stable/ CARE A1+"
        assert items[1]["rating"] == "[ICRA AAA] (Stable)"
        assert items[2]["rating"] == "Crisil AAA/Stable"
        assert all(item["outlook"] == "Stable" for item in items)

    def test_iifl_two_agencies_not_swallowed_by_wrong_terminal_phrase(self):
        """
        This document closes each item with "for the Issue" instead
        of "in respect of" - a version of the extractor bounded on
        that specific phrase silently consumed both items (and 4
        more unrelated ones after them) as a single match, dropping
        the ICRA rating entirely.
        """
        text = (
            'Credit rating letter dated December 6, 2023 by CRISIL assigning a '
            'rating of “CRISIL AA/Stable” (pronounced as CRISIL double '
            'A rating with Stable outlook) for the Issue with rating rationale '
            'dated December 5, 2023. '
            '13. Credit rating letter dated December 4, 2023 by ICRA Limited '
            'assigning a rating of “[ICRA]AA (Stable)” for the Issue '
            'with rating rationale dated December 7, 2023.'
        )

        items = de.extract_rating_letter_items(text)

        assert len(items) == 2
        assert items[0]["agency"] == "CRISIL"
        assert items[0]["rating"] == "CRISIL AA/Stable"
        assert items[1]["agency"] == "ICRA Limited"
        assert items[1]["rating"] == "[ICRA]AA (Stable)"


class TestCompanyNameDetection:

    def test_company_name_with_leading_digit(self):
        """
        "360" is a purely numeric token - it must NOT be rejected
        the way a mixed alphanumeric code (CIN/ISIN/PAN) is.
        """
        text = (
            "360 ONE Prime Limited (formerly known as IIFL Wealth Prime "
            "Limited) was incorporated as Chephis Capital Markets Limited"
        )
        names = de.find_company_names_in_text(text)
        assert "360 ONE Prime Limited" in names

    def test_cin_code_still_rejected(self):
        """A mixed letter+digit registration code is not a company name."""
        text = "Tel: 011 2345 6000; CIN: L65910DL1986GOI024862; PAN: AAACP1570H"
        names = de.find_company_names_in_text(text)
        assert all("CIN" not in name and "L65910" not in name for name in names)


class TestGreenShoeOption:

    def test_term_then_amount(self):
        text = 'WITH A GREEN SHOE OPTION OF ₹4,500 CRORE AMOUNTING TO...'
        assert de.extract_green_shoe_option(text) == "₹4,500 CRORE"

    def test_amount_then_term(self):
        """
        The reverse phrasing - amount stated first, "Green Shoe
        Option" as a trailing defined-term alias - must resolve to
        the SAME amount it's labeling, not fall through to an
        unrelated figure elsewhere in the document.
        """
        text = (
            'WITH AN OPTION TO RETAIN OVERSUBSCRIPTION OF UPTO ₹ 800 CRORE '
            '(“GREEN SHOE OPTION”) AGGREGATING UP TO ₹ 1,000 CRORE'
        )
        assert de.extract_green_shoe_option(text) == "₹ 800 CRORE"

    def test_absent_returns_none(self):
        assert de.extract_green_shoe_option("No such option in this text.") is None


class TestIssueDateExtractors:

    def test_issue_open_and_close_dates(self):
        text = (
            "TRANCHE I ISSUE OPENS ON: FRIDAY, JANUARY 16, 2026 "
            "TRANCHE I ISSUE CLOSES ON: FRIDAY, JANUARY 30, 2026"
        )
        assert de.extract_issue_open_date(text) == "FRIDAY, JANUARY 16, 2026"
        assert de.extract_issue_close_date(text) == "FRIDAY, JANUARY 30, 2026"

    def test_document_date_ignores_referenced_shelf_prospectus_date(self):
        text = (
            'THE SHELF PROSPECTUS DATED JANUARY 2, 2026 AND IS BEING OFFERED '
            'BY WAY OF THIS TRANCHE I PROSPECTUS DATED JANUARY 9, 2026'
        )
        assert de.extract_document_date(text) == "JANUARY 9, 2026"


class TestShelfLimit:

    def test_cover_page_style(self):
        text = "WHICH IS WITHIN THE SHELF LIMIT OF ₹10,000 CRORE AND IS"
        assert de.extract_shelf_limit(text) == "₹10,000 CRORE"
