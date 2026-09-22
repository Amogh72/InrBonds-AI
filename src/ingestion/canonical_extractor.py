import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------
# Make src/ available for imports when running this file
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schema.canonical_schema import (
    BondDocument,
    BondIssue,
    BondSeries,
    DocumentMetadata,
    Fact,
    InvestorCategory,
    IssueTerms,
    Issuer,
    Provenance,
    SeriesTerms,
    Rating,
    validate_canonical_document,
)

from ingestion.table_extractor import extract_series_terms


# =========================================================
# HELPERS
# =========================================================

def make_provenance(
    document_id: str,
    page: int,
    source_text: Optional[str] = None,
) -> Provenance:
    return Provenance(
        document_id=document_id,
        page=page,
        source_text=source_text,
    )


def make_fact(
    value: Any,
    document_id: str,
    page: int,
    unit: Optional[str] = None,
    raw_value: Optional[str] = None,
) -> Fact:
    return Fact(
        value=value,
        unit=unit,
        raw_value=raw_value,
        provenance=[
            make_provenance(
                document_id=document_id,
                page=page,
                source_text=raw_value,
            )
        ],
    )


def first_page(series_data: Dict[str, Any]) -> int:
    pages = series_data.get("provenance", {}).get("pages", [])

    if pages:
        return min(pages)

    return 1


def normalize_category_name(category_id: str) -> str:
    names = {
        "I_II": "Category I and Category II",
        "III": "Category III",
        "IV": "Category IV",
    }

    return names.get(category_id, category_id)


# =========================================================
# SERIES CONVERSION
# =========================================================

def convert_series(
    series_id: str,
    data: Dict[str, Any],
    document_id: str,
) -> BondSeries:

    page = first_page(data)

    provenance = [
        make_provenance(
            document_id=document_id,
            page=p,
        )
        for p in data.get("provenance", {}).get("pages", [])
    ]

    terms = SeriesTerms()

    # -----------------------------------------------------
    # Basic series terms
    # -----------------------------------------------------

    if data.get("tenor") is not None:
        terms.tenor = make_fact(
            value=data["tenor"],
            document_id=document_id,
            page=page,
            raw_value=str(data["tenor"]),
        )

    if data.get("frequency") is not None:
        terms.frequency = make_fact(
            value=data["frequency"],
            document_id=document_id,
            page=page,
            raw_value=str(data["frequency"]),
        )

    if data.get("face_value_inr") is not None:
        terms.face_value = make_fact(
            value=data["face_value_inr"],
            unit="INR",
            document_id=document_id,
            page=page,
            raw_value=str(data["face_value_inr"]),
        )

    if data.get("minimum_application") is not None:
        terms.minimum_application = make_fact(
            value=data["minimum_application"],
            document_id=document_id,
            page=page,
            raw_value=str(data["minimum_application"]),
        )

    if data.get("mode_of_interest_payment") is not None:
        terms.mode_of_interest_payment = make_fact(
            value=data["mode_of_interest_payment"],
            document_id=document_id,
            page=page,
            raw_value=str(data["mode_of_interest_payment"]),
        )

    if data.get("maturity_redemption") is not None:
        terms.maturity_redemption = make_fact(
            value=data["maturity_redemption"],
            document_id=document_id,
            page=page,
            raw_value=str(data["maturity_redemption"]),
        )

    if data.get("nature_of_indebtedness") is not None:
        terms.nature_of_indebtedness = make_fact(
            value=data["nature_of_indebtedness"],
            document_id=document_id,
            page=page,
            raw_value=str(data["nature_of_indebtedness"]),
        )

    if data.get("put_call_option") is not None:
        terms.put_call_option = make_fact(
            value=data["put_call_option"],
            document_id=document_id,
            page=page,
            raw_value=str(data["put_call_option"]),
        )

    # -----------------------------------------------------
    # Investor-category-specific terms
    # -----------------------------------------------------

    category_ids = set()

    category_ids.update(
        data.get("issue_price_by_category", {}).keys()
    )

    category_ids.update(
        data.get("coupon_by_category", {}).keys()
    )

    category_ids.update(
        data.get("effective_yield_by_category", {}).keys()
    )

    category_ids.update(
        data.get("maturity_amount_by_category", {}).keys()
    )

    investor_categories = []

    for category_id in sorted(category_ids):

        category = InvestorCategory(
            category_id=category_id,
            category_name=normalize_category_name(category_id),
            terms={},
        )

        # Issue price
        if category_id in data.get("issue_price_by_category", {}):

            value = data["issue_price_by_category"][category_id]

            category.terms["issue_price"] = make_fact(
                value=value,
                unit="INR",
                document_id=document_id,
                page=page,
                raw_value=str(value),
            )

        # Coupon
        if category_id in data.get("coupon_by_category", {}):

            value = data["coupon_by_category"][category_id]

            category.terms["coupon"] = make_fact(
                value=value,
                unit="percent",
                document_id=document_id,
                page=page,
                raw_value=f"{value}%",
            )

        # Effective yield
        if category_id in data.get(
            "effective_yield_by_category", {}
        ):

            value = data["effective_yield_by_category"][category_id]

            category.terms["effective_yield"] = make_fact(
                value=value,
                unit="percent",
                document_id=document_id,
                page=page,
                raw_value=f"{value}%",
            )

        # Maturity amount
        if category_id in data.get(
            "maturity_amount_by_category", {}
        ):

            value = data["maturity_amount_by_category"][category_id]

            category.terms["maturity_amount"] = make_fact(
                value=value,
                unit="INR",
                document_id=document_id,
                page=page,
                raw_value=str(value),
            )

        investor_categories.append(category)

    return BondSeries(
        series_id=f"{document_id}_series_{series_id}",
        series_name=f"Series {series_id}",
        series_number=series_id,
        terms=terms,
        investor_categories=investor_categories,
        provenance=provenance,
    )


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_canonical_document(
    pdf_path: str,
    document_id: str,
    document_name: Optional[str] = None,
) -> BondDocument:

    pdf_path = Path(pdf_path)

    if document_name is None:
        document_name = pdf_path.name

    # -----------------------------------------------------
    # 1. Existing geometry-aware table extraction
    # -----------------------------------------------------

    series_data, tables_found = extract_series_terms(
        pdf_path,
        verbose=True,
    )

    if not series_data:
        raise RuntimeError(
            "No series terms were extracted from the PDF."
        )

    # -----------------------------------------------------
    # 2. Convert every extracted series into canonical form
    # -----------------------------------------------------

    series = []

    for series_id, data in sorted(series_data.items()):

        series.append(
            convert_series(
                series_id=series_id,
                data=data,
                document_id=document_id,
            )
        )

    # -----------------------------------------------------
    # 3. Build issue
    #
    # We deliberately keep issue metadata minimal for now.
    # Text/domain extraction will populate additional
    # document-level information in the next pass.
    # -----------------------------------------------------

    issue = BondIssue(
        issue_id=f"{document_id}_issue_1",
        issue_name=None,
        issue_type="NCD",
        terms=IssueTerms(),
        ratings=[],
        series=series,
        provenance=[],
    )

    # -----------------------------------------------------
    # 4. Issuer
    #
    # Temporary generic placeholder until the textual
    # domain extractor is connected.
    # -----------------------------------------------------

    issuer = Issuer(
        issuer_id=f"{document_id}_issuer",
        name="UNKNOWN",
        legal_name=None,
        issuer_type=None,
        issues=[issue],
        provenance=[],
    )

    # -----------------------------------------------------
    # 5. Document metadata
    # -----------------------------------------------------

    document = DocumentMetadata(
        document_id=document_id,
        document_name=document_name,
        document_type="Bond/NCD Prospectus",
        source_file=str(pdf_path),
    )

    # -----------------------------------------------------
    # 6. Canonical document
    # -----------------------------------------------------

    canonical = BondDocument(
        schema_version="1.0",
        document=document,
        issuer=issuer,
        provenance=[],
    )

    return validate_canonical_document(canonical)


# =========================================================
# SAVE
# =========================================================

def save_canonical_document(
    document: BondDocument,
    output_path: str,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            document.model_dump(
                mode="json",
                exclude_none=True,
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Extract one canonical JSON from a bond/NCD prospectus."
    )

    parser.add_argument(
        "pdf",
        help="Path to source PDF",
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="Stable document ID",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Canonical JSON output path",
    )

    args = parser.parse_args()

    document = extract_canonical_document(
        pdf_path=args.pdf,
        document_id=args.document_id,
    )

    save_canonical_document(
        document=document,
        output_path=args.output,
    )

    print()
    print("=" * 60)
    print("CANONICAL DOCUMENT CREATED")
    print("=" * 60)
    print(f"Document ID : {args.document_id}")
    print(f"Output      : {args.output}")
    print(
        f"Series      : {len(document.issuer.issues[0].series)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()