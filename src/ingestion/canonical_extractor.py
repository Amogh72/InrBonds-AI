import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Chunk,
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
from ingestion import chunker
from ingestion import pdf_parser
from ingestion import domain_extractor
from ingestion import structure_detector


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


def classify_coupon_type(frequency_raw_value: Optional[str]) -> str:
    """
    Generic bond-terminology classification of the "Frequency of
    Interest Payment" column into a small, industry-standard
    coupon-structure vocabulary. Indian NCD prospectuses commonly
    reuse this exact column for genuinely different concepts: a
    true payment cadence (Annual/Semi-Annual/Quarterly/Monthly)
    for coupon-bearing series, versus a non-periodic payoff
    structure (Cumulative, Zero Coupon) for series that pay no
    periodic interest at all. This is standard industry
    terminology, not specific to any one issuer, and it never
    replaces the raw `frequency` value - it's an additional,
    normalized classification derived from the same source text.
    """

    text = (frequency_raw_value or "").lower()

    if "zero coupon" in text:
        return "Zero Coupon"

    if "cumulative" in text:
        return "Cumulative"

    return "Coupon-Bearing"


def normalize_category_name(category_id: str) -> str:
    """
    category_id is table_extractor.extract_category()'s output: one
    or more Roman numerals joined by "_" (e.g. "III", "I_II",
    "I_II_III_IV"), in whatever combination the source document's
    row label actually named. Built generically from that list
    rather than a fixed lookup, so any combination is covered, not
    just the ones seen in past documents.
    """

    if not category_id:
        return category_id

    numerals = category_id.split("_")

    if len(numerals) == 1:
        return f"Category {numerals[0]}"

    if len(numerals) == 2:
        return f"Category {numerals[0]} and Category {numerals[1]}"

    return "Category " + ", ".join(numerals[:-1]) + f" and {numerals[-1]}"


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
        terms.coupon_type = make_fact(
            value=classify_coupon_type(str(data["frequency"])),
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
# SHARED vs. SERIES-SPECIFIC TERMS
# =========================================================

# Scalar (non-category) SeriesTerms fields that CAN be shared
# across every series of one issue. tenor/frequency/face_value
# are deliberately excluded here - those genuinely vary per
# series in every real prospectus seen so far, so they are never
# even considered for hoisting. This list only decides which
# fields are ELIGIBLE to be hoisted; a field is only actually
# hoisted if every series' extracted value for it is identical
# (see hoist_shared_series_terms), so a document where these
# fields genuinely do vary per series is unaffected.
HOISTABLE_SERIES_FIELDS = [
    "minimum_application",
    "mode_of_interest_payment",
    "maturity_redemption",
    "nature_of_indebtedness",
    "put_call_option",
]


def hoist_shared_series_terms(
    series_list: list,
    issue_terms: "IssueTerms",
) -> None:
    """
    A merged table cell that spans every series column (e.g. one
    "Minimum Application" cell drawn across all 5 series) produces
    the SAME Fact value on every BondSeries after table extraction.
    Duplicating that fact 5 times misrepresents it as 5 independent
    per-series facts, when it's actually one issue-level term.

    For each field in HOISTABLE_SERIES_FIELDS, if every series
    carries the exact same value for it, move it once onto
    `issue_terms` (merging provenance from every series it came
    from) and clear it from each series. A field that genuinely
    differs across series - even for just one series - is left
    exactly where it was.
    """

    if not series_list:
        return

    for field_name in HOISTABLE_SERIES_FIELDS:

        facts = [getattr(series.terms, field_name) for series in series_list]

        if any(fact is None for fact in facts):
            continue

        distinct_values = {fact.value for fact in facts}

        if len(distinct_values) != 1:
            continue

        merged_provenance = []
        seen_pages = set()

        for fact in facts:
            for prov in fact.provenance:
                key = (prov.document_id, prov.page)
                if key not in seen_pages:
                    seen_pages.add(key)
                    merged_provenance.append(prov)

        template_fact = facts[0]

        merged_fact = Fact(
            value=template_fact.value,
            unit=template_fact.unit,
            raw_value=template_fact.raw_value,
            provenance=merged_provenance,
        )

        if field_name == "nature_of_indebtedness":
            issue_terms.security_type = merged_fact
        else:
            issue_terms.additional_terms[field_name] = merged_fact

        for series in series_list:
            setattr(series.terms, field_name, None)


# =========================================================
# TEXT/DOMAIN EXTRACTION INTEGRATION
#
# Combines table-derived series data (above) with text-derived
# issuer/issue/rating data, using the existing domain_extractor.py
# rather than a second, competing extraction system. This section
# is the only place responsible for turning raw PDF page text into
# the inputs domain_extractor.py expects (and into the smaller set
# of document-level facts - issue dates, listing, ratings - that
# domain_extractor.py's new page-scanning functions produce).
# =========================================================

# Target size for the lightweight text windows fed to
# domain_extractor.extract_domain_knowledge(). This is NOT the
# project's future retrieval/vector-DB chunker (chunker.py) - it
# is a minimal, page-range-aware windowing utility that exists
# only to give the entity/relationship extractor reasonably-sized
# text to scan, with page-range provenance. It has no overlap and
# no section-awareness; chunker.py's structure-aware chunks remain
# the real chunking system for retrieval, to be wired in later.
DOMAIN_CHUNK_TARGET_CHARS = 3500


def _page_texts_from_layout(layout_document: Dict[str, Any]) -> Dict[int, str]:

    page_texts = {}

    for page in layout_document["pages"]:

        lines = [
            line["text"].strip()
            for line in page["lines"]
            if line["text"].strip()
        ]

        page_texts[page["page_number"]] = "\n".join(lines)

    return page_texts


def _build_domain_chunks(
    layout_document: Dict[str, Any],
    target_chars: int = DOMAIN_CHUNK_TARGET_CHARS,
) -> Dict[str, Any]:

    page_texts = _page_texts_from_layout(layout_document)

    chunks = []
    current_text = []
    current_pages = []
    current_length = 0

    def flush():
        if not current_text:
            return
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_page": min(current_pages),
            "end_page": max(current_pages),
            "text": "\n\n".join(current_text),
        })

    for page_number in sorted(page_texts):

        text = page_texts[page_number]

        if not text:
            continue

        if current_length and current_length + len(text) > target_chars:
            flush()
            current_text = []
            current_pages = []
            current_length = 0

        current_text.append(text)
        current_pages.append(page_number)
        current_length += len(text)

    flush()

    return {
        "document": layout_document["document"],
        "chunks": chunks,
    }


def _pick_canonical_entity(
    entities: list,
    entity_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Mirrors domain_extractor.pick_canonical_entity_of_type's
    selection rule (most mentions, shorter value breaks ties),
    applied to the flat, already-grouped `entities` list that
    domain_extractor.extract_domain_knowledge() returns.
    """

    candidates = [e for e in entities if e["entity_type"] == entity_type]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda e: (len(e["mentions"]), -len(e["value"])),
    )


def _entity_to_fact(
    entity: Optional[Dict[str, Any]],
    document_id: str,
) -> Optional[Fact]:
    """
    Convert one domain_extractor.py entity (value + mentions) into a
    Fact, citing the page of its first mention. Returns None if no
    entity was passed in, so callers can chain this directly.
    """

    if entity is None:
        return None

    page = min(mention["start_page"] for mention in entity["mentions"])

    return make_fact(
        value=entity["value"],
        document_id=document_id,
        page=page,
        raw_value=entity["value"],
    )


def _scan_pages_for_first_match(
    page_texts: Dict[int, str],
    extractor_fn,
):
    """
    Run a single-value page-level extractor (e.g.
    domain_extractor.extract_issue_open_date) over every page in
    order and return (value, page_number) for the first page that
    produces a value, or (None, None) if no page does.
    """

    for page_number in sorted(page_texts):

        normalized = domain_extractor.normalize_text(page_texts[page_number])
        value = extractor_fn(normalized)

        if value:
            return value, page_number

    return None, None


def collect_ratings_from_pages(
    page_texts: Dict[int, str],
    alias_map: Dict[str, str],
) -> list:
    """
    Scan every page for itemized credit-rating-letter facts (see
    domain_extractor.extract_rating_letter_items), grouping repeat
    occurrences of the SAME (agency, rating) pair into one item
    with merged page provenance, and never merging two different
    agencies' ratings into one item.
    """

    grouped: Dict[tuple, Dict[str, Any]] = {}

    for page_number in sorted(page_texts):

        normalized = domain_extractor.normalize_text(page_texts[page_number])

        for item in domain_extractor.extract_rating_letter_items(normalized):

            resolved_agency = alias_map.get(
                domain_extractor.normalize_entity_value(item["agency"]),
                item["agency"],
            )

            key = (
                domain_extractor.normalize_entity_value(resolved_agency),
                item["rating"],
            )

            if key not in grouped:
                grouped[key] = {**item, "agency": resolved_agency, "pages": []}

            if page_number not in grouped[key]["pages"]:
                grouped[key]["pages"].append(page_number)

    return list(grouped.values())


def gather_domain_knowledge(
    pdf_path: Path,
    layout_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the text/domain side of extraction: parse the PDF's text
    layout (or reuse an already-parsed one, if the caller has one -
    parsing a several-hundred-page PDF twice for the same run is
    pure waste), build lightweight windows for domain_extractor.py's
    entity/relationship pipeline, and separately scan every page
    for the document/issue-level facts (dates, listing, ratings)
    that need page-precise provenance rather than chunk ranges.
    """

    if layout_document is None:
        layout_document = pdf_parser.parse_pdf_with_layout(pdf_path)

    page_texts = _page_texts_from_layout(layout_document)

    domain_chunks_doc = _build_domain_chunks(layout_document)
    chunk_texts = [chunk["text"] for chunk in domain_chunks_doc["chunks"]]

    entities, _relationships = domain_extractor.extract_domain_knowledge(
        domain_chunks_doc
    )

    alias_map = domain_extractor.build_alias_map(chunk_texts)

    issuer_entity = _pick_canonical_entity(entities, "Issuer")
    base_issue_size_entity = _pick_canonical_entity(entities, "Base Issue Size")
    security_cover_entity = _pick_canonical_entity(entities, "Security Cover")

    # Exactly one trustee is mandated per issue (see BondIssue.debenture_trustee),
    # so the canonical (most-mentioned) entity is the right choice here.
    debenture_trustee_entity = _pick_canonical_entity(
        entities, "Debenture Trustee"
    )

    # Several Lead Managers are routinely appointed together, so - unlike
    # the single-valued entity types above - every distinct one found is
    # kept, not just the most-mentioned.
    lead_manager_entities = [
        e for e in entities if e["entity_type"] == "Lead Manager"
    ]

    issue_open_date, issue_open_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_issue_open_date
    )
    issue_close_date, issue_close_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_issue_close_date
    )
    document_date, document_date_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_document_date
    )
    issue_name, _issue_name_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_issue_name
    )
    listing_info, listing_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_listing_exchange
    )
    shelf_limit, shelf_limit_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_shelf_limit
    )
    green_shoe_option, green_shoe_page = _scan_pages_for_first_match(
        page_texts, domain_extractor.extract_green_shoe_option
    )

    ratings_raw = collect_ratings_from_pages(page_texts, alias_map)

    return {
        "total_pages": layout_document["total_pages"],
        "issuer_entity": issuer_entity,
        "base_issue_size_entity": base_issue_size_entity,
        "security_cover_entity": security_cover_entity,
        "debenture_trustee_entity": debenture_trustee_entity,
        "lead_manager_entities": lead_manager_entities,
        "issue_open_date": issue_open_date,
        "issue_open_page": issue_open_page,
        "issue_close_date": issue_close_date,
        "issue_close_page": issue_close_page,
        "document_date": document_date,
        "document_date_page": document_date_page,
        "issue_name": issue_name,
        "listing_info": listing_info,
        "listing_page": listing_page,
        "shelf_limit": shelf_limit,
        "shelf_limit_page": shelf_limit_page,
        "green_shoe_option": green_shoe_option,
        "green_shoe_page": green_shoe_page,
        "ratings_raw": ratings_raw,
    }


# =========================================================
# STRUCTURED-FACT CHUNK GENERATION
#
# One retrieval-oriented Chunk per meaningful group of Facts already
# present on the BondDocument we just built - never a second,
# independent extraction of the source PDF. See Chunk's docstring in
# canonical_schema.py for the "structured_fact" vs "prose" split;
# only "structured_fact" chunks are generated here.
# =========================================================

def _humanize_field_name(field_name: str) -> str:
    return field_name.replace("_", " ").strip().title()


def _fact_lines_and_provenance(
    labeled_facts: List[Tuple[str, Optional[Fact]]],
) -> Tuple[List[str], List[Provenance]]:
    """
    Turn a list of (label, Fact-or-None) pairs into readable text
    lines and merged, deduplicated provenance. A Fact that is None
    (the field simply doesn't apply to this series/issue) is
    silently skipped, never rendered as "N/A" - only real extracted
    values become chunk text.
    """

    lines = []
    provenance: List[Provenance] = []
    seen_pages = set()

    for label, fact in labeled_facts:

        if fact is None:
            continue

        # fact.value is this Fact's actual canonical value; raw_value
        # is documented as an audit trail back to the original source
        # text (which sometimes deliberately differs from value - see
        # coupon_type, whose raw_value is the un-classified source
        # text it was derived from) and is NOT a display preference,
        # so it's never used here.
        display_value = str(fact.value)

        if fact.unit and fact.unit not in display_value:
            display_value = f"{display_value} {fact.unit}"

        lines.append(f"{label}: {display_value}")

        for prov in fact.provenance:
            key = (prov.document_id, prov.page)
            if key not in seen_pages:
                seen_pages.add(key)
                provenance.append(prov)

    return lines, provenance


SERIES_TERM_LABELS = [
    ("tenor", "Tenor"),
    ("frequency", "Frequency of Interest Payment"),
    ("coupon_type", "Coupon Type"),
    ("face_value", "Face Value"),
    ("minimum_application", "Minimum Application"),
    ("mode_of_interest_payment", "Mode of Interest Payment"),
    ("maturity_redemption", "Maturity/Redemption"),
    ("nature_of_indebtedness", "Nature of Indebtedness"),
    ("put_call_option", "Put/Call Option"),
]

CATEGORY_TERM_LABELS = {
    "issue_price": "Issue Price",
    "coupon": "Coupon",
    "effective_yield": "Effective Yield",
    "maturity_amount": "Maturity Amount",
}


def generate_series_category_chunks(
    document_id: str,
    issuer_name: str,
    issue: BondIssue,
    series: BondSeries,
) -> List[Chunk]:
    """
    One structured_fact chunk per (series, investor category),
    combining that series' own terms with that category's
    category-specific terms - matching the shape a comparison
    question ("effective yield of Series III for Category III") is
    actually asking for. Falls back to a single series-level chunk
    if the series has no investor categories at all, so those facts
    are never silently dropped.
    """

    series_labeled = [
        (label, getattr(series.terms, field))
        for field, label in SERIES_TERM_LABELS
    ]

    header = [
        f"Issuer: {issuer_name}",
        f"Issue: {issue.issue_name or issue.issue_id}",
        f"Series: {series.series_name}",
    ]

    if not series.investor_categories:

        lines, provenance = _fact_lines_and_provenance(series_labeled)

        if not lines:
            return []

        text = "\n".join(header + lines)

        return [Chunk(
            chunk_id=f"{document_id}_chunk_series_{series.series_number}",
            chunk_type="structured_fact",
            text=text,
            series_id=series.series_id,
            provenance=provenance,
        )]

    chunks = []

    for category in series.investor_categories:

        category_labeled = [
            (
                CATEGORY_TERM_LABELS.get(field_name, _humanize_field_name(field_name)),
                fact,
            )
            for field_name, fact in category.terms.items()
        ]

        lines, provenance = _fact_lines_and_provenance(
            series_labeled + category_labeled
        )

        if not lines:
            continue

        text = "\n".join(
            header
            + [f"Investor Category: {category.category_name or category.category_id}"]
            + lines
        )

        chunks.append(Chunk(
            chunk_id=(
                f"{document_id}_chunk_series_{series.series_number}"
                f"_cat_{category.category_id}"
            ),
            chunk_type="structured_fact",
            text=text,
            series_id=series.series_id,
            category_id=category.category_id,
            provenance=provenance,
        ))

    return chunks


ISSUE_TERM_LABELS = [
    ("issue_size", "Issue Size"),
    ("issue_open_date", "Issue Open Date"),
    ("issue_close_date", "Issue Close Date"),
    ("allotment_date", "Deemed Date of Allotment"),
    ("listing", "Listing"),
    ("exchange", "Stock Exchange"),
    ("security_type", "Security Type"),
    ("shelf_limit", "Shelf Limit"),
    ("green_shoe_option", "Green Shoe Option"),
    ("security_cover", "Minimum Security Cover"),
]


def generate_issue_overview_chunk(
    document_id: str,
    issuer: Issuer,
    issue: BondIssue,
) -> List[Chunk]:
    """
    One structured_fact chunk for issuer identity + issue-level terms
    that aren't specific to any one series (dates, sizing, listing,
    trustee, lead managers) - so a question like "who is the
    debenture trustee" or "what is the security cover" can be
    answered from a structured fact, not only from prose.
    """

    header = [
        f"Issuer: {issuer.name}",
        f"Issue: {issue.issue_name or issue.issue_id}",
    ]

    term_labeled = [
        (label, getattr(issue.terms, field))
        for field, label in ISSUE_TERM_LABELS
    ]

    extra_labeled = [
        (_humanize_field_name(field_name), fact)
        for field_name, fact in issue.terms.additional_terms.items()
    ]

    party_labeled = [("Debenture Trustee", issue.debenture_trustee)]
    party_labeled += [
        ("Lead Manager", lead_manager) for lead_manager in issue.lead_managers
    ]

    lines, provenance = _fact_lines_and_provenance(
        term_labeled + extra_labeled + party_labeled
    )

    if not lines:
        return []

    text = "\n".join(header + lines)

    return [Chunk(
        chunk_id=f"{document_id}_chunk_issue_overview",
        chunk_type="structured_fact",
        text=text,
        provenance=provenance,
    )]


def generate_rating_chunks(
    document_id: str,
    issuer: Issuer,
    issue: BondIssue,
) -> List[Chunk]:
    """One structured_fact chunk per credit rating - never merged."""

    chunks = []

    for index, rating in enumerate(issue.ratings, start=1):

        lines = [
            f"Issuer: {issuer.name}",
            f"Credit Rating Agency: {rating.agency}",
            f"Rating: {rating.rating}",
        ]

        if rating.outlook:
            lines.append(f"Outlook: {rating.outlook}")

        if rating.rating_date:
            lines.append(f"Rating Date: {rating.rating_date}")

        chunks.append(Chunk(
            chunk_id=f"{document_id}_chunk_rating_{index}",
            chunk_type="structured_fact",
            text="\n".join(lines),
            provenance=rating.provenance,
        ))

    return chunks


def generate_structured_fact_chunks(document: BondDocument) -> List[Chunk]:
    """
    Every structured_fact Chunk this document currently has facts
    for: one per (series, investor category), one issue overview, and
    one per rating. Purely derived from Facts already on `document` -
    generates no new values and never re-reads the source PDF.
    """

    document_id = document.document.document_id
    issuer = document.issuer
    chunks: List[Chunk] = []

    for issue in issuer.issues:

        chunks.extend(
            generate_issue_overview_chunk(document_id, issuer, issue)
        )
        chunks.extend(
            generate_rating_chunks(document_id, issuer, issue)
        )

        for series in issue.series:
            chunks.extend(
                generate_series_category_chunks(
                    document_id, issuer.name, issue, series
                )
            )

    return chunks


# =========================================================
# PROSE CHUNK GENERATION
#
# Reuses structure_detector.py + chunker.py as-is (both already
# validated against both fixture documents) rather than a competing
# extraction system. Two things needed fixing before this could be
# merged into the same canonical JSON as structured_fact chunks - see
# chunker.py's create_chunks() and collect_section_pages() docstrings
# for the document_id-consistency and table-page-exclusion fixes.
# =========================================================

def generate_prose_chunks(
    pdf_path: Path,
    layout_document: Dict[str, Any],
    document_id: str,
    table_pages: set,
) -> List[Chunk]:
    """
    One "prose" Chunk per structure_detector-identified leaf section
    window (Risk Factors, Objects of the Issue, covenants, etc.) -
    the free-text content structured_fact chunks can't cover, since
    none of it is modeled as a Fact anywhere in this schema.

    table_pages are excluded from chunking here: those pages already
    have a clean, structured extraction (table_extractor.py's series
    table), and re-chunking their raw line text would just be an
    out-of-order restatement of the same data, likely to read as
    garbled since a table's cells don't reconstruct into prose.
    """

    structure = structure_detector.detect_structure(
        pdf_path, layout_document, verbose=False
    )

    raw_chunks = chunker.create_chunks(
        layout_document,
        structure,
        document_id=document_id,
        excluded_pages=table_pages,
    )

    return [
        Chunk(
            chunk_id=raw["chunk_id"],
            chunk_type="prose",
            text=raw["text"],
            start_page=raw["start_page"],
            end_page=raw["end_page"],
            section_path=raw["section_path"],
            provenance=[
                make_provenance(document_id, raw["start_page"])
            ],
        )
        for raw in raw_chunks
    ]


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_canonical_document(
    pdf_path: str,
    document_id: str,
    document_name: Optional[str] = None,
    layout_document: Optional[Dict[str, Any]] = None,
) -> BondDocument:
    """
    layout_document: an already-parsed pdf_parser.parse_pdf_with_layout()
    result, if the caller has one (e.g. also running structure_detector
    against the same PDF). Avoids parsing a several-hundred-page PDF
    twice for one pipeline run; parsed internally if not given.
    """

    pdf_path = Path(pdf_path)

    if document_name is None:
        document_name = pdf_path.name

    if layout_document is None:
        layout_document = pdf_parser.parse_pdf_with_layout(pdf_path)

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
    # 3. Text/domain extraction (issuer, issue terms, ratings)
    # -----------------------------------------------------

    domain = gather_domain_knowledge(pdf_path, layout_document=layout_document)

    issue_terms = IssueTerms()

    issue_terms.issue_size = _entity_to_fact(
        domain["base_issue_size_entity"], document_id
    )
    issue_terms.security_cover = _entity_to_fact(
        domain["security_cover_entity"], document_id
    )

    if domain["shelf_limit"] is not None:

        issue_terms.shelf_limit = make_fact(
            value=domain["shelf_limit"],
            document_id=document_id,
            page=domain["shelf_limit_page"],
            raw_value=domain["shelf_limit"],
        )

    if domain["green_shoe_option"] is not None:

        issue_terms.green_shoe_option = make_fact(
            value=domain["green_shoe_option"],
            document_id=document_id,
            page=domain["green_shoe_page"],
            raw_value=domain["green_shoe_option"],
        )

    if domain["issue_open_date"] is not None:

        issue_terms.issue_open_date = make_fact(
            value=domain["issue_open_date"],
            document_id=document_id,
            page=domain["issue_open_page"],
            raw_value=domain["issue_open_date"],
        )

    if domain["issue_close_date"] is not None:

        issue_terms.issue_close_date = make_fact(
            value=domain["issue_close_date"],
            document_id=document_id,
            page=domain["issue_close_page"],
            raw_value=domain["issue_close_date"],
        )

    if domain["listing_info"] is not None:

        issue_terms.exchange = make_fact(
            value=domain["listing_info"]["exchange"],
            document_id=document_id,
            page=domain["listing_page"],
            raw_value=domain["listing_info"]["exchange"],
        )

        issue_terms.listing = make_fact(
            value=(
                f"Listed on {domain['listing_info']['exchange']}"
                + (
                    f" (\"{domain['listing_info']['exchange_alias']}\")"
                    if domain["listing_info"]["exchange_alias"]
                    else ""
                )
            ),
            document_id=document_id,
            page=domain["listing_page"],
        )

    # -----------------------------------------------------
    # 4. Hoist issue-level terms that were extracted onto
    #    every series only because one merged table cell
    #    physically spans all of them (see
    #    hoist_shared_series_terms's docstring).
    # -----------------------------------------------------

    hoist_shared_series_terms(series, issue_terms)

    # -----------------------------------------------------
    # 5. Ratings - one Fact-like Rating object per agency,
    #    never merging different agencies' ratings together.
    # -----------------------------------------------------

    ratings = [
        Rating(
            agency=item["agency"],
            rating=item["rating"],
            outlook=item["outlook"],
            instrument="NCD",
            rating_date=item["rating_date"],
            provenance=[
                make_provenance(
                    document_id=document_id,
                    page=page,
                    source_text=item["source_text"],
                )
                for page in item["pages"]
            ],
        )
        for item in domain["ratings_raw"]
    ]

    # -----------------------------------------------------
    # 6. Build issue
    # -----------------------------------------------------

    debenture_trustee = _entity_to_fact(
        domain["debenture_trustee_entity"], document_id
    )

    lead_managers = [
        fact
        for fact in (
            _entity_to_fact(entity, document_id)
            for entity in domain["lead_manager_entities"]
        )
        if fact is not None
    ]

    issue = BondIssue(
        issue_id=f"{document_id}_issue_1",
        issue_name=domain["issue_name"],
        issue_type="NCD",
        terms=issue_terms,
        ratings=ratings,
        debenture_trustee=debenture_trustee,
        lead_managers=lead_managers,
        series=series,
        provenance=[],
    )

    # -----------------------------------------------------
    # 7. Issuer
    #
    # Resolved from domain_extractor.py's generic, structural
    # company-name + label-proximity detection - never a
    # hardcoded issuer name. Falls back to "UNKNOWN" if no
    # Issuer-labeled company name was found anywhere in the
    # document, rather than guessing.
    # -----------------------------------------------------

    issuer_entity = domain["issuer_entity"]

    if issuer_entity is not None:

        issuer_page = min(
            mention["start_page"] for mention in issuer_entity["mentions"]
        )

        issuer_name = issuer_entity["value"]
        issuer_provenance = [
            make_provenance(document_id=document_id, page=issuer_page)
        ]

    else:

        issuer_name = "UNKNOWN"
        issuer_provenance = []

    issuer = Issuer(
        issuer_id=f"{document_id}_issuer",
        name=issuer_name,
        legal_name=None,
        issuer_type=None,
        issues=[issue],
        provenance=issuer_provenance,
    )

    # -----------------------------------------------------
    # 8. Document metadata
    # -----------------------------------------------------

    document = DocumentMetadata(
        document_id=document_id,
        document_name=document_name,
        document_type="Bond/NCD Prospectus",
        document_date=domain["document_date"],
        source_file=str(pdf_path),
        page_count=domain["total_pages"],
    )

    # -----------------------------------------------------
    # 9. Canonical document
    # -----------------------------------------------------

    canonical = BondDocument(
        schema_version="1.0",
        document=document,
        issuer=issuer,
        provenance=[],
    )

    # -----------------------------------------------------
    # 10. Chunks: structured_fact (derived from the Facts just
    #     assembled above) and prose (from structure_detector.py +
    #     chunker.py, excluding pages already covered by the series
    #     table). One combined list, per Chunk's docstring in
    #     canonical_schema.py.
    # -----------------------------------------------------

    table_pages = {t["page"] for t in tables_found}

    canonical.chunks = generate_structured_fact_chunks(canonical) + (
        generate_prose_chunks(
            pdf_path, layout_document, document_id, table_pages
        )
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