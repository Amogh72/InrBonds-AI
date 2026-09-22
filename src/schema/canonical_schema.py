from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Page-level evidence for an extracted fact."""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    page: int = Field(..., ge=1)
    source_text: Optional[str] = None
    bbox: Optional[List[float]] = None


class Fact(BaseModel):
    """A canonical extracted value plus its source evidence."""
    model_config = ConfigDict(extra="forbid")

    value: Any
    unit: Optional[str] = None
    raw_value: Optional[str] = None
    provenance: List[Provenance] = Field(default_factory=list)


class InvestorCategory(BaseModel):
    """Investor category with category-specific bond terms."""
    model_config = ConfigDict(extra="forbid")

    category_id: str
    category_name: Optional[str] = None
    terms: Dict[str, Fact] = Field(default_factory=dict)


class Rating(BaseModel):
    """Credit rating information."""
    model_config = ConfigDict(extra="forbid")

    agency: str
    rating: str
    outlook: Optional[str] = None
    instrument: Optional[str] = None
    rating_date: Optional[str] = None
    provenance: List[Provenance] = Field(default_factory=list)


class SeriesTerms(BaseModel):
    """Terms that apply to a whole bond/NCD series."""
    model_config = ConfigDict(extra="forbid")

    tenor: Optional[Fact] = None
    frequency: Optional[Fact] = None
    # Generic classification of coupon structure (e.g. "Coupon-Bearing",
    # "Cumulative", "Zero Coupon"), derived from `frequency` where that
    # column actually describes a non-periodic payoff type rather than a
    # true payment cadence. `frequency` always keeps the raw source value
    # unchanged; this field never replaces it.
    coupon_type: Optional[Fact] = None
    face_value: Optional[Fact] = None
    minimum_application: Optional[Fact] = None
    issue_price: Optional[Fact] = None
    coupon: Optional[Fact] = None
    effective_yield: Optional[Fact] = None
    maturity_amount: Optional[Fact] = None
    maturity_redemption: Optional[Fact] = None
    mode_of_interest_payment: Optional[Fact] = None
    nature_of_indebtedness: Optional[Fact] = None
    put_call_option: Optional[Fact] = None
    additional_terms: Dict[str, Fact] = Field(default_factory=dict)


class BondSeries(BaseModel):
    """One series belonging to one specific issue."""
    model_config = ConfigDict(extra="forbid")

    series_id: str
    series_name: str
    series_number: Optional[str] = None
    terms: SeriesTerms = Field(default_factory=SeriesTerms)
    investor_categories: List[InvestorCategory] = Field(default_factory=list)
    provenance: List[Provenance] = Field(default_factory=list)


class IssueTerms(BaseModel):
    """Terms applying to the complete issue."""
    model_config = ConfigDict(extra="forbid")

    issue_size: Optional[Fact] = None
    issue_open_date: Optional[Fact] = None
    issue_close_date: Optional[Fact] = None
    allotment_date: Optional[Fact] = None
    listing: Optional[Fact] = None
    exchange: Optional[Fact] = None
    security_type: Optional[Fact] = None
    additional_terms: Dict[str, Fact] = Field(default_factory=dict)


class BondIssue(BaseModel):
    """One specific bond/NCD issue made by an issuer."""
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    issue_name: Optional[str] = None
    issue_type: Optional[str] = None
    terms: IssueTerms = Field(default_factory=IssueTerms)
    ratings: List[Rating] = Field(default_factory=list)
    series: List[BondSeries] = Field(default_factory=list)
    provenance: List[Provenance] = Field(default_factory=list)


class Issuer(BaseModel):
    """Issuer that can have multiple separate bond issues."""
    model_config = ConfigDict(extra="forbid")

    issuer_id: str
    name: str
    legal_name: Optional[str] = None
    issuer_type: Optional[str] = None
    issues: List[BondIssue] = Field(default_factory=list)
    provenance: List[Provenance] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Identity and metadata for the source PDF/document."""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_name: str
    document_type: Optional[str] = None
    document_date: Optional[str] = None
    source_file: Optional[str] = None
    page_count: Optional[int] = Field(default=None, ge=1)
    version: Optional[str] = None


class BondDocument(BaseModel):
    """Root canonical representation: one source document."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    document: DocumentMetadata
    issuer: Issuer
    provenance: List[Provenance] = Field(default_factory=list)


def fact(
    value: Any,
    *,
    unit: Optional[str] = None,
    raw_value: Optional[str] = None,
    provenance: Optional[List[Provenance]] = None,
) -> Fact:
    return Fact(
        value=value,
        unit=unit,
        raw_value=raw_value,
        provenance=provenance or [],
    )


def provenance(
    document_id: str,
    page: int,
    *,
    source_text: Optional[str] = None,
    bbox: Optional[List[float]] = None,
) -> Provenance:
    return Provenance(
        document_id=document_id,
        page=page,
        source_text=source_text,
        bbox=bbox,
    )


def validate_canonical_document(
    data: Union[BondDocument, Dict[str, Any]]
) -> BondDocument:
    if isinstance(data, BondDocument):
        return data
    return BondDocument.model_validate(data)


def canonical_json(
    data: Union[BondDocument, Dict[str, Any]],
    *,
    indent: int = 2,
) -> str:
    document = validate_canonical_document(data)
    return document.model_dump_json(indent=indent, exclude_none=True)
