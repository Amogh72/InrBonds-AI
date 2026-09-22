"""
Generic domain-knowledge extraction layer for bond/NCD prospectuses.

Takes a chunked document JSON (produced by an earlier chunking step,
e.g. pfc_ncd_2026_chunks.json) and produces a structured JSON of
entities and co-occurrence relationships suitable as input to a later
Graph DB loading script.

This module deliberately does NOT hardcode any issuer, company name,
lead manager, or page number. Every pattern here is structural
(defined-term markers, company-name suffixes, currency/percentage
shapes, ISIN shape, rating-agency vocabulary) so the same script works
across different issuers' bond/NCD prospectuses, not just one.

    PDF -> chunks -> THIS SCRIPT -> domain_knowledge.json -> Graph DB

This script does not build the Graph DB itself, and does not touch the
SQL DB, Vector DB, or MCP server code.
"""

import argparse
import json
import re
from pathlib import Path


# =========================================================
# LOAD / SAVE
# =========================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """
    Normalize whitespace while keeping the original
    meaning/content intact.
    """

    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalize_entity_value(value):
    """
    Canonical form of an entity's value, used ONLY to group
    repeated mentions of the same real-world entity into one
    graph node (e.g. "Power Finance Corporation Limited" seen
    in 40 chunks should become one node with 40 mentions, not
    40 separate near-duplicate nodes).

    This does not change what is stored/displayed - the first
    original-cased value seen is kept as the display value.
    """

    value = normalize_text(value)

    # Drop a trailing parenthetical annotation, e.g.
    # "Power Finance Corporation Limited (\"PFC\")" -> the
    # parenthetical is usually a defined-term alias, not part
    # of the entity's own name.
    value = re.sub(r"\s*\([^)]{0,120}\)\s*$", "", value)

    value = value.strip(" .,;:\"'")

    value = value.lower()

    # Collapse whitespace between single-letter initials, e.g.
    # "A. K. Capital..." and "A.K. Capital..." both occur in the
    # real PDF for the same entity - normalize both to the same
    # grouping key ("a.k. capital...") without touching the
    # display value shown to the user.
    value = re.sub(r"\.\s+(?=[a-z]\.)", ".", value)

    # Canonicalize the "Ltd"/"Ltd." abbreviation to "limited"
    # for grouping purposes only - confirmed against the real
    # PDF, where a rating agency's OWN annexure consistently
    # abbreviates its name as "CARE Ratings Ltd." throughout,
    # while the main prospectus body uses "CARE Ratings
    # Limited" - unmistakably the same organization, just an
    # abbreviation used inconsistently across different parts
    # of a 274-page document.
    value = re.sub(r"\bltd\.?$", "limited", value)

    return value


def slugify(value):
    value = normalize_entity_value(value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


# =========================================================
# ALIAS DETECTION (generic textual-evidence based merging)
#
# Only merges two extracted values when the SOURCE TEXT
# itself states they refer to the same organization - e.g.
# "CARE Ratings Limited (\"CARE\")" or a defined-terms table
# row like '"CRISIL"/ "CRISIL Ratings" CRISIL Ratings
# Limited'. Both patterns are standard legal-drafting/SEBI
# defined-terms conventions used across Indian prospectuses
# generally, not specific to this issuer. Two names are NEVER
# merged just because they look similar or because of outside
# knowledge about the real organizations - only textual
# alias-definition evidence triggers a merge.
# =========================================================

ALIAS_AFTER_NAME_PATTERN = re.compile(
    r'\(\s*["“\']([\w .&\'-]{1,40})["”\']\s*\)'
)

ALIAS_BEFORE_NAME_PATTERN = re.compile(
    r'((?:["“\']\s*[\w .&\'-]{1,40}?\s*["”\']\s*/?\s*){1,3})\s*$'
)


def build_alias_map(all_chunk_texts):
    """
    Scan every chunk's text for alias-definition patterns and
    return {normalized_alias: canonical_display_name}.

    Two shapes are recognized:
      1. "Full Company Name Limited (\"Alias\")"
      2. '"Alias1"/ "Alias2" Full Company Name Limited'

    Both require the FULL, suffix-bearing name to be found by
    the same company-name matcher used elsewhere, so an alias
    is only ever recorded when the document itself states the
    full name it refers to - never invented.
    """

    alias_map = {}

    for text in all_chunk_texts:

        for start, end, full_name in find_company_names_with_spans(text):

            # Shape 1: full name immediately followed by a
            # parenthetical quoted alias.
            after_match = ALIAS_AFTER_NAME_PATTERN.match(text, end)

            if after_match:

                alias = normalize_text(after_match.group(1))
                key = normalize_entity_value(alias)

                if key and key not in alias_map:
                    alias_map[key] = full_name

            # Shape 2: one or more quoted aliases immediately
            # before the full name.
            before_match = ALIAS_BEFORE_NAME_PATTERN.search(
                text[max(0, start - 80):start]
            )

            if before_match:

                for alias in re.findall(
                    r'["“\']([\w .&\'-]{1,40})["”\']', before_match.group(1)
                ):

                    alias = normalize_text(alias)
                    key = normalize_entity_value(alias)

                    if key and key not in alias_map:
                        alias_map[key] = full_name

    return alias_map


# =========================================================
# GENERIC STRUCTURAL PATTERNS
#
# These are shape-based, not name-based: they match the
# *structure* an entity name or fact takes in a prospectus,
# not a specific literal company/agency name. This is what
# makes the extractor issuer-agnostic.
# =========================================================

# Company-name detection uses a token-run approach rather
# than a single sprawling regex. A single lazy regex like
# "[A-Z]...{1,90}? followed by a suffix" is unsafe: given
# "Power Finance Corporation Limited", the word "Corporation"
# is itself a valid suffix, so a lazy match stops there and
# truncates before "Limited" (confirmed against the real PDF
# text). Walking whole whitespace-delimited tokens and only
# chaining together tokens that look like part of a proper
# noun (capitalized, or a small connector word) avoids that,
# and naturally stops at ordinary sentence filler ("was",
# "incorporated", "as", ...) instead of swallowing it.

# STRONG suffixes always end a company name at that word -
# "Limited"/"Ltd"/"LLP"/"LLC" essentially never have more of
# the same name after them.
#
# WEAK suffixes ("Corporation", "Corp", "Inc", "PLC", ...) are
# treated differently: a name like "Power Finance Corporation
# Limited" has "Corporation" sitting in the MIDDLE of the real
# name, with "Limited" as the true final suffix. Cutting at
# "Corporation" unconditionally truncates the name (confirmed
# against the real PDF text). So a weak suffix only ends the
# name if the *next* token isn't itself part of a continuing
# name - otherwise we keep accumulating and let the stronger
# suffix close it.
STRONG_SINGLE_SUFFIXES = {"limited", "ltd", "ltd.", "llp", "llc"}

WEAK_SINGLE_SUFFIXES = {
    "corporation", "corp", "corp.",
    "inc", "inc.", "plc", "ag", "n.v.",
}

TWO_WORD_SUFFIXES = {
    ("private", "limited"),
    ("pvt", "ltd"),
    ("pvt.", "ltd."),
    ("pvt", "ltd."),
    ("pvt.", "ltd"),
}

NAME_CONNECTORS = {"of", "and", "&"}

MAX_COMPANY_NAME_TOKENS = 8

# A legal-entity suffix ("Limited", "Corporation", ...) that is
# immediately preceded by a plain grammatical determiner is
# almost certainly NOT a company name - real company names
# essentially never end "...The Limited" or "...Our Limited".
# This specifically catches ALL-CAPS disclaimer boilerplate
# coincidentally containing the ordinary word "limited" as an
# adjective rather than a company suffix (confirmed against the
# real PDF: "THE LEAD MANAGERS ACCEPT NO RESPONSIBILITY, SAVE TO
# THE LIMITED EXTENT AS PROVIDED..." - every word here is
# capitalized because the whole disclaimer paragraph is
# typeset in capitals for emphasis, not because it's a proper
# noun; "LIMITED" is directly preceded by "THE").
DETERMINERS = {
    "the", "a", "an", "our", "its", "their", "his", "her",
    "this", "that", "these", "those", "any", "no",
}


QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"
FOOTNOTE_CHARS = "*\u2020\u2021"


def _clean_token(token):
    return token.strip(",;:" + QUOTE_CHARS + FOOTNOTE_CHARS + "()")


# Double-quote characters (straight or curly) mark a legal
# DEFINED TERM in these documents - e.g. '"Lead Managers" or
# "LMs" Nuvama Wealth Management Limited...'. A quoted term
# immediately before a real company name is a defined-term
# label, never part of that company's own name (confirmed
# against the real PDF: the token "LMs”" swept into "Nuvama
# Wealth Management Limited" as a garbage prefix). Only DOUBLE
# quotes trigger this - a single quote/apostrophe is left alone
# since it's more often a legitimate part of a name (Moody's).
DEFINED_TERM_QUOTE_CHARS = "\"\u201c\u201d"

# Bare (unquoted) self-referential defined terms. SEBI-style
# prospectuses define "the Company" / "the Issuer" / "the
# Promoter" as shorthand for the ISSUER itself, and then use
# them unquoted throughout the running text - e.g. "the Issue
# Agreement... entered between the Company and Nuvama Wealth
# Management Limited..." (confirmed against the real PDF).
# Treating these three words as never-nameable stops them from
# being swept into an unrelated THIRD PARTY's name as a prefix.
# Trade-off, stated plainly: a real organization whose own name
# happens to contain the bare word "Company" as an early word
# (rather than as part of a "... Limited"/"... LLP" suffix
# phrase) would not be captured. None of the entities actually
# present in this document have that shape, and it's a fairly
# rare naming pattern, so this is judged a good trade for
# eliminating a confirmed, real false-entity class.
SELF_REFERENCE_WORDS = {"company", "issuer", "promoter"}

# Common English words used as REPORT/DOCUMENT SECTION HEADINGS
# in credit-rating-agency annexures attached to Indian bond
# prospectuses (CRISIL, CARE, and ICRA all use similar
# boilerplate report templates - this is an industry-wide
# convention, not specific to this issuer). Confirmed against
# the real PDF: "About Crisil Ratings Limited (A subsidiary of
# Crisil Limited...)" and "DISCLAIMER STATEMENT OF CRISIL
# RATINGS LIMITED" both swept the heading word(s) into the
# company name as a garbage prefix, because every word in an
# ALL-CAPS or Title-Case heading is capitalized and there's no
# lowercase word to break the chain before the real name
# starts.
REPORT_HEADING_WORDS = {"about", "disclaimer", "statement", "website"}

# Common job-title words seen in signature-block / contact-
# details listings ("Rohit Arun Dhanuka\nManager\nCrisil
# Ratings Limited") - confirmed against the real PDF. A
# person's name is capitalized the same way a company name is,
# so there's no way to tell them apart directly; excluding the
# job-title word that sits between the name and the company
# breaks the chain before the person's name gets swept in.
JOB_TITLE_WORDS = {"manager", "director"}

# A 2-token candidate whose FIRST word is a generic corporate-
# structure word (rather than a distinctive brand/proper noun)
# is far more likely to be a fragment of some longer name that
# got cut off earlier (e.g. by a hyphenated word like
# "e-Governance" breaking the chain) than a genuine 2-word
# company name - confirmed against the real PDF: "National
# e-Governance Services Limited" got truncated to just
# "Services Limited" this way. Real short company names in
# this corpus (e.g. "Crisil Limited", "ICRA Limited") start
# with a distinctive brand name, never a bare structural word.
GENERIC_FIRST_WORDS = {
    "services", "consultancy", "group", "holdings",
    "enterprises", "industries",
}


def _ends_sentence(token):
    """
    A token ending in a literal period is treated as the end of
    a sentence/defined-term entry - a hard break in the
    nameable-token chain - UNLESS the word before the period is
    short enough to plausibly be an abbreviation/initial ("A.",
    "K.", "Co.") or is itself a known legal suffix ("Ltd.",
    "Corp.", "Inc.", already handled as suffixes elsewhere).

    Without this, two unrelated, back-to-back defined-term
    sentences that both happen to be built from capitalized
    words merge into one candidate (confirmed against the real
    PDF: "...the SEBI NCS Regulations.\\nEESL Energy Efficiency
    Services Limited." - a sentence ending in "Regulations."
    immediately followed by a completely unrelated company name
    - with nothing lowercase in between to signal the break).
    """

    if not token.endswith("."):
        return False

    before_period = _clean_token(token).rstrip(".")

    if len(before_period) <= 3:
        return False

    if before_period.lower() in STRONG_SINGLE_SUFFIXES | WEAK_SINGLE_SUFFIXES:
        return False

    return True


def _is_nameable_token(token):
    """A token that could plausibly be part of a proper noun:
    starts with a capital letter and contains no digits, or is
    a small connector word/ampersand commonly found inside
    company names.

    Digit-bearing tokens are deliberately excluded even though
    they're often capitalized (e.g. "CIN:", ISIN codes like
    "INE001A01036", registration numbers) - real company names
    essentially never contain digits, and without this check
    those codes chain onto a nearby real company name and
    produce garbage captures (confirmed against the real PDF,
    where a CIN/ISIN block sits directly next to "Beacon
    Trusteeship Limited").
    """

    if any(char in DEFINED_TERM_QUOTE_CHARS for char in token):
        return False

    if _ends_sentence(token):
        return False

    cleaned = _clean_token(token)

    if not cleaned:
        return False

    if cleaned.lower() in NAME_CONNECTORS:
        return True

    if cleaned.lower() in SELF_REFERENCE_WORDS:
        return False

    if cleaned.lower() in REPORT_HEADING_WORDS:
        return False

    if cleaned.lower() in JOB_TITLE_WORDS:
        return False

    if any(char.isdigit() for char in cleaned):
        return False

    return cleaned[0].isupper()


def _strip_duplicated_alias_prefix(tokens_slice):
    """
    Defined-terms tables sometimes render as '"Alias" Full
    Name' with the quote marks silently dropped by the PDF
    text layer, leaving the alias sitting bare and
    unpunctuated right before the real name it aliases - and
    when the alias is (or starts with) the same word(s) as the
    real name, that produces a literally duplicated leading
    prefix (confirmed against the real PDF: "ICRA ICRA
    Limited", "A. K. Capital A. K. Capital Services Limited").
    If the first half of the token slice is an exact repeat of
    the immediately following block, the duplicated leading
    copy is dropped.
    """

    n = len(tokens_slice)

    for prefix_len in range(n // 2, 0, -1):

        first = [
            _clean_token(t).rstrip(".").lower()
            for t in tokens_slice[:prefix_len]
        ]
        second = [
            _clean_token(t).rstrip(".").lower()
            for t in tokens_slice[prefix_len:2 * prefix_len]
        ]

        if first and first == second:
            return tokens_slice[prefix_len:]

    return tokens_slice


def find_company_names_with_spans(text):
    """
    Same matching logic as find_company_names_in_text, but also
    returns each candidate's character offsets in `text`. Needed
    so callers can measure distance to nearby label keywords
    and disambiguate which label a name actually belongs to.

    Returns a list of (start, end, name) tuples, in the order
    found, with duplicates NOT removed (callers that want
    per-chunk deduping do it themselves after label resolution).
    """

    tokens = text.split()
    n = len(tokens)

    # Precompute the character offset of each token's start in
    # `text`, since tokens came from a plain .split().
    offsets = []
    cursor = 0

    for token in tokens:
        start = text.index(token, cursor)
        offsets.append(start)
        cursor = start + len(token)

    results = []

    segment_start = None
    i = 0

    while i < n:

        token = tokens[i]
        cleaned = _clean_token(token).rstrip(".")
        cleaned_lower = cleaned.lower()

        if not _is_nameable_token(token):
            segment_start = None
            i += 1
            continue

        if segment_start is None:
            segment_start = i

        suffix_end = None
        next_is_nameable = (
            i + 1 < n and _is_nameable_token(tokens[i + 1])
        )

        if i + 1 < n:

            next_cleaned = _clean_token(tokens[i + 1]).rstrip(".").lower()

            if (cleaned_lower, next_cleaned) in TWO_WORD_SUFFIXES:
                suffix_end = i + 1

        if suffix_end is None and cleaned_lower in STRONG_SINGLE_SUFFIXES:
            suffix_end = i

        if (
            suffix_end is None
            and cleaned_lower in WEAK_SINGLE_SUFFIXES
            and not next_is_nameable
        ):
            suffix_end = i

        if suffix_end is not None:

            span_len = suffix_end - segment_start + 1

            preceded_by_determiner = (
                i - 1 >= segment_start
                and _clean_token(tokens[i - 1]).rstrip(".").lower()
                in DETERMINERS
            )

            if (
                2 <= span_len <= MAX_COMPANY_NAME_TOKENS
                and not preceded_by_determiner
            ):

                token_slice = _strip_duplicated_alias_prefix(
                    tokens[segment_start:suffix_end + 1]
                )

                first_word_is_generic = (
                    len(token_slice) == 2
                    and _clean_token(token_slice[0]).rstrip(".").lower()
                    in GENERIC_FIRST_WORDS
                )

                candidate = (
                    "" if first_word_is_generic
                    else " ".join(token_slice)
                )

                candidate = candidate.strip(
                    " ,.;:" + QUOTE_CHARS + FOOTNOTE_CHARS + "()"
                )

                candidate = re.sub(
                    r"^(?:and|of|&)\s+", "", candidate, flags=re.IGNORECASE
                )

                if candidate:

                    effective_start = (
                        segment_start
                        + (suffix_end + 1 - segment_start - len(token_slice))
                    )

                    span_start = offsets[effective_start]
                    span_end = (
                        offsets[suffix_end] + len(tokens[suffix_end])
                    )

                    results.append((span_start, span_end, candidate))

            segment_start = None
            i = suffix_end + 1
            continue

        i += 1

    return results


def find_company_names_in_text(text):
    """
    Convenience wrapper over find_company_names_with_spans that
    drops offsets and deduplicates - used where only the names
    themselves matter (offsets aren't needed).
    """

    seen = set()
    unique_names = []

    for _, _, name in find_company_names_with_spans(text):

        key = normalize_entity_value(name)

        if key in seen:
            continue

        seen.add(key)
        unique_names.append(name)

    return unique_names

# ISIN: 2-letter country code + 9 alphanumeric + 1 check
# digit. Universal bond identifier format, not tied to any
# issuer or market.
ISIN_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")

# A broad (but still non-exhaustive) vocabulary of credit
# rating agency names. Organization names are proper nouns
# and can't be inferred from pure regex shape the way a
# company suffix can, so this list is used only as a
# secondary/fallback signal used as a validator (see
# _looks_like_rating_agency below), not as the primary
# detection mechanism.
KNOWN_RATING_AGENCY_HINTS = [
    "CRISIL", "ICRA", "CARE", "India Ratings", "IND-RA",
    "Brickwork", "Acuite", "ACUITE", "SMERA", "INFOMERICS",
    "Moody's", "Fitch", "S&P", "Standard & Poor's",
]

CURRENCY_AMOUNT = (
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*"
    r"(?:crore|lakh|million|billion)?"
)


# =========================================================
# UNIFIED NEAREST-LABEL ENTITY RESOLUTION
#
# Earlier versions of this extractor ran each field
# (Issuer/Promoter/Debenture Trustee/Lead Manager/Credit
# Rating Agency) as an INDEPENDENT window search. That is
# unsafe: a dense cover-page directory listing (confirmed in
# the real PDF - "DEBENTURE TRUSTEE CREDIT RATING AGENCIES
# STATUTORY AUDITORS ... Beacon Trusteeship Limited Crisil
# Ratings Limited CARE Ratings Limited ICRA Limited") puts
# several different labels and several different company
# names within reach of EVERY label's window, so more than one
# label independently claims the same name, producing false
# relationships like "Issuer HAS_DEBENTURE_TRUSTEE -> CARE
# Ratings Limited".
#
# The fix here: find every company name ONCE, find every label
# keyword occurrence ONCE, and assign each name to whichever
# label occurrence is textually closest to it - so a name is
# only ever given a single type per occurrence, rather than
# being claimed by every label within range.
#
# This measurably improves things but does not fully solve
# multi-column cover-page directories: confirmed against the
# real PDF, raw character distance in the block above actually
# favors the WRONG label for some names, because pdfplumber's
# text linearization does not reliably preserve left-to-right
# column order for that kind of layout, and the true column
# geometry is not available at this stage (this script
# consumes already-chunked plain text, not the PDF itself).
# Where that residual ambiguity exists for Credit Rating
# Agency specifically, a secondary vocabulary-based check
# (see _looks_like_rating_agency) is used to reject clearly
# wrong matches; there's no equally reliable secondary check
# available for the other label types, so a small amount of
# residual mislabeling there remains a known limitation - see
# the module docstring / final report.
# =========================================================

LABEL_TYPES = {
    "Issuer": [r'["“\']Issuer["”\']', r"incorporated as"],
    "Promoter": [r'["“\']Promoter["”\']'],
    "Debenture Trustee": [r"Debenture\s+Trustee"],
    "Lead Manager": [r"Lead\s+Managers?"],
    "Credit Rating Agency": [
        r"Credit Rating Agenc(?:y|ies)", r"\brat(?:ed|ing)\b"
    ],
}

# Other standard SEBI-prescribed prospectus cover-page roles
# that this extractor deliberately does NOT model as entity
# types. These matter anyway: a company name sitting next to
# one of these (e.g. "REGISTRAR TO THE ISSUE KFIN Technologies
# Limited") must NOT silently fall through to whichever
# TRACKED label happens to be next closest ("Lead Manager", in
# the real PDF) - that produced a genuine false relationship
# (confirmed: KFIN Technologies Limited is the Registrar, not
# a Lead Manager). Treating these as a distinct "ignore" zone
# means a name found near one of them is dropped instead of
# mis-attributed. This list is the same kind of generic,
# industry-wide convention as LABEL_TYPES - not specific to
# this issuer.
IGNORE_LABEL_PATTERNS = [
    r"Registrar(?:\s+to\s+the\s+Issue)?",
    r"Statutory\s+Auditors?",
    r"Bankers?\s+to\s+the\s+(?:Issue|Company)",
    r"Consortium\s+Members?",
    r"(?:Escrow\s+Collection|Public\s+Issue\s+Account|Refund|Sponsor)\s+Bank",
    r"Legal\s+(?:Counsel|Advisors?)",
]

MAX_LABEL_ASSOCIATION_DISTANCE = 180

# In "Label: value1, value2, value3 and value4" style listings
# (confirmed in the real PDF: "Lead Managers Nuvama Wealth
# Management Limited, A. K. Capital Services Limited, Tipsons
# Consultancy Services Private Limited and Trust Investment
# Advisors Private Limited Debenture Trustee Beacon
# Trusteeship Limited"), the LAST item(s) in the list can sit
# textually closer to the NEXT different label than to their
# own - pure nearest-distance would misattribute "Tipsons..."
# and "Trust Investment..." to "Debenture Trustee" instead of
# "Lead Manager". Since "one label introduces a list of
# values" is the dominant convention in these documents (far
# more common than a value appearing before its own label),
# a label that PRECEDES a name is discounted relative to one
# that follows it, so the list-introducing label wins unless
# a following label is dramatically closer. This is a general
# document-convention judgment, not tuned to any one issuer's
# specific wording.
PRECEDING_LABEL_DISCOUNT = 0.15


def _looks_like_rating_agency(name):
    """
    Secondary vocabulary check used ONLY to reject a
    Credit-Rating-Agency-labeled candidate that clearly isn't
    one (e.g. a debenture trustee's name that ended up
    nearest to a "Credit Rating Agencies" heading purely
    because of PDF text-linearization order, not because it
    actually is a rating agency). This is a sanity filter, not
    the primary detection mechanism - a name can still become
    a Credit Rating Agency entity via the ordinary nearest-
    label path without appearing here, as long as nothing
    contradicts it structurally elsewhere.
    """

    normalized = normalize_entity_value(name)

    return any(
        normalize_entity_value(hint) in normalized
        for hint in KNOWN_RATING_AGENCY_HINTS
    )


def _all_label_spans(text):
    """
    Collect every match span, from EVERY tracked and ignored
    label pattern, against the ORIGINAL text - never against
    a partially-masked copy. Matching against an
    already-modified copy is what caused an earlier bug: the
    short pattern for "rating" matched the middle word of
    "Credit Rating Agencies" first, which broke the literal
    text the longer, more specific pattern needed to match,
    leaving "Credit" and "Agencies" unmasked on either side
    (confirmed against the real PDF). Doing one collection
    pass over the untouched text and merging overlapping spans
    avoids that entirely, regardless of pattern order.

    Returns a list of (start, end, label_type_or_None) tuples;
    label_type is None for IGNORE_LABEL_PATTERNS matches.
    """

    spans = []

    for label_type, patterns in LABEL_TYPES.items():

        for pattern in patterns:

            for match in re.finditer(pattern, text, re.IGNORECASE):
                spans.append((match.start(), match.end(), label_type))

    for pattern in IGNORE_LABEL_PATTERNS:

        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((match.start(), match.end(), None))

    return spans


def _mask_label_keywords(text, spans):
    """
    Blank out every given (start, end, label_type) span with
    same-length '#' filler, WITHOUT shifting any character
    offsets. Overlapping spans are merged first so a
    character is never processed twice. Used only to stop
    label headings from being swept into an adjacent company
    name as a garbage prefix (confirmed against the real PDF
    for both "Credit Rating Agencies Crisil Ratings Limited"
    and "Debenture Trustee Beacon Trusteeship Limited").
    """

    if not spans:
        return text

    merged = []

    for start, end, _ in sorted(spans, key=lambda item: item[0]):

        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    chars = list(text)

    for start, end in merged:

        for i in range(start, end):

            if chars[i] != " ":
                chars[i] = "#"

    return "".join(chars)


def resolve_labeled_company_names(text):
    """
    Find every company-suffix-shaped name in `text`, and assign
    each one to whichever tracked label keyword occurrence is
    textually nearest to it (within MAX_LABEL_ASSOCIATION_DISTANCE).
    A name whose nearest label is an IGNORE_LABEL_PATTERNS match
    (a real prospectus role this extractor doesn't model, e.g.
    "Registrar to the Issue") is dropped rather than falling
    through to the next-nearest TRACKED label - that fallback is
    exactly what mis-attributed a Registrar's name to "Lead
    Manager" before this fix.

    Returns a list of {"entity_type", "value"} dicts.
    """

    label_spans = _all_label_spans(text)

    if not label_spans:
        return []

    label_positions = [
        (label_type, (start + end) / 2)
        for start, end, label_type in label_spans
    ]

    # Company-name detection runs on a MASKED copy (every
    # tracked AND ignored label heading blanked out) so a
    # heading can't be swept into an adjacent name as a
    # garbage prefix. Label-distance resolution still uses the
    # ORIGINAL text's label positions, computed above.
    masked_text = _mask_label_keywords(text, label_spans)

    results = []

    for start, end, name in find_company_names_with_spans(masked_text):

        center = (start + end) / 2

        nearest_type = None
        nearest_effective_distance = None
        nearest_raw_distance = None

        for label_type, midpoint in label_positions:

            raw_distance = abs(center - midpoint)

            if midpoint <= center:
                effective_distance = raw_distance * PRECEDING_LABEL_DISCOUNT
            else:
                effective_distance = raw_distance

            if (
                nearest_effective_distance is None
                or effective_distance < nearest_effective_distance
            ):
                nearest_effective_distance = effective_distance
                nearest_raw_distance = raw_distance
                nearest_type = label_type

        if nearest_type is None:
            continue

        if nearest_raw_distance > MAX_LABEL_ASSOCIATION_DISTANCE:
            continue

        if (
            nearest_type == "Credit Rating Agency"
            and not _looks_like_rating_agency(name)
        ):
            continue

        results.append({"entity_type": nearest_type, "value": name})

    return results


# =========================================================
# PER-CHUNK ENTITY EXTRACTION
#
# Every function below returns a list of
# {"entity_type", "value"} dicts found in ONE chunk of text.
# Nothing here is invented: a value is only returned if a
# matching pattern is literally present in the chunk text.
# =========================================================

def extract_labeled_entities(text):
    """
    Single entry point for all company-name-shaped entity
    types (Issuer, Promoter, Debenture Trustee, Lead Manager,
    Credit Rating Agency). Replaces what used to be five
    independent window searches with one pass that resolves
    each company name to its single nearest label - see the
    "UNIFIED NEAREST-LABEL ENTITY RESOLUTION" section above for
    why that matters.
    """

    seen_per_type = {}
    results = []

    for entity in resolve_labeled_company_names(text):

        key = (entity["entity_type"], normalize_entity_value(entity["value"]))

        if key in seen_per_type:
            continue

        seen_per_type[key] = True
        results.append(entity)

    return results


def extract_credit_rating_grade(text):
    """
    The rating grade itself, e.g. "Crisil AAA/Stable" -
    captured generically as a short quoted token following the
    word "rated", regardless of which agency issued it. This is
    independent of the company-name resolution above (it's a
    quoted value, not a company name, so it isn't subject to
    the same cross-label contamination risk).
    """

    entities = []

    grade_pattern = re.compile(
        r"\brated[^\"'“”‘’]{0,40}[\"'“‘](.{3,60}?)[\"'”’]",
        re.IGNORECASE
    )

    for match in grade_pattern.finditer(text):

        entities.append({
            "entity_type": "Credit Rating Grade",
            "value": normalize_text(match.group(1)),
        })

    return entities


def extract_base_issue_size(text):

    match = re.search(
        rf"Base Issue Size\s*(?:[:\-]\s*)?({CURRENCY_AMOUNT})",
        text,
        re.IGNORECASE
    )

    if not match:
        return []

    return [{
        "entity_type": "Base Issue Size",
        "value": normalize_text(match.group(1)),
    }]


def extract_face_value(text):

    match = re.search(
        r"face value\s*(?:of|:)?\s*(₹?\s*[\d,]+)",
        text,
        re.IGNORECASE
    )

    if not match:
        return []

    return [{
        "entity_type": "Face Value",
        "value": normalize_text(match.group(1)),
    }]


def extract_coupon(text):

    patterns = [
        r"(?:Coupon|Interest Rate)[^%]{0,100}?(\d+(?:\.\d+)?\s*%)",
        r"(\d+(?:\.\d+)?\s*%)[^%]{0,100}?(?:Coupon|Interest Rate)",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return [{
                "entity_type": "Coupon / Interest Rate",
                "value": normalize_text(match.group(1)),
            }]

    return []


def extract_security_cover(text):

    match = re.search(
        r"(?:Minimum\s+)?Security Cover.{0,150}?(\d+(?:\.\d+)?\s*%)",
        text,
        re.IGNORECASE
    )

    if not match:
        return []

    return [{
        "entity_type": "Security Cover",
        "value": normalize_text(match.group(1)),
    }]


def extract_maturity(text):

    month_names = (
        r"(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)"
    )

    # "Asset Liability Maturity" / "Residual Maturity Profile"
    # are standard risk-disclosure headings in NBFC/bond
    # prospectuses describing the ISSUER's own balance-sheet
    # tenor buckets - a completely different concept from the
    # NCD's own redemption term, but textually indistinguishable
    # from it by the word "Maturity" alone (confirmed against
    # the real PDF: "The Asset Liability Maturity (ALM) profile
    # ... had few mismatches in some buckets upto 1 year").
    # Excluding matches immediately preceded by "Liability" or
    # followed by "Profile" filters these out.
    patterns = [
        rf"(?:Maturity Date|Redemption Date).{{0,150}}?"
        rf"({month_names}\s+\d{{1,2}},?\s+\d{{4}})",

        r"(?<!Liability )(?:Maturity|Redemption)(?!\s+Profile)"
        r".{0,100}?(\d+\s*(?:years?|months?))",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return [{
                "entity_type": "Maturity / Redemption",
                "value": normalize_text(match.group(1)),
            }]

    return []


def extract_isin(text):

    return [
        {"entity_type": "ISIN", "value": match.group(1)}
        for match in ISIN_PATTERN.finditer(text)
    ]


ENTITY_EXTRACTORS = [
    extract_labeled_entities,
    extract_credit_rating_grade,
    extract_base_issue_size,
    extract_face_value,
    extract_coupon,
    extract_security_cover,
    extract_maturity,
    extract_isin,
]


def extract_entities_from_chunk(chunk):
    """
    Run every extractor against one chunk's text and attach
    chunk-level provenance to each result.
    """

    text = normalize_text(chunk.get("text", ""))

    if not text:
        return []

    chunk_id = chunk.get("chunk_id")
    start_page = chunk.get("start_page")
    end_page = chunk.get("end_page")

    results = []

    for extractor in ENTITY_EXTRACTORS:

        for entity in extractor(text):

            results.append({
                "entity_type": entity["entity_type"],
                "value": entity["value"],
                "source_chunk": chunk_id,
                "start_page": start_page,
                "end_page": end_page,
            })

    return results


# =========================================================
# RELATIONSHIP PREDICATES
# =========================================================

RELATIONSHIP_PREDICATES = {
    "Promoter": "HAS_PROMOTER",
    "Debenture Trustee": "HAS_DEBENTURE_TRUSTEE",
    "Lead Manager": "HAS_LEAD_MANAGER",
    "Credit Rating Grade": "HAS_CREDIT_RATING",
    "Credit Rating Agency": "RATED_BY",
    "Base Issue Size": "HAS_BASE_ISSUE_SIZE",
    "ISIN": "HAS_ISIN",
}

# Face Value, Coupon/Interest Rate, Security Cover, and
# Maturity/Redemption are deliberately NOT promoted to
# Issuer-level relationships, even though they're still
# extracted as entities (with provenance). A bond issuer
# commonly has SEVERAL NCD series, each with its own coupon,
# face value, security cover, and maturity - confirmed by the
# separate table-extraction step in this same project, which
# found 5 series on this document with 5 different coupon/
# maturity combinations. Asserting a single "Issuer HAS_COUPON
# X%" fact from whichever value a text chunk happens to
# mention first would misrepresent a multi-series bond as
# having one uniform term. The per-series JSON produced by the
# table-extraction step is the correct source of truth for
# these figures; this relationship set intentionally matches
# that boundary.


# =========================================================
# ENTITY GROUPING (dedup into graph-ready nodes)
# =========================================================

def group_entities(raw_entities, alias_map=None):
    """
    Group raw per-chunk entity hits into one record per
    real-world entity (same entity_type + normalized value),
    each carrying a list of mentions (chunk/page provenance).

    If alias_map is given, any entity whose normalized value is
    a known alias is grouped under its canonical full name
    instead (see build_alias_map) - textual-evidence-only,
    never a blind similarity merge.

    This is the shape a Graph DB loader wants: MERGE one node
    per (entity_type, normalized value), then attach a
    MENTIONED_IN edge per mention - not one exploded node per
    chunk occurrence.

    Returns:
        {
            (entity_type, normalized_value): {
                "entity_id": ...,
                "entity_type": ...,
                "value": ...,          # first original-cased value seen
                "mentions": [ {source_chunk, start_page, end_page}, ... ]
            }
        }
    """

    alias_map = alias_map or {}
    grouped = {}

    for entity in raw_entities:

        display_value = entity["value"]
        normalized_value = normalize_entity_value(display_value)

        canonical_full_name = alias_map.get(normalized_value)

        if canonical_full_name:
            display_value = canonical_full_name
            normalized_value = normalize_entity_value(canonical_full_name)

        key = (entity["entity_type"], normalized_value)

        mention = {
            "source_chunk": entity["source_chunk"],
            "start_page": entity["start_page"],
            "end_page": entity["end_page"],
        }

        if key not in grouped:

            entity_id = f"{slugify(entity['entity_type'])}:{slugify(display_value)}"

            grouped[key] = {
                "entity_id": entity_id,
                "entity_type": entity["entity_type"],
                "value": display_value,
                "mentions": [],
            }

        # Dedupe identical mentions (same chunk seen twice).
        existing_mentions = grouped[key]["mentions"]

        if mention not in existing_mentions:
            existing_mentions.append(mention)

    return grouped


def pick_canonical_entity_of_type(grouped_entities, entity_type):
    """
    Generalized version of pick_canonical_issuer's "most
    mentions, shorter value wins ties" rule, for any entity
    type that should structurally be single-valued for a given
    document. Returns None if no entity of that type exists.
    """

    candidates = [
        entity
        for entity in grouped_entities.values()
        if entity["entity_type"] == entity_type
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda entity: (len(entity["mentions"]), -len(entity["value"]))
    )


def pick_canonical_issuer(grouped_entities):
    """
    Pick the Issuer entity with the most mentions across the
    document as the canonical subject for co-occurrence
    relationships. Returns None if no Issuer entity was found
    anywhere - relationships are simply not generated in that
    case, rather than guessing at a subject.

    Ties are broken in favor of the SHORTER value. This is a
    deliberate hedge against repeated PDF page headers/footers
    running directly into the real company name with no
    lowercase word breaking them apart (e.g. a running header
    "Tranche I Prospectus) POWER FINANCE CORPORATION LIMITED"
    next to a defined-term marker) - the noisy, longer variant
    can still end up as its own entity, but a shorter, cleaner
    match is preferred as the canonical one relationships
    attach to whenever mention counts are equal.
    """

    return pick_canonical_entity_of_type(grouped_entities, "Issuer")


# A bond issue has exactly one Debenture Trustee under SEBI NCS
# Regulations - this is an industry-wide regulatory fact true
# for any Indian NCD issuer, not something specific to this
# document. Restricting HAS_DEBENTURE_TRUSTEE to a single
# canonical trustee (same "most mentions, shorter value" rule
# as the canonical Issuer) is defense-in-depth: even if some
# other document's text produces a stray second Debenture-
# Trustee-typed entity from noisy layout, it can't turn into a
# second false relationship - it would just remain an unused,
# visible entity in the entities list.
SINGLE_VALUED_RELATIONSHIP_TYPES = {"Debenture Trustee"}


# =========================================================
# RELATIONSHIP EXTRACTION (co-occurrence based)
#
# IMPORTANT: this is intentionally a lightweight,
# co-occurrence-based signal, not a deep relation-extraction
# model. A relationship is only ever created when both
# entities were literally found together in the same source
# chunk - it is a grounded structural fact ("these two things
# were mentioned in the same passage"), never an inference
# about meaning that isn't backed by the text.
# =========================================================

def extract_relationships(
    document_chunks, grouped_entities, raw_entities_by_chunk, alias_map=None
):

    alias_map = alias_map or {}

    canonical_issuer = pick_canonical_issuer(grouped_entities)

    if canonical_issuer is None:
        return []

    issuer_id = canonical_issuer["entity_id"]

    # For entity types that should be single-valued for the
    # whole document (see SINGLE_VALUED_RELATIONSHIP_TYPES),
    # only the canonical entity's id is allowed to become a
    # relationship object - any other same-type entity is left
    # in the entities list but never linked to the issuer.
    canonical_ids_by_type = {
        entity_type: pick_canonical_entity_of_type(grouped_entities, entity_type)
        for entity_type in SINGLE_VALUED_RELATIONSHIP_TYPES
    }

    relationships = []
    seen = set()

    for chunk in document_chunks:

        chunk_id = chunk.get("chunk_id")
        chunk_entities = raw_entities_by_chunk.get(chunk_id, [])

        for entity in chunk_entities:

            if entity["entity_type"] not in RELATIONSHIP_PREDICATES:
                continue

            if entity["entity_type"] == "Issuer":
                continue

            normalized_value = normalize_entity_value(entity["value"])
            canonical_full_name = alias_map.get(normalized_value)

            if canonical_full_name:
                normalized_value = normalize_entity_value(canonical_full_name)

            key = (entity["entity_type"], normalized_value)

            object_entity = grouped_entities.get(key)

            if object_entity is None:
                continue

            if object_entity["entity_id"] == issuer_id:
                continue

            if entity["entity_type"] in SINGLE_VALUED_RELATIONSHIP_TYPES:

                canonical = canonical_ids_by_type.get(entity["entity_type"])

                if canonical is None or object_entity["entity_id"] != canonical["entity_id"]:
                    continue

            predicate = RELATIONSHIP_PREDICATES[entity["entity_type"]]

            dedup_key = (
                issuer_id,
                predicate,
                object_entity["entity_id"],
                chunk_id,
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            relationships.append({
                "subject_entity_id": issuer_id,
                "predicate": predicate,
                "object_entity_id": object_entity["entity_id"],
                "source_chunk": chunk_id,
                "start_page": entity["start_page"],
                "end_page": entity["end_page"],
            })

    return relationships


# =========================================================
# MAIN PIPELINE
# =========================================================

def extract_domain_knowledge(document):

    chunks = document.get("chunks", [])

    chunk_texts = [normalize_text(chunk.get("text", "")) for chunk in chunks]
    alias_map = build_alias_map(chunk_texts)

    raw_entities = []
    raw_entities_by_chunk = {}

    for chunk in chunks:

        chunk_entities = extract_entities_from_chunk(chunk)

        raw_entities.extend(chunk_entities)
        raw_entities_by_chunk[chunk.get("chunk_id")] = chunk_entities

    grouped = group_entities(raw_entities, alias_map)

    relationships = extract_relationships(
        chunks, grouped, raw_entities_by_chunk, alias_map
    )

    # Deterministic ordering: sort entities by type then value,
    # and sort each entity's mentions by chunk id. Sort
    # relationships by subject/predicate/object/chunk.
    entities = sorted(
        grouped.values(),
        key=lambda entity: (entity["entity_type"], entity["value"])
    )

    for entity in entities:
        entity["mentions"] = sorted(
            entity["mentions"],
            key=lambda mention: (
                str(mention.get("source_chunk")),
            )
        )

    relationships = sorted(
        relationships,
        key=lambda rel: (
            rel["subject_entity_id"],
            rel["predicate"],
            rel["object_entity_id"],
            str(rel["source_chunk"]),
        )
    )

    return entities, relationships


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract generic domain entities and co-occurrence "
            "relationships from a chunked bond/NCD prospectus, "
            "for later loading into a Graph DB."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the chunked document JSON "
             "(e.g. pfc_ncd_2026_chunks.json)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the domain knowledge JSON to"
    )

    args = parser.parse_args()

    print("Loading chunks...")

    document = load_json(args.input)

    chunk_count = len(document.get("chunks", []))

    print(f"Processing {chunk_count} chunks...")

    entities, relationships = extract_domain_knowledge(document)

    result = {
        "document": document.get("document"),
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entities": entities,
        "relationships": relationships,
    }

    save_json(result, args.output)

    print(f"Extracted {len(entities)} distinct domain entities.")
    print(f"Extracted {len(relationships)} co-occurrence relationships.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()