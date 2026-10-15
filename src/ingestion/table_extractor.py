import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import pdfplumber


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_label(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).replace("\n", " ")
    ).strip()


def clean_value(value):
    value = clean_label(value)
    return value if value else None


# =========================================================
# ROMAN NUMERALS / SERIES DETECTION
# =========================================================

ROMAN_PATTERN = r"[IVXLCDM]+"


def normalize_series_id(value):
    """
    Examples:
        I    -> I
        II*  -> II
        III  -> III
    """

    if not value:
        return None

    value = clean_label(value).upper()
    value = value.replace("*", "")
    value = re.sub(r"[^IVXLCDM]", "", value)

    if not value:
        return None

    if not re.fullmatch(ROMAN_PATTERN, value):
        return None

    return value


def is_series_cell(value):
    """
    Only accept standalone Roman numerals, optionally followed by
    footnote markers (one or more asterisks/daggers - a "default
    allocation" series confirmed in a real prospectus was marked
    "V**", which a single optional "*" doesn't cover).

    Valid:
        I
        II
        II*
        III
        V**

    Invalid:
        ISIN
        Category III
        Series III
    """

    if not value:
        return False

    text = clean_label(value)

    return bool(
        re.fullmatch(
            rf"{ROMAN_PATTERN}[*†‡]*",
            text,
            re.IGNORECASE
        )
    )


def row_contains_series_word(row):
    """
    A valid header row should explicitly contain
    the word 'Series'.
    """

    if not row:
        return False

    for cell in row:
        text = clean_label(cell).lower()

        if text == "series":
            return True

        if text.startswith("series "):
            return True

    return False


def detect_explicit_series_columns(table):
    """
    Find a row containing 'Series' and standalone
    Roman numeral series identifiers.

    Returns:

        (
            {"I": 4, "II": 7, "III": 10},   # column-index mapping
            header_row_index
        )

    The column-index mapping is retained for debug output
    and as a fallback signal. The header_row_index is the
    critical addition: it lets the caller go back to the
    *geometric* table (find_tables()) and read the header
    cells' actual bounding boxes, which is what makes
    group-aware, merge-safe extraction possible.
    """

    best_mapping = {}
    best_row_index = None

    for row_index, row in enumerate(table):

        if not row:
            continue

        if not row_contains_series_word(row):
            continue

        mapping = {}

        for col_index, cell in enumerate(row):

            if not is_series_cell(cell):
                continue

            series_id = normalize_series_id(cell)

            if series_id:
                mapping[series_id] = col_index

        if len(mapping) >= 2:

            positions = list(mapping.values())

            if len(positions) != len(set(positions)):
                continue

            if len(mapping) > len(best_mapping):
                best_mapping = mapping
                best_row_index = row_index

    return best_mapping, best_row_index


# =========================================================
# REFERENCE DETECTION
# =========================================================

REFERENCE_PATTERNS = re.compile(
    r"("
    r"please\s+(see|refer)"
    r"|kindly\s+(see|refer)"
    r"|refer\s+to"
    r"|as\s+(mentioned|specified|set\s+out|stated)"
    r"|see\s+[\"']?issue\s+structure"
    r"|see\s+(page|section|annexure|para)"
    r"|refer\s+(page|section|annexure|para)"
    r"|\bpage\s+no\.?\s*\d+"
    r")",
    re.IGNORECASE
)


def is_reference_text(value):

    if not value:
        return False

    text = clean_label(value).lower()

    return bool(
        REFERENCE_PATTERNS.search(text)
    )


# =========================================================
# TERM DETECTION (used for candidate-table scoring only)
# =========================================================

TERM_KEYWORDS = [
    "tenor",
    "frequency of interest payment",
    "frequency",
    "minimum application",
    "face value",
    "issue price",
    "coupon",
    "effective yield",
    "yield",
    "mode of interest payment",
    "amount on maturity",
    "maturity",
    "redemption",
    "nature of indebtedness",
    "put and call option",
    "put option",
    "call option",
]


def get_table_text(table):

    values = []

    for row in table:

        if not row:
            continue

        for cell in row:

            value = clean_label(cell)

            if value:
                values.append(value.lower())

    return " ".join(values)


def count_term_matches(table):

    text = get_table_text(table)

    matches = []

    for keyword in TERM_KEYWORDS:

        if keyword in text:
            matches.append(keyword)

    return matches


def looks_like_series_terms_table(table):
    """
    A candidate table must have:

    1. Explicit Series header row
    2. At least 2 series
    3. At least 1 recognizable bond/NCD term

    This keeps table *detection* generic and prevents false
    positives from unrelated tables elsewhere in the
    prospectus that merely happen to contain words like
    "Series" or "Coupon".
    """

    if not table:
        return False

    series_columns, _ = detect_explicit_series_columns(table)

    if len(series_columns) < 2:
        return False

    term_matches = count_term_matches(table)

    if len(term_matches) < 1:
        return False

    return True


def score_table(table):

    if not table:
        return 0

    score = 0

    series_columns, _ = detect_explicit_series_columns(table)
    term_matches = count_term_matches(table)

    score += len(series_columns) * 2
    score += len(term_matches)

    return score


# =========================================================
# GEOMETRY HELPERS
#
# This is the actual fix. pdfplumber's extract_tables()
# collapses the table into a flat text grid and throws away
# merge information: a merged cell simply appears once, and
# every other grid position it covers becomes None/blank.
#
# find_tables() keeps the underlying Table object, whose
# cells carry real bounding boxes (x0, top, x1, bottom).
# A merged cell's bounding box is *physically wider or
# taller* than a normal single cell - that's the structural
# proof of a merge, and it's what we use instead of guessing
# from blank/non-blank text.
# =========================================================

def build_series_ranges(geom_table, header_row_index, series_columns):
    """
    For each detected series, read the ACTUAL x0/x1 of its
    header cell from the geometric table. This gives the
    true physical column-group boundaries for that series,
    rather than an approximation based on column index
    (e.g. "next header column - 1").

    Returns:
        { "I": (x0, x1), "II": (x0, x1), ... }
    """

    ranges = {}

    if header_row_index is None:
        return ranges

    if header_row_index >= len(geom_table.rows):
        return ranges

    header_cells = geom_table.rows[header_row_index].cells

    for series_id, col_index in series_columns.items():

        if col_index >= len(header_cells):
            continue

        cell = header_cells[col_index]

        if cell is None:
            continue

        x0, top, x1, bottom = cell

        ranges[series_id] = (x0, x1)

    return ranges


def find_label_boundary_x(series_ranges):
    """
    Everything at or after this x-coordinate is "value
    space" (it belongs to some series' column). Everything
    entirely to the left of it is "label space".

    IMPORTANT: this must be a physical x-coordinate, not a
    column index. pdfplumber's column grid is not guaranteed
    to line up between the header row and data rows - in
    practice, a thin "spacer" column that sits by itself in
    the header row is often merged into the first data
    cell of each row instead (no vertical rule is drawn
    there in the data rows). Using a column-index cutoff
    would misclassify that merged spacer+value cell as
    "label space" purely because of where the header's
    Roman numeral happened to sit. Using the true x0 of the
    left-most series' header cell sidesteps that entirely:
    a cell only counts as a value cell if it *physically
    reaches into* the series column area.
    """

    if not series_ranges:
        return 0.0

    return min(x0 for x0, _ in series_ranges.values())


def extract_row_label(geom_row_cells, text_row, boundary_x):
    """
    The row label is the right-most non-empty cell whose
    bounding box sits entirely to the left of the value
    area (cell.x1 <= boundary_x). In these tables the label
    is a single wrapped cell (e.g. "Tenor", "Issue Price of
    NCDs (Rs/ NCD) for NCD Holders in Category III."), so we
    scan the label region right-to-left and take the first
    non-empty text we find whose geometry confirms it is
    genuinely in label space.
    """

    if not text_row:
        return ""

    num_cols = min(len(geom_row_cells), len(text_row))

    for col_index in range(num_cols - 1, -1, -1):

        cell = geom_row_cells[col_index]

        if cell is None:
            continue

        _, _, cell_x1, _ = cell

        if cell_x1 > boundary_x + 1e-6:
            # This cell reaches into value space - not a
            # label cell (or we've scanned past the label
            # region already).
            continue

        label = clean_label(text_row[col_index])

        if label:
            return label

    return ""


def build_row_label_ranges(geom_table, text_grid, boundary_x):
    """
    For every row, determine:
        - its label text
        - its own vertical extent (top, bottom)

    The vertical extent is taken from whichever cell in the
    label region actually carries geometry for that row. This
    is what lets us later detect when a VALUE cell's vertical
    span covers more than one row (i.e. a value that is
    shared across several category rows), because we compare
    the value cell's (top, bottom) against each row's own
    (top, bottom) midpoint - not against a fixed row height.
    """

    row_info = {}

    num_rows = min(len(geom_table.rows), len(text_grid))

    for row_index in range(num_rows):

        cells = geom_table.rows[row_index].cells

        label = extract_row_label(
            cells,
            text_grid[row_index],
            boundary_x
        )

        if not label:
            continue

        top = None
        bottom = None

        # Prefer geometry from cells that are genuinely in
        # label space.
        for cell in cells:

            if cell is None:
                continue

            cell_x0, cell_top, cell_x1, cell_bottom = cell

            if cell_x1 > boundary_x + 1e-6:
                continue

            if top is None or cell_top < top:
                top = cell_top

            if bottom is None or cell_bottom > bottom:
                bottom = cell_bottom

        # Fall back to any cell in the row if the label
        # region itself has no geometry for some reason.
        if top is None or bottom is None:

            for cell in cells:

                if cell is None:
                    continue

                _, cell_top, _, cell_bottom = cell

                if top is None or cell_top < top:
                    top = cell_top

                if bottom is None or cell_bottom > bottom:
                    bottom = cell_bottom

        if top is None or bottom is None:
            continue

        row_info[row_index] = {
            "label": label,
            "top": top,
            "bottom": bottom,
        }

    return row_info


def collect_value_cells(geom_table, text_grid, boundary_x):
    """
    Walk every grid position whose cell reaches into "value
    space" (cell.x1 > boundary_x) and collect the ones that
    carry BOTH geometry and non-empty, non-reference text.
    Each of these is one physical cell as printed in the PDF
    - possibly spanning multiple series and/or multiple rows
    if merged.

    Returns a list of dicts:
        {
            "row_index": int,   # topmost row this cell starts at
            "text": str,
            "x0": float, "x1": float,
            "top": float, "bottom": float,
        }
    """

    value_cells = []

    num_rows = min(len(geom_table.rows), len(text_grid))

    for row_index in range(num_rows):

        row_cells = geom_table.rows[row_index].cells
        text_row = text_grid[row_index]

        num_cols = min(len(row_cells), len(text_row))

        for col_index in range(num_cols):

            cell = row_cells[col_index]

            if cell is None:
                # Either genuinely empty, or absorbed into a
                # merged cell that starts at an earlier
                # row/column - in both cases there is nothing
                # new to read at this grid position.
                continue

            x0, top, x1, bottom = cell

            if x1 <= boundary_x + 1e-6:
                # Entirely in label space - not a value cell.
                continue

            text = clean_value(text_row[col_index])

            if not text:
                continue

            if is_reference_text(text):
                continue

            value_cells.append({
                "row_index": row_index,
                "text": text,
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
            })

    return value_cells


def match_series_for_cell(cell, series_ranges):
    """
    A series is considered covered by this value cell if the
    series' header-cell CENTER point falls inside the value
    cell's horizontal span. This is what correctly handles
    merged cells: a cell that is twice the normal width will
    contain two series' header centers, and both are matched.
    A normal single-width cell contains exactly one.
    """

    matched = []

    for series_id, (header_x0, header_x1) in series_ranges.items():

        header_center = (header_x0 + header_x1) / 2

        if cell["x0"] <= header_center <= cell["x1"]:
            matched.append(series_id)

    return matched


def match_rows_for_cell(cell, row_label_ranges):
    """
    A row is considered covered by this value cell if the
    row's OWN vertical center falls inside the value cell's
    vertical span. A normal single-row cell only contains its
    own row's center. A vertically-merged cell (e.g. one
    "Issue Price" value that applies across all three
    category rows because it doesn't actually vary by
    category) will contain multiple rows' centers.
    """

    matched = []

    for row_index, info in row_label_ranges.items():

        row_center = (info["top"] + info["bottom"]) / 2

        if cell["top"] <= row_center <= cell["bottom"]:
            matched.append(row_index)

    return matched


# =========================================================
# NUMBER CLEANING
# =========================================================

NULL_TOKENS = {
    "NA",
    "N/A",
    "N.A.",
    "N.A",
    "-",
    "—",
    "--",
}


def clean_percentage(value):

    if not value:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value)

    normalized = value_str.upper().replace(" ", "")

    if normalized in NULL_TOKENS:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*%", value_str)

    if match:
        return float(match.group(1))

    return None


def clean_amount(value):

    if not value:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value)

    normalized = value_str.upper().replace(" ", "")

    if normalized in NULL_TOKENS:
        return None

    value_str = (
        value_str
        .replace("₹", "")
        .replace(",", "")
        .replace("INR", "")
        .replace("Rs.", "")
        .replace("RS.", "")
        .strip()
    )

    match = re.search(r"(\d+(?:\.\d+)?)", value_str)

    if not match:
        return None

    try:
        return float(match.group(1))

    except ValueError:
        return None


# =========================================================
# CATEGORY DETECTION
# =========================================================

CATEGORY_NUMERAL_ORDER = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
CATEGORY_NUMERAL_TOKEN_PATTERN = re.compile(
    r"\b(" + "|".join(CATEGORY_NUMERAL_ORDER) + r")\b",
    re.IGNORECASE,
)


def extract_category(row_label):
    """
    A row label names whichever Investor Category/Categories its
    value applies to - anywhere from one ("Category IV") up to all
    of them sharing a single value ("... for NCD Holders in
    Category I, II, III & IV"), joined by ",", "&", "and", or a
    repeated "Category"/"Cat" prefix per numeral. Rather than
    matching known fixed combinations, this scans everything after
    the first "category"/"cat" keyword for standalone Roman-numeral
    words and joins whichever ones actually appear - so it covers
    any combination generically, not just the ones seen so far.
    """

    label = clean_label(row_label).lower()

    keyword_match = re.search(r"\bcat(?:egory)?\.?\s*", label)

    if not keyword_match:
        return None

    tail = label[keyword_match.end():]

    seen = []

    for token in CATEGORY_NUMERAL_TOKEN_PATTERN.findall(tail):

        numeral = token.upper()

        if numeral not in seen:
            seen.append(numeral)

    if not seen:
        return None

    seen.sort(key=lambda numeral: CATEGORY_NUMERAL_ORDER.index(numeral))

    return "_".join(seen)


# =========================================================
# SERIES INITIALIZATION
# =========================================================

def initialize_series(series_ids):

    series_data = {}

    for series_id in series_ids:

        series_data[series_id] = {

            "entity_type": "NCDSeries",

            "series_id": series_id,

            "tenor": None,

            "frequency": None,

            "minimum_application": None,

            "face_value_inr": None,

            "issue_price_by_category": {},

            "coupon_by_category": {},

            "effective_yield_by_category": {},

            "mode_of_interest_payment": None,

            "maturity_amount_by_category": {},

            "maturity_redemption": None,

            "nature_of_indebtedness": None,

            "put_call_option": None,

            "provenance": {
                "pages": []
            }
        }

    return series_data


def add_provenance(series_data, series_id, page_number):

    if series_id not in series_data:
        return

    pages = series_data[series_id]["provenance"]["pages"]

    if page_number not in pages:
        pages.append(page_number)


# =========================================================
# FIELD ROUTING
#
# Applied once per (row, series) pair that a value cell was
# structurally matched to. This is the same field -> schema
# mapping the original code used, just invoked per single
# resolved value instead of per whole-row dict, since a
# single physical cell may now resolve to several
# (row, series) pairs when it is merged.
# =========================================================

def route_field(series_data, label, series_id, value, page_number):

    if series_id not in series_data:
        return

    label_lower = label.lower()

    if "tenor" in label_lower:
        series_data[series_id]["tenor"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if (
        "frequency" in label_lower
        and ("interest" in label_lower or "payment" in label_lower)
    ):
        series_data[series_id]["frequency"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if "minimum" in label_lower and "application" in label_lower:
        series_data[series_id]["minimum_application"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if "face value" in label_lower:
        amount = clean_amount(value)
        if amount is not None:
            series_data[series_id]["face_value_inr"] = amount
            add_provenance(series_data, series_id, page_number)
        return

    if (
        "issue price" in label_lower
        or ("issue" in label_lower and "price" in label_lower)
    ):
        category = extract_category(label) or "default"
        amount = clean_amount(value)
        if amount is not None:
            series_data[series_id]["issue_price_by_category"][category] = amount
            add_provenance(series_data, series_id, page_number)
        return

    if "coupon" in label_lower:
        category = extract_category(label) or "default"
        percentage = clean_percentage(value)
        if percentage is not None:
            series_data[series_id]["coupon_by_category"][category] = percentage
            add_provenance(series_data, series_id, page_number)
        return

    if "effective" in label_lower and "yield" in label_lower:
        category = extract_category(label) or "default"
        percentage = clean_percentage(value)
        if percentage is not None:
            series_data[series_id]["effective_yield_by_category"][category] = percentage
            add_provenance(series_data, series_id, page_number)
        return

    if "mode" in label_lower and "interest" in label_lower:
        series_data[series_id]["mode_of_interest_payment"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if "amount" in label_lower and "maturity" in label_lower:
        category = extract_category(label) or "default"
        amount = clean_amount(value)
        if amount is not None:
            series_data[series_id]["maturity_amount_by_category"][category] = amount
            add_provenance(series_data, series_id, page_number)
        return

    if "maturity" in label_lower and "redemption" in label_lower:
        series_data[series_id]["maturity_redemption"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if "nature" in label_lower and "indebtedness" in label_lower:
        series_data[series_id]["nature_of_indebtedness"] = value
        add_provenance(series_data, series_id, page_number)
        return

    if "put" in label_lower and "call" in label_lower:
        series_data[series_id]["put_call_option"] = value
        add_provenance(series_data, series_id, page_number)
        return


# =========================================================
# MERGE SERIES DATA (across tables/pages for same document)
# =========================================================

def merge_series_data(destination, source):

    for series_id, source_data in source.items():

        if series_id not in destination:
            destination[series_id] = source_data
            continue

        destination_data = destination[series_id]

        for key, value in source_data.items():

            if key in {"entity_type", "series_id", "provenance"}:
                continue

            if isinstance(value, dict):
                destination_data[key].update(value)

            elif destination_data.get(key) is None and value is not None:
                destination_data[key] = value

        destination_pages = destination_data["provenance"]["pages"]
        source_pages = source_data["provenance"]["pages"]

        destination_data["provenance"]["pages"] = sorted(
            set(destination_pages + source_pages)
        )


# =========================================================
# MAIN EXTRACTION
# =========================================================

def extract_series_terms(pdf_path, verbose=True):

    all_series_data = {}
    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            try:
                text_tables = page.extract_tables()
            except Exception as exc:
                print(
                    f"Warning: could not extract tables (text) "
                    f"from page {page_number}: {exc}"
                )
                continue

            if not text_tables:
                continue

            try:
                geom_tables = page.find_tables()
            except Exception as exc:
                print(
                    f"Warning: could not extract tables (geometry) "
                    f"from page {page_number}: {exc}"
                )
                continue

            # text_tables and geom_tables come from the same
            # underlying table finder with default settings, so
            # they should correspond 1:1 in the same order. We
            # verify dimensions before trusting that pairing -
            # if they disagree for some page, we skip geometric
            # extraction for that page's table rather than risk
            # silently misaligning cells.
            if len(text_tables) != len(geom_tables):
                print(
                    f"Warning: text/geometry table count mismatch "
                    f"on page {page_number} "
                    f"({len(text_tables)} vs {len(geom_tables)}). "
                    f"Skipping geometric extraction for this page."
                )
                continue

            for table_index, table in enumerate(text_tables):

                if not table:
                    continue

                if not looks_like_series_terms_table(table):
                    continue

                geom_table = geom_tables[table_index]

                if len(geom_table.rows) != len(table):
                    print(
                        f"Warning: row-count mismatch for table "
                        f"{table_index} on page {page_number}. "
                        f"Skipping this table."
                    )
                    continue

                series_columns, header_row_index = (
                    detect_explicit_series_columns(table)
                )

                if len(series_columns) < 2 or header_row_index is None:
                    continue

                series_ids = list(series_columns.keys())

                series_ranges = build_series_ranges(
                    geom_table,
                    header_row_index,
                    series_columns
                )

                if len(series_ranges) < 2:
                    print(
                        f"Warning: could not resolve geometric "
                        f"series ranges on page {page_number}, "
                        f"table {table_index}. Skipping."
                    )
                    continue

                boundary_x = find_label_boundary_x(series_ranges)

                row_label_ranges = build_row_label_ranges(
                    geom_table,
                    table,
                    boundary_x
                )

                value_cells = collect_value_cells(
                    geom_table,
                    table,
                    boundary_x
                )

                current_series_data = initialize_series(series_ids)

                row_lengths = [len(row) for row in table if row]
                table_width = max(row_lengths) if row_lengths else 0

                if verbose:
                    print()
                    print(
                        f"Processing table on page {page_number}, "
                        f"index {table_index}"
                    )
                    print(f"Confirmed series: {series_ids}")
                    print(f"Series columns (index): {series_columns}")
                    print(
                        "Series ranges (x0, x1): "
                        + str({
                            sid: (round(r[0], 1), round(r[1], 1))
                            for sid, r in series_ranges.items()
                        })
                    )
                    print(f"Label boundary x-coordinate: {round(boundary_x, 1)}")
                    print(
                        f"Table dimensions: {len(table)} rows x "
                        f"{table_width} cols"
                    )

                # -----------------------------------------------
                # ASSIGN EVERY VALUE CELL TO ITS TRUE (row, series)
                # PAIRS, DERIVED FROM GEOMETRY - NOT ASSUMED.
                # -----------------------------------------------

                for cell in value_cells:

                    matched_series = match_series_for_cell(
                        cell, series_ranges
                    )

                    if not matched_series:
                        continue

                    matched_rows = match_rows_for_cell(
                        cell, row_label_ranges
                    )

                    if not matched_rows:
                        continue

                    if verbose and (
                        len(matched_series) > 1
                        or len(matched_rows) > 1
                    ):
                        row_labels = [
                            row_label_ranges[r]["label"]
                            for r in matched_rows
                        ]
                        print(
                            f"  Merged cell '{cell['text']}' -> "
                            f"series {matched_series}, "
                            f"rows {row_labels}"
                        )

                    for row_index in matched_rows:

                        label = row_label_ranges[row_index]["label"]

                        for series_id in matched_series:

                            route_field(
                                current_series_data,
                                label,
                                series_id,
                                cell["text"],
                                page_number
                            )

                merge_series_data(all_series_data, current_series_data)

                tables_found.append({
                    "page": page_number,
                    "table_index": table_index,
                    "series_columns": series_columns,
                    "score": score_table(table),
                })

    for series_id in all_series_data:
        pages = all_series_data[series_id]["provenance"]["pages"]
        all_series_data[series_id]["provenance"]["pages"] = sorted(set(pages))

    return all_series_data, tables_found


# =========================================================
# SAVE JSON
# =========================================================

def save_json(data, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract NCD/Bond Series terms dynamically from a PDF."
    )

    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output JSON")

    args = parser.parse_args()

    print("Scanning PDF for NCD/Bond Series terms tables...")

    series_data, tables_found = extract_series_terms(args.pdf, verbose=True)

    result = {
        "document": Path(args.pdf).name,
        "series_count": len(series_data),
        "tables_found": tables_found,
        "series": list(series_data.values()),
    }

    save_json(result, args.output)

    print()
    print(f"Done! Extracted {len(series_data)} NCD series.")
    print(f"Found {len(tables_found)} candidate series tables.")
    print()

    for series in series_data.values():
        print(f"Series {series['series_id']}: {series['tenor']}")

    print()
    print(f"Saved to: {args.output}")