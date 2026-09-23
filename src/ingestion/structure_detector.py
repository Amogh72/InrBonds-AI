import fitz
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PDF_PATH = "data/raw/pfc_ncd_2026.pdf"
LAYOUT_JSON = "data/processed/pfc_ncd_2026_layout.json"
OUTPUT_JSON = "data/processed/pfc_ncd_2026_structure.json"


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_text(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{2,}", " ", text)
    return text.strip()


def similarity(text1, text2):
    a = normalize_text(text1)
    b = normalize_text(text2)
    return SequenceMatcher(None, a, b).ratio()


def load_layout_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# STEP 1 — EMBEDDED PDF OUTLINE / BOOKMARKS
# ---------------------------------------------------------

def extract_embedded_toc(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    raw_toc = document.get_toc()
    document.close()

    entries = []
    for item in raw_toc:
        if len(item) < 3:
            continue
        level, title, page = item[0], item[1], item[2]
        entries.append({
            "level": level,
            "title": title.strip(),
            "pdf_page": page,
            "source": "embedded_toc",
        })
    return entries


# ---------------------------------------------------------
# STEP 2 — VISIBLE TABLE OF CONTENTS
# ---------------------------------------------------------

def find_visible_toc_pages(document, max_pages=15):
    toc_start = None
    for page in document["pages"][:max_pages]:
        for line in page["lines"]:
            if normalize_text(line["text"]) == "table of contents":
                toc_start = page["page_number"]
                break
        if toc_start is not None:
            break
    return [toc_start] if toc_start is not None else []


def extract_visible_toc(document, start_page=2, max_scan_pages=5):
    pages_by_number = {p["page_number"]: p for p in document["pages"]}
    entries = []

    # FIX: allow optional whitespace between dot leaders and the page
    # number -- real ToC lines have a single space there, which matched
    # neither the "\.{2,}" branch (needs digit immediately after dots)
    # nor "\s{2,}" (needs 2+ spaces), so every line silently failed.
    entry_pattern = re.compile(
        r"(.+?)(?:\.{2,}\s*|\s{2,})(\d{1,4})(?=\s|$)"
    )

    pending_parts = []
    pending_x0 = None

    for page_num in range(start_page, start_page + max_scan_pages):
        page = pages_by_number.get(page_num)

        if not page:
            break

        for line in page["lines"]:
            text = line["text"].strip()

            if not text:
                continue

            if text.upper() == "TABLE OF CONTENTS":
                pending_parts = []
                pending_x0 = None
                continue

            matches = list(entry_pattern.finditer(text))

            if matches:
                cursor = 0

                for match in matches:
                    title_part = match.group(1).strip()
                    printed_page = int(match.group(2))

                    if cursor:
                        title_part = text[
                            cursor:match.start(2)
                        ].strip()

                        title_part = re.sub(
                            r"\.{2,}\s*$",
                            "",
                            title_part
                        ).strip()

                    if pending_parts:
                        full_title = " ".join(
                            pending_parts + [title_part]
                        ).strip()

                        x0 = pending_x0
                    else:
                        full_title = title_part
                        x0 = line["bbox"][0]

                    full_title = re.sub(
                        r"\.{2,}",
                        " ",
                        full_title
                    )

                    full_title = re.sub(
                        r"\s+",
                        " ",
                        full_title
                    ).strip()

                    if len(full_title) > 3:
                        entries.append({
                            "title": full_title,
                            "printed_page": printed_page,
                            "x0": round(x0, 1),
                            "source": "visible_toc"
                        })

                    pending_parts = []
                    pending_x0 = None
                    cursor = match.end()

                remainder = text[cursor:].strip()

                if remainder:
                    pending_parts = [remainder]
                    pending_x0 = line["bbox"][0]

            else:
                if pending_x0 is None:
                    pending_x0 = line["bbox"][0]

                pending_parts.append(text)

    return entries

def infer_toc_levels(entries, tolerance=5):
    """
    Infer hierarchy levels from the horizontal indentation
    of visible TOC entries.

    Smaller x0 = less indentation = higher-level heading.
    Similar x0 values are treated as the same level.
    """

    if not entries:
        return []

    # Collect all x positions.
    x_positions = sorted(
        set(
            round(entry["x0"], 1)
            for entry in entries
        )
    )

    # Group nearby x positions together.
    groups = []

    for x in x_positions:

        if not groups:
            groups.append([x])
            continue

        previous_group = groups[-1]

        representative = sum(
            previous_group
        ) / len(previous_group)

        if abs(x - representative) <= tolerance:
            previous_group.append(x)
        else:
            groups.append([x])

    # The leftmost indentation is level 1.
    level_by_x = {}

    for level, group in enumerate(
        groups,
        start=1
    ):

        representative = sum(group) / len(group)

        for x in group:
            level_by_x[x] = level

    result = []

    for entry in entries:

        x = round(entry["x0"], 1)

        closest_x = min(
            level_by_x.keys(),
            key=lambda candidate: abs(
                candidate - x
            )
        )

        result.append({
            **entry,
            "level": level_by_x[closest_x]
        })

    return result

# ---------------------------------------------------------
# STEP 3 — LAYOUT HEADING CANDIDATES (verification signal only)
# ---------------------------------------------------------

def get_body_font_size(document):
    sizes = [ln["size"] for page in document["pages"] for ln in page["lines"]]
    return Counter(sizes).most_common(1)[0][0] if sizes else 0


def find_heading_candidates(document, size_threshold=1.5):
    """
    IMPORTANT: these are candidates only. Never treated as a final
    detector on their own — bold/large all-caps disclaimer text
    (as seen in this document's pages 54-58) looks identical to a
    real heading by font signal alone. Used only to confirm that
    something heading-shaped physically exists near a page a TOC
    (embedded or visible) claims a section starts on.
    """
    body_size = get_body_font_size(document)
    candidates = []

    for page in document["pages"]:
        top_band = page["height"] * 0.08
        bottom_band = page["height"] * 0.92

        for line in page["lines"]:
            text = line["text"].strip()
            if not text:
                continue
            y0 = line["bbox"][1]
            if y0 < top_band or y0 > bottom_band:
                continue

            is_bigger = line["size"] >= body_size + size_threshold
            is_bold = line["bold"]
            is_reasonable_length = 3 <= len(text) <= 180

            if (is_bigger or is_bold) and is_reasonable_length:
                candidates.append({
                    "page_number": page["page_number"],
                    "text": text,
                    "size": line["size"],
                    "bold": line["bold"],
                    "bbox": line["bbox"],
                })
    return candidates


# ---------------------------------------------------------
# STEP 4 — PAGE OFFSET (two-pass, window-constrained)
# ---------------------------------------------------------

def estimate_page_offset(toc_entries, candidates, similarity_threshold=0.80):
    """
    Two-pass estimation. Pass 1 finds a rough offset the same way
    as before (global best match + majority vote across entries) —
    this is already fairly robust to a few duplicate-heading
    mismatches because outliers get out-voted. Pass 2 re-resolves
    each entry using ONLY candidates within a page window around
    the pass-1 offset, which is what actually fixes the duplicate-
    heading problem: a same-named heading far from the expected
    page can no longer win just because SequenceMatcher liked it.
    """

    def rough_pass(window=None, base_offset=0):
        offsets = []
        for entry in toc_entries:
            best_candidate, best_score = None, 0
            for candidate in candidates:
                if window is not None:
                    implied_offset = candidate["page_number"] - entry["printed_page"]
                    if abs(implied_offset - base_offset) > window:
                        continue
                score = similarity(entry["title"], candidate["text"])
                if score > best_score:
                    best_score, best_candidate = score, candidate
            if best_candidate and best_score >= similarity_threshold:
                offset = best_candidate["page_number"] - entry["printed_page"]
                if -20 <= offset <= 20:
                    offsets.append(offset)
        if not offsets:
            return 0
        return Counter(offsets).most_common(1)[0][0]

    pass1_offset = rough_pass()
    pass2_offset = rough_pass(window=10, base_offset=pass1_offset)
    return pass2_offset if pass2_offset else pass1_offset


# ---------------------------------------------------------
# STEP 5 — RECONCILE VISIBLE TOC WITH BODY (no embedded outline case)
# ---------------------------------------------------------

def _best_candidate_for_entry(
    entry, candidates, expected_pdf_page, page_tolerance, claimed_ids,
):
    """
    Score every layout candidate within the page window against this
    TOC entry's title, tracking the best UNCLAIMED match and the best
    match overall (ties broken by proximity to the expected page,
    not iteration order) separately. The overall best is kept even
    when it scores below the caller's similarity threshold, purely so
    resolve_sections can still report a near-miss similarity score
    for an unconfirmed entry, matching prior behavior.

    Returns (best_unclaimed_candidate_or_None, its_score,
             best_overall_candidate_or_None, its_score).
    """

    best_unclaimed, unclaimed_score, unclaimed_distance = None, 0, None
    best_overall, overall_score, overall_distance = None, 0, None

    for candidate in candidates:

        distance = abs(candidate["page_number"] - expected_pdf_page)

        if distance > page_tolerance:
            continue

        score = similarity(entry["title"], candidate["text"])

        if (
            best_overall is None
            or score > overall_score
            or (score == overall_score and distance < overall_distance)
        ):
            best_overall, overall_score, overall_distance = (
                candidate, score, distance
            )

        if id(candidate) in claimed_ids:
            continue

        if (
            best_unclaimed is None
            or score > unclaimed_score
            or (score == unclaimed_score and distance < unclaimed_distance)
        ):
            best_unclaimed, unclaimed_score, unclaimed_distance = (
                candidate, score, distance
            )

    return best_unclaimed, unclaimed_score, best_overall, overall_score


def resolve_sections(toc_entries, candidates, page_offset, page_tolerance=3, similarity_threshold=0.65):
    """
    A visible-TOC entry's title is not always unique within the
    document - some prospectuses print an identically-titled heading
    on several consecutive pages (e.g. a separate "DECLARATION" page
    per signing director). Matching purely on best similarity score
    within the page window, with no memory of what earlier entries
    already matched, collapses every such entry onto the SAME single
    best-scoring candidate (confirmed against a real second
    prospectus: 4 consecutive "DECLARATION" TOC entries, each with
    its own distinct heading on pages 194-197, all resolved to page
    194). Tracking claimed candidates and preferring an unclaimed one
    fixes this without weakening matching for the (far more common)
    case of a uniquely-titled entry, which is unaffected either way.
    """
    resolved = []
    claimed_ids = set()

    for entry in toc_entries:
        expected_pdf_page = entry["printed_page"] + page_offset

        best_unclaimed, unclaimed_score, best_overall, overall_score = (
            _best_candidate_for_entry(
                entry, candidates, expected_pdf_page, page_tolerance,
                claimed_ids,
            )
        )

        if best_unclaimed is not None and unclaimed_score >= similarity_threshold:
            # An unclaimed match good enough to confirm - always
            # preferred, so a repeated title spreads across its
            # distinct physical occurrences instead of collapsing.
            best_candidate, best_score = best_unclaimed, unclaimed_score
        else:
            # Nothing new clears the bar; fall back to the overall
            # best match (possibly already claimed, possibly below
            # threshold) - identical to this function's behavior
            # before claim-tracking existed.
            best_candidate, best_score = best_overall, overall_score

        confirmed = best_candidate is not None and best_score >= similarity_threshold

        if confirmed:
            claimed_ids.add(id(best_candidate))

        resolved.append({
            "level": entry.get("level", 1),
            "title": entry["title"],
            "printed_page": entry["printed_page"],
            "expected_pdf_page": expected_pdf_page,
            "resolved_pdf_page": best_candidate["page_number"] if confirmed else expected_pdf_page,
            "confirmed": confirmed,
            "similarity": round(best_score, 3),
            "matched_text": best_candidate["text"] if confirmed else None,
            "source": "visible_toc+layout",
        })
    return resolved


def build_section_boundaries(sections, total_pages):
    """
    Build hierarchical start/end page boundaries using section levels.

    A section ends when the next section with the same or higher
    hierarchy level begins.
    """

    if not sections:
        return []

    sections = [
        s for s in sections
        if 1 <= s["resolved_pdf_page"] <= total_pages
    ]

    sections.sort(
        key=lambda x: x["resolved_pdf_page"]
    )

    result = []

    for i, section in enumerate(sections):

        start_page = section["resolved_pdf_page"]
        current_level = section.get("level", 1)

        end_page = total_pages

        for next_section in sections[i + 1:]:

            next_level = next_section.get("level", 1)
            next_page = next_section["resolved_pdf_page"]

            if next_level <= current_level:

                if next_page > start_page:
                    end_page = next_page - 1
                else:
                    end_page = start_page

                break

        result.append({
            **section,
            "start_page": start_page,
            "end_page": end_page
        })

    return result


# ---------------------------------------------------------
# STEP 6 — EMBEDDED OUTLINE HIERARCHY + BOUNDARIES
# ---------------------------------------------------------

def add_hierarchical_boundaries(entries, total_pages):
    result = []
    for i, entry in enumerate(entries):
        current_level = entry["level"]
        start_page = entry["pdf_page"]
        end_page = total_pages

        for next_entry in entries[i + 1:]:
            if next_entry["level"] <= current_level:
                next_page = next_entry["pdf_page"]
                end_page = next_page - 1 if next_page > start_page else start_page
                break

        result.append({**entry, "start_page": start_page, "end_page": end_page})
    return result


# ---------------------------------------------------------
# STEP 7 — CROSS-VALIDATE EMBEDDED OUTLINE AGAINST
#          VISIBLE TOC + LAYOUT (evidence priority, not
#          blind trust in the embedded outline)
# ---------------------------------------------------------

def cross_validate_embedded(embedded_entries, visible_toc, candidates,
                             toc_similarity_threshold=0.75,
                             layout_similarity_threshold=0.75,
                             layout_page_tolerance=2):
    """
    The embedded outline still supplies hierarchy and page numbers
    (it's the strongest single signal — usually issuer/printer
    generated deliberately, not inferred). But we no longer trust
    it blindly: every entry gets checked against the visible TOC
    (if one exists) and against a real heading-shaped line of text
    physically near its claimed page. Entries that fail both checks
    are flagged low-confidence rather than silently accepted.
    """
    # Build a normalized-title lookup of visible TOC entries for
    # quick cross-checking, independent of page numbers (embedded
    # outline pages are physical PDF pages; visible TOC pages are
    # printed page numbers, so we compare titles, not pages, here).
    visible_titles = [(e["title"], e) for e in visible_toc]

    validated = []
    for entry in embedded_entries:
        # Check 1: does a similarly-titled visible TOC entry exist at all?
        best_toc_score = 0
        for title, _ in visible_titles:
            score = similarity(entry["title"], title)
            if score > best_toc_score:
                best_toc_score = score
        confirmed_by_toc = best_toc_score >= toc_similarity_threshold

        # Check 2: does a heading-shaped line of text actually exist
        # on or near the page the embedded outline claims?
        best_layout_score = 0
        for cand in candidates:
            if abs(cand["page_number"] - entry["pdf_page"]) > layout_page_tolerance:
                continue
            score = similarity(entry["title"], cand["text"])
            if score > best_layout_score:
                best_layout_score = score
        confirmed_by_layout = best_layout_score >= layout_similarity_threshold

        validated.append({
            **entry,
            "confirmed_by_visible_toc": confirmed_by_toc,
            "toc_similarity": round(best_toc_score, 3),
            "confirmed_by_layout": confirmed_by_layout,
            "layout_similarity": round(best_layout_score, 3),
            "confidence": (
                "high" if (confirmed_by_toc and confirmed_by_layout) else
                "medium" if (confirmed_by_toc or confirmed_by_layout) else
                "low"
            ),
        })
    return validated


# ---------------------------------------------------------
# STEP 8 — LAYOUT-ONLY CONSERVATIVE FALLBACK
#          (neither embedded outline nor visible TOC exists)
# ---------------------------------------------------------

def build_conservative_layout_structure(candidates, total_pages, top_n=40):
    """
    Weakest evidence tier. No titles to trust, so we don't try to
    assign section names beyond what layout shows, and we keep the
    set small (largest/boldest candidates only) rather than treating
    every bold line as a section — this is explicitly a low-confidence,
    best-effort structure, not equivalent to the other two paths.
    """
    ranked = sorted(candidates, key=lambda c: (c["size"], c["bold"]), reverse=True)
    chosen = ranked[:top_n]
    chosen.sort(key=lambda c: c["page_number"])

    result = []
    for i, cand in enumerate(chosen):
        start_page = cand["page_number"]
        end_page = chosen[i + 1]["page_number"] - 1 if i + 1 < len(chosen) else total_pages
        end_page = max(start_page, end_page)
        result.append({
            "title": cand["text"],
            "start_page": start_page,
            "end_page": end_page,
            "confidence": "low",
            "source": "layout_only",
        })
    return result


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# MAIN — evidence-priority orchestration
# ---------------------------------------------------------

# An embedded outline that mostly fails cross-validation isn't a
# slightly-noisy version of the truth -- it's evidence this outline
# doesn't belong to this document's actual structure (e.g. stale
# bookmarks carried over from merging source PDFs together). Below
# this trust threshold we don't use it as the primary source at all.
EMBEDDED_TRUST_THRESHOLD = 0.30


def detect_structure(pdf_path, document, verbose=True):
    """
    Run the full evidence-priority structure-detection pipeline
    against one already-parsed layout document (see pdf_parser.py),
    and return the resulting structure dict - the same shape
    previously only produced by running this file as a script.
    Callable directly by other pipeline stages (e.g. a future
    chunker) without going through a file on disk.
    """

    def log(message):
        if verbose:
            print(message)

    total_pages = document["total_pages"]

    log("Checking embedded PDF outline...")
    embedded_toc = extract_embedded_toc(pdf_path)
    log(f"Embedded TOC entries: {len(embedded_toc)}")

    log("Extracting visible Table of Contents (always attempted)...")
    visible_toc = extract_visible_toc(document)

    if visible_toc:
        visible_toc = infer_toc_levels(visible_toc)

    log(f"Visible TOC entries: {len(visible_toc)}")

    log("Generating layout heading candidates (verification signal)...")
    candidates = find_heading_candidates(document)
    log(f"Layout candidates: {len(candidates)}")

    use_embedded = False
    validated_sections = None
    confirmed_ratio = None

    if embedded_toc:
        log("\nCross-validating embedded outline against visible TOC + layout...")
        sections_with_boundaries = add_hierarchical_boundaries(embedded_toc, total_pages)
        validated_sections = cross_validate_embedded(sections_with_boundaries, visible_toc, candidates)

        low_conf = [s for s in validated_sections if s["confidence"] == "low"]
        confirmed_ratio = 1 - (len(low_conf) / len(validated_sections))
        log(f"{len(validated_sections)} sections total, {len(low_conf)} flagged low-confidence "
            f"({confirmed_ratio:.0%} confirmed)")

        if confirmed_ratio >= EMBEDDED_TRUST_THRESHOLD:
            use_embedded = True
        else:
            log(f"Confirmed ratio {confirmed_ratio:.0%} is below the "
                f"{EMBEDDED_TRUST_THRESHOLD:.0%} trust threshold -- the embedded "
                f"outline does not reliably describe this document. "
                f"Falling back instead of using it as primary.")

    if use_embedded:
        return {
            "document": document["document"],
            "strategy": "embedded_pdf_outline_validated",
            "confirmed_ratio": round(confirmed_ratio, 3),
            "visible_toc_available": bool(visible_toc),
            "sections": validated_sections,
        }

    if visible_toc:
        log("\nUsing visible TOC + layout confirmation as primary source...")

        page_offset = estimate_page_offset(visible_toc, candidates)
        log(f"Estimated page offset: {page_offset:+d}")

        resolved = resolve_sections(visible_toc, candidates, page_offset)
        for section in resolved:
            status = "✓" if section["confirmed"] else "?"
            log(f"{status} PDF {section['resolved_pdf_page']:>3} | {section['title']}")

        boundaries = build_section_boundaries(resolved, total_pages)

        return {
            "document": document["document"],
            "strategy": "visible_toc_plus_layout",
            "embedded_toc_rejected": bool(embedded_toc),
            "embedded_toc_confirmed_ratio": round(confirmed_ratio, 3) if confirmed_ratio is not None else None,
            "estimated_page_offset": page_offset,
            "toc_entries": visible_toc,
            "sections": boundaries,
        }

    log("\nNo embedded outline or visible TOC usable. Falling back to conservative layout-only structure.")

    conservative = build_conservative_layout_structure(candidates, total_pages)

    return {
        "document": document["document"],
        "strategy": "layout_only_conservative",
        "sections": conservative,
    }


if __name__ == "__main__":

    print("\nLoading document layout...")
    _document = load_layout_json(LAYOUT_JSON)

    _structure = detect_structure(PDF_PATH, _document, verbose=True)

    save_json(_structure, OUTPUT_JSON)
    print(f"\nSaved structure to: {OUTPUT_JSON}")