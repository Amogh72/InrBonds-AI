import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------

# Target chunk size.
# This is a target, not a hard limit.
TARGET_CHARS = 3500

# Context overlap between consecutive chunks.
OVERLAP_CHARS = 300


# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ---------------------------------------------------------
# DOCUMENT ID
# ---------------------------------------------------------

def make_document_id(document_name):
    """
    Derive a filesystem/ID-safe slug from a document name.

    Examples:

        "pfc_ncd_2026.pdf"            -> "pfc_ncd_2026"
        "REC NCD (Tranche II) 2026"   -> "rec_ncd_tranche_ii_2026"
        "PFC-NCD-2026.pdf"            -> "pfc_ncd_2026"
    """

    stem = Path(document_name).stem

    doc_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        stem.lower()
    ).strip("_")

    return doc_id or "document"


# ---------------------------------------------------------
# BUILD PAGE DATA
# ---------------------------------------------------------

def build_page_data(layout_document):
    """
    Preserve text separately for every PDF page.

    Returns:

        {
            1: {
                "text": "...",
                "lines": [...]
            },
            2: {
                "text": "...",
                "lines": [...]
            }
        }
    """

    pages = {}

    for page in layout_document["pages"]:

        page_number = page["page_number"]

        lines = []

        for line in page["lines"]:

            text = line["text"].strip()

            if text:
                lines.append(text)

        pages[page_number] = {
            "text": "\n".join(lines),
            "lines": lines
        }

    return pages


# ---------------------------------------------------------
# STRUCTURE PATH
# ---------------------------------------------------------

def build_structure_path(
    sections,
    section_index
):
    """
    Build the hierarchy path for a section.

    Example:

        SECTION III
        └── TERMS OF THE ISSUE

    becomes:

        [
            "SECTION III",
            "TERMS OF THE ISSUE"
        ]
    """

    section = sections[section_index]

    current_level = section["level"]

    path = [
        section["title"]
    ]

    # Walk backwards looking for parents.
    for previous in reversed(
        sections[:section_index]
    ):

        if previous["level"] < current_level:

            path.insert(
                0,
                previous["title"]
            )

            current_level = previous["level"]

    return path


# ---------------------------------------------------------
# FIND LEAF SECTIONS
# ---------------------------------------------------------

def is_leaf_section(
    sections,
    index
):
    """
    A leaf section is a section that has no
    lower-level child in its page range.

    These are the sections whose actual text
    we will chunk.
    """

    section = sections[index]

    current_level = section["level"]

    for other_index, other in enumerate(sections):

        if other_index == index:
            continue

        if (
            other["level"] > current_level
            and other["start_page"] >= section["start_page"]
            and other["start_page"] <= section["end_page"]
        ):
            return False

    return True


# ---------------------------------------------------------
# COLLECT PAGE CONTENT
# ---------------------------------------------------------

def collect_section_pages(
    page_data,
    start_page,
    end_page,
    excluded_pages=None
):
    """
    Return page-by-page text for a section.

    Each item retains its original PDF page number.

    excluded_pages: an optional set of page numbers to leave out
    entirely (e.g. pages a table extractor already turned into clean
    structured facts - re-chunking that same page's raw line text as
    prose would just be a garbled, out-of-order restatement of data
    that's already correctly represented elsewhere).
    """

    excluded_pages = excluded_pages or set()

    pages = []

    for page_number in range(
        start_page,
        end_page + 1
    ):

        if page_number in excluded_pages:
            continue

        page = page_data.get(
            page_number
        )

        if not page:
            continue

        text = page["text"].strip()

        if not text:
            continue

        pages.append({
            "page_number": page_number,
            "text": text
        })

    return pages


# ---------------------------------------------------------
# CREATE TEXT PIECES
# ---------------------------------------------------------

def split_large_text(
    text,
    target_chars=TARGET_CHARS,
    overlap_chars=OVERLAP_CHARS
):
    """
    Split a single large page/paragraph into smaller
    pieces while preserving overlap.

    Prefer newline boundaries where possible.
    """

    text = text.strip()

    if len(text) <= target_chars:
        return [text]

    pieces = []

    start = 0

    while start < len(text):

        desired_end = min(
            start + target_chars,
            len(text)
        )

        # If this isn't the final piece, try to
        # find a natural line/space boundary.
        end = desired_end

        if desired_end < len(text):

            newline_position = text.rfind(
                "\n",
                start,
                desired_end
            )

            space_position = text.rfind(
                " ",
                start,
                desired_end
            )

            best_break = max(
                newline_position,
                space_position
            )

            # Only use the boundary if it isn't
            # too close to the beginning.
            if best_break > start + 500:
                end = best_break

        piece = text[
            start:end
        ].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        next_start = end - overlap_chars

        # Prevent infinite loops.
        if next_start <= start:
            next_start = end

        start = next_start

    return pieces


# ---------------------------------------------------------
# CHUNK A SECTION
# ---------------------------------------------------------

def chunk_section_pages(
    section_pages,
    section_path,
    target_chars=TARGET_CHARS,
    overlap_chars=OVERLAP_CHARS
):
    """
    Turn page-level content into chunks while preserving
    the actual page range of every chunk.
    """

    chunks = []

    current_text = []
    current_pages = []

    current_length = 0

    for page in section_pages:

        page_number = page["page_number"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        # -------------------------------------------------
        # If the page itself is larger than the target,
        # split it before adding it to the current chunk.
        # -------------------------------------------------

        if len(page_text) > target_chars:

            # Save anything currently accumulated.
            if current_text:

                chunks.append({
                    "section_path": section_path.copy(),
                    "start_page": min(current_pages),
                    "end_page": max(current_pages),
                    "text": "\n\n".join(current_text)
                })

                current_text = []
                current_pages = []
                current_length = 0

            # Split the large page.
            page_pieces = split_large_text(
                page_text,
                target_chars,
                overlap_chars
            )

            # Every piece comes from the same PDF page.
            for piece_index, piece in enumerate(page_pieces):

                if piece_index < len(page_pieces) - 1:

                    chunks.append({
                        "section_path": section_path.copy(),
                        "start_page": page_number,
                        "end_page": page_number,
                        "text": piece
                    })

                else:

                    current_text = [piece]
                    current_pages = [page_number]
                    current_length = len(piece)

            continue

        # -------------------------------------------------
        # Normal page-sized content.
        # -------------------------------------------------

        if (
            current_length == 0
            or current_length + len(page_text) + 2 <= target_chars
        ):

            current_text.append(page_text)

            current_pages.append(page_number)

            current_length += len(page_text) + 2

            continue

        # -------------------------------------------------
        # Current chunk is full.
        # Save it first.
        # -------------------------------------------------

        chunks.append({
            "section_path": section_path.copy(),
            "start_page": min(current_pages),
            "end_page": max(current_pages),
            "text": "\n\n".join(current_text)
        })

        # -------------------------------------------------
        # Start next chunk with overlap.
        # -------------------------------------------------

        previous_text = "\n\n".join(
            current_text
        )

        last_page_of_previous_chunk = max(
            current_pages
        )

        overlap_text = previous_text[
            max(
                0,
                len(previous_text)
                - overlap_chars
            ):
        ].strip()

        current_text = []
        current_pages = []
        current_length = 0

        if overlap_text:

            current_text.append(
                overlap_text
            )

            current_pages.append(
                last_page_of_previous_chunk
            )

            current_length = len(
                overlap_text
            )

        # -------------------------------------------------
        # Add current page.
        # -------------------------------------------------

        if (
            current_length
            + len(page_text)
            + 2
            <= target_chars
        ):

            current_text.append(
                page_text
            )

            current_pages.append(
                page_number
            )

            current_length += (
                len(page_text) + 2
            )

        else:

            pieces = split_large_text(
                page_text,
                target_chars,
                overlap_chars
            )

            for piece_index, piece in enumerate(
                pieces
            ):

                if piece_index < len(pieces) - 1:

                    chunks.append({
                        "section_path":
                            section_path.copy(),

                        "start_page":
                            page_number,

                        "end_page":
                            page_number,

                        "text":
                            piece
                    })

                else:

                    current_text = [
                        piece
                    ]

                    current_pages = [
                        page_number
                    ]

                    current_length = len(
                        piece
                    )

    # -----------------------------------------------------
    # Save final chunk
    # -----------------------------------------------------

    if current_text:

        chunks.append({
            "section_path":
                section_path.copy(),

            "start_page":
                min(current_pages),

            "end_page":
                max(current_pages),

            "text":
                "\n\n".join(current_text)
        })

    return chunks


# ---------------------------------------------------------
# CREATE ALL CHUNKS
# ---------------------------------------------------------

def create_chunks(
    layout_document,
    structure_document,
    target_chars=TARGET_CHARS,
    overlap_chars=OVERLAP_CHARS,
    document_id=None,
    excluded_pages=None
):
    """
    Create structure-aware, page-aware chunks.

    document_id: use this exact id for chunk_id generation instead of
    re-deriving one from layout_document["document"] (the PDF
    filename). A caller that already has an authoritative document_id
    (e.g. canonical_extractor.py, which may assign a document_id
    unrelated to the PDF's filename) should always pass it, so chunk
    ids stay consistent with the rest of that same canonical document
    instead of silently drifting from a different, filename-derived id.

    excluded_pages: see collect_section_pages.
    """

    page_data = build_page_data(
        layout_document
    )

    sections = structure_document[
        "sections"
    ]

    all_chunks = []

    for section_index, section in enumerate(
        sections
    ):

        # Only chunk leaf sections.
        if not is_leaf_section(
            sections,
            section_index
        ):
            continue

        start_page = section[
            "start_page"
        ]

        end_page = section[
            "end_page"
        ]

        section_pages = collect_section_pages(
            page_data,
            start_page,
            end_page,
            excluded_pages
        )

        if not section_pages:
            continue

        section_path = build_structure_path(
            sections,
            section_index
        )

        section_chunks = chunk_section_pages(
            section_pages,
            section_path,
            target_chars,
            overlap_chars
        )

        all_chunks.extend(
            section_chunks
        )

    # -----------------------------------------------------
    # Add IDs and common metadata
    # -----------------------------------------------------

    final_chunks = []

    document_name = layout_document[
        "document"
    ]

    if document_id is None:
        document_id = make_document_id(
            document_name
        )

    for index, chunk in enumerate(
        all_chunks,
        start=1
    ):

        section_path = chunk[
            "section_path"
        ]

        final_chunks.append({

            "chunk_id": (
                f"{document_id}_"
                f"{index:04d}"
            ),

            "document":
                document_name,

            "section_path":
                section_path,

            "section":
                section_path[0],

            "subsection":
                section_path[-1],

            "start_page":
                chunk["start_page"],

            "end_page":
                chunk["end_page"],

            "text":
                chunk["text"],

            "char_count":
                len(chunk["text"])
        })

    return final_chunks


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

def save_chunks(
    chunks,
    output_path,
    document_name,
    target_chars=TARGET_CHARS,
    overlap_chars=OVERLAP_CHARS
):
    """
    Save chunks to JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {

        "document":
            document_name,

        "total_chunks":
            len(chunks),

        "chunking_config": {

            "target_chars":
                target_chars,

            "overlap_chars":
                overlap_chars
        },

        "chunks":
            chunks
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create structure-aware, page-aware chunks "
            "from a parsed layout + structure JSON pair."
        )
    )

    parser.add_argument(
        "--layout",
        required=True,
        help="Path to the *_layout.json file."
    )

    parser.add_argument(
        "--structure",
        required=True,
        help="Path to the *_structure.json file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the *_chunks.json file."
    )

    parser.add_argument(
        "--target-chars",
        type=int,
        default=TARGET_CHARS,
        help=(
            f"Target chunk size in characters "
            f"(default: {TARGET_CHARS})."
        )
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=OVERLAP_CHARS,
        help=(
            f"Overlap size in characters "
            f"(default: {OVERLAP_CHARS})."
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    args = parse_args()

    print(
        "Loading parsed document..."
    )

    layout_document = load_json(
        args.layout
    )

    print(
        "Loading document structure..."
    )

    structure_document = load_json(
        args.structure
    )

    print(
        "Creating page-aware "
        "structure-aware chunks..."
    )

    chunks = create_chunks(
        layout_document,
        structure_document,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars
    )

    save_chunks(
        chunks,
        args.output,
        layout_document["document"],
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars
    )

    print(
        f"Done! Created "
        f"{len(chunks)} chunks."
    )

    print(
        f"Saved chunks to: "
        f"{args.output}"
    )