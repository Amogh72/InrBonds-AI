import fitz
import json
from pathlib import Path


def parse_pdf_with_layout(pdf_path, page_offset=0):
    """
    Extract PDF content page-by-page while preserving
    layout information useful for later structure detection.

    For each text line we preserve:
    - text
    - font size
    - font names
    - whether a bold font was detected
    - bounding box / position on page
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    document = fitz.open(pdf_path)

    pages = []

    for page_index, page in enumerate(document):

        # "dict" mode gives us blocks, lines, spans,
        # font information and coordinates.
        raw = page.get_text("dict")

        lines_out = []

        for block in raw["blocks"]:

            # Blocks may contain images instead of text.
            if "lines" not in block:
                continue

            for line in block["lines"]:

                spans = line["spans"]

                if not spans:
                    continue

                # Combine all spans belonging to this line.
                text = "".join(
                    span["text"]
                    for span in spans
                ).strip()

                if not text:
                    continue

                # Largest font size appearing in the line.
                max_size = max(
                    span["size"]
                    for span in spans
                )

                # Preserve all font names used in the line.
                fonts = sorted(set(
                    span["font"]
                    for span in spans
                ))

                # Basic bold detection using font names.
                is_bold = any(
                    "bold" in span["font"].lower()
                    for span in spans
                )

                # Bounding box:
                # [x0, y0, x1, y1]
                bbox = line["bbox"]

                lines_out.append({
                    "text": text,

                    "size": round(
                        max_size,
                        1
                    ),

                    "bold": is_bold,

                    "fonts": fonts,

                    "bbox": [
                        round(value, 1)
                        for value in bbox
                    ]
                })

        pages.append({
            "page_number": page_index + 1,

            "width": round(
                page.rect.width,
                1
            ),

            "height": round(
                page.rect.height,
                1
            ),

            "lines": lines_out
        })

    result = {
        "document": pdf_path.name,
        "total_pages": len(document),
        "pages": pages
    }

    document.close()

    return result


def save_to_json(data, output_path):
    """
    Save parsed PDF data to a JSON file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    input_pdf = (
        "data/raw/"
        "pfc_ncd_2026.pdf"
    )

    output_json = (
        "data/processed/"
        "pfc_ncd_2026_layout.json"
    )

    print(
        "Extracting PDF with layout information..."
    )

    parsed_document = parse_pdf_with_layout(
        input_pdf
    )

    save_to_json(
        parsed_document,
        output_json
    )

    print(
        f"Done! Extracted "
        f"{parsed_document['total_pages']} pages."
    )

    print(
        f"Saved layout data to: "
        f"{output_json}"
    )