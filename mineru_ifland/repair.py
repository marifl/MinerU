"""Fill gaps the acceptance check found, from the PDF text layer, and mark every filled line.

MinerU loses text on some pages: a table transcription interleaves multi-line cells ("K
Prlaüufusnugr" for "Klausur"/"Prüfung"), a slide line comes back truncated. The words are in the
PDF text layer, so they can be recovered deterministically - no model, no guessing.

A repaired line is a normal `text` block, so every 3.x reader keeps working, plus the field
`"source": "text_layer_repair"` so anyone can tell MinerU's reading from the filled-in one. Nothing
is replaced or deleted; only lines whose words are missing from the page are appended.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from .gate import BBOX_SCALE, blob, block_blob, tokens

REPAIR_SOURCE = "text_layer_repair"
# a line is filled in when this share of its words is missing from the page; measured on the
# corpus: 0.2 recovers clearly more than 0.5, below that nothing changes
MIN_MISSING_SHARE = 0.2
MIN_LINE_TOKENS = 1


def _page_lines(page: pdfium.PdfPage) -> list[tuple[str, list[float]]]:
    """Text-layer lines of a page with their box, scaled like content_list bboxes."""
    width, height = page.get_width(), page.get_height()
    text_page = page.get_textpage()
    text = text_page.get_text_range()
    lines: list[tuple[str, list[float]]] = []
    current: list[str] = []
    box: list[float] | None = None

    def flush() -> None:
        nonlocal box
        line = "".join(current).strip()
        if line and box is not None:
            lines.append((line, box))
        current.clear()
        box = None

    for index, char in enumerate(text):
        if char in "\r\n":
            flush()
            continue
        current.append(char)
        try:
            left, bottom, right, top = text_page.get_charbox(index)
        except Exception:
            continue
        scaled = [
            left / width * BBOX_SCALE,
            (height - top) / height * BBOX_SCALE,
            right / width * BBOX_SCALE,
            (height - bottom) / height * BBOX_SCALE,
        ]
        box = scaled if box is None else [
            min(box[0], scaled[0]), min(box[1], scaled[1]), max(box[2], scaled[2]), max(box[3], scaled[3])
        ]
    flush()
    return lines


def _on_page(box: list[float]) -> bool:
    centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return 0 <= centre_x <= BBOX_SCALE and 0 <= centre_y <= BBOX_SCALE


def repair_pages(content_list: list[dict], pdf_path: Path, pages: list[int]) -> list[dict[str, Any]]:
    """Append the missing text-layer lines of `pages` (1-based) to `content_list`, in place.

    Returns one record per repaired page for the acceptance report.
    """
    if not pages:
        return []
    wanted = {page - 1 for page in pages}
    page_blob: dict[int, str] = {}
    for block in content_list:
        index = block.get("page_idx", -1)
        if index in wanted:
            page_blob[index] = page_blob.get(index, "") + block_blob(block)

    records: list[dict[str, Any]] = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in sorted(wanted):
            if index >= len(document):
                continue
            existing = page_blob.get(index, "")
            filled: list[dict[str, Any]] = []
            for line, box in _page_lines(document[index]):
                if not _on_page(box):
                    continue
                line_tokens = tokens(line)
                if len(line_tokens) < MIN_LINE_TOKENS:
                    continue
                missing = [token for token in line_tokens if token not in existing]
                if len(missing) < MIN_MISSING_SHARE * len(line_tokens):
                    continue
                filled.append({
                    "type": "text",
                    "text": re.sub(r"\s+", " ", line).strip(),
                    "page_idx": index,
                    "bbox": [round(value) for value in box],
                    "source": REPAIR_SOURCE,
                })
                existing += blob(line)
            if filled:
                content_list.extend(filled)
                records.append({"page": index + 1, "lines": len(filled),
                                "text": [entry["text"][:80] for entry in filled[:5]]})
    finally:
        document.close()
    content_list.sort(key=lambda block: (block.get("page_idx", 0), block.get("source") == REPAIR_SOURCE))
    return records


__all__ = ["repair_pages", "REPAIR_SOURCE"]
