"""Acceptance check for one parsed document: prove it is complete, or say exactly where it is not.

The PDF text layer is the only ground truth available without a human, so every word it carries must
show up in the output - except where it sits inside a picture, chart or formula, which MinerU keeps
as an image or as LaTeX on purpose. Pages without a text layer (scans) are reported as unchecked
rather than silently counted as fine. Nothing here repairs anything.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

# a page with fewer reference tokens than this carries no usable text layer (scan, full-page image)
MIN_REFERENCE_TOKENS = 5
# tokens of at least three characters, letters or digits, so table numbers count too
TOKEN_CHAR = re.compile(r"[^\W_]", re.UNICODE)
MIN_TOKEN_LENGTH = 3
HYPHENS = "\u00ad\u2010\u2011\u2012\u2013\u2014-"
HTML_TAG = re.compile(r"<[^>]+>")
LATEX_COMMAND = re.compile(r"\\[A-Za-z]+|[{}$&\\^_]")
# content_list bboxes are the page scaled to 0..1000
BBOX_SCALE = 1000.0
FRAME_TYPES = {"header", "footer", "page_number", "page_footnote", "aside_text"}
# text inside these blocks is kept as picture or as LaTeX, so the text layer is not the contract
VISUAL_TYPES = {"image", "chart", "equation"}
TEXT_FIELDS = ("text", "table_body", "code_body")
LIST_FIELDS = (
    "list_items", "image_caption", "image_footnote", "table_caption", "table_footnote",
    "chart_caption", "chart_footnote", "code_caption", "code_footnote",
)


def tokens(text: str) -> set[str]:
    """Comparable word tokens of a rendered string."""
    text = HTML_TAG.sub(" ", LATEX_COMMAND.sub(" ", unicodedata.normalize("NFC", text)))
    found: set[str] = set()
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if TOKEN_CHAR.match(char):
            current.append(char)
        elif char in HYPHENS and _line_break_follows(text, index):
            index = _skip_line_break(text, index)
            continue
        else:
            _flush(current, found)
        index += 1
    _flush(current, found)
    return found


LINE_BREAK = re.compile(r"[ \t]*[\r\n]+[ \t]*")


def _line_break_follows(text: str, index: int) -> bool:
    """A hyphen at a line end joins the word; a soft hyphen joins it anywhere."""
    return text[index] == "\u00ad" or bool(LINE_BREAK.match(text, index + 1))


def _skip_line_break(text: str, index: int) -> int:
    match = LINE_BREAK.match(text, index + 1)
    return match.end() if match else index + 1


def _flush(current: list[str], found: set[str]) -> None:
    if len(current) >= MIN_TOKEN_LENGTH:
        found.add("".join(current).casefold())
    current.clear()


def blob(text: str) -> str:
    """Letters and digits of a string, in order: comparing against this ignores how words are split."""
    text = HTML_TAG.sub(" ", LATEX_COMMAND.sub(" ", unicodedata.normalize("NFC", text)))
    return "".join(char.casefold() for char in text if TOKEN_CHAR.match(char))


def block_blob(block: dict[str, Any]) -> str:
    """What a block contributes; MinerU's own picture descriptions (`content`) never count."""
    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = block.get(field)
        if isinstance(value, str):
            parts.append(blob(value))
    for field in LIST_FIELDS:
        value = block.get(field) or []
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                parts.append(blob(item))
    return "".join(parts)


def _page_reference(page: pdfium.PdfPage) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Text-layer tokens of a page with their box, scaled like content_list bboxes."""
    width, height = page.get_width(), page.get_height()
    text_page = page.get_textpage()
    text = text_page.get_text_range()
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    current: list[str] = []
    box: list[float] | None = None
    char_index = 0

    def flush() -> None:
        nonlocal box
        if len(current) >= MIN_TOKEN_LENGTH and box is not None:
            result.append(("".join(current).casefold(), (box[0], box[1], box[2], box[3])))
        current.clear()
        box = None

    index = 0
    while index < len(text):
        char = text[index]
        if TOKEN_CHAR.match(char):
            current.append(char)
            try:
                left, bottom, right, top = text_page.get_charbox(char_index)
            except Exception:
                left = bottom = right = top = 0.0
            scaled = (
                left / width * BBOX_SCALE,
                (height - top) / height * BBOX_SCALE,
                right / width * BBOX_SCALE,
                (height - bottom) / height * BBOX_SCALE,
            )
            box = scaled if box is None else [
                min(box[0], scaled[0]), min(box[1], scaled[1]), max(box[2], scaled[2]), max(box[3], scaled[3])
            ]
        elif char in HYPHENS and _line_break_follows(text, index):
            skipped = _skip_line_break(text, index)
            char_index += skipped - index
            index = skipped
            continue
        else:
            flush()
        char_index += 1
        index += 1
    flush()
    return result


def _on_page(box: tuple[float, float, float, float]) -> bool:
    """False for text placed outside the page area (below the crop, off to the side): it is not visible."""
    centre_x = (box[0] + box[2]) / 2
    centre_y = (box[1] + box[3]) / 2
    return 0 <= centre_x <= BBOX_SCALE and 0 <= centre_y <= BBOX_SCALE


def _inside(box: tuple[float, float, float, float], area: list[float]) -> bool:
    centre_x = (box[0] + box[2]) / 2
    centre_y = (box[1] + box[3]) / 2
    return area[0] <= centre_x <= area[2] and area[1] <= centre_y <= area[3]


def check_completeness(content_list: list[dict], pdf_path: Path, *, missing_sample: int = 25) -> dict[str, Any]:
    """Per page: which text-layer tokens outside pictures and formulas are missing from the output."""
    blob_per_page: dict[int, str] = {}
    visual_per_page: dict[int, list[list[float]]] = {}
    blocks_per_page: dict[int, int] = {}
    for block in content_list:
        page = block.get("page_idx", -1)
        blob_per_page[page] = blob_per_page.get(page, "") + block_blob(block)
        blocks_per_page[page] = blocks_per_page.get(page, 0) + (block.get("type") not in FRAME_TYPES)
        if block.get("type") in VISUAL_TYPES and block.get("bbox"):
            visual_per_page.setdefault(page, []).append(block["bbox"])

    pages: list[dict[str, Any]] = []
    unchecked = 0
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for index, page in enumerate(document):
            reference = _page_reference(page)
            areas = visual_per_page.get(index, [])
            wanted = {
                token
                for token, box in reference
                if _on_page(box) and not any(_inside(box, area) for area in areas)
            }
            if len(wanted) < MIN_REFERENCE_TOKENS:
                unchecked += 1
                continue
            # compare against the page's text without separators: a word the text layer glues
            # ("aufdem") or splits is still present when the output carries the same letters
            missing = sorted(token for token in wanted if token not in blob_per_page.get(index, ""))
            pages.append({
                "page": index + 1,
                "reference_tokens": len(wanted),
                "recall": round(1 - len(missing) / len(wanted), 4),
                "missing": missing[:missing_sample],
                "missing_count": len(missing),
                "blocks": blocks_per_page.get(index, 0),
            })
    finally:
        document.close()

    checked = [page["recall"] for page in pages]
    return {
        "pages_total": len(checked) + unchecked,
        "pages_checked": len(pages),
        "pages_without_text_layer": unchecked,
        "recall_min": min(checked) if checked else None,
        "recall_mean": round(sum(checked) / len(checked), 4) if checked else None,
        "pages": pages,
    }


# a page below this share of its text-layer words counts as incomplete (1 % of corpus pages, 2026-09)
MIN_PAGE_RECALL = 0.95
KNOWN_TYPES = {
    "text", "header", "footer", "aside_text", "page_footnote", "ref_text", "list", "index",
    "equation", "code", "image", "chart", "table", "page_number", "algorithm",
}


def check_structure(content_list: list[dict], target: Path) -> list[dict[str, Any]]:
    """Defects that make an output unusable: unknown types, missing images, empty visuals."""
    problems: list[dict[str, Any]] = []
    unknown = sorted({b.get("type") for b in content_list} - KNOWN_TYPES)
    if unknown:
        problems.append({"problem": "unknown_block_types", "types": unknown})
    missing_images = sorted(
        b["img_path"] for b in content_list if b.get("img_path") and not (target / b["img_path"]).exists()
    )
    if missing_images:
        problems.append({"problem": "missing_image_files", "paths": missing_images[:10], "count": len(missing_images)})
    empty_tables = [b.get("page_idx") for b in content_list if b.get("type") == "table" and not (b.get("table_body") or b.get("img_path"))]
    if empty_tables:
        problems.append({"problem": "empty_tables", "pages": sorted({p + 1 for p in empty_tables if p is not None})})
    return problems


def check_headings(content_list: list[dict]) -> list[dict[str, Any]]:
    """Heading levels must form a tree: never more than one level deeper than the heading before."""
    levels = [(b.get("page_idx", 0) + 1, b.get("text", ""), b["text_level"]) for b in content_list if b.get("text_level")]
    jumps = [
        {"page": page, "title": title[:60], "from": previous, "to": level}
        for (_, _, previous), (page, title, level) in zip(levels, levels[1:])
        if level - previous > 1
    ]
    return [{"problem": "heading_level_jumps", "count": len(jumps), "examples": jumps[:5]}] if jumps else []


def check_document(
    content_list: list[dict],
    pdf_path: Path,
    target: Path,
    *,
    min_page_recall: float = MIN_PAGE_RECALL,
) -> dict[str, Any]:
    """Full acceptance report for one parsed document: ok, incomplete or failed, and why."""
    completeness = check_completeness(content_list, pdf_path) if pdf_path.suffix.lower() == ".pdf" else None
    problems = check_structure(content_list, target) + check_headings(content_list)

    incomplete_pages: list[dict[str, Any]] = []
    if completeness:
        incomplete_pages = [p for p in completeness["pages"] if p["recall"] < min_page_recall]

    if any(p["problem"] in {"unknown_block_types", "missing_image_files"} for p in problems):
        status = "failed"
    elif incomplete_pages:
        status = "incomplete"
    elif completeness is None:
        status = "unchecked"
    elif completeness["pages_checked"] == 0:
        status = "unchecked"  # scan or images only: no text layer to check against
    else:
        status = "ok"

    return {
        "status": status,
        "min_page_recall": min_page_recall,
        "completeness": completeness and {k: v for k, v in completeness.items() if k != "pages"},
        "incomplete_pages": incomplete_pages,
        "problems": problems,
    }
