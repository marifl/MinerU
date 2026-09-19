from __future__ import annotations

from docvortex.content.inline import inline_plain_text
from docvortex.schema import Producer

from mineru.integrations.docvortex import build_metadata
from mineru.types import BlockType, DocTitleBlock, MiddleJson, PageInfo, ParagraphTitleBlock, TextBlock
from mineru_ifland.title_levels import apply_title_levels


BBOX = (0.1, 0.1, 0.9, 0.2)


def _span(text: str) -> list[dict]:
    return [{"type": "text", "content": text}]


def _title(index: int, text: str, *, doc: bool = False):
    if doc:
        return DocTitleBlock(type=BlockType.DOC_TITLE, index=index, bbox=BBOX, content=_span(text), level=1)
    return ParagraphTitleBlock(type=BlockType.PARAGRAPH_TITLE, index=index, bbox=BBOX, content=_span(text), level=2)


def _text(index: int, text: str) -> TextBlock:
    return TextBlock(type=BlockType.TEXT, index=index, bbox=BBOX, content=_span(text))


def _doc(*pages: list) -> MiddleJson:
    return MiddleJson(
        pages=[PageInfo(page_idx=i, blocks=blocks) for i, blocks in enumerate(pages)],
        is_full_document=True,
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="test")},
        extensions=build_metadata(effort="flash", parse_mode="txt"),
    )


def _levels(middle_json: MiddleJson) -> list[tuple[str, int]]:
    return [
        (inline_plain_text(b.content), b.level)
        for page in middle_json.pages
        for b in page.blocks
        if isinstance(b, (DocTitleBlock, ParagraphTitleBlock))
    ]


def test_slides_running_title_above_slide_titles_regardless_of_block_type():
    doc = _doc(
        [_title(0, "Lecture", doc=True), _title(1, "Agenda")],
        [_title(0, "EFA", doc=True), _title(1, "Loadings"), _title(2, "Note")],
        [_title(0, "EFA"), _title(1, "Eigenvalue")],
        [_title(0, "Summary")],
    )

    assert apply_title_levels(doc, "auto", landscape=True) == "slides"

    assert _levels(doc) == [
        ("Lecture", 1),
        ("Agenda", 2),
        ("EFA", 2),
        ("Loadings", 3),
        ("Note", 4),
        ("EFA", 2),
        ("Eigenvalue", 3),
        ("Summary", 2),
    ]
    assert all(isinstance(b, ParagraphTitleBlock) for b in doc.pages[1].blocks)


def test_numbered_dotted_depth_and_recurring_boxes_get_one_level():
    doc = _doc(
        [_title(0, "Motion vision", doc=True), _title(1, "5.1 Systems"), _title(2, "5.1.1 Cortical")],
        [_title(0, "Case study"), _text(1, "body"), _title(2, "5.1.2 Subcortical")],
        [_title(0, "Case study"), _title(1, "5.2 Models"), _title(2, "Stray heading")],
    )

    assert apply_title_levels(doc) == "numbered"

    assert _levels(doc) == [
        ("Motion vision", 1),
        ("5.1 Systems", 2),
        ("5.1.1 Cortical", 3),
        ("Case study", 4),
        ("5.1.2 Subcortical", 3),
        ("Case study", 4),
        ("5.2 Models", 2),
        ("Stray heading", 3),
    ]


def test_legal_numbering_ranks_sections_above_paragraphs_and_merges_bare_numbers():
    doc = _doc(
        [_title(0, "Ordnung", doc=True), _title(1, "Inhalt")],
        [_title(0, "I. Allgemeines"), _title(1, "§ 1 Geltung"), _title(2, "§ 2"), _title(3, "Ziel"), _text(4, "x")],
        [_title(0, "II. Prüfungen"), _title(1, "§ 3 Ausschuss")],
    )

    assert apply_title_levels(doc) == "numbered"

    assert _levels(doc) == [
        ("Ordnung", 1),
        ("Inhalt", 2),
        ("I. Allgemeines", 2),
        ("§ 1 Geltung", 3),
        ("§ 2 Ziel", 3),
        ("II. Prüfungen", 2),
        ("§ 3 Ausschuss", 3),
    ]
    assert [b.index for b in doc.pages[1].blocks] == [0, 1, 2, 4]


def test_auto_leaves_unnumbered_portrait_documents_untouched():
    doc = _doc([_title(0, "Intro"), _title(1, "Method")], [_title(0, "Results")])

    assert apply_title_levels(doc) == "off"

    assert _levels(doc) == [("Intro", 2), ("Method", 2), ("Results", 2)]


def test_off_changes_nothing():
    doc = _doc([_title(0, "§ 1"), _title(1, "Ziel")])

    assert apply_title_levels(doc, "off") == "off"

    assert _levels(doc) == [("§ 1", 2), ("Ziel", 2)]
