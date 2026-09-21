from __future__ import annotations

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from mineru_ifland.gate import check_document
from mineru_ifland.repair import REPAIR_SOURCE, repair_pages

WIDTH, HEIGHT = A4


@pytest.fixture
def pdf(tmp_path):
    def build(*lines: str) -> object:
        path = tmp_path / "page.pdf"
        page = canvas.Canvas(str(path), pagesize=A4)
        for number, text in enumerate(lines):
            page.drawString(72, HEIGHT - 100 - number * 20, text)
        page.save()
        return path

    return build


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text, "page_idx": 0, "bbox": [100, 100, 900, 200]}


def test_a_lost_line_is_filled_in_and_marked(pdf, tmp_path):
    source = pdf("Am sexuell aktivsten waren Frauen zwischen 26 und 35 Jahren", "Maenner zwischen 36 und 45 Jahren")
    content_list = [_text_block("en 26 und 35 Jahren und")]  # MinerU kept only a fragment

    records = repair_pages(content_list, source, [1])

    filled = [b for b in content_list if b.get("source") == REPAIR_SOURCE]
    assert [b["text"] for b in filled] == [
        "Am sexuell aktivsten waren Frauen zwischen 26 und 35 Jahren",
        "Maenner zwischen 36 und 45 Jahren",
    ]
    assert records == [{"page": 1, "lines": 2, "text": [b["text"][:80] for b in filled]}]
    assert check_document(content_list, source, tmp_path)["status"] == "ok"


def test_nothing_is_added_when_the_page_is_already_complete(pdf, tmp_path):
    source = pdf("Vollstaendige Zeile mit allen Woertern", "Zweite Zeile ebenfalls vollstaendig")
    content_list = [_text_block("Vollstaendige Zeile mit allen Woertern Zweite Zeile ebenfalls vollstaendig")]

    records = repair_pages(content_list, source, [1])

    assert records == []
    assert not [b for b in content_list if b.get("source") == REPAIR_SOURCE]


def test_words_split_across_lines_are_recovered(pdf, tmp_path):
    # the text layer splits a heading; MinerU's table reading lost it entirely
    source = pdf("Kognitiv", "-affektive Neurowissenschaften", "weitere Zeile mit Inhalt")
    content_list = [_text_block("weitere Zeile mit Inhalt")]

    repair_pages(content_list, source, [1])

    assert check_document(content_list, source, tmp_path)["status"] == "ok"


def test_existing_blocks_are_never_replaced_or_reordered(pdf, tmp_path):
    source = pdf("Erste Zeile der Seite", "Zweite verlorene Zeile hier")
    content_list = [_text_block("Erste Zeile der Seite"), {"type": "image", "page_idx": 0, "img_path": "a.jpg"}]
    original = [dict(block) for block in content_list]

    repair_pages(content_list, source, [1])

    assert content_list[:2] == original
    assert content_list[2]["source"] == REPAIR_SOURCE
