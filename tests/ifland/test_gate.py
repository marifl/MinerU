from __future__ import annotations

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from mineru_ifland.gate import check_completeness, check_document

WIDTH, HEIGHT = A4


@pytest.fixture
def pdf(tmp_path):
    def build(*lines: tuple[float, float, str]) -> object:
        path = tmp_path / "page.pdf"
        page = canvas.Canvas(str(path), pagesize=A4)
        for x, y, text in lines:
            page.drawString(x, y, text)
        page.save()
        return path

    return build


def _text_block(text: str, bbox=(100, 100, 900, 200)) -> dict:
    return {"type": "text", "text": text, "page_idx": 0, "bbox": list(bbox)}


def test_line_break_hyphenation_counts_as_present(pdf):
    # the text layer of these PDFs breaks lines with \r\n; the output joins the word
    source = pdf((72, HEIGHT - 100, "Arbeits-"), (72, HEIGHT - 120, "gedaechtnis belastet die Aufgabe"))

    report = check_completeness([_text_block("Arbeitsgedaechtnis belastet die Aufgabe")], source)

    assert report["pages"][0]["missing"] == []


def test_glued_or_split_words_in_the_text_layer_count_as_present(pdf):
    # some PDFs carry a broken text layer ("aufdem"); the output reads the rendered page correctly
    source = pdf((72, HEIGHT - 100, "Ergebnisse aufdem Pruefstand derStudie mit Teilnehmenden"))

    report = check_completeness([_text_block("Ergebnisse auf dem Pruefstand der Studie mit Teilnehmenden")], source)

    assert report["pages"][0]["missing"] == []


def test_text_outside_the_page_area_is_not_required(pdf):
    source = pdf((72, HEIGHT - 100, "Sichtbarer Text auf dieser gedruckten Seite"), (72, -80, "verstecktes englisches Formular"))

    report = check_completeness([_text_block("Sichtbarer Text auf dieser gedruckten Seite")], source)

    assert report["pages"][0]["missing"] == []


def test_words_inside_a_picture_are_not_required(pdf):
    source = pdf(
        (72, HEIGHT - 100, "Ueberschrift der Seite mit weiterem Fliesstext"),
        (72, HEIGHT - 300, "Beschriftung im Diagramm"),
    )
    picture = {"type": "image", "page_idx": 0, "img_path": "images/a.jpg", "bbox": [0, 250, 1000, 450]}

    report = check_completeness([_text_block("Ueberschrift der Seite mit weiterem Fliesstext"), picture], source)

    assert report["pages"][0]["missing"] == []


def test_missing_words_are_reported_per_page(pdf):
    source = pdf((72, HEIGHT - 100, "Christi Himmelfahrt Feiertag"), (72, HEIGHT - 130, "Corpus Christi Wiesbaden Termin"))

    report = check_completeness([_text_block("Himmelfahrt Feiertag Termin")], source)

    page = report["pages"][0]
    assert page["missing"] == ["christi", "corpus", "wiesbaden"]
    assert page["recall"] < 0.95


def test_document_status_incomplete_names_the_pages(pdf, tmp_path):
    source = pdf((72, HEIGHT - 100, "Christi Himmelfahrt Feiertag Corpus Wiesbaden Termin"))

    report = check_document([_text_block("Himmelfahrt Feiertag")], source, tmp_path)

    assert report["status"] == "incomplete"
    assert [p["page"] for p in report["incomplete_pages"]] == [1]


def test_document_status_failed_on_unknown_type_or_missing_image(pdf, tmp_path):
    source = pdf((72, HEIGHT - 100, "Ein vollstaendiger Satz steht hier"))
    blocks = [
        _text_block("Ein vollstaendiger Satz steht hier"),
        {"type": "hologram", "page_idx": 0},
        {"type": "image", "page_idx": 0, "img_path": "images/gone.jpg", "bbox": [0, 0, 10, 10]},
    ]

    report = check_document(blocks, source, tmp_path)

    assert report["status"] == "failed"
    assert {p["problem"] for p in report["problems"]} == {"unknown_block_types", "missing_image_files"}


def test_document_status_ok_and_heading_jumps_are_reported(pdf, tmp_path):
    source = pdf((72, HEIGHT - 100, "Kapitel eins"), (72, HEIGHT - 130, "Unterabschnitt mit Inhalt"))
    blocks = [
        {"type": "text", "text": "Kapitel eins", "text_level": 2, "page_idx": 0, "bbox": [100, 100, 900, 150]},
        {"type": "text", "text": "Unterabschnitt mit Inhalt", "text_level": 4, "page_idx": 0, "bbox": [100, 160, 900, 200]},
    ]

    report = check_document(blocks, source, tmp_path)

    assert report["status"] == "ok"
    assert report["problems"] == [{"problem": "heading_level_jumps", "count": 1, "examples": [
        {"page": 1, "title": "Unterabschnitt mit Inhalt", "from": 2, "to": 4}
    ]}]


def test_pages_without_text_layer_are_unchecked_not_ok(pdf, tmp_path):
    source = pdf((72, HEIGHT - 100, "ab"))  # too little text to check against

    report = check_document([], source, tmp_path)

    assert report["status"] == "unchecked"
    assert report["completeness"]["pages_without_text_layer"] == 1
