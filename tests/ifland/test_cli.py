from __future__ import annotations

import json

import pypdfium2 as pdfium
import pytest

from mineru.parser.base import ParseResult
from mineru_ifland import cli

from test_title_levels import _doc, _text, _title


@pytest.fixture
def landscape_pdf(tmp_path):
    doc = pdfium.PdfDocument.new()
    for _ in range(2):
        doc.new_page(800, 600)
    path = tmp_path / "Vorlesung 3.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_writes_3x_layout_with_levels_and_provenance(tmp_path, landscape_pdf, monkeypatch):
    calls = []

    def fake_parse(path, **kwargs):
        calls.append(kwargs)
        return ParseResult(
            middle_json=_doc(
                [_title(0, "EFA", doc=True), _title(1, "Ladungen"), _text(2, "Text")],
                [_title(0, "EFA"), _title(1, "Eigenwert")],
            )
        )

    monkeypatch.setattr(cli, "parse", fake_parse)
    out = tmp_path / "out"

    assert cli.main(["-p", str(landscape_pdf), "-o", str(out), "-l", "latin", "-s", "1", "-e", "1"]) == 0

    target = out / "Vorlesung 3" / "hybrid_auto"
    names = sorted(p.name for p in target.iterdir())
    assert names == [
        "Vorlesung 3.md",
        "Vorlesung 3_content_list.json",
        "Vorlesung 3_content_list_v2.json",
        "Vorlesung 3_middle_v4.json",
        "Vorlesung 3_mineru.json",
        "Vorlesung 3_origin.pdf",
    ]
    content_list = json.loads((target / "Vorlesung 3_content_list.json").read_text())
    assert [(b["text"].strip(), b.get("text_level")) for b in content_list if b["type"] == "text"] == [
        ("EFA", 2),
        ("Ladungen", 3),
        ("Text", None),
        ("EFA", 2),
        ("Eigenwert", 3),
    ]
    provenance = json.loads((target / "Vorlesung 3_mineru.json").read_text())
    assert provenance["parse"] == {"tier": "standard", "ocr_mode": "auto", "image_analysis": True, "page_range": "2-2"}
    assert provenance["title_levels"] == {"requested": "auto", "applied": "slides"}
    assert provenance["ignored"] == {"lang": "latin"}
    assert calls == [{"tier": "standard", "ocr_mode": "auto", "image_analysis": True, "page_range": "2-2"}]


def test_pipeline_backend_maps_to_basic_tier_and_method_dir(tmp_path, landscape_pdf, monkeypatch):
    monkeypatch.setattr(cli, "parse", lambda path, **kwargs: ParseResult(middle_json=_doc([_text(0, "x")])))

    cli.main(["-p", str(landscape_pdf), "-o", str(tmp_path / "out"), "-b", "pipeline", "-m", "txt"])

    provenance = json.loads((tmp_path / "out" / "Vorlesung 3" / "txt" / "Vorlesung 3_mineru.json").read_text())
    assert provenance["parse"]["tier"] == "basic"
    assert provenance["parse"]["ocr_mode"] == "txt"


def test_refuses_switching_off_formula_recognition(tmp_path, landscape_pdf):
    with pytest.raises(SystemExit, match="formula or table"):
        cli.main(["-p", str(landscape_pdf), "-o", str(tmp_path), "-f", "false"])


def test_office_inputs_keep_their_own_heading_levels(tmp_path, monkeypatch):
    source = tmp_path / "Skript.docx"
    source.write_bytes(b"placeholder, parse is faked")
    calls = []

    def fake_parse(path, **kwargs):
        calls.append(kwargs["tier"])
        return ParseResult(middle_json=_doc([_title(0, "§ 1"), _title(1, "Ziel"), _title(2, "§ 2 Zweck")]))

    monkeypatch.setattr(cli, "parse", fake_parse)

    cli.main(["-p", str(source), "-o", str(tmp_path / "out")])

    provenance = json.loads((tmp_path / "out" / "Skript" / "hybrid_auto" / "Skript_mineru.json").read_text())
    assert calls == ["flash"]
    assert provenance["title_levels"] == {"requested": "auto", "applied": "off"}
