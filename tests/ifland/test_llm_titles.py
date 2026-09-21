from __future__ import annotations

import json

import pytest

from mineru_ifland import llm_titles
from mineru_ifland.title_levels import apply_title_levels, uncertain_titles
from test_title_levels import _doc, _levels, _title


@pytest.fixture
def answers(monkeypatch, tmp_path):
    """Replace the model call; records the prompts and returns queued answers."""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    prompts: list[str] = []
    queue: list[object] = []

    def fake_ask(profile, prompt):
        prompts.append(prompt)
        if not queue:
            raise AssertionError("asked more often than expected")
        return queue.pop(0), False

    monkeypatch.setattr(llm_titles, "_ask", fake_ask)
    return {"prompts": prompts, "queue": queue}


def test_only_open_headings_are_asked_and_anchors_stay(answers):
    doc = _doc(
        [_title(0, "Was ist das?", doc=True), _title(1, "1.1 Einführung"), _title(2, "Fallbeispiel")],
        [_title(0, "1.2 Geschichte"), _title(1, "Unter der Lupe"), _title(2, "1.3 Ausblick")],
    )
    mode = apply_title_levels(doc)
    assert mode == "numbered"
    assert uncertain_titles(doc, mode) == {2, 4}
    answers["queue"].append({"2": 2, "4": 2})  # the model lifts both boxes to section level

    outcome = llm_titles.apply_llm_levels(doc, mode, "local")

    assert outcome.accepted and outcome.asked == 2 and outcome.changed == 2
    assert _levels(doc) == [
        ("Was ist das?", 1), ("1.1 Einführung", 2), ("Fallbeispiel", 2),
        ("1.2 Geschichte", 2), ("Unter der Lupe", 2), ("1.3 Ausblick", 2),
    ]
    entries = json.loads(answers["prompts"][0].split("Überschriften:\n", 1)[1])
    assert [e["frei"] for e in entries] == [False, False, True, False, True, False]


def test_an_answer_that_touches_a_fixed_heading_is_rejected(answers):
    doc = _doc([_title(0, "1.1 Einführung"), _title(1, "Kasten"), _title(2, "1.2 Methode"), _title(3, "1.3 Ausblick")])
    mode = apply_title_levels(doc)
    answers["queue"].extend([{"0": 5, "1": 3}, {"1": 3}])  # first answer moves a fixed heading

    outcome = llm_titles.apply_llm_levels(doc, mode, "local")

    assert outcome.accepted
    assert _levels(doc) == [("1.1 Einführung", 2), ("Kasten", 3), ("1.2 Methode", 2), ("1.3 Ausblick", 2)]


def test_incomplete_or_jumping_answers_keep_the_rule_result(answers):
    doc = _doc([_title(0, "1.1 Einführung"), _title(1, "Kasten"), _title(2, "Zweiter Kasten"),
                _title(3, "1.2 Methode"), _title(4, "1.3 Ausblick")])
    mode = apply_title_levels(doc)
    before = _levels(doc)
    answers["queue"].extend([{"1": 6}, {"1": 3}])  # jump 2 -> 6, then an answer missing heading 2

    outcome = llm_titles.apply_llm_levels(doc, mode, "local")

    assert not outcome.accepted
    assert "level jump" in outcome.reason or "without a level" in outcome.reason
    assert _levels(doc) == before


def test_a_failing_model_leaves_the_document_untouched(answers, monkeypatch):
    doc = _doc([_title(0, "1.1 Einführung"), _title(1, "Kasten"), _title(2, "1.2 Methode"), _title(3, "1.3 Ausblick")])
    mode = apply_title_levels(doc)
    before = _levels(doc)
    monkeypatch.setattr(llm_titles, "_ask", lambda *_: (_ for _ in ()).throw(ConnectionError("no server")))

    outcome = llm_titles.apply_llm_levels(doc, mode, "local")

    assert not outcome.accepted and "ConnectionError" in outcome.reason
    assert _levels(doc) == before


def test_slides_ask_only_about_the_extra_headings_of_a_page(answers):
    doc = _doc(
        [_title(0, "EFA", doc=True), _title(1, "Ladungen"), _title(2, "Fördert"), _title(3, "Hemmt")],
        [_title(0, "EFA"), _title(1, "Eigenwert")],
    )
    mode = apply_title_levels(doc, "auto", landscape=True)

    assert mode == "slides"
    assert uncertain_titles(doc, mode) == {2, 3}


def test_unknown_profile_is_refused(answers):
    doc = _doc([_title(0, "Kasten")])
    with pytest.raises(SystemExit, match="unknown LLM profile"):
        llm_titles.apply_llm_levels(doc, "off", "quantum")


def test_answers_are_cached_per_model_and_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    calls = []

    class _Client:
        def __init__(self, **kwargs):
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("M", (), {"content": '{"1": 3}'})
            return type("R", (), {"choices": [type("C", (), {"message": message})]})

    monkeypatch.setattr("openai.OpenAI", _Client)
    profile = llm_titles.DEFAULT_PROFILES["local"]

    first, cached_first = llm_titles._ask(profile, "prompt")
    second, cached_second = llm_titles._ask(profile, "prompt")

    assert first == second == {"1": 3}
    assert (cached_first, cached_second) == (False, True)
    assert len(calls) == 1 and calls[0]["temperature"] == 0
