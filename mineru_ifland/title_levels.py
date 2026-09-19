"""Deterministic heading levels for a MinerU 4.x MiddleJson, without an LLM.

Without LLM post-processing MinerU puts every paragraph title on level 2. These rules restore
a hierarchy from what the document itself states:

- slides: a running title (same text is the first title on at least two pages) is level 2;
  the first own title of a page is level 3 below a running title, else 2; further titles on
  the page sit one level deeper. A doc title on the first page that is not running stays level 1.
- numbered: numbering decides. Roman sections ("II.", "Abschnitt 2") rank above dotted numbers
  ("5", "5.1", "5.1.2"), which rank above "§ n". Unnumbered titles sit one level below the last
  numbered one; a recurring unnumbered title ("Fallbeispiel", "Zusammenfassung") gets the level
  most of its occurrences would get, so it is the same everywhere. A bare number ("§ 2")
  directly followed by an unnumbered title is merged with it.
- auto: slides if most PDF pages are landscape, numbered if enough titles carry numbers,
  otherwise the levels stay as MinerU produced them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from docvortex.content.inline import inline_plain_text
from docvortex.schema import TextSpan

from mineru.types import BlockType, DocTitleBlock, MiddleJson, ParagraphTitleBlock

Mode = Literal["auto", "slides", "numbered", "off"]
AppliedMode = Literal["slides", "numbered", "off"]

MIN_LEVEL = 2
MAX_LEVEL = 6
# auto picks "numbered" when at least this many titles, and this share of titles, carry numbers
MIN_NUMBERED_TITLES = 3
MIN_NUMBERED_SHARE = 0.3

ROMAN = re.compile(r"^(?:[IVX]{1,5}\.|(?:Abschnitt|Teil)\s+(?:[IVX]{1,5}|\d{1,2})\b)", re.IGNORECASE)
DOTTED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?(?:\s+\S|$)")
PARAGRAPH = re.compile(r"^§\s*\d+")
BARE_NUMBER = re.compile(r"^(?:§\s*\d+[a-z]?|\d{1,2}(?:\.\d{1,2})*\.?)$")


def apply_title_levels(middle_json: MiddleJson, mode: Mode = "auto", *, landscape: bool = False) -> AppliedMode:
    """Rewrite title levels in place and return the mode that was applied.

    `landscape` states whether most pages of the source are landscape; only `auto` uses it.
    """
    if mode == "off":
        return "off"
    if mode == "slides" or (mode == "auto" and landscape):
        _level_slides(middle_json)
        return "slides"
    _merge_bare_numbers(middle_json)
    if mode == "numbered" or _looks_numbered(middle_json):
        _level_numbered(middle_json)
        return "numbered"
    return "off"


def _text(block: DocTitleBlock | ParagraphTitleBlock) -> str:
    return " ".join(inline_plain_text(block.content).split())


def _key(text: str) -> str:
    return text.casefold()


def _as_paragraph_title(block: DocTitleBlock | ParagraphTitleBlock, level: int) -> ParagraphTitleBlock:
    if isinstance(block, ParagraphTitleBlock):
        block.level = level
        return block
    return ParagraphTitleBlock.model_validate(
        {**block.model_dump(), "type": BlockType.PARAGRAPH_TITLE, "level": level}
    )


def _level_slides(middle_json: MiddleJson) -> None:
    first_titles = Counter()
    for page in middle_json.pages:
        titles = [b for b in page.blocks if isinstance(b, (DocTitleBlock, ParagraphTitleBlock))]
        if titles:
            first_titles[_key(_text(titles[0]))] += 1
    running = {key for key, count in first_titles.items() if count >= 2}

    for page in middle_json.pages:
        has_running = False
        own_level: int | None = None
        for position, block in enumerate(page.blocks):
            if not isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
                continue
            key = _key(_text(block))
            if key in running:
                has_running = True
                level = MIN_LEVEL
            elif isinstance(block, DocTitleBlock) and page.page_idx == 0 and own_level is None:
                own_level = MIN_LEVEL - 1
                continue
            elif own_level is None:
                own_level = MIN_LEVEL + 1 if has_running else MIN_LEVEL
                level = own_level
            else:
                level = min(own_level + 1, MAX_LEVEL)
            page.blocks[position] = _as_paragraph_title(block, level)


def _numbering(text: str) -> tuple[str, int] | None:
    """Rank key of a numbered title: ("roman", 0), ("dotted", depth) or ("paragraph", 0)."""
    if ROMAN.match(text):
        return ("roman", 0)
    if PARAGRAPH.match(text):
        return ("paragraph", 0)
    match = DOTTED.match(text)
    if match:
        return ("dotted", match.group(1).count(".") + 1)
    return None


def _paragraph_titles(middle_json: MiddleJson) -> list[ParagraphTitleBlock]:
    return [b for page in middle_json.pages for b in page.blocks if isinstance(b, ParagraphTitleBlock)]


def _looks_numbered(middle_json: MiddleJson) -> bool:
    titles = _paragraph_titles(middle_json)
    numbered = sum(1 for b in titles if _numbering(_text(b)))
    return numbered >= MIN_NUMBERED_TITLES and numbered >= MIN_NUMBERED_SHARE * len(titles)


def _merge_bare_numbers(middle_json: MiddleJson) -> None:
    for page in middle_json.pages:
        merged: list = []
        for block in page.blocks:
            previous = merged[-1] if merged else None
            if (
                isinstance(block, ParagraphTitleBlock)
                and isinstance(previous, ParagraphTitleBlock)
                and BARE_NUMBER.match(_text(previous))
                and not _numbering(_text(block))
            ):
                previous.content = [*previous.content, TextSpan(type="text", content=" "), *block.content]
                continue
            merged.append(block)
        page.blocks[:] = merged


def _level_numbered(middle_json: MiddleJson) -> None:
    titles = _paragraph_titles(middle_json)
    numbering = [_numbering(_text(b)) for b in titles]
    ranks = sorted({n for n in numbering if n}, key=lambda n: ({"roman": 0, "dotted": 1, "paragraph": 2}[n[0]], n[1]))
    level_of = {n: min(MIN_LEVEL + i, MAX_LEVEL) for i, n in enumerate(ranks)}
    last_numbered_level: int | None = None
    candidates: dict[str, Counter] = {}
    for block, n in zip(titles, numbering):
        if n:
            block.level = level_of[n]
            last_numbered_level = block.level
        else:
            block.level = MIN_LEVEL if last_numbered_level is None else min(last_numbered_level + 1, MAX_LEVEL)
            candidates.setdefault(_key(_text(block)), Counter())[block.level] += 1

    # a recurring box ("Fallbeispiel") gets one level everywhere: the most frequent, the higher one on ties
    for block, n in zip(titles, numbering):
        votes = None if n else candidates[_key(_text(block))]
        if votes and sum(votes.values()) >= 2:
            block.level = min(votes, key=lambda level: (-votes[level], level))
