"""Deterministic heading levels for a MinerU 4.x MiddleJson, without an LLM.

Without LLM post-processing MinerU puts every paragraph title on level 2. These rules restore
a hierarchy from what the document itself states:

- slides: a running title (same text is the first title on at least two pages) is level 2;
  the first own title of a page is level 3 below a running title, else 2; further titles on
  the page sit one level deeper. A doc title on the first page that is not running stays level 1.
- numbered: numbering decides, whether MinerU saw the title as doc title or paragraph title.
  Roman sections ("II.", "Abschnitt 2") rank above letters ("A)"), above dotted numbers
  ("5", "Kapitel 5", "5.1", "5.1.2"), above "§ n". Table-of-contents lines carry no rank.
  An unnumbered title right before "x.y.1" is that section's parent; an unnumbered doc title
  (book or chapter title) stays level 1, and when such doc titles head "x.y" sections, numbered
  chapters ("2 Das Gehirn") become level-1 doc titles as well; other unnumbered titles, boxes like "Fallbeispiel"
  included, sit one level below the section they appear in. A bare number ("§ 2") directly
  followed by an unnumbered title is merged with it.
- auto: slides if most PDF pages are landscape; numbered if at least 3 titles carry numbers and
  they are 30 % of all titles or 5 distinct numbers; otherwise the levels stay as MinerU made them.
"""

from __future__ import annotations

import re
from bisect import bisect_left
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
# ... or when at least this many distinct numbers occur (books numbering only their chapters)
MIN_DISTINCT_NUMBERS = 5
# a numbering kind (roman, letters, dotted depth, §) shapes the outline only with this much support
MIN_KIND_TITLES = 10
MIN_KIND_SHARE = 0.05

ROMAN = re.compile(r"^(?:[IVX]{1,5}\.|(?:Abschnitt|Teil)\s+(?:[IVX]{1,5}|\d{1,2})\b)", re.IGNORECASE)
DOTTED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)(?:\.?\s+\S|\.(?=[^\d\s.])|\.?$)")
CHAPTER = re.compile(r"^(?:Kapitel|Chapter|Kap\.)\s+(\d{1,2})\b", re.IGNORECASE)
ALPHA = re.compile(r"^(?:[A-H]\)\s+\S|[A-H]\.\s+[A-ZÄÖÜ][a-zäöüß]{2,})")
PARAGRAPH = re.compile(r"^§\s*\d+")
DOT_LEADER = re.compile(r"(?:\.\s*){2,}\d{1,4}\s*$")
TRAILING_PAGE = re.compile(r"[^\d\s§]\.?\s+\d{1,4}\s*$")
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
    return " ".join(inline_plain_text(block.content).replace("．", ".").split())


def _key(text: str) -> str:
    return text.casefold()


def _as_paragraph_title(block: DocTitleBlock | ParagraphTitleBlock, level: int) -> ParagraphTitleBlock:
    if isinstance(block, ParagraphTitleBlock):
        block.level = level
        return block
    return ParagraphTitleBlock.model_validate(
        {**block.model_dump(), "type": BlockType.PARAGRAPH_TITLE, "level": level}
    )


def _as_doc_title(block: DocTitleBlock | ParagraphTitleBlock) -> DocTitleBlock:
    if isinstance(block, DocTitleBlock):
        return block
    return DocTitleBlock.model_validate({**block.model_dump(), "type": BlockType.DOC_TITLE, "level": MIN_LEVEL - 1})


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
    """Rank key of a numbered title: ("roman", 0), ("alpha", 0), ("dotted", depth) or ("paragraph", 0)."""
    if ROMAN.match(text):
        return ("roman", 0)
    if ALPHA.match(text):
        return ("alpha", 0)
    if PARAGRAPH.match(text):
        return ("paragraph", 0)
    if CHAPTER.match(text):
        return ("dotted", 1)
    match = DOTTED.match(text)
    if match:
        return ("dotted", match.group(1).count(".") + 1)
    return None


def _number_token(text: str) -> str | None:
    match = CHAPTER.match(text) or DOTTED.match(text)
    if match:
        return match.group(1)
    if ROMAN.match(text) or ALPHA.match(text) or PARAGRAPH.match(text):
        return " ".join(text.split()[:2]) if text.startswith("§") else text.split()[0]
    return None


ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}


def _sort_value(kind: str, token: str) -> tuple[int, ...]:
    """Position of a number within its own sequence, for the order check."""
    if kind == "dotted":
        return tuple(int(part) for part in token.split("."))
    if kind == "paragraph":
        return (int(re.sub(r"\D", "", token) or 0),)
    if kind == "alpha":
        return (ord(token[0].upper()),)
    letters = [ROMAN_VALUES.get(c, 0) for c in token.rstrip(".").upper() if c in ROMAN_VALUES]
    value = sum(-v if i + 1 < len(letters) and v < letters[i + 1] else v for i, v in enumerate(letters))
    return (value,)


def _longest_increasing(indices: list[int], values: list[tuple[int, ...]]) -> set[int]:
    """Indices of a longest strictly increasing subsequence of `values` (patience sorting)."""
    tails: list[tuple[int, ...]] = []
    tail_at: list[int] = []
    previous: list[int | None] = []
    for position, value in enumerate(values):
        slot = bisect_left(tails, value)
        if slot == len(tails):
            tails.append(value)
            tail_at.append(position)
        else:
            tails[slot] = value
            tail_at[slot] = position
        previous.append(tail_at[slot - 1] if slot else None)
    keep: set[int] = set()
    position = tail_at[-1] if tail_at else None
    while position is not None:
        keep.add(indices[position])
        position = previous[position]
    return keep


def _document_numbering(texts: list[str], chapter_starts: list[bool]) -> list[tuple[str, int] | None]:
    """Numbering per title, keeping only numbers that structure the document.

    Dropped, so they rank nothing and count as unnumbered:
    - table-of-contents lines: dot leaders ("1 Einleitung .... 5"), or a trailing page number while
      a later heading repeats the number ("2.1 Methode 12" before "2.1 Methode");
    - numbers outside the longest strictly increasing run of their kind, like a numbered list
      MinerU took for headings ("1. Exponentielle Dynamik" amid "7.3.x"), a stray TOC line before
      chapter 1, or a repeated running head; an unnumbered doc title (`chapter_starts`) starts a new run, for books whose
      chapters each count from 1;
    - kinds too rare to be the document's outline: fewer than MIN_KIND_SHARE of all numbered
      titles and fewer than MIN_KIND_TITLES (five stray "I." boxes in a book of 500 "x.y" sections).
    """
    tokens = [_number_token(text) for text in texts]
    numbering: list[tuple[str, int] | None] = []
    for i, (text, token) in enumerate(zip(texts, tokens)):
        n = _numbering(text) if token is not None else None
        toc = n is not None and (
            DOT_LEADER.search(text) or (TRAILING_PAGE.search(text) and token in tokens[i + 1 :])
        )
        numbering.append(None if toc else n)

    # per chapter and kind, keep the longest strictly increasing run of numbers
    segment = 0
    sequences: dict[tuple[int, str], list[int]] = {}
    for i, n in enumerate(numbering):
        segment += chapter_starts[i]
        if n:
            sequences.setdefault((segment, n[0]), []).append(i)
    for (_, kind), indices in sequences.items():
        keep = _longest_increasing(indices, [_sort_value(kind, tokens[i]) for i in indices])
        for i in indices:
            if i not in keep:
                numbering[i] = None

    counts = Counter(kind for kind, _ in filter(None, numbering))
    total = sum(counts.values())
    minimum = min(MIN_KIND_TITLES, max(2, MIN_KIND_SHARE * total))
    return [n if n and counts[n[0]] >= minimum else None for n in numbering]


def _first_child_depth(text: str, numbering: tuple[str, int] | None) -> int | None:
    """Depth of a dotted number that opens a section ("2.1.1" -> 3), else None."""
    if numbering is None or numbering[0] != "dotted" or numbering[1] < 2:
        return None
    return numbering[1] if DOTTED.match(text).group(1).endswith(".1") else None


TitleBlock = DocTitleBlock | ParagraphTitleBlock


def _title_positions(middle_json: MiddleJson) -> list[tuple[list, int, TitleBlock]]:
    return [
        (page.blocks, position, block)
        for page in middle_json.pages
        for position, block in enumerate(page.blocks)
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock))
    ]


def _chapter_starts(positions: list[tuple[list, int, TitleBlock]], texts: list[str]) -> list[bool]:
    return [isinstance(block, DocTitleBlock) and _numbering(text) is None for (_, _, block), text in zip(positions, texts)]


def _looks_numbered(middle_json: MiddleJson) -> bool:
    positions = _title_positions(middle_json)
    texts = [_text(block) for _, _, block in positions]
    numbered = [text for text, n in zip(texts, _document_numbering(texts, _chapter_starts(positions, texts))) if n]
    distinct = {_number_token(text) for text in numbered}
    return len(numbered) >= MIN_NUMBERED_TITLES and (
        len(numbered) >= MIN_NUMBERED_SHARE * len(texts) or len(distinct) >= MIN_DISTINCT_NUMBERS
    )


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


def _numbered_chapters_are_doc_titles(
    positions: list[tuple[list, int, TitleBlock]],
    numbering: list[tuple[str, int] | None],
    ranks: list[tuple[str, int]],
) -> bool:
    """True when MinerU made some chapter titles unnumbered doc titles over "x.y" sections, so the
    numbered chapter titles ("2 Das Gehirn") belong on the same level 1 instead of level 2.

    Needs at least two such doc titles and no roman or letter numbering above the chapters.
    """
    if ("dotted", 1) not in ranks or any(kind in ("roman", "alpha") for kind, _ in ranks):
        return False
    chapters, open_chapter = 0, False
    for (_, _, block), n in zip(positions, numbering):
        if isinstance(block, DocTitleBlock) and n is None:
            open_chapter = True
        elif n == ("dotted", 1):
            open_chapter = False
        elif open_chapter and n == ("dotted", 2):
            chapters += 1
            open_chapter = False
    return chapters >= 2


def _level_numbered(middle_json: MiddleJson) -> None:
    positions = _title_positions(middle_json)
    texts = [_text(block) for _, _, block in positions]
    numbering = _document_numbering(texts, _chapter_starts(positions, texts))
    ranks = sorted({n for n in numbering if n}, key=lambda n: ({"roman": 0, "alpha": 1, "dotted": 2, "paragraph": 3}[n[0]], n[1]))
    chapter_level = _numbered_chapters_are_doc_titles(positions, numbering, ranks)
    if chapter_level:
        ranks.remove(("dotted", 1))
    level_of = {n: min(MIN_LEVEL + i, MAX_LEVEL) for i, n in enumerate(ranks)}

    last_numbered_level: int | None = None
    for i, ((blocks, position, block), n) in enumerate(zip(positions, numbering)):
        if n and chapter_level and n == ("dotted", 1):
            blocks[position] = _as_doc_title(block)
            last_numbered_level = MIN_LEVEL - 1
            continue
        if n:
            level = level_of[n]
            last_numbered_level = level
        elif isinstance(block, DocTitleBlock):
            last_numbered_level = None  # an unnumbered doc title (book or chapter title) opens a new context
            continue
        elif i + 1 < len(texts) and (depth := _first_child_depth(texts[i + 1], numbering[i + 1])):
            # an unnumbered title right before "x.y.1" is the section that number belongs to
            level = max(MIN_LEVEL, level_of[("dotted", depth)] - 1)
            last_numbered_level = level
        else:
            # boxes and other unnumbered titles belong to the section they appear in
            level = MIN_LEVEL if last_numbered_level is None else min(last_numbered_level + 1, MAX_LEVEL)
        blocks[position] = _as_paragraph_title(block, level)


def iter_titles(middle_json: MiddleJson) -> list[tuple[int, list, int, TitleBlock]]:
    """All headings in reading order as (page_idx, page blocks, position, block)."""
    return [
        (page.page_idx, page.blocks, position, block)
        for page in middle_json.pages
        for position, block in enumerate(page.blocks)
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock))
    ]


def uncertain_titles(middle_json: MiddleJson, mode: AppliedMode) -> set[int]:
    """Indices (into `iter_titles`) of headings whose level the rules did not decide.

    Those are the ones an LLM stage may re-rank: everything in an untouched document, the
    unnumbered ones in a numbered document, and the extra headings of a slide below its own title.
    """
    titles = iter_titles(middle_json)
    if mode == "off":
        return {i for i, (_, _, _, block) in enumerate(titles) if isinstance(block, ParagraphTitleBlock)}
    if mode == "numbered":
        texts = [_text(block) for _, _, _, block in titles]
        chapter_starts = [
            isinstance(block, DocTitleBlock) and _numbering(text) is None
            for (_, _, _, block), text in zip(titles, texts)
        ]
        numbering = _document_numbering(texts, chapter_starts)
        return {
            i
            for i, ((_, _, _, block), n) in enumerate(zip(titles, numbering))
            if n is None and isinstance(block, ParagraphTitleBlock)
        }
    seen_on_page: set[int] = set()
    uncertain: set[int] = set()
    for i, (page_idx, _, _, block) in enumerate(titles):
        if not isinstance(block, ParagraphTitleBlock):
            continue
        if page_idx in seen_on_page and block.level > MIN_LEVEL + 1:
            uncertain.add(i)
        seen_on_page.add(page_idx)
    return uncertain
