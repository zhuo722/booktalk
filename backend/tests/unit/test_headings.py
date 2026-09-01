"""Heading-family voting (SPEC §14.2, tier 2.5)."""

from __future__ import annotations

from app.ingest.headings import (
    MIN_VOTES,
    any_family_level,
    family_levels,
    is_heading_like,
    normalize_levels,
    tally,
    vote,
)

ENGLISH_BOOK = [
    "PART ONE. THE OUTER ISLES",
    "Chapter 1. Isle of Basalt",
    "the ferry leaves the quay at dawn and returns before the tide turns",
    "Chapter 2. Isle of Granite",
    "granite was quarried here until the last kiln closed",
    "Chapter 3. Isle of Salt",
]

CJK_BOOK = [
    "第一章 灯塔与港口",
    "渡轮在黎明时分离开码头，在退潮之前返回。",
    "第二章 采石场",
    "这座岛屿的花岗岩一直开采到最后一座窑关闭为止。",
    "第三章 盐田",
]

NUMBERED_BOOK = [
    "1. Harbours",
    "1.1 The ferry terminal",
    "the ferry leaves the quay at dawn",
    "2. Quarries",
    "2.1 The basalt face",
    "3. Salt pans",
]


def test_english_family_wins_an_english_book():
    winner = vote(ENGLISH_BOOK)

    assert winner is not None
    assert winner.name == "english"
    assert winner.matches == 4  # one part heading plus three chapters
    assert winner.counts["cjk"] == 0


def test_cjk_family_wins_a_chinese_book():
    winner = vote(CJK_BOOK)

    assert winner is not None
    assert winner.name == "cjk"
    assert winner.matches == 3


#: 西遊記 prints its tenth chapter as 第一○回, where ○ is U+25CB WHITE CIRCLE
#: rather than the ideographic zero — the convention a real Chinese book uses.
WHITE_CIRCLE_BOOK = [
    "第九回　陳光蕊赴任逢災",
    "話說那唐僧的父親陳光蕊，中了狀元，跨馬遊街。",
    "第一○回　老龍王拙計犯天條",
    "卻說長安城外涇河岸邊，有兩個賢人，一個是漁翁，一個是樵子。",
    "第一一回　遊地府太宗還魂",
]


def test_white_circle_counts_as_a_zero():
    """Every chapter whose number contains a zero is lost otherwise."""
    winner = vote(WHITE_CIRCLE_BOOK)

    assert winner is not None
    assert winner.name == "cjk"
    assert winner.matches == 3


def test_numeric_family_wins_and_ranks_its_levels():
    winner = vote(NUMBERED_BOOK)

    assert winner is not None
    assert winner.name == "numeric"
    assert winner.level_of("2. Quarries") == 1
    assert winner.level_of("2.1 The basalt face") == 2  # deeper pattern wins
    assert winner.level_of("the ferry leaves the quay at dawn") is None


def test_a_book_with_no_convention_elects_nobody():
    prose = [
        "the ferry leaves the quay at dawn and returns before the tide turns",
        "granite was quarried here until the last kiln closed in 1911",
        "Isle of Basalt",
        "Isle of Granite",
    ]

    assert vote(prose) is None
    assert tally(prose) == {"english": 0, "cjk": 0, "numeric": 0, "markdown": 0}


def test_too_few_matches_do_not_carry_an_election():
    assert vote(["Chapter 1. Isle", "Chapter 2. Isle"], min_votes=MIN_VOTES) is None
    assert vote(["Chapter 1. Isle", "Chapter 2. Isle"], min_votes=2) is not None


def test_body_lines_that_open_with_a_number_are_screened_out():
    """The shape test runs before any pattern: only short lines can be headings."""
    long_line = "1. " + " ".join(["harbour"] * 40)

    assert not is_heading_like(long_line)
    assert any_family_level(long_line) is None
    assert any_family_level("1. Harbours") == 1
    assert any_family_level("a clause that trails off,") is None


def test_markdown_family_is_recognized():
    winner = vote(["# Part One", "## Chapter 1", "## Chapter 2", "body text here"])

    assert winner is not None
    assert winner.name == "markdown"
    assert winner.level_of("## Chapter 1") == 2


def test_levels_are_compacted_to_start_at_one():
    assert normalize_levels([2, 2, 3, 2]) == [1, 1, 2, 1]
    assert normalize_levels([1, 2, 4]) == [1, 2, 3]


def test_family_levels_assigns_a_hierarchy_to_titles():
    titles = ["PART ONE", "Chapter 1. Isle", "Chapter 2. Isle", "Chapter 3. Isle"]

    assert family_levels(titles) == [1, 2, 2, 2]


def test_family_levels_gives_up_when_no_family_matches():
    assert family_levels(["Preface", "Isle of Basalt", "Index"]) == []
