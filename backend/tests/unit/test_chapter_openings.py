"""The chapter opening a reader sees, in a language with spaces and in one without.

`get_chapter_openings` used to measure a passage in whitespace-separated words
and rejoin its lines with a space. Both are latin assumptions. Two hundred
characters of Chinese come back from `str.split()` as one or two tokens, so
every chapter of a Chinese book failed the length test and the reader printed
its empty state over a chapter with plenty of text.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from app.index.store import (
    STATUS_READY,
    Store,
    _flatten,
    _substantial,
)
from app.ingest.chunker import Chunk
from app.ingest.structure import BookStructure, Chapter
from app.services.openai_client import FakeAIClient

# 老殘遊記's own opening, which is what a real ingest of it stores: one
# heading line, then prose with the extractor's line breaks still in it.
CH1_TITLE = "第一回　土不制水歷年成患　風能鼓浪到處可危"
CH1_BODY = (
    "話說山東登州府東門外有一座大山，名叫蓬萊山。山上有個閣子，名叫蓬萊閣。\n"
    "這閣造得畫棟飛雲，珠簾捲雨，十分壯麗。西面看城中人戶，煙雨萬家；\n"
    "東面看海上波濤，崢嶸千里。所以城中人士往往於下午攜尊挈酒，在閣中住宿。"
)
CH2_TITLE = "第二回　歷山山下古帝遺蹤　明湖湖邊美人絕調"
CH2_BODY = (
    "話說老殘在漁船上被眾人砸得沉下海去，自知萬無生理，只好閉著眼睛，聽他怎樣。\n"
    "覺得身體如落葉一般，飄飄蕩蕩，頃刻工夫，沉到水底。忽然又見有人拉他一把，\n"
    "睜眼看時，卻是一個老者，鬚眉皆白，笑嘻嘻的站在面前。"
)
TITLE_PAGE = "老殘遊記\n\nProject Gutenberg 文本，公有领域"


def _seed(
    store: Store,
    ai: FakeAIClient,
    chapters: Sequence[Tuple[int, str]],
    passages: Sequence[Tuple[int, str]],
    book_id: str = "book-laocan",
) -> str:
    store.create_book(
        book_id=book_id,
        title="老殘遊記",
        filename="laocan.pdf",
        sha256="sha-laocan",
        status=STATUS_READY,
    )
    store.replace_chapters(
        book_id,
        BookStructure(
            title="老殘遊記",
            chapters=[
                Chapter(
                    idx=idx,
                    title=title,
                    page_start=idx * 7,
                    page_end=idx * 7 + 6,
                    anchor=0,
                    sections=[],
                )
                for idx, title in chapters
            ],
            source="toc",
        ),
    )
    chunks: List[Chunk] = [
        Chunk(
            id="{0}-c{1}".format(book_id, position),
            ord=position,
            chapter_idx=chapter_idx,
            chapter_title="",
            section_title=None,
            page_start=chapter_idx * 7,
            page_end=chapter_idx * 7,
            text=text,
            header=text,
            parent_text=text,
            char_span=(0, len(text)),
        )
        for position, (chapter_idx, text) in enumerate(passages)
    ]
    store.replace_chunks(
        book_id, chunks, ai.embed([chunk.header for chunk in chunks])
    )
    return book_id


def test_chinese_chapters_report_their_own_opening(
    store: Store, fake_ai: FakeAIClient
) -> None:
    book_id = _seed(
        store,
        fake_ai,
        [(1, CH1_TITLE), (2, CH2_TITLE)],
        [(1, CH1_TITLE + "\n\n" + CH1_BODY), (2, CH2_TITLE + "\n\n" + CH2_BODY)],
    )
    openings = store.get_chapter_openings(book_id)

    assert set(openings) == {1, 2}
    assert openings[1].startswith("話說山東登州府東門外")
    assert openings[2].startswith("話說老殘在漁船上")


def test_the_chapter_heading_is_stripped_despite_its_ideographic_spaces(
    store: Store, fake_ai: FakeAIClient
) -> None:
    """The stored title separates its two halves with U+3000, which the
    flattened body no longer has — comparing the raw strings leaves the whole
    heading printed above prose that already carries it."""
    book_id = _seed(
        store, fake_ai, [(1, CH1_TITLE)], [(1, CH1_TITLE + "\n\n" + CH1_BODY)]
    )
    opening = store.get_chapter_openings(book_id)[1]

    assert "土不制水歷年成患" not in opening
    assert "第一回" not in opening


def test_a_title_page_is_not_mistaken_for_a_chapter_opening(
    store: Store, fake_ai: FakeAIClient
) -> None:
    book_id = _seed(
        store,
        fake_ai,
        [(1, CH1_TITLE)],
        [(1, TITLE_PAGE), (1, CH1_BODY)],
    )
    opening = store.get_chapter_openings(book_id)[1]

    assert "Project Gutenberg" not in opening
    assert opening.startswith("話說山東登州府東門外")


def test_flatten_joins_by_script() -> None:
    assert _flatten("蓬萊山。\n山上有個閣子") == "蓬萊山。山上有個閣子"
    assert _flatten("the ferry\ntimetable") == "the ferry timetable"
    # A boundary between the two scripts keeps its space.
    assert _flatten("見 Project\nGutenberg 文本") == "見 Project Gutenberg 文本"
    assert _flatten("   ") == ""


def test_substantial_counts_characters_only_where_words_are_unavailable() -> None:
    assert _substantial(_flatten(CH1_BODY))
    assert not _substantial(_flatten(TITLE_PAGE))
    assert _substantial(" ".join(["word"] * 40))
    assert not _substantial(" ".join(["word"] * 39))
