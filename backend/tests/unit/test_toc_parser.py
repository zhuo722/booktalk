"""The printed contents page and LLM level assignment (SPEC §14.2)."""

from __future__ import annotations

from typing import Any, Dict, List

from app.ingest.pdf import Line, PageText
from app.ingest.toc import assign_toc_levels, parse_contents_pages

LINE_HEIGHT = 20.0


def page(page_no: int, *texts: str) -> PageText:
    """A page of plain body-size lines, laid out top to bottom."""
    return PageText(
        page_no=page_no,
        width=595.0,
        height=842.0,
        lines=[
            Line(text=text, size=11.0, bold=False, y=100.0 + LINE_HEIGHT * index, x0=72.0)
            for index, text in enumerate(texts)
        ],
    )


def test_contents_page_entries_are_read_with_their_page_numbers():
    pages = [
        page(
            1,
            "Contents",
            "Isle of Basalt . . . . . . . 2",
            "Isle of Granite . . . . . . 3",
            "Isle of Salt . . . . . . . . 4",
        ),
        page(2, "Isle of Basalt", "the ferry leaves at dawn"),
        page(3, "Isle of Granite", "the quarry closed in 1911"),
        page(4, "Isle of Salt", "the pans flood every spring"),
    ]

    entries = parse_contents_pages(pages, page_count=4)

    assert [(entry.title, entry.page) for entry in entries] == [
        ("Isle of Basalt", 2),
        ("Isle of Granite", 3),
        ("Isle of Salt", 4),
    ]
    assert {entry.level for entry in entries} == {1}


def test_printed_numbers_are_shifted_onto_physical_pages():
    """Front matter offsets the printed numbering; the offset is measured."""
    pages = [
        page(
            1,
            "Table of Contents",
            "Isle of Basalt . . . 1",
            "Isle of Granite . . . 5",
            "Isle of Salt . . . . 9",
        ),
        page(2, "a half title and a dedication"),
        page(3, "Isle of Basalt", "the ferry leaves at dawn"),
        page(4, "more about the ferry"),
        page(5, "still more about the ferry"),
        page(6, "even more about the ferry"),
        page(7, "Isle of Granite", "the quarry closed in 1911"),
        page(8, "more about granite"),
        page(9, "yet more about granite"),
        page(10, "still more about granite"),
        page(11, "Isle of Salt", "the pans flood every spring"),
    ]

    entries = parse_contents_pages(pages, page_count=11)

    assert [(entry.title, entry.page) for entry in entries] == [
        ("Isle of Basalt", 3),
        ("Isle of Granite", 7),
        ("Isle of Salt", 11),
    ]


def test_entries_that_go_backwards_are_dropped():
    pages = [
        page(
            1,
            "Contents",
            "Harbours . . . 2",
            "Quarries . . . 9",
            "A stray line . . . 5",
            "Salt pans . . . 12",
        )
    ]

    entries = parse_contents_pages(pages, page_count=20)

    assert [(entry.title, entry.page) for entry in entries] == [
        ("Harbours", 2),
        ("Quarries", 9),
        ("Salt pans", 12),
    ]


def test_collection_stops_when_the_contents_list_ends():
    pages = [
        page(1, "Contents", "Harbours . . . 2", "Quarries . . . 3", "Salt pans . . . 4"),
        page(2, "Harbours", "the ferry leaves at dawn"),
        page(3, "a stray body line that ends in 12"),
        page(4, "Salt pans"),
    ]

    entries = parse_contents_pages(pages, page_count=4)

    assert [entry.title for entry in entries] == ["Harbours", "Quarries", "Salt pans"]


def test_a_book_without_a_contents_page_yields_nothing():
    pages = [page(1, "the ferry leaves at dawn"), page(2, "the quarry closed in 1911")]

    assert parse_contents_pages(pages, page_count=2) == []


def test_too_few_entries_are_not_a_contents_page():
    pages = [page(1, "Contents", "Harbours . . . 2", "Quarries . . . 3")]

    assert parse_contents_pages(pages, page_count=10) == []


def test_lines_without_a_page_number_are_not_entries():
    pages = [
        page(
            1,
            "Contents",
            "Harbours",  # no page number at all
            "Quarries . . . 3",
            "Salt pans . . . 4",
            "Ferries . . . 6",
        )
    ]

    entries = parse_contents_pages(pages, page_count=10)

    assert [entry.title for entry in entries] == ["Quarries", "Salt pans", "Ferries"]


def test_page_numbers_beyond_the_document_are_rejected():
    pages = [
        page(1, "Contents", "Harbours . . . 200", "Quarries . . . 300", "Salt . . . 400")
    ]

    assert parse_contents_pages(pages, page_count=10) == []


# ------------------------------------------------------------ level assignment


def test_assign_toc_levels_uses_the_llm_reply(fake_ai):
    titles = ["PART ONE", "Chapter 1. Isle", "Chapter 2. Isle", "Chapter 3. Isle"]

    levels = assign_toc_levels(fake_ai, titles, book_title="Fixture Book")

    assert levels == [1, 2, 2, 2]
    assert fake_ai.chat_calls, "the assignment must go through the AI client"


class _BadClient:
    """A client whose reply is not the strict JSON the prompt demands."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kw):
        self.calls.append(kw)
        return self.reply


def test_a_malformed_reply_is_discarded_wholesale():
    titles = ["Chapter 1", "Chapter 2", "Chapter 3"]

    assert assign_toc_levels(_BadClient("sure! levels: 1, 2, 3"), titles) == []
    assert assign_toc_levels(_BadClient('[{"level": 9, "title": "x"}]'), titles) == []
    assert assign_toc_levels(_BadClient('[{"level": 1, "title": "x"}]'), titles) == []


def test_no_client_means_no_call():
    assert assign_toc_levels(None, ["Chapter 1", "Chapter 2"]) == []
