"""Tokenizer and BM25 behaviour, including the jieba path for CJK."""

from __future__ import annotations

from app.index.bm25 import BM25Index, has_cjk, tokenize


def test_latin_text_is_lowercased_and_punctuation_dropped():
    assert tokenize("The Vell Family, 1873 — harbour!") == [
        "the",
        "vell",
        "family",
        "1873",
        "harbour",
    ]


def test_decimal_numbers_stay_one_token():
    assert "3.5" in tokenize("The channel is 3.5 metres deep")


def test_cjk_text_goes_through_jieba():
    tokens = tokenize("灯塔建于一八七三年")

    assert tokens
    assert all(token.strip() for token in tokens)
    assert "".join(tokens) == "灯塔建于一八七三年"
    assert len(tokens) > 1


def test_mixed_scripts_are_segmented_separately():
    tokens = tokenize("Meridian 群岛 guide 2024")

    assert "meridian" in tokens
    assert "guide" in tokens
    assert "2024" in tokens
    assert "群岛" in tokens


def test_has_cjk_detects_ideographs():
    assert has_cjk("第三章")
    assert not has_cjk("Chapter Three")


def test_empty_text_yields_no_tokens():
    assert tokenize("") == []
    assert tokenize("   ...  ") == []


def test_bm25_ranks_the_document_containing_the_query_terms():
    documents = [
        "The ferry timetable for the northern islands and their harbours.",
        "Basalt quarries supplied the stone for the lighthouse at Vell Point.",
        "Orchards on the eastern slope produce cider every autumn.",
        "The Saturday market fills the square with fishermen and their catch.",
        "Three sea lanes to the inner sound are marked on the old chart.",
    ]
    index = BM25Index(documents)

    ranked = index.top_k("lighthouse basalt quarry", k=3)

    assert ranked
    assert ranked[0][0] == 1
    assert ranked[0][1] > 0


def test_bm25_ranks_cjk_queries():
    documents = [
        "渡轮每天清晨从旧石码头出发，前往北方的岛屿。",
        "采石场为灯塔提供了建造所需的玄武岩。",
        "果园在东坡，每年秋天酿造苹果酒。",
        "市集每周六在广场举行，渔民带来当日的渔获。",
        "旧地图上标注了三条通往内海的航道。",
    ]
    index = BM25Index(documents)

    ranked = index.top_k("灯塔 玄武岩", k=3)

    assert ranked
    assert ranked[0][0] == 1


def test_unmatched_query_returns_nothing():
    index = BM25Index(["harbour", "orchard", "quarry", "ferry", "market"])

    assert index.top_k("submarine", k=2) == []
    assert index.top_k("harbour", k=0) == []


def test_empty_corpus_and_empty_documents_are_safe():
    assert BM25Index([]).top_k("anything", k=3) == []

    index = BM25Index(["", "the harbour lantern", "orchard cider", "ferry timetable"])
    ranked = index.top_k("harbour", k=4)

    assert [row for row, _ in ranked] == [1]


def test_top_k_is_capped_by_the_number_of_matching_documents():
    index = BM25Index(
        ["harbour lantern", "harbour ferry", "orchard", "quarry", "market"]
    )

    assert len(index.top_k("harbour", k=99)) == 2
