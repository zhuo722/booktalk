"""Harvesting the book's proper nouns at ingest, and storing them (SPEC §14.9)."""

from __future__ import annotations

from conftest import FixtureChapter, lorem

from app.ingest.ingestor import ingest_book
from app.ingest.vocab import MAX_TERMS, extract_vocabulary, vocabulary_of


def terms(*texts: str, **kwargs) -> list:
    return [term for term, _ in extract_vocabulary(texts, **kwargs)]


def test_a_name_used_mid_sentence_is_a_proper_noun():
    found = terms(
        "The harbour at Kestrel Cove was built in 1873.",
        "Every ferry to Kestrel Cove leaves at dawn.",
    )

    assert "Kestrel" in found


def test_a_word_that_also_appears_lowercase_is_not_a_name():
    """"harbour" the noun must not become "Harbour" the place."""
    found = terms(
        "The Harbour was rebuilt twice, and the Harbour still stands.",
        "Every harbour on the coast silted up. The harbour here did not. "
        "A harbour needs dredging, and this harbour never got it.",
    )

    assert "Harbour" not in found


def test_a_sentence_initial_capital_is_not_evidence():
    """"However" opens sentences all day and is not a name."""
    found = terms(
        "However, the ferry was late. However, nobody minded. "
        "However, the tide turned. However, the keeper waited."
    )

    assert "However" not in found


def test_one_mid_sentence_appearance_is_not_enough():
    found = terms("The keeper met Halvard once and never again.")

    assert "Halvard" not in found

    found = terms(
        "The keeper met Halvard once.",
        "Later the keeper wrote to Halvard about the lantern.",
    )

    assert "Halvard" in found


def test_structural_words_are_excluded():
    found = terms(
        "See Chapter 4 for the tide tables. Chapter 4 also lists the ferries. "
        "The Appendix repeats them, and the Appendix adds distances."
    )

    assert "Chapter" not in found
    assert "Appendix" not in found


def test_terms_are_ranked_by_frequency():
    ranked = extract_vocabulary(
        [
            "The Sable ferry passes Kestrel. The Sable ferry is old. "
            "The Sable ferry runs daily, near Kestrel and past Kestrel, "
            "and it always calls at Kestrel.",
        ]
    )

    assert [term for term, _ in ranked] == ["Kestrel", "Sable"]
    assert ranked[0][1] > ranked[1][1]


def test_ties_are_broken_alphabetically_so_the_stored_list_is_stable():
    ranked = extract_vocabulary(["Near Sable and near Kestrel. Past Sable and past Kestrel."])

    assert [term for term, _ in ranked] == ["Kestrel", "Sable"]


def test_the_canonical_form_is_the_books_own_spelling():
    ranked = extract_vocabulary(
        ["A MacLeod chart, another MacLeod chart, and a third MacLeod chart."]
    )

    assert ranked[0][0] == "MacLeod"


def test_cjk_and_short_tokens_are_skipped():
    found = terms(
        "渡轮每天清晨从旧石码头出发，无论风雨都准时抵达对岸的小镇。"
        "渡轮每天清晨从旧石码头出发，无论风雨都准时抵达对岸。",
        "The Ox pulled the cart. The Ox pulled it again.",
    )

    assert found == []


def test_the_vocabulary_is_capped():
    def coined(index: int) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return "Nam" + alphabet[index // 26] + alphabet[index % 26]

    text = " ".join(
        "the {0} was seen and the {0} was noted.".format(coined(index))
        for index in range(MAX_TERMS + 40)
    )

    assert len(extract_vocabulary([text])) == MAX_TERMS
    assert len(extract_vocabulary([text], max_terms=5)) == 5


def test_vocabulary_of_reads_chunk_text():
    class Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    ranked = vocabulary_of(
        [
            Chunk("The Sable ferry left. The Sable ferry returned."),
            Chunk("Nobody boarded the Sable ferry that day."),
        ]
    )

    assert ranked[0][0] == "Sable"


# ------------------------------------------------------------- persistence

BOOK_ID = "book-vocab"
NAME_SENTENCES = (
    "The keeper of Kestrel Cove wrote to Halvard about the lantern.",
    "Halvard rebuilt the Kestrel light after the storm that winter.",
    "Every ferry to Kestrel Cove was met by Halvard on the stone quay.",
    "The Halvard ledger lists each night the Kestrel lamp was lit.",
)


def test_the_vocabulary_round_trips_through_the_database(store):
    store.create_book(BOOK_ID, "Guide", "guide.pdf", "sha", "ready")

    store.replace_vocabulary(BOOK_ID, [("Kestrel", 9), ("Halvard", 4), ("Sable", 2)])

    assert store.get_vocabulary(BOOK_ID) == ["Kestrel", "Halvard", "Sable"]
    assert store.get_vocabulary(BOOK_ID, limit=2) == ["Kestrel", "Halvard"]
    assert store.get_vocabulary("other-book") == []


def test_re_ingesting_replaces_the_vocabulary_rather_than_appending(store):
    store.create_book(BOOK_ID, "Guide", "guide.pdf", "sha", "ready")
    store.replace_vocabulary(BOOK_ID, [("Kestrel", 9), ("Halvard", 4)])

    store.replace_vocabulary(BOOK_ID, [("Thornbury", 3)])

    assert store.get_vocabulary(BOOK_ID) == ["Thornbury"]


def test_deleting_a_book_forgets_its_names(store):
    store.create_book(BOOK_ID, "Guide", "guide.pdf", "sha", "ready")
    store.replace_vocabulary(BOOK_ID, [("Kestrel", 9)])

    store.delete_book(BOOK_ID)

    assert store.get_vocabulary(BOOK_ID) == []


def test_ingestion_stores_the_books_own_names(store, fake_ai, book_pdf_factory):
    """The end of §14.9's ingest half: a real PDF in, a usable name list out."""
    store.create_book(BOOK_ID, "pending", "meridian.pdf", "sha256", "parsing")
    path = book_pdf_factory(
        [
            FixtureChapter(
                title="Chapter 1. The Northern Light",
                paragraphs=[
                    "{0} {1}".format(lorem("p{0}".format(index), 60), sentence)
                    for index, sentence in enumerate(NAME_SENTENCES)
                ],
            )
        ],
        title="Meridian Fixture",
    )

    result = ingest_book(store, fake_ai, BOOK_ID, path, "meridian.pdf")

    stored = store.get_vocabulary(BOOK_ID)
    assert result.vocabulary == len(stored) > 0
    assert "Kestrel" in stored
    assert "Halvard" in stored
