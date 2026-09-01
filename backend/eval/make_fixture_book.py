"""Render the fixture book and its golden question set (SPEC §10, §14.10).

Two artefacts come out of one run, both derived from :mod:`facts` so they cannot
drift apart:

``fixture/meridian_archipelago.pdf``
    A real book: title page, twelve chapters of prose set in 11 pt with 20 pt
    chapter headings and 14 pt section headings, a running head, folios, and an
    **embedded outline** so the ingest ladder's tier 1 (``get_toc()``) is what
    detects the structure. Every fact in the table is printed inside a sentence
    of ordinary prose, surrounded by near-miss prose about the same island —
    finding the right chapter has to be earned, not read off a list.

``golden.json``
    88 questions in the five categories of SPEC §14.10, each carrying the
    chapters that answer it, the substrings a correct answer must contain, and
    the exact sentences a perfect retriever would return (the evidence spans the
    token-IoU metric scores against).

Nothing here is random. Run it twice and you get byte-identical prose and an
identical question set; the only thing that varies is how many pages of field
notes are appended to reach the ≥25-page bar, and that is a function of the
font metrics, not of chance.

Generation fails loudly rather than emitting a broken fixture. Before writing
anything it checks that every expected substring is actually printed in the
book, that the ASR corruptions are repairable against the vocabulary the real
ingest path would extract, and that the finished PDF re-parses into twelve
chapters through the app's own structure detector.

    python eval/make_fixture_book.py
    python eval/make_fixture_book.py --pdf /tmp/book.pdf --golden /tmp/golden.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
for _path in (str(BACKEND_DIR), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import facts  # noqa: E402  (after the sys.path bootstrap above)
from facts import (  # noqa: E402
    ABSENT_QUESTIONS,
    BOOK_AUTHOR,
    BOOK_SUBTITLE,
    BOOK_TITLE,
    COMPARISONS,
    ISLANDS,
    ISLANDS_BY_IDX,
    Island,
)

DEFAULT_PDF = EVAL_DIR / "fixture" / "meridian_archipelago.pdf"
DEFAULT_GOLDEN = EVAL_DIR / "golden.json"
#: SPEC §15's definition of done for this phase: ≥12 chapters, ≥25 pages.
MIN_PAGES = 25
MIN_CHAPTERS = 12

CATEGORY_SINGLE = "single-fact"
CATEGORY_CROSS = "cross-chapter"
CATEGORY_SUMMARY = "chapter-summary"
CATEGORY_ABSENT = "not-in-book"
CATEGORY_ASR = "asr-corrupted"
CATEGORIES = (
    CATEGORY_SINGLE,
    CATEGORY_CROSS,
    CATEGORY_SUMMARY,
    CATEGORY_ABSENT,
    CATEGORY_ASR,
)

# Page geometry, in points. The frame stops short of the running head and the
# folio so that neither is ever set inside a paragraph.
PAGE_MARGIN = 62.0
HEAD_OFFSET = 42.0
FOOT_OFFSET = 34.0
BODY_SIZE = 11.0
BODY_LEADING = 15.5
SECTION_SIZE = 14.0
CHAPTER_SIZE = 20.0
#: Field-note paragraphs added per retry when the book comes up short.
PAD_STEP = 3
PAD_LIMIT = 30


class FixtureError(RuntimeError):
    """The fixture could not be generated in a state worth evaluating."""


# ------------------------------------------------------------------- the prose


def chapter_paragraphs(island: Island, padding: int = 0) -> List[Tuple[int, str]]:
    """``(section_index, paragraph)`` for one chapter, in printed order.

    Facts are spread across paragraphs rather than gathered into one, and each
    sits among prose about the same island. That is deliberate: a chunker that
    keeps a section together and a chunker that cuts every 800 characters
    produce measurably different evidence for the same question only when the
    evidence is embedded in continuous prose.
    """
    fact = {key: facts.fact_sentence(island, key) for key in facts.FACT_KEYS}
    filler = facts.filler_sentences(island)
    echo = facts.echo_sentences(island)
    note = [facts.note_sentence(island, n) for n in range(16 + padding)]

    paragraphs: List[Tuple[int, str]] = [
        (0, _join(fact["founder"], filler[0], echo[0], note[0])),
        (0, _join(fact["founded"], fact["population"], filler[1], note[1])),
        (0, _join(island.custom, fact["distance"], filler[2], note[2])),
        (0, _join(filler[3], note[3], filler[4], note[4])),
        (1, _join(fact["landmark"], echo[1], filler[5], note[5])),
        (1, _join(fact["dish"], filler[6], note[6], filler[7])),
        (1, _join(fact["festival"], fact["export"], echo[2], note[7])),
        (1, _join(filler[8], note[8], filler[9], note[9])),
        (1, _join(filler[10], note[10], filler[11], note[11])),
        (1, _join(filler[12], note[12], filler[13], note[13])),
        (1, _join(filler[14], note[14], filler[15], note[15])),
    ]
    for step in range(padding):
        index = 16 + step
        paragraphs.append((1, _join(note[index], note[(index + 1) % len(note)])))
    return paragraphs


def _join(*sentences: str) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def book_text(padding: int = 0) -> str:
    """Everything the book prints, as one string (for substring validation)."""
    parts: List[str] = [BOOK_TITLE, BOOK_SUBTITLE, BOOK_AUTHOR]
    for island in ISLANDS:
        parts.append(island.chapter_title)
        parts.extend(island.sections)
        parts.extend(text for _, text in chapter_paragraphs(island, padding))
    return "\n\n".join(parts)


# -------------------------------------------------------------------- the PDF


def _styles():
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle

    body = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=BODY_SIZE,
        leading=BODY_LEADING,
        spaceAfter=6,
        firstLineIndent=14,
        alignment=TA_JUSTIFY,
    )
    chapter = ParagraphStyle(
        "ChapterHeading",
        fontName="Helvetica-Bold",
        fontSize=CHAPTER_SIZE,
        leading=CHAPTER_SIZE * 1.3,
        spaceAfter=18,
    )
    section = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=SECTION_SIZE,
        leading=SECTION_SIZE * 1.35,
        spaceBefore=14,
        spaceAfter=8,
    )
    title = ParagraphStyle(
        "Title",
        fontName="Times-Bold",
        fontSize=26,
        leading=32,
        spaceAfter=18,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        fontName="Times-Italic",
        fontSize=14,
        leading=20,
        spaceAfter=10,
    )
    return body, chapter, section, title, subtitle


def _template_class():
    """A doc template that writes an outline entry for every heading.

    The embedded outline is the point of the whole exercise: it is what makes
    the fixture exercise tier 1 of the structure ladder (SPEC §14.2) rather than
    the font-size fallback, and it is what pins chapter indices 1–12 to the
    twelve islands of the fact table.
    """
    from reportlab.platypus import BaseDocTemplate

    class BookTemplate(BaseDocTemplate):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            BaseDocTemplate.__init__(self, *args, **kwargs)
            self._outline_seq = 0

        def afterFlowable(self, flowable: Any) -> None:
            level = getattr(flowable, "outline_level", None)
            if level is None:
                return
            self._outline_seq += 1
            key = "h{0:04d}".format(self._outline_seq)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(
                flowable.getPlainText(), key, level=int(level), closed=False
            )

    return BookTemplate


def _decorate(canvas: Any, doc: Any) -> None:
    """Running head and folio — repeated furniture the ingester must strip."""
    if canvas.getPageNumber() <= 1:
        return
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(PAGE_MARGIN, doc.pagesize[1] - HEAD_OFFSET, BOOK_TITLE)
    canvas.drawCentredString(
        doc.pagesize[0] / 2.0, FOOT_OFFSET, str(canvas.getPageNumber())
    )
    canvas.restoreState()


def render_pdf(path: Path, padding: int = 0) -> int:
    """Write the PDF and return its page count."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Frame, PageBreak, PageTemplate, Paragraph, Spacer

    body, chapter_style, section_style, title_style, subtitle_style = _styles()
    width, height = A4
    frame = Frame(
        PAGE_MARGIN,
        PAGE_MARGIN,
        width - 2 * PAGE_MARGIN,
        height - 2 * PAGE_MARGIN - 14,
        id="body",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _template_class()(
        str(path),
        pagesize=A4,
        title=BOOK_TITLE,
        author=BOOK_AUTHOR,
        subject="Eval fixture for BookTalk retrieval",
    )
    doc.addPageTemplates([PageTemplate(id="book", frames=[frame], onPage=_decorate)])

    story: List[Any] = [
        Spacer(1, 140),
        Paragraph(_escape(BOOK_TITLE), title_style),
        Paragraph(_escape(BOOK_SUBTITLE), subtitle_style),
        Paragraph(_escape(BOOK_AUTHOR), subtitle_style),
    ]
    for island in ISLANDS:
        story.append(PageBreak())
        heading = Paragraph(_escape(island.chapter_title), chapter_style)
        heading.outline_level = 0
        story.append(heading)
        current = -1
        for section_index, text in chapter_paragraphs(island, padding):
            if section_index != current:
                current = section_index
                section = Paragraph(
                    _escape(island.sections[section_index]), section_style
                )
                section.outline_level = 1
                story.append(section)
            story.append(Paragraph(_escape(text), body))

    doc.build(story)
    return int(getattr(doc, "page", 0))


def _escape(text: str) -> str:
    """Paragraph markup is XML; the prose is plain text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(path: Path, min_pages: int = MIN_PAGES, verbose: bool = True) -> Tuple[int, int]:
    """Render, adding field notes until the book reaches ``min_pages``.

    Font metrics decide how much prose fits on a page, so the page target is met
    by measuring rather than by guessing: render, count, append more notes to
    every chapter, render again. Returns ``(pages, padding)``.
    """
    padding = 0
    while True:
        pages = render_pdf(path, padding)
        if verbose:
            print(
                "  rendered {0} pages (padding {1})".format(pages, padding),
                file=sys.stderr,
            )
        if pages >= min_pages or padding >= PAD_LIMIT:
            if pages < min_pages:
                raise FixtureError(
                    "book is {0} pages after {1} padding paragraphs per chapter; "
                    "expected at least {2}".format(pages, padding, min_pages)
                )
            return pages, padding
        padding += PAD_STEP


# ---------------------------------------------------------------- the golden set


def build_golden(pdf_path: Path, pages: int, padding: int) -> Dict[str, Any]:
    """Assemble every golden question from the fact table."""
    questions: List[Dict[str, Any]] = []
    questions.extend(_single_fact())
    questions.extend(_cross_chapter())
    questions.extend(_chapter_summary())
    questions.extend(_not_in_book())
    questions.extend(_asr_corrupted())

    counts: Dict[str, int] = {name: 0 for name in CATEGORIES}
    for item in questions:
        counts[item["category"]] += 1

    return {
        "book": {
            "title": BOOK_TITLE,
            "subtitle": BOOK_SUBTITLE,
            "author": BOOK_AUTHOR,
            "pdf": _relative(pdf_path),
            "pages": pages,
            "padding_paragraphs": padding,
            "chapters": [
                {"idx": island.idx, "title": island.chapter_title, "island": island.name}
                for island in ISLANDS
            ],
        },
        "categories": list(CATEGORIES),
        "counts": counts,
        "questions": questions,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EVAL_DIR))
    except ValueError:
        return str(path)


def _question(
    qid: str,
    category: str,
    question: str,
    substrings: Sequence[str],
    chapters: Sequence[int],
    evidence: Sequence[str] = (),
    expect_refusal: bool = False,
    **extra: Any
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": qid,
        "category": category,
        "question": question,
        "expected_answer_substrings": list(substrings),
        # SPEC §10 keys a question to one chapter; the cross-chapter category of
        # §14.10 needs two, so the single value stays (it is the primary
        # chapter) and the full set travels beside it.
        "chapter_idx": int(chapters[0]) if chapters else 0,
        "gold_chapters": [int(idx) for idx in chapters],
        "gold_chapter_titles": [ISLANDS_BY_IDX[idx].name for idx in chapters],
        "expect_refusal": bool(expect_refusal),
        "evidence": list(evidence),
    }
    record.update(extra)
    return record


def _single_fact() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for island in ISLANDS:
        for key in facts.questioned_keys(island):
            items.append(
                _question(
                    "single-{0:02d}-{1}".format(island.idx, key),
                    CATEGORY_SINGLE,
                    facts.fact_question(island, key),
                    facts.fact_answers(island, key),
                    [island.idx],
                    evidence=[facts.fact_sentence(island, key)],
                    fact=key,
                )
            )
    return items


def _cross_chapter() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for position, (left, right, kind) in enumerate(COMPARISONS, start=1):
        question, substrings = facts.comparison(left, right, kind)
        evidence = [
            facts.fact_sentence(ISLANDS_BY_IDX[idx], _EVIDENCE_KEY[kind])
            for idx in (left, right)
        ]
        items.append(
            _question(
                "cross-{0:02d}-{1}".format(position, kind),
                CATEGORY_CROSS,
                question,
                substrings,
                [left, right],
                evidence=evidence,
                comparison=kind,
                # An arithmetic answer ("166 years apart") is derived from two
                # chapters rather than quoted from either, so it is exempt from
                # the "must be printed in the book" check — and it is the only
                # question type in the set that genuinely cannot be answered by
                # copying a sentence.
                derived=kind in _DERIVED_COMPARISONS,
            )
        )
    return items


#: Comparisons whose expected answer is computed, not quoted.
_DERIVED_COMPARISONS = frozenset({"charter_gap"})


#: Which fact each comparison is decided on — its evidence in both chapters.
_EVIDENCE_KEY = {
    "older": "founded",
    "charter_gap": "founded",
    "more_residents": "population",
    "farther": "distance",
    "taller": "landmark",
    "same_month": "festival",
    "populations": "population",
}


def _chapter_summary() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for island in ISLANDS:
        question, substrings = facts.summary_question(island)
        items.append(
            _question(
                "summary-{0:02d}".format(island.idx),
                CATEGORY_SUMMARY,
                question,
                substrings,
                [island.idx],
                # A synopsis question has no single sentence that answers it —
                # that is what makes it a summary question — so it carries no
                # evidence span and is excluded from the IoU average.
                evidence=[],
                theme=island.theme,
            )
        )
    return items


def _not_in_book() -> List[Dict[str, Any]]:
    return [
        _question(
            "absent-{0:02d}".format(position),
            CATEGORY_ABSENT,
            question,
            [],
            [],
            expect_refusal=True,
        )
        for position, question in enumerate(ABSENT_QUESTIONS, start=1)
    ]


def _asr_corrupted() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for island in ISLANDS:
        heard, spoken, key = facts.asr_question(island)
        items.append(
            _question(
                "asr-{0:02d}-{1}".format(island.idx, key),
                CATEGORY_ASR,
                heard,
                facts.fact_answers(island, key),
                [island.idx],
                evidence=[facts.fact_sentence(island, key)],
                fact=key,
                asr={
                    "spoken": spoken,
                    "term": island.key_term,
                    "heard_term": island.heard,
                },
            )
        )
    return items


# ------------------------------------------------------------------ validation


def validate_golden(golden: Dict[str, Any], prose: Optional[str] = None) -> List[str]:
    """Check the question set against its own schema and the book's text.

    Returns the list of problems found; empty means the fixture is sound. Used
    both here (generation aborts on any problem) and by ``run_eval.py`` on load,
    so a hand-edited ``golden.json`` cannot quietly produce meaningless numbers.
    """
    problems: List[str] = []
    questions = golden.get("questions")
    if not isinstance(questions, list) or not questions:
        return ["golden set has no questions"]

    normalized = _normalize(prose) if prose is not None else None
    seen: set = set()
    chapter_indices = {
        int(entry["idx"]) for entry in golden.get("book", {}).get("chapters", [])
    }

    for item in questions:
        qid = item.get("id", "?")
        if not isinstance(item, dict):
            problems.append("{0}: not an object".format(qid))
            continue
        if qid in seen:
            problems.append("{0}: duplicate id".format(qid))
        seen.add(qid)
        for field in ("category", "question", "expected_answer_substrings",
                      "chapter_idx", "gold_chapters", "expect_refusal"):
            if field not in item:
                problems.append("{0}: missing field {1}".format(qid, field))
        if item.get("category") not in CATEGORIES:
            problems.append("{0}: unknown category {1!r}".format(qid, item.get("category")))
        if not str(item.get("question", "")).strip():
            problems.append("{0}: empty question".format(qid))

        gold = [int(idx) for idx in item.get("gold_chapters", [])]
        if item.get("expect_refusal"):
            if gold or item.get("expected_answer_substrings"):
                problems.append(
                    "{0}: a refusal question must have no gold chapters and no "
                    "expected substrings".format(qid)
                )
        else:
            if not gold:
                problems.append("{0}: answerable question with no gold chapter".format(qid))
            if not item.get("expected_answer_substrings"):
                problems.append("{0}: answerable question with no expected substrings".format(qid))
        if chapter_indices and any(idx not in chapter_indices for idx in gold):
            problems.append("{0}: gold chapter outside the book".format(qid))

        if normalized is None:
            continue
        for substring in (
            [] if item.get("derived") else item.get("expected_answer_substrings", [])
        ):
            if _normalize(substring) not in normalized:
                problems.append(
                    "{0}: expected substring {1!r} is never printed in the book".format(
                        qid, substring
                    )
                )
        for sentence in item.get("evidence", []):
            if _normalize(sentence) not in normalized:
                problems.append(
                    "{0}: evidence sentence is not printed in the book".format(qid)
                )
    return problems


def validate_asr(golden: Dict[str, Any], vocabulary: Sequence[str]) -> List[str]:
    """Every ASR corruption must be repairable against the book's vocabulary.

    Run against the vocabulary :mod:`app.ingest.vocab` extracts from the same
    prose, so the check is the real repair path and not an approximation of it.
    """
    from app.speech.phonetics import repair_transcript

    problems: List[str] = []
    for item in golden.get("questions", []):
        asr = item.get("asr")
        if not asr:
            continue
        repaired = repair_transcript(item["question"], vocabulary).text
        if repaired != asr["spoken"]:
            problems.append(
                "{0}: {1!r} repaired to {2!r}, expected {3!r}".format(
                    item["id"], item["question"], repaired, asr["spoken"]
                )
            )
    return problems


def validate_pdf(path: Path, min_pages: int, min_chapters: int) -> List[str]:
    """Re-parse the finished PDF through the app's own ingest front end."""
    from app.ingest.markdown import extract_blocks
    from app.ingest.pdf import PdfDocument, chars_per_page
    from app.ingest.structure import detect_structure

    problems: List[str] = []
    with PdfDocument(path) as document:
        pages = list(document.iter_pages())
        page_count = document.page_count
        title = document.metadata_title
        toc = document.toc()
    if page_count < min_pages:
        problems.append("pdf has {0} pages, expected ≥{1}".format(page_count, min_pages))
    if title != BOOK_TITLE:
        problems.append("pdf metadata title is {0!r}".format(title))
    if not toc:
        problems.append("pdf has no embedded outline")
    density = chars_per_page(pages)
    if density < 400:
        problems.append("only {0:.0f} characters per page".format(density))

    structure = detect_structure(pages, toc, title=title, page_count=page_count)
    if structure.source != "toc":
        problems.append(
            "structure came from {0!r}, expected the embedded outline".format(
                structure.source
            )
        )
    if len(structure.chapters) < min_chapters:
        problems.append(
            "detected {0} chapters, expected ≥{1}".format(
                len(structure.chapters), min_chapters
            )
        )
    for island, chapter in zip(ISLANDS, structure.chapters):
        if chapter.idx != island.idx or island.name not in chapter.title:
            problems.append(
                "chapter {0} detected as {1!r} ({2})".format(
                    island.idx, chapter.title, chapter.idx
                )
            )
        if len(chapter.sections) != 2:
            problems.append(
                "chapter {0} has {1} sections, expected 2".format(
                    island.idx, len(chapter.sections)
                )
            )

    # Every evidence span has to survive extraction into the block stream the
    # chunker is handed — not merely exist in this generator's own strings. A
    # sentence that a page break splits around a running head, or that a
    # ligature or an end-of-line hyphen mangles, would silently zero the IoU
    # metric for its question and nothing else would notice.
    corpus = _normalize(
        " ".join(
            block.text for block in extract_blocks(path, structure, pages=pages)
        )
    )
    for island in ISLANDS:
        for key in facts.FACT_KEYS:
            if _normalize(facts.fact_sentence(island, key)) not in corpus:
                problems.append(
                    "chapter {0}: the {1} sentence does not survive extraction".format(
                        island.idx, key
                    )
                )
    return problems


def book_vocabulary(prose: str) -> List[str]:
    """The proper nouns ingestion would extract from this book's prose."""
    from app.ingest.vocab import extract_vocabulary

    return [term for term, _ in extract_vocabulary([prose])]


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace so a line break cannot fail a substring check."""
    return _WHITESPACE.sub(" ", text or "").strip()


# ------------------------------------------------------------------------ main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="output PDF path")
    parser.add_argument(
        "--golden", type=Path, default=DEFAULT_GOLDEN, help="output golden.json path"
    )
    parser.add_argument("--min-pages", type=int, default=MIN_PAGES)
    parser.add_argument("--min-chapters", type=int, default=MIN_CHAPTERS)
    parser.add_argument(
        "--skip-pdf-check",
        action="store_true",
        help="skip re-parsing the PDF (fast path for prose-only edits)",
    )
    args = parser.parse_args(argv)

    print("Rendering {0}".format(args.pdf), file=sys.stderr)
    pages, padding = build_pdf(args.pdf, min_pages=args.min_pages)

    prose = book_text(padding)
    golden = build_golden(args.pdf, pages, padding)

    problems = validate_golden(golden, prose)
    problems += validate_asr(golden, book_vocabulary(prose))
    if not args.skip_pdf_check:
        problems += validate_pdf(args.pdf, args.min_pages, args.min_chapters)
    if problems:
        for problem in problems:
            print("  FAIL {0}".format(problem), file=sys.stderr)
        raise FixtureError("{0} problem(s) with the generated fixture".format(len(problems)))

    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.golden.write_text(
        json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = golden["counts"]
    print(
        "OK  {0} pages, {1} chapters, {2} questions ({3})".format(
            pages,
            len(ISLANDS),
            len(golden["questions"]),
            ", ".join("{0} {1}".format(counts[name], name) for name in CATEGORIES),
        ),
        file=sys.stderr,
    )
    print("    pdf    {0}".format(args.pdf), file=sys.stderr)
    print("    golden {0}".format(args.golden), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
