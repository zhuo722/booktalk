"""Typeset a real public-domain book from Project Gutenberg into a PDF.

The fixture book in this directory is invented on purpose, so answers about
it prove retrieval rather than recall. This builds the other kind of test
material: a real book, long enough and structured enough to try the
ingestion ladder on something nobody wrote for it.

    python eval/make_real_book.py 1342 "Pride and Prejudice" out.pdf


The text is the book's own; this only strips Gutenberg's licence wrapper,
finds the chapter breaks, and lays the result out with an embedded outline so
BookTalk's tier-1 structure detection has a real table of contents to read.
Each PDF keeps the first CHAPTER_LIMIT complete chapters — whole chapters, never
truncated mid-way — which keeps ingestion cost sane while staying above the
ten-chapter floor.
"""

import re
import sys
import unicodedata
import urllib.request
from typing import List

from reportlab.lib.pagesizes import A5
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

CHAPTER_LIMIT = 12
#: Below this the assignment's ten-chapter floor is not met.
MIN_CHAPTERS = 10

START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)

HEADING_PATTERNS = [
    re.compile(r"^(chapter|letter|book|adventure)\s+([ivxlcdm]+|\d+)\b\.?(.*)$", re.I),
    re.compile(r"^([ivxlcdm]+)\.\s+(\S.*)$"),
]

#: 第一回, 第十二章, 第20節 — the heading convention of a Chinese chaptered book.
#: Kept apart from HEADING_PATTERNS because its number sits in a different group
#: and is written in a different numeral system.
CJK_HEADING = re.compile(r"^第\s*([〇零○一二三四五六七八九十百千\d]+)\s*[回章節节篇卷](\s.*)?$")
#: ○ is U+25CB WHITE CIRCLE, not the ideographic zero, and Chinese typesetting
#: uses it as one: 西遊記 prints chapter 10 as 第一○回. A parser that only knows
#: 〇 and 零 silently loses every chapter whose number contains a zero.
CJK_DIGITS = {
    "〇": 0, "零": 0, "○": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CJK_UNITS = {"十": 10, "百": 100, "千": 1000}
#: Han, plus the CJK punctuation a typeset Chinese book actually uses.
CJK_RANGE = (
    "　-〿"  # 、。〈〉《》「」and the ideographic space
    "㐀-䶿"
    "一-鿿"
    "豈-﫿"
    "＀-￯"  # full-width forms
)
CJK_CHAR = re.compile("[" + CJK_RANGE + "]")


def body_text(raw):
    """Drop Gutenberg's licence header and footer."""
    m = START.search(raw)
    if m:
        raw = raw[m.end():]
    m = END.search(raw)
    if m:
        raw = raw[: m.start()]
    return raw.strip("\n")


def is_cjk(text):
    """True when the text is written in Chinese rather than a latin script."""
    sample = text[:20000]
    if not sample:
        return False
    return len(CJK_CHAR.findall(sample)) / len(sample) > 0.2


def is_heading(line):
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    if CJK_HEADING.match(stripped):
        return stripped
    for pattern in HEADING_PATTERNS:
        if pattern.match(stripped):
            return stripped
    return None


ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def cjk_number(token):
    """一 → 1, 十二 → 12, 二十 → 20, 一一 → 11. None when not a numeral.

    Chinese chapter numbers come in two conventions and a single book will
    mix them: 第十四回 is positional, where a digit before a unit character
    multiplies it and a digit after is added; 第一一回 is a digit string,
    where the characters are read one place each — that is chapter 11, not
    chapter 1. The two are told apart by whether a unit character (十百千)
    appears at all, which is the only signal available and is unambiguous for
    every number a book chapter reaches.
    """
    if token.isdigit():
        return int(token)
    if not token:
        return None
    if any(ch in CJK_UNITS for ch in token):
        total, current = 0, 0
        for ch in token:
            if ch in CJK_DIGITS:
                current = CJK_DIGITS[ch]
            elif ch in CJK_UNITS:
                total += (current or 1) * CJK_UNITS[ch]
                current = 0
            else:
                return None
        return total + current
    if all(ch in CJK_DIGITS for ch in token):
        return int("".join(str(CJK_DIGITS[ch]) for ch in token))
    return None


def chapter_number(heading):
    """The number a chapter heading claims, or None."""
    m = CJK_HEADING.match(heading)
    if m:
        return cjk_number(m.group(1))
    for pattern in HEADING_PATTERNS:
        m = pattern.match(heading)
        if not m:
            continue
        token = m.group(2) if pattern is HEADING_PATTERNS[0] else m.group(1)
        token = token.strip().lower()
        if token.isdigit():
            return int(token)
        if all(ch in ROMAN for ch in token):
            total = 0
            prev = 0
            for ch in reversed(token):
                value = ROMAN[ch]
                total += -value if value < prev else value
                prev = max(prev, value)
            return total
    return None


def substantial(body):
    """Long enough to be a chapter rather than a line in a contents list.

    Word counting does not survive the trip to Chinese, which is written
    without spaces: a whole chapter of it can come back as a handful of
    whitespace-separated tokens. Chinese is measured in characters instead,
    at roughly the ratio a translation would have.
    """
    if is_cjk(body):
        return len(CJK_CHAR.findall(body)) >= 300
    return len(body.split()) >= 120


def split_chapters(text):
    """Return [(heading, body)] for every chapter heading found, in order."""
    lines = text.split("\n")
    marks = []
    for i, line in enumerate(lines):
        heading = is_heading(line)
        if heading:
            marks.append((i, heading))
    chapters = []
    for n, (index, heading) in enumerate(marks):
        stop = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[index + 1 : stop]).strip("\n")
        if substantial(body):  # a real chapter, not a contents-page echo
            chapters.append((heading, body))
    # A Gutenberg file usually prints its own contents list before the text,
    # and those lines match the same heading pattern. The body always comes
    # after the list, so for any chapter number the LAST occurrence is the
    # real one; numbering then puts them back in reading order.
    by_number = {}
    unnumbered = []
    for heading, body in chapters:
        number = chapter_number(heading)
        if number is None:
            unnumbered.append((heading, body))
        else:
            by_number[number] = (heading, body)
    ordered = [by_number[n] for n in sorted(by_number)]
    return ordered or unnumbered


def paragraphs(body):
    """One string per paragraph, with the source's own line breaks removed.

    Rejoining with a space is right for a language written with spaces and
    wrong for Chinese, where it would leave a gap at every line break the
    typesetter happened to make — visible on the page, and a word boundary
    that BM25 would then tokenize around.
    """
    out = []
    for block in re.split(r"\n\s*\n", body):
        lines = block.split()
        if not lines:
            continue
        joiner = "" if CJK_CHAR.search(block) else " "
        para = joiner.join(lines)
        if para:
            out.append(para)
    return out


def safe(text):
    """reportlab's built-in fonts are WinAnsi; drop what cannot be encoded."""
    text = unicodedata.normalize("NFC", text)
    return text.encode("cp1252", "replace").decode("cp1252")


#: STSong-Light is one of the CID fonts reportlab ships a metrics table for,
#: so a Chinese PDF needs no font file on disk. It has no bold or italic
#: companion, which is why the Chinese styles below separate by size alone.
CJK_FONT = "STSong-Light"

LATIN_STYLE = {
    "title": ("Times-Bold", 20),
    "subtitle": ("Times-Italic", 10),
    "heading": ("Times-Bold", 14),
    "body": ("Times-Roman", 10.5),
    "folio": ("Times-Italic", 8),
    "leading": 5.2 * mm,
    "credit": "Project Gutenberg text, public domain",
}
CJK_STYLE = {
    "title": (CJK_FONT, 20),
    "subtitle": (CJK_FONT, 10),
    "heading": (CJK_FONT, 14),
    "body": (CJK_FONT, 10.5),
    "folio": (CJK_FONT, 8),
    "leading": 6.0 * mm,  # Han glyphs are full-width and need the extra air
    "credit": "Project Gutenberg 文本，公有领域",
}


def wrap(c, text, font, size, measure):
    """Break one paragraph into lines that fit ``measure``.

    Latin text breaks at spaces. Chinese has none, so it breaks between
    characters, which is what Chinese typesetting does anyway.
    """
    lines = []
    if CJK_CHAR.search(text):
        line = ""
        for ch in text:
            trial = line + ch
            if c.stringWidth(trial, font, size) > measure and line:
                lines.append(line)
                line = ch
            else:
                line = trial
        if line:
            lines.append(line)
        return lines
    line = ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if c.stringWidth(trial, font, size) > measure and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def render(title, chapters, path, cjk=False):
    style = CJK_STYLE if cjk else LATIN_STYLE
    if cjk:
        pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))
    clean = (lambda s: s) if cjk else safe

    page_w, page_h = A5
    left, right = 18 * mm, 18 * mm
    top, bottom = 22 * mm, 20 * mm
    measure = page_w - left - right
    body_font, body_size = style["body"]
    leading = style["leading"]
    c = canvas.Canvas(path, pagesize=A5)
    c.setTitle(title)

    c.setFont(*style["title"])
    c.drawCentredString(page_w / 2, page_h - 70 * mm, clean(title))
    c.setFont(*style["subtitle"])
    c.drawCentredString(page_w / 2, page_h - 80 * mm, style["credit"])
    c.showPage()

    page_no = 2

    def folio():
        c.setFont(*style["folio"])
        c.drawCentredString(page_w / 2, bottom - 6 * mm, str(page_no))

    for idx, (heading, body) in enumerate(chapters, start=1):
        key = "ch%d" % idx
        c.bookmarkPage(key)
        c.addOutlineEntry(clean(heading), key, level=0)

        y = page_h - top
        c.setFont(*style["heading"])
        c.drawString(left, y, clean(heading))
        y -= 10 * mm

        for para in paragraphs(body):
            c.setFont(body_font, body_size)
            for line in wrap(c, clean(para), body_font, body_size, measure):
                c.drawString(left, y, line)
                y -= leading
                if y < bottom:
                    folio()
                    c.showPage()
                    page_no += 1
                    c.setFont(body_font, body_size)
                    y = page_h - top
            y -= 2.4 * mm
            if y < bottom:
                folio()
                c.showPage()
                page_no += 1
                y = page_h - top

        folio()
        c.showPage()
        page_no += 1

    c.showOutline()
    c.save()
    return page_no - 1


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    gid, title, out = argv[0], argv[1], argv[2]
    url = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt".format(gid)
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read().decode("utf-8", "replace")
    text = body_text(raw)
    chapters = split_chapters(text)
    if len(chapters) < MIN_CHAPTERS:
        print("only %d chapters detected; this text needs its own rule" % len(chapters))
        return 1
    kept = chapters[:CHAPTER_LIMIT]
    cjk = is_cjk(text)
    pages = render(title, kept, out, cjk=cjk)
    print("%s — %d chapters (of %d), %d pages, %s -> %s"
          % (title, len(kept), len(chapters), pages,
             "Chinese" if cjk else "latin", out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
