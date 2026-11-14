#!/usr/bin/env python3
"""Build three simplified-Chinese public-domain example books as PDFs.

The texts come from Chinese Wikisource (zh.wikisource.org), one page per
chapter, fetched through the MediaWiki API and cached under ``--cache-dir``.
The script keeps only the body paragraphs (no page header, navigation or
editorial footnotes), converts traditional characters to simplified with
OpenCC (``t2s``), and lays the result out with the same template as
``build_public_domain_examples.py``: one PDF bookmark per chapter so BookTalk
detects the structure.

    backend/.venv/bin/pip install opencc-python-reimplemented==0.1.7
    backend/.venv/bin/python scripts/build_simplified_chinese_examples.py

Works: 鲁迅《呐喊》(1923) and《朝花夕拾》(1928); 沈复《浮生六记》(c. 1808,
books 1–4 only — books 5–6 are generally regarded as later forgeries). All
are public domain; the Wikisource transcriptions are CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_public_domain_examples as base  # noqa: E402

from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # noqa: E402
from reportlab.pdfbase.pdfmetrics import registerFont  # noqa: E402
from reportlab.platypus import Frame, PageBreak, PageTemplate, Paragraph, Spacer  # noqa: E402

API = "https://zh.wikisource.org/w/api.php"
USER_AGENT = "booktalk-example-builder/1.0 (public-domain example PDFs)"
DEFAULT_CACHE = Path("/tmp/booktalk_wikisource")
DEFAULT_OUTPUT_DIR = base.ROOT / "examples" / "books"


@dataclass(frozen=True)
class Chapter:
    page: str  # Wikisource page title
    heading: str  # simplified heading printed in the PDF and its outline
    #: Take only the paragraphs after this section heading on the page (the
    #: 呐喊 preface sits on the collection's index page, below its contents).
    section: Optional[str] = None


@dataclass(frozen=True)
class ZhBook:
    output_name: str
    title: str
    author: str
    year: str
    index_page: str
    chapters: Tuple[Chapter, ...]


BOOKS: Tuple[ZhBook, ...] = (
    ZhBook(
        output_name="01-呐喊-鲁迅.pdf",
        title="呐喊",
        author="鲁迅",
        year="1923",
        index_page="吶喊",
        chapters=(
            Chapter("吶喊", "自序", section="自序"),
            Chapter("狂人日記", "狂人日记"),
            Chapter("孔乙己", "孔乙己"),
            Chapter("藥", "药"),
            Chapter("明天", "明天"),
            Chapter("一件小事", "一件小事"),
            Chapter("頭髮的故事", "头发的故事"),
            Chapter("風波", "风波"),
            Chapter("故鄉", "故乡"),
            Chapter("阿Ｑ正傳", "阿Q正传"),
            Chapter("端午節", "端午节"),
            Chapter("白光", "白光"),
            Chapter("兔和貓", "兔和猫"),
            Chapter("鴨的喜劇", "鸭的喜剧"),
            Chapter("社戲", "社戏"),
        ),
    ),
    ZhBook(
        output_name="02-朝花夕拾-鲁迅.pdf",
        title="朝花夕拾",
        author="鲁迅",
        year="1928",
        index_page="朝花夕拾",
        chapters=(
            Chapter("朝花夕拾/小引", "小引"),
            Chapter("狗·猫·鼠", "狗·猫·鼠"),
            Chapter("阿長與《山海經》", "阿长与《山海经》"),
            Chapter("二十四孝圖", "《二十四孝图》"),
            Chapter("五猖會", "五猖会"),
            Chapter("无常", "无常"),
            Chapter("從百草園到三味書屋", "从百草园到三味书屋"),
            Chapter("父亲的病", "父亲的病"),
            Chapter("琐记", "琐记"),
            Chapter("藤野先生", "藤野先生"),
            Chapter("范愛農", "范爱农"),
            Chapter("朝花夕拾/後記", "后记"),
        ),
    ),
    ZhBook(
        output_name="03-浮生六记-沈复.pdf",
        title="浮生六记",
        author="沈复",
        year="约 1808",
        index_page="浮生六記",
        chapters=(
            Chapter("浮生六記/卷一 閨房記樂", "卷一 闺房记乐"),
            Chapter("浮生六記/卷二 閑情記趣", "卷二 闲情记趣"),
            Chapter("浮生六記/卷三 坎坷記愁", "卷三 坎坷记愁"),
            Chapter("浮生六記/卷四 浪遊記快", "卷四 浪游记快"),
        ),
    ),
)


#: Old printed forms in the 1938《鲁迅全集》and Qing woodblock texts that OpenCC
#: keeps as-is, mapped to the standard simplified character. Without this a
#: reader asking about 故乡 never matches the book's 故鄕. Genuinely rare
#: characters (喤, 窸窣, 魆 ...) are left alone.
VARIANTS = str.maketrans({
    "擡": "抬", "靑": "青", "偸": "偷", "淸": "清", "揑": "捏", "鄕": "乡",
    "朶": "朵", "尙": "尚", "簷": "檐", "拏": "拿", "吿": "告", "寃": "冤",
    "鎗": "枪", "硏": "研", "嬾": "懒", "牠": "它", "囘": "回", "鬭": "斗",
    "煑": "煮", "躱": "躲", "呪": "咒", "燄": "焰", "覩": "睹", "怱": "匆",
    "粧": "妆", "牕": "窗", "汚": "污", "尅": "克", "値": "值", "盌": "碗",
    "輭": "软", "郤": "却", "僊": "仙", "煖": "暖", "壻": "婿", "隄": "堤",
    "茍": "苟", "麺": "面", "衞": "卫", "敍": "叙", "儍": "傻", "顚": "颠",
    "遶": "绕", "緖": "绪", "舖": "铺", "幷": "并", "槪": "概", "甁": "瓶",
    "攷": "考", "啣": "衔", "倂": "并", "賸": "剩", "繋": "系", "貍": "狸",
    "緜": "绵", "覊": "羁", "慇": "殷", "倣": "仿", "傚": "效", "搾": "榨",
    "姪": "侄", "粃": "秕", "嘷": "嗥",
})


def to_simplified(t2s):
    """OpenCC ``t2s`` followed by the variant table above."""
    return lambda text: t2s(text).translate(VARIANTS)


class BodyText(HTMLParser):
    """Collect paragraph text from rendered wikitext, skipping page chrome.

    Kept: ``<p>``, headings and ``<dd>`` lines (verse, quoted letters).
    List items are not kept: on these pages they are contents and navigation.
    Skipped wholesale: tables and the header container (navigation, source
    note), footnote markers, reference lists, edit links and hidden text.
    """

    BLOCKS = {"p", "h2", "h3", "h4", "dd"}
    SKIP_TAGS = {"table", "style", "script", "sup", "ol"}
    SKIP_CLASSES = (
        "reference", "references", "mw-editsection", "noprint", "ws-noexport",
        "licenseContainer", "licensetpl",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: List[str] = []
        self._skip_depth = 0
        self._stack: List[Tuple[str, bool]] = []
        self._buffer: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):  # type: ignore[no-untyped-def]
        if tag in {"br", "img", "hr", "meta", "link", "wbr"}:
            if tag == "br" and self._buffer is not None and not self._skip_depth:
                self._buffer.append("\n")
            return
        attr = dict(attrs)
        classes = attr.get("class") or ""
        skip = (
            tag in self.SKIP_TAGS
            or attr.get("id") in {"headerContainer", "plainSister", "toc"}
            or any(name in classes.split() for name in self.SKIP_CLASSES)
            or "display:none" in (attr.get("style") or "").replace(" ", "")
        )
        self._stack.append((tag, skip))
        if skip:
            self._skip_depth += 1
        elif tag in self.BLOCKS and self._buffer is None and not self._skip_depth:
            self._buffer = []

    def handle_startendtag(self, tag, attrs):  # type: ignore[no-untyped-def]
        # Self-closing tags (<img/>, <br/>, <hr/>) open nothing: never let them
        # reach handle_endtag, which would unwind the open-tag stack.
        if tag == "br" and self._buffer is not None and not self._skip_depth:
            self._buffer.append("\n")

    def handle_endtag(self, tag):  # type: ignore[no-untyped-def]
        if tag in {"br", "img", "hr", "meta", "link", "wbr"}:
            return
        if not any(open_tag == tag for open_tag, _ in self._stack):
            return
        while self._stack:
            open_tag, skipped = self._stack.pop()
            if skipped:
                self._skip_depth -= 1
            if open_tag in self.BLOCKS and self._buffer is not None and not self._skip_depth:
                text = "".join(self._buffer)
                for line in text.split("\n"):
                    line = re.sub(r"\s+", " ", line).strip()
                    if line:
                        self.paragraphs.append(line)
                self._buffer = None
            if open_tag == tag:
                break

    def handle_data(self, data):  # type: ignore[no-untyped-def]
        if self._buffer is not None and not self._skip_depth:
            self._buffer.append(data)


def fetch(page: str, cache_dir: Path) -> Tuple[str, str]:
    """Return ``(resolved title, rendered HTML)`` for a page, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (re.sub(r"[/\s]", "_", page) + ".json")
    if not cache.is_file():
        query = urllib.parse.urlencode(
            {
                "action": "parse",
                "page": page,
                "prop": "text",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
        )
        request = urllib.request.Request(
            API + "?" + query, headers={"User-Agent": USER_AGENT}
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    cache.write_bytes(response.read())
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))
        time.sleep(2)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    if "error" in payload:
        raise ValueError("{0}: {1}".format(page, payload["error"].get("info")))
    return payload["parse"]["title"], payload["parse"]["text"]


def chapter_paragraphs(chapter: Chapter, cache_dir: Path, convert) -> List[str]:
    title, rendered = fetch(chapter.page, cache_dir)
    if "消歧义" in rendered or "消歧義" in rendered:
        raise ValueError("{0} resolved to a disambiguation page".format(chapter.page))
    parser = BodyText()
    parser.feed(rendered)
    paragraphs = [convert(text.replace("\u200b", "")) for text in parser.paragraphs]
    paragraphs = [text for text in paragraphs if text]
    if chapter.section is not None:
        start = paragraphs.index(convert(chapter.section))
        paragraphs = paragraphs[start + 1 :]
    # Drop a leading repeat of the chapter title, and trailing editorial notes.
    while paragraphs and paragraphs[0].replace(" ", "") in {
        chapter.heading.replace(" ", ""),
        convert(title.split("/")[-1]).replace(" ", ""),
    }:
        paragraphs.pop(0)
    for marker in ("注释", "注解", "註釋"):
        if marker in paragraphs:
            paragraphs = paragraphs[: paragraphs.index(marker)]
    if sum(len(p) for p in paragraphs) < 300:
        raise ValueError("{0}: only {1} characters of body text".format(
            chapter.page, sum(len(p) for p in paragraphs)))
    return paragraphs


def render(book: ZhBook, cache_dir: Path, output_dir: Path, convert) -> Path:
    body_style, heading_style, title_style = base.styles(_style_key())
    note_style = ParagraphStyle(
        "Source-note-zh", fontName="STSong-Light", fontSize=9, leading=14,
        alignment=TA_CENTER, textColor="#555555",
    )
    output = output_dir / book.output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(48, 45, A4[0] - 96, A4[1] - 92, id="content")
    doc = base.BookTemplate(
        str(output), pagesize=A4, title=book.title, author=book.author,
        subject="Public-domain BookTalk example", running_title=book.title,
    )
    doc.addPageTemplates([PageTemplate(id="book", frames=[frame], onPage=base.decorate)])
    source = "https://zh.wikisource.org/wiki/" + urllib.parse.quote(book.index_page)
    story = [
        Spacer(1, 165),
        Paragraph(html.escape(book.title), title_style),
        Paragraph(html.escape("{0}（{1}）".format(book.author, book.year)), note_style),
        Spacer(1, 36),
        Paragraph(
            html.escape("公有领域作品。文本来源：维基文库（CC BY-SA 4.0），经 OpenCC 转为简体。")
            + "<br/>" + html.escape(source),
            note_style,
        ),
        PageBreak(),
    ]
    for chapter in book.chapters:
        heading = Paragraph(html.escape(chapter.heading), heading_style)
        heading.is_chapter_heading = True
        story.append(heading)
        for text in chapter_paragraphs(chapter, cache_dir, convert):
            story.append(Paragraph(html.escape(text), body_style))
    doc.build(story)
    return output


def _style_key():
    """``base.styles`` only reads ``.language``; give it a zh stand-in."""
    from types import SimpleNamespace

    return SimpleNamespace(language="zh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dump", action="store_true", help="print paragraphs, build nothing")
    args = parser.parse_args()

    from opencc import OpenCC

    convert = to_simplified(OpenCC("t2s").convert)
    if args.dump:
        for book in BOOKS:
            for chapter in book.chapters:
                paragraphs = chapter_paragraphs(chapter, args.cache_dir, convert)
                print("##", book.title, chapter.heading, len(paragraphs), "paras",
                      sum(map(len, paragraphs)), "chars")
                print("   first:", paragraphs[0][:60])
                print("   last: ", paragraphs[-1][:60])
        return

    registerFont(UnicodeCIDFont("STSong-Light"))
    outputs = []
    for book in BOOKS:
        outputs.append((book, render(book, args.cache_dir, args.output_dir, convert)))
    base.validate(outputs)


if __name__ == "__main__":
    main()
