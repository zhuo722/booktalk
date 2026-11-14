# Public-domain example books

The PDFs in `books/` are five real, complete public-domain works prepared for
BookTalk. The wording is not abridged or rewritten; the build scripts only drop
the source's page furniture (distribution wrapper, navigation, licence banner,
editorial footnotes), lay out the text, and add PDF chapter bookmarks so
BookTalk can detect structure.

| Language | Work | Author | Source |
|---|---|---|---|
| Chinese (simplified) | 《呐喊》(1923) | 鲁迅 Lu Xun | [Wikisource: 吶喊](https://zh.wikisource.org/wiki/%E5%90%B6%E5%96%8A) |
| Chinese (simplified) | 《朝花夕拾》(1928) | 鲁迅 Lu Xun | [Wikisource: 朝花夕拾](https://zh.wikisource.org/wiki/%E6%9C%9D%E8%8A%B1%E5%A4%95%E6%8B%BE) |
| Chinese (simplified) | 《浮生六记》卷一–四 (c. 1808) | 沈复 Shen Fu | [Wikisource: 浮生六記](https://zh.wikisource.org/wiki/%E6%B5%AE%E7%94%9F%E5%85%AD%E8%A8%98) |
| English | *Alice's Adventures in Wonderland* | Lewis Carroll | [Project Gutenberg #11](https://www.gutenberg.org/ebooks/11) |
| English | *The Strange Case of Dr. Jekyll and Mr. Hyde* | Robert Louis Stevenson | [Project Gutenberg #43](https://www.gutenberg.org/ebooks/43) |

**Chinese books.** Wikisource transcribes these works in traditional
characters (Lu Xun from the 1938《鲁迅全集》). The build converts them to
simplified Chinese with OpenCC `t2s`, then maps the old printed variant forms
OpenCC keeps (鄕, 靑, 覩, …) to the standard character, so a reader asking in
simplified Chinese matches the text. 《浮生六记》 includes books 1–4 only; books
5–6 are generally regarded as later forgeries. The Wikisource transcriptions
are CC BY-SA 4.0.

**English books.** Project Gutenberg's UTF-8 transcriptions.

All five works are public domain in the USA and in China. Check the copyright
law that applies in your jurisdiction before redistributing files.

Regenerate:

```bash
# Chinese: fetches from the Wikisource API (cached), needs OpenCC
backend/.venv/bin/pip install opencc-python-reimplemented==0.1.7
backend/.venv/bin/python scripts/build_simplified_chinese_examples.py

# English: download the two source files listed in the script first
backend/.venv/bin/python scripts/build_public_domain_examples.py
```
