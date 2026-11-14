# Public-domain example books

The PDFs in `books/` contain five real, complete public-domain works prepared
for BookTalk from Project Gutenberg's UTF-8 transcriptions. The wording is not
abridged or rewritten; the build script only removes the distribution wrapper,
lays out the text, and adds PDF chapter bookmarks.

| Language | Work | Author | Source |
|---|---|---|---|
| Chinese | 《論語》 | Confucius and disciples | [Project Gutenberg #23839](https://www.gutenberg.org/ebooks/23839) |
| Chinese | 《道德經》 | Laozi | [Project Gutenberg #7337](https://www.gutenberg.org/ebooks/7337) |
| Chinese | 《孫子兵法》 | Sunzi | [Project Gutenberg #12407](https://www.gutenberg.org/ebooks/12407) |
| English | *Alice's Adventures in Wonderland* | Lewis Carroll | [Project Gutenberg #11](https://www.gutenberg.org/ebooks/11) |
| English | *The Strange Case of Dr. Jekyll and Mr. Hyde* | Robert Louis Stevenson | [Project Gutenberg #43](https://www.gutenberg.org/ebooks/43) |

Each catalog entry identifies its text as public domain in the USA. Check the
copyright law that applies in your jurisdiction before redistributing files.

Regenerate after downloading the five source text files listed in
`scripts/build_public_domain_examples.py`:

```bash
backend/.venv/bin/python scripts/build_public_domain_examples.py
```
