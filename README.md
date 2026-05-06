# Search Engine Tool

COMP3011 Coursework 2 — a Python CLI search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index, and answers single- and multi-word queries.

## Status

- ✅ Crawler ([`src/crawler.py`](src/crawler.py)) — breadth-first, single-host, robots.txt-aware, 6 s minimum politeness window, retry-aware, fully unit-tested with mocked HTTP.
- ✅ Indexer ([`src/indexer.py`](src/indexer.py)) — two-pass tokenisation (BS4 + `\W+` split with apostrophe stripping), inverted index with per-posting positions, document field extents, quote tags, and single-file JSON persistence with strict load-time validation.
- ✅ Search ([`src/search.py`](src/search.py)) — conjunctive query processing with tag filters, quoted phrase matching, simple ranked output, formatted single-word entry printing, and query explanations.
- ✅ CLI ([`src/main.py`](src/main.py)) — interactive shell with build/load/print/find/explain commands, optional build limits, concise crawl progress, and verbose crawl diagnostics.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

A plain `requirements.txt` is also provided for environments without uv:

```bash
pip install -r requirements.txt
```

## Usage

Launch the interactive shell:

```bash
make run
```

Equivalent direct command:

```bash
uv run python -m src.main
```

Commands:

| Command | Description |
| --- | --- |
| `build [max_pages]` | Crawl the target site, build the inverted index, save to disk. Optional page limit is useful for quick demos/tests |
| `load` | Load a previously built index from disk |
| `print <word>` | Print the inverted-index entry for a word |
| `find <query>` | Return ranked pages matching all query terms, quoted phrases, and `tag:<name>` filters |
| `explain <query>` | Show parsed terms, phrases, tags, matching pages, and scores |
| `verbose on\|off` | Toggle detailed crawler diagnostics during `build` |
| `help` | Print available commands |
| `exit` / `quit` | Leave the shell |

Example session:

```
> build 10
> print indifference
> find good friends
> find "good friends" tag:life
> explain love tag:inspirational
> verbose on
> build 3
> quit
```

Search syntax:

- `find good friends` returns pages containing both words.
- `find "good friends"` requires the words to appear next to each other.
- `find love tag:life` filters results to pages with quotes tagged `life`.
- Results are ranked by a simple explainable score using term frequency, phrase matches, tag matches, and title hits.

`build` respects `robots.txt` when present. Any `Crawl-delay` greater than the coursework's 6 s minimum is honoured; smaller values do not reduce the minimum politeness window.

## Testing

```bash
make check                 # lint + typecheck + tests
uv run pytest              # full suite
uv run pytest --cov=src    # with coverage
```

## Dependencies

- `requests` — HTTP client
- `beautifulsoup4` — HTML parsing
- `lxml` — fast parser backend for BeautifulSoup
- `pytest`, `pytest-cov`, `requests-mock` — testing (dev)
