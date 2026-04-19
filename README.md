# Search Engine Tool

COMP3011 Coursework 2 — a Python CLI search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index, and answers single- and multi-word queries.

## Status

- ✅ Crawler ([`src/crawler.py`](src/crawler.py)) — breadth-first, single-host, 6 s politeness window, retry-aware, fully unit-tested with mocked HTTP.
- ⏳ Indexer, search, CLI — not yet implemented.

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
uv run python -m src.main
```

Commands:

| Command | Description |
| --- | --- |
| `build` | Crawl the target site, build the inverted index, save to disk |
| `load` | Load a previously built index from disk |
| `print <word>` | Print the inverted-index entry for a word |
| `find <w1> [w2 ...]` | Return pages containing **all** given words |

Example session:

```
> build
> print indifference
> find good friends
```

## Testing

```bash
uv run pytest              # full suite
uv run pytest --cov=src    # with coverage
```

## Dependencies

- `requests` — HTTP client
- `beautifulsoup4` — HTML parsing
- `lxml` — fast parser backend for BeautifulSoup
- `pytest`, `pytest-cov`, `requests-mock` — testing (dev)
