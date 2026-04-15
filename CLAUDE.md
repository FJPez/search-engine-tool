# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP3011 (Web Services and Web Data) Coursework 2: a Python search engine tool that crawls [https://quotes.toscrape.com/](https://quotes.toscrape.com/), builds an inverted index, and answers queries via a CLI. Individual assessment, due **8 May 2026**, 30% of module mark. Assessed via a 5-minute video demo + GitHub repo + index file.

No code exists yet — only the brief (`COMP3011_Coursework2_Brief__2025_2026.pdf`). Expected layout per the brief:

```
src/         crawler.py  indexer.py  search.py  main.py
tests/       test_crawler.py  test_indexer.py  test_search.py
data/        [compiled index file]
requirements.txt  README.md
```

## Hard Requirements (from the brief)

- **Language/libs**: Python, using `requests` + `beautifulsoup4`.
- **Politeness window**: at least **6 seconds** between successive HTTP requests to quotes.toscrape.com. Non-negotiable — enforce in the crawler, not at call sites.
- **Target site only**: crawl `quotes.toscrape.com`. Stay on-domain.
- **Inverted index** must store per-word statistics (frequency, position, etc.) per page, not just page membership.
- **Case-insensitive** matching (`Good` == `good`).
- **Single-file index** persisted to disk (acceptable simplification per brief).
- **CLI commands** (exact names) — the tool is a shell that accepts:
  - `build` — crawl, build index, save to disk
  - `load`  — load previously-saved index
  - `print <word>` — print inverted index entry for a word
  - `find <w1> [w2 ...]` — return pages containing **all** given words (multi-word = conjunctive)
- **Edge cases** to handle explicitly: non-existent words, empty queries, multi-word queries, network failures.

## Architecture Notes

Four-module split is prescribed by the brief and should be respected:

- `crawler.py` — fetches pages, enforces the 6s politeness delay, extracts links + text. Owns all network I/O and retry/error handling.
- `indexer.py` — tokenises (lowercase), builds the inverted index data structure (word → list of {page, freq, positions, ...}), and handles save/load serialisation.
- `search.py` — query logic: `print` (lookup one word) and `find` (intersect posting lists for multi-word AND queries).
- `main.py` — CLI shell dispatching `build`/`load`/`print`/`find`.

The index is the contract between indexer and search — pick its shape deliberately and document the choice (it's explicitly called out in the marking scheme and the video walkthrough).

## Grading Signals to Optimise For

Marks are weighted: Testing 20%, GenAI critical evaluation 15%, Search 12%, Crawler/Indexer/Docs 10% each, Storage 8%, Video 10%, Git 5%. Practical implications:

- **Tests are the single largest slice.** Write `tests/test_*.py` alongside each module, not at the end. Target >85% coverage for the top band. Mock network in crawler tests (do not hit the live site from the test suite).
- **Commit incrementally with meaningful messages** — git history is graded directly.
- **GenAI use is Green-category** (permitted, encouraged) but the video must include a specific critical evaluation. If AI is used during development, keep notes on where it helped/hindered so the reflection is concrete, not generic.

## Commands (once implemented)

```bash
pip install -r requirements.txt
python src/main.py         # launches the interactive shell: build / load / print <w> / find <w>...
pytest                     # run full test suite
pytest tests/test_indexer.py::test_name   # single test
```

## Commit Discipline

- Work lands in **small, independently reviewable commits**. The user reviews each commit before the next is written — do not batch unrelated changes.
- Every commit must leave `make check` green (ruff, ty, pytest).
- Use conventional-commit prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`) scoped by module where relevant (e.g. `feat(crawler): ...`).
- Tests live in the same commit as the code they cover, not a separate "add tests" commit at the end.
- **Write tests as flat `test_*` functions, never grouped inside `Test*` classes.** Parametrise with `@pytest.mark.parametrize` rather than reaching for class-based organisation.
- Never `--amend` or force-push a commit the user has already reviewed.

## Submission Artefacts

Three things go to Minerva: video link (unlisted, not private; test in incognito), public GitHub URL, and the compiled index file. README must cover overview, setup, usage for all four commands, testing instructions, and dependencies.
