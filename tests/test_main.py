"""Unit tests for the interactive CLI in :mod:`src.main`."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pytest
from requests_mock import Mocker

from src.indexer import InvertedIndex
from src.main import SearchShell
from tests.conftest import HtmlFactory

BASE_URL = "http://quotes.toscrape.com/"


def _shell(tmp_path: Path) -> tuple[SearchShell, StringIO]:
    output = StringIO()
    shell = SearchShell(index_path=tmp_path / "index.json", output=output)
    return shell, output


def test_build_crawls_builds_saves_and_keeps_index_loaded(
    tmp_path: Path, html: HtmlFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [
        ("http://quotes.toscrape.com/", html(body="good friends")),
        ("http://quotes.toscrape.com/page/2", html(body="indifference")),
    ]

    class FakeCrawler:
        def __init__(self, start_url: str, *, max_pages: int | None = None) -> None:
            assert start_url == "https://quotes.toscrape.com/"
            assert max_pages is None

        def crawl(self) -> list[tuple[str, str]]:
            return pages

    monkeypatch.setattr("src.main.PoliteCrawler", FakeCrawler)
    shell, output = _shell(tmp_path)

    assert shell.run_command("build") is True

    assert shell.index is not None
    assert len(shell.index) == 2
    assert shell.index_path.exists()
    text = output.getvalue()
    assert "Building index from https://quotes.toscrape.com/" in text
    assert "Limit: none" in text
    assert "Politeness delay: 6 seconds between requests" in text
    assert "Indexed 1: http://quotes.toscrape.com/" in text
    assert "Indexed 2: http://quotes.toscrape.com/page/2" in text
    assert "Built index for 2 documents" in text


def test_build_accepts_max_pages_limit(
    tmp_path: Path, html: HtmlFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_limits: list[int | None] = []

    class FakeCrawler:
        def __init__(self, _start_url: str, *, max_pages: int | None = None) -> None:
            seen_limits.append(max_pages)

        def crawl(self) -> list[tuple[str, str]]:
            return [("http://quotes.toscrape.com/", html(body="limited"))]

    monkeypatch.setattr("src.main.PoliteCrawler", FakeCrawler)
    shell, output = _shell(tmp_path)

    assert shell.run_command("build 10") is True

    assert seen_limits == [10]
    assert "Limit: 10 pages" in output.getvalue()


@pytest.mark.parametrize("command", ["build abc", "build 0", "build -1", "build 1 2"])
def test_build_rejects_invalid_max_pages(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    instantiated = False

    class FakeCrawler:
        def __init__(self, _start_url: str, *, max_pages: int | None = None) -> None:
            nonlocal instantiated
            instantiated = True

    monkeypatch.setattr("src.main.PoliteCrawler", FakeCrawler)
    shell, output = _shell(tmp_path)

    assert shell.run_command(command) is True

    assert instantiated is False
    assert "Usage: build [max_pages]" in output.getvalue()


def test_load_reads_existing_index_and_enables_print(tmp_path: Path, html: HtmlFactory) -> None:
    index_path = tmp_path / "index.json"
    index = InvertedIndex.from_pages([("http://q/1", html(body="good good"))])
    index.save(index_path)

    output = StringIO()
    shell = SearchShell(index_path=index_path, output=output)

    assert shell.run_command("load") is True
    assert shell.index is not None
    assert shell.run_command("print good") is True

    text = output.getvalue()
    assert "Loaded index with 1 document" in text
    assert "good (1 documents)" in text
    assert "count=2" in text


def test_print_requires_loaded_index(tmp_path: Path) -> None:
    shell, output = _shell(tmp_path)

    assert shell.run_command("print good") is True

    assert "No index loaded" in output.getvalue()


def test_print_requires_a_word(tmp_path: Path, html: HtmlFactory) -> None:
    shell, output = _shell(tmp_path)
    shell.index = InvertedIndex.from_pages([("http://q/1", html(body="good"))])

    assert shell.run_command("print") is True

    assert "Usage: print <word>" in output.getvalue()


def test_find_requires_loaded_index(tmp_path: Path) -> None:
    shell, output = _shell(tmp_path)

    assert shell.run_command("find good") is True

    assert "No index loaded" in output.getvalue()


def test_find_requires_query_terms(tmp_path: Path, html: HtmlFactory) -> None:
    shell, output = _shell(tmp_path)
    shell.index = InvertedIndex.from_pages([("http://q/1", html(body="good"))])

    assert shell.run_command("find") is True

    assert "Usage: find <word> [word ...]" in output.getvalue()


def test_find_prints_matching_urls_one_per_line(tmp_path: Path, html: HtmlFactory) -> None:
    shell, output = _shell(tmp_path)
    shell.index = InvertedIndex.from_pages(
        [
            ("http://q/1", html(body="good friends")),
            ("http://q/2", html(body="good")),
        ]
    )

    assert shell.run_command("find good friends") is True

    lines = output.getvalue().splitlines()
    assert "http://q/1" in lines
    assert "http://q/2" not in lines


def test_find_prints_zero_results_message(tmp_path: Path, html: HtmlFactory) -> None:
    shell, output = _shell(tmp_path)
    shell.index = InvertedIndex.from_pages([("http://q/1", html(body="good"))])

    assert shell.run_command("find missing") is True

    assert "No pages found" in output.getvalue()


def test_unknown_command_prints_help_hint(tmp_path: Path) -> None:
    shell, output = _shell(tmp_path)

    assert shell.run_command("nonsense") is True

    assert "Unknown command" in output.getvalue()
    assert "build [max_pages]" in output.getvalue()


def test_verbose_commands_toggle_debug_output(tmp_path: Path) -> None:
    shell, output = _shell(tmp_path)

    assert shell.run_command("verbose") is True
    assert shell.run_command("verbose on") is True
    assert shell.run_command("verbose") is True
    assert shell.run_command("verbose off") is True

    text = output.getvalue()
    assert "Verbose mode is off" in text
    assert "Verbose mode is on" in text


def test_verbose_rejects_unknown_state(tmp_path: Path) -> None:
    shell, output = _shell(tmp_path)

    assert shell.run_command("verbose maybe") is True

    assert "Usage: verbose on|off" in output.getvalue()


def test_verbose_mode_controls_crawler_debug_output(
    tmp_path: Path, html: HtmlFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCrawler:
        def __init__(self, _start_url: str, *, max_pages: int | None = None) -> None:
            pass

        def crawl(self) -> list[tuple[str, str]]:
            logging.getLogger("src.crawler").debug("debug crawl detail")
            return [("http://quotes.toscrape.com/", html(body="debug"))]

    monkeypatch.setattr("src.main.PoliteCrawler", FakeCrawler)
    normal_shell, normal_output = _shell(tmp_path)
    verbose_shell, verbose_output = _shell(tmp_path)
    verbose_shell.run_command("verbose on")

    assert normal_shell.run_command("build 1") is True
    assert verbose_shell.run_command("build 1") is True

    assert "debug crawl detail" not in normal_output.getvalue()
    assert "debug crawl detail" in verbose_output.getvalue()


def test_verbose_build_shows_real_crawler_fetch_diagnostics(
    tmp_path: Path, html: HtmlFactory, requests_mock: Mocker
) -> None:
    requests_mock.get(BASE_URL, text=html(body="visible diagnostics"))
    normal_output = StringIO()
    verbose_output = StringIO()
    normal_shell = SearchShell(
        index_path=tmp_path / "normal.json",
        start_url=BASE_URL,
        output=normal_output,
    )
    verbose_shell = SearchShell(
        index_path=tmp_path / "verbose.json",
        start_url=BASE_URL,
        output=verbose_output,
    )
    verbose_shell.run_command("verbose on")

    assert normal_shell.run_command("build 1") is True
    assert verbose_shell.run_command("build 1") is True

    assert "Fetching http://quotes.toscrape.com/" not in normal_output.getvalue()
    assert "Fetched http://quotes.toscrape.com/: HTTP 200" not in normal_output.getvalue()
    assert "Fetching http://quotes.toscrape.com/" in verbose_output.getvalue()
    assert "Fetched http://quotes.toscrape.com/: HTTP 200" in verbose_output.getvalue()


@pytest.mark.parametrize("command", ["exit", "quit"])
def test_exit_commands_stop_loop(tmp_path: Path, command: str) -> None:
    shell, _output = _shell(tmp_path)

    assert shell.run_command(command) is False
