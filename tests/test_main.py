"""Unit tests for the interactive CLI in :mod:`src.main`."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from src.indexer import InvertedIndex
from src.main import SearchShell
from tests.conftest import HtmlFactory


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
        def __init__(self, start_url: str) -> None:
            assert start_url == "https://quotes.toscrape.com/"

        def crawl(self) -> list[tuple[str, str]]:
            return pages

    monkeypatch.setattr("src.main.PoliteCrawler", FakeCrawler)
    shell, output = _shell(tmp_path)

    assert shell.run_command("build") is True

    assert shell.index is not None
    assert len(shell.index) == 2
    assert shell.index_path.exists()
    assert "Built index for 2 documents" in output.getvalue()


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
    assert "build" in output.getvalue()


@pytest.mark.parametrize("command", ["exit", "quit"])
def test_exit_commands_stop_loop(tmp_path: Path, command: str) -> None:
    shell, _output = _shell(tmp_path)

    assert shell.run_command(command) is False
