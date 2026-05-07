"""Interactive command-line shell for the COMP3011 search engine tool."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from src.crawler import DEFAULT_DELAY, PoliteCrawler
from src.indexer import InvertedIndex
from src.search import explain, print_entry, result_details

DEFAULT_START_URL = "https://quotes.toscrape.com/"
DEFAULT_INDEX_PATH = Path("data/index.json")
_SHELL_HANDLER_ATTR = "_search_shell_handler"


class SearchShell:
    """Stateful dispatcher for the coursework CLI commands."""

    def __init__(
        self,
        *,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        start_url: str = DEFAULT_START_URL,
        output: TextIO = sys.stdout,
    ) -> None:
        self.index_path = Path(index_path)
        self.start_url = start_url
        self.output = output
        self.index: InvertedIndex | None = None
        self.verbose = False
        self._logger = logging.getLogger(f"{__name__}.SearchShell.{id(self)}")
        self._log_handler = logging.StreamHandler(output)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(self._log_handler, _SHELL_HANDLER_ATTR, True)
        self._configure_logging()

    def run_command(self, line: str) -> bool:
        """Run one command line. Return ``False`` when the shell should exit."""
        stripped = line.strip()
        if not stripped:
            return True

        command, _, argument = stripped.partition(" ")
        command = command.lower()
        argument = argument.strip()

        match command:
            case "build":
                self._build(argument)
            case "load":
                self._load()
            case "print":
                self._print(argument)
            case "find":
                self._find(argument)
            case "explain":
                self._explain(argument)
            case "verbose":
                self._verbose(argument)
            case "help":
                self._help()
            case "exit" | "quit":
                return False
            case _:
                self._write(f"Unknown command: {command}")
                self._help()

        return True

    def _build(self, argument: str) -> None:
        max_pages = self._parse_max_pages(argument)
        if max_pages == 0:
            self._write("Usage: build [max_pages]")
            return

        crawler_logger_state = self._attach_crawler_logger()
        try:
            crawler = PoliteCrawler(self.start_url, max_pages=max_pages)
            self._logger.info("Building index from %s", self.start_url)
            self._logger.info("Limit: %s", _format_limit(max_pages))
            self._logger.info("Politeness delay: %.0f seconds between requests", DEFAULT_DELAY)
            index = InvertedIndex.from_pages(self._report_pages(crawler.crawl()))
            index.save(self.index_path)
        except Exception as exc:
            self._write(f"Build failed: {exc}")
            return
        finally:
            self._restore_crawler_logger(crawler_logger_state)

        self.index = index
        self._logger.info(
            "Built index for %d %s; saved to %s",
            len(index),
            _document_label(len(index)),
            self.index_path,
        )

    def _parse_max_pages(self, argument: str) -> int | None:
        if not argument:
            return None

        parts = argument.split()
        if len(parts) != 1:
            return 0

        try:
            max_pages = int(parts[0])
        except ValueError:
            return 0

        if max_pages < 1:
            return 0
        return max_pages

    def _report_pages(self, pages: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
        for count, (url, html) in enumerate(pages, start=1):
            self._logger.info("Indexed %d: %s", count, url)
            yield url, html

    def _load(self) -> None:
        try:
            self.index = InvertedIndex.load(self.index_path)
        except FileNotFoundError:
            self._write(f"No index file found at {self.index_path}. Run 'build' first.")
            return
        except ValueError as exc:
            self._write(f"Could not load index: {exc}")
            return

        self._write(f"Loaded index with {len(self.index)} {_document_label(len(self.index))}")

    def _print(self, word: str) -> None:
        if self.index is None:
            self._write("No index loaded. Run 'build' or 'load' first.")
            return
        if not word:
            self._write("Usage: print <word>")
            return

        self._write(print_entry(self.index, word))

    def _find(self, query: str) -> None:
        if self.index is None:
            self._write("No index loaded. Run 'build' or 'load' first.")
            return
        if not query:
            self._write("Usage: find <word> [word ...]")
            return

        results = result_details(self.index, query)
        if not results:
            self._write(f"No pages found for: {query}")
            return

        for result in results:
            self._write(f"{result.url}  score={result.score:.1f}")
            if result.snippet:
                self._write(f"  {result.snippet}")

    def _explain(self, query: str) -> None:
        if self.index is None:
            self._write("No index loaded. Run 'build' or 'load' first.")
            return
        if not query:
            self._write("Usage: explain <query>")
            return

        self._write(explain(self.index, query))

    def _verbose(self, argument: str) -> None:
        if not argument:
            state = "on" if self.verbose else "off"
            self._write(f"Verbose mode is {state}")
            return

        match argument.lower():
            case "on":
                self.verbose = True
                self._configure_logging()
                self._write("Verbose mode is on")
            case "off":
                self.verbose = False
                self._configure_logging()
                self._write("Verbose mode is off")
            case _:
                self._write("Usage: verbose on|off")

    def _help(self) -> None:
        self._write(
            "Commands: build [max_pages], load, print <word>, "
            "find <query>, explain <query>, verbose on|off, help, exit"
        )

    def _write(self, message: str) -> None:
        print(message, file=self.output)

    def _configure_logging(self) -> None:
        level = logging.DEBUG if self.verbose else logging.INFO
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._log_handler.setLevel(level)
        if self._log_handler not in self._logger.handlers:
            self._logger.addHandler(self._log_handler)

    def _attach_crawler_logger(self) -> _LoggerState:
        logger = logging.getLogger("src.crawler")
        state = _LoggerState(
            level=logger.level,
            propagate=logger.propagate,
            handlers=list(logger.handlers),
        )
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.propagate = False
        logger.handlers = [
            handler
            for handler in logger.handlers
            if not getattr(handler, _SHELL_HANDLER_ATTR, False)
        ]
        logger.addHandler(self._log_handler)
        return state

    def _restore_crawler_logger(self, state: _LoggerState) -> None:
        logger = logging.getLogger("src.crawler")
        logger.setLevel(state.level)
        logger.propagate = state.propagate
        logger.handlers = state.handlers


@dataclass(frozen=True, slots=True)
class _LoggerState:
    level: int
    propagate: bool
    handlers: list[logging.Handler]


def _document_label(count: int) -> str:
    return "document" if count == 1 else "documents"


def _format_limit(max_pages: int | None) -> str:
    if max_pages is None:
        return "none"
    label = "page" if max_pages == 1 else "pages"
    return f"{max_pages} {label}"


def main() -> int:
    """Launch the interactive shell."""
    shell = SearchShell()
    shell._help()

    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not shell.run_command(line):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
