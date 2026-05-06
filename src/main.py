"""Interactive command-line shell for the COMP3011 search engine tool."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from src.crawler import PoliteCrawler
from src.indexer import InvertedIndex
from src.search import find, print_entry

DEFAULT_START_URL = "https://quotes.toscrape.com/"
DEFAULT_INDEX_PATH = Path("data/index.json")


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
                self._build()
            case "load":
                self._load()
            case "print":
                self._print(argument)
            case "find":
                self._find(argument)
            case "help":
                self._help()
            case "exit" | "quit":
                return False
            case _:
                self._write(f"Unknown command: {command}")
                self._help()

        return True

    def _build(self) -> None:
        try:
            crawler = PoliteCrawler(self.start_url)
            index = InvertedIndex.from_pages(crawler.crawl())
            index.save(self.index_path)
        except Exception as exc:
            self._write(f"Build failed: {exc}")
            return

        self.index = index
        self._write(
            f"Built index for {len(index)} {_document_label(len(index))}; "
            f"saved to {self.index_path}"
        )

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

        results = find(self.index, query)
        if not results:
            self._write(f"No pages found for {query!r}")
            return

        for url in results:
            self._write(url)

    def _help(self) -> None:
        self._write("Commands: build, load, print <word>, find <word> [word ...], help, exit")

    def _write(self, message: str) -> None:
        print(message, file=self.output)


def _document_label(count: int) -> str:
    return "document" if count == 1 else "documents"


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
