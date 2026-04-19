"""Unit tests for :mod:`src.indexer`."""

from __future__ import annotations

import pytest

from src.indexer import tokenise


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Empty / whitespace-only inputs produce no tokens
        ("", []),
        ("   ", []),
        ("\n\t  ", []),
        # Lowercasing (brief requires case-insensitive matching)
        ("Hello World", ["hello", "world"]),
        ("GOOD Good good", ["good", "good", "good"]),
        # Apostrophes stripped so contractions are a single token
        ("can't", ["cant"]),
        ("don't panic", ["dont", "panic"]),
        ("o'donnell", ["odonnell"]),
        # Curly-quote apostrophes normalised the same way
        ("can\u2019t", ["cant"]),
        # Hyphens split into separate tokens — \W+ consumes them
        ("t-shirt", ["t", "shirt"]),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        # Numbers are kept; periods/decimals split
        ("Nokia 3250", ["nokia", "3250"]),
        ("92.3", ["92", "3"]),
        # Abbreviations with periods split (documented cost)
        ("I.B.M.", ["i", "b", "m"]),
        # Short (1-2 char) tokens are preserved; they matter for phrases
        ("j lo el paso", ["j", "lo", "el", "paso"]),
        # Stopwords are kept; removing them would break phrase-style queries
        (
            "To be or not to be",
            ["to", "be", "or", "not", "to", "be"],
        ),
        # Unicode letters survive via re.UNICODE
        ("café", ["café"]),
        ("naïve résumé", ["naïve", "résumé"]),
        # Leading/trailing separators don't yield empty tokens
        ("  hello,   world!  ", ["hello", "world"]),
        ("...dots...", ["dots"]),
        # Underscores are kept — they're part of \w
        ("snake_case_name", ["snake_case_name"]),
        # Mixed punctuation collapses into one split
        ("foo---bar???baz", ["foo", "bar", "baz"]),
    ],
)
def test_tokenise(text: str, expected: list[str]) -> None:
    assert tokenise(text) == expected


def test_tokenise_is_idempotent_on_its_own_output() -> None:
    """Tokens round-trip: re-tokenising a joined token list yields the same
    tokens. Indexer and search.py share this function, so a query and its
    indexed form must produce identical tokens after any number of passes."""
    text = "The Quick Brown Fox — can't stop, won't stop!"
    once = tokenise(text)
    twice = tokenise(" ".join(once))
    assert once == twice
