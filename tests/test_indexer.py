"""Unit tests for :mod:`src.indexer`."""

from __future__ import annotations

import pytest

from src.indexer import extract_fields, tokenise


def _html(title: str = "", body: str = "") -> str:
    """Build a minimal HTML document with optional title and body."""
    title_tag = f"<title>{title}</title>" if title else ""
    return f"<!doctype html><html><head>{title_tag}</head><body>{body}</body></html>"


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


# --------------------------------------------------------------------------
# extract_fields
# --------------------------------------------------------------------------


def test_extract_fields_title_and_body() -> None:
    result = extract_fields(_html(title="Quotes to Scrape", body="<p>Hello world</p>"))
    assert result["title"] == "Quotes to Scrape"
    assert "Hello world" in result["body"]
    # Title is not duplicated into the body stream
    assert "Quotes to Scrape" not in result["body"]


def test_extract_fields_missing_title_is_empty_string() -> None:
    result = extract_fields(_html(body="<p>no title here</p>"))
    assert result["title"] == ""
    assert "no title here" in result["body"]


def test_extract_fields_strips_script_and_style_and_noscript() -> None:
    body = (
        "<script>var secret='indexthis';</script>"
        "<style>.a { content: 'cssleak'; }</style>"
        "<noscript>enable javascript</noscript>"
        "<p>real content</p>"
    )
    result = extract_fields(_html(title="T", body=body))
    assert "real content" in result["body"]
    for leaked in ("secret", "indexthis", "cssleak", "enable javascript"):
        assert leaked not in result["body"]


def test_extract_fields_flattens_nested_inline_tags() -> None:
    # Adjacent inline tags must not glue into a single word
    result = extract_fields(_html(body="<p>hello <b>world</b></p>"))
    body_tokens = tokenise(result["body"])
    assert body_tokens == ["hello", "world"]


def test_extract_fields_tolerates_malformed_html() -> None:
    # lxml should recover gracefully; we only require no exception + sane body
    result = extract_fields("<html><body><p>oops <b>unclosed</p></body>")
    assert "oops" in result["body"]
    assert "unclosed" in result["body"]


def test_extract_fields_empty_input() -> None:
    result = extract_fields("")
    assert result == {"title": "", "body": ""}


def test_extract_fields_whitespace_only_title_is_empty() -> None:
    # <title>   </title> should not leak whitespace into the title field
    result = extract_fields(_html(title="   ", body="content"))
    assert result["title"] == ""
