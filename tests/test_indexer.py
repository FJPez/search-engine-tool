"""Unit tests for :mod:`src.indexer`."""

from __future__ import annotations

import pytest

from src.indexer import Document, InvertedIndex, Posting, extract_fields, tokenise


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


# --------------------------------------------------------------------------
# InvertedIndex.add_document / lookup / field_of
# --------------------------------------------------------------------------


def test_add_document_returns_sequential_doc_ids() -> None:
    idx = InvertedIndex()
    assert idx.add_document("http://a/", _html(body="alpha")) == 0
    assert idx.add_document("http://b/", _html(body="beta")) == 1
    assert idx.add_document("http://c/", _html(body="gamma")) == 2
    assert len(idx) == 3


def test_add_document_is_idempotent_on_url() -> None:
    idx = InvertedIndex()
    first = idx.add_document("http://dup/", _html(body="hello world"))
    second = idx.add_document("http://dup/", _html(body="something else entirely"))
    assert first == second
    assert len(idx) == 1
    # The second call must not re-tokenise: posting list for "hello" should
    # still point at one doc with count 1 — not doubled, not replaced.
    postings = idx.lookup("hello")
    assert len(postings) == 1
    assert postings[0].count == 1
    # And the later body ("something") never got indexed
    assert idx.lookup("something") == []


def test_lookup_counts_and_positions_for_single_document() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(title="T", body="good good friends"))
    # Stream: [T, good, good, friends] -> positions 0, 1, 2, 3
    good = idx.lookup("good")
    assert good == [Posting(doc_id=0, count=2, positions=(1, 2))]
    assert idx.lookup("friends") == [Posting(doc_id=0, count=1, positions=(3,))]
    assert idx.lookup("t") == [Posting(doc_id=0, count=1, positions=(0,))]


def test_lookup_aggregates_across_documents_sorted_by_doc_id() -> None:
    idx = InvertedIndex()
    idx.add_document("http://a/", _html(body="good good"))
    idx.add_document("http://b/", _html(body="bad"))
    idx.add_document("http://c/", _html(body="good"))
    good = idx.lookup("good")
    assert [p.doc_id for p in good] == [0, 2]
    assert good[0].count == 2
    assert good[1].count == 1


def test_lookup_is_case_insensitive_on_query() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="Hello World"))
    assert idx.lookup("HELLO") == idx.lookup("hello")
    assert idx.lookup("Hello")[0].count == 1


def test_lookup_unknown_word_returns_empty_list() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="only this word"))
    assert idx.lookup("absent") == []


def test_contains_is_case_insensitive() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="Indifference"))
    assert "indifference" in idx
    assert "INDIFFERENCE" in idx
    assert "missing" not in idx
    # Non-string comparisons don't explode
    assert 42 not in idx


def test_field_extents_mark_title_before_body() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(title="alpha beta", body="gamma delta"))
    doc = idx.documents[0]
    assert doc.fields == {"title": (0, 2), "body": (2, 4)}


def test_field_of_resolves_title_and_body_positions() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(title="alpha beta", body="gamma"))
    # Positions 0-1 are title, 2 is body
    assert idx.field_of(0, 0) == "title"
    assert idx.field_of(0, 1) == "title"
    assert idx.field_of(0, 2) == "body"
    # Outside any extent -> None
    assert idx.field_of(0, 99) is None
    # Unknown doc -> None
    assert idx.field_of(999, 0) is None


def test_field_of_reports_title_when_term_appears_in_title() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(title="unique headline", body="filler words"))
    posting = idx.lookup("unique")[0]
    assert idx.field_of(posting.doc_id, posting.positions[0]) == "title"


def test_documents_property_is_a_defensive_copy() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="x"))
    snapshot = idx.documents
    snapshot[999] = Document(doc_id=999, url="http://evil/", fields={})
    # Mutating the snapshot must not affect the real catalogue
    assert 999 not in idx.documents
    assert len(idx) == 1


def test_empty_document_tokenises_to_no_postings() -> None:
    idx = InvertedIndex()
    doc_id = idx.add_document("http://empty/", _html())
    assert doc_id == 0
    assert len(idx) == 1
    # Empty title + empty body => both extents collapse to (0, 0)
    assert idx.documents[0].fields == {"title": (0, 0), "body": (0, 0)}


def test_unicode_tokens_round_trip_through_the_index() -> None:
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="café naïve"))
    assert idx.lookup("café")[0].count == 1
    assert idx.lookup("naïve")[0].count == 1


def test_apostrophe_stripping_applied_during_indexing() -> None:
    # Query must use the post-tokenisation form (same rule search.py will
    # follow): "can't" tokenises to ["cant"], so that's what we look up.
    idx = InvertedIndex()
    idx.add_document("http://p/", _html(body="I can't see why"))
    assert idx.lookup("cant")[0].count == 1
    assert idx.lookup("can't") == []
