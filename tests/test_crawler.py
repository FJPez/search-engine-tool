"""Unit tests for the pure URL helpers in :mod:`src.crawler`."""

from __future__ import annotations

import pytest

from src.crawler import canonicalise, in_scope


@pytest.mark.parametrize(
    ("url", "base", "expected"),
    [
        # Trailing slash collapsed on non-root paths
        (
            "http://quotes.toscrape.com/page/1/",
            None,
            "http://quotes.toscrape.com/page/1",
        ),
        (
            "http://quotes.toscrape.com/page/1",
            None,
            "http://quotes.toscrape.com/page/1",
        ),
        # Root path always keeps exactly one slash
        ("http://quotes.toscrape.com/", None, "http://quotes.toscrape.com/"),
        ("http://quotes.toscrape.com", None, "http://quotes.toscrape.com/"),
        # Fragments stripped
        ("http://quotes.toscrape.com/#section", None, "http://quotes.toscrape.com/"),
        (
            "http://quotes.toscrape.com/page/1#top",
            None,
            "http://quotes.toscrape.com/page/1",
        ),
        # Scheme and host lowercased
        (
            "HTTP://Quotes.ToScrape.Com/",
            None,
            "http://quotes.toscrape.com/",
        ),
        # Default ports dropped
        ("http://quotes.toscrape.com:80/", None, "http://quotes.toscrape.com/"),
        ("https://quotes.toscrape.com:443/", None, "https://quotes.toscrape.com/"),
        # Non-default port preserved
        (
            "http://quotes.toscrape.com:8080/",
            None,
            "http://quotes.toscrape.com:8080/",
        ),
        # Relative resolution against a base
        (
            "/page/2",
            "http://quotes.toscrape.com/page/1",
            "http://quotes.toscrape.com/page/2",
        ),
        (
            "author/foo/",
            "http://quotes.toscrape.com/page/1",
            "http://quotes.toscrape.com/page/author/foo",
        ),
        # Fragment-only link against a base resolves to the base page
        (
            "#section",
            "http://quotes.toscrape.com/page/1",
            "http://quotes.toscrape.com/page/1",
        ),
        # Query strings preserved
        (
            "http://quotes.toscrape.com/search?q=fish",
            None,
            "http://quotes.toscrape.com/search?q=fish",
        ),
    ],
)
def test_canonicalise_returns_canonical_form(url: str, base: str | None, expected: str) -> None:
    assert canonicalise(url, base) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "mailto:author@example.com",
        "javascript:void(0)",
        "tel:+441234567890",
        "ftp://quotes.toscrape.com/file",
    ],
)
def test_canonicalise_rejects_uncrawlable_schemes(url: str) -> None:
    assert canonicalise(url) is None


def test_canonicalise_rejects_relative_url_without_base() -> None:
    # Without a base, a path-only string has no scheme and cannot be crawled.
    assert canonicalise("/page/2") is None


@pytest.mark.parametrize(
    ("url", "allowed", "expected"),
    [
        ("http://quotes.toscrape.com/", "quotes.toscrape.com", True),
        ("https://quotes.toscrape.com/page/1", "quotes.toscrape.com", True),
        # Case-insensitive on both sides
        ("http://Quotes.Toscrape.Com/", "quotes.toscrape.com", True),
        ("http://quotes.toscrape.com/", "QUOTES.TOSCRAPE.COM", True),
        # Subdomain walking is rejected — the brief says stay on-domain.
        ("http://www.quotes.toscrape.com/", "quotes.toscrape.com", False),
        ("http://example.com/", "quotes.toscrape.com", False),
        # Malformed input is out of scope, not an exception.
        ("not a url", "quotes.toscrape.com", False),
    ],
)
def test_in_scope(url: str, allowed: str, expected: bool) -> None:
    assert in_scope(url, allowed) is expected
