"""Unit tests for the pure URL helpers in :mod:`src.crawler`."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import requests
from requests_mock import Mocker

from src.crawler import PoliteCrawler, canonicalise, extract_links, in_scope


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


def test_extract_links_resolves_relative_urls() -> None:
    html = """
    <html><body>
        <a href="http://quotes.toscrape.com/page/1/">Absolute</a>
        <a href="/page/2">Root-relative</a>
        <a href="author/einstein">Doc-relative</a>
    </body></html>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/page/1",
        "http://quotes.toscrape.com/page/2",
        "http://quotes.toscrape.com/author/einstein",
    ]


def test_extract_links_preserves_first_appearance_order() -> None:
    # Links are interleaved and one is duplicated. The returned order must
    # match the order each canonical URL *first* appears in the document so
    # that BFS traversal is deterministic.
    html = """
    <a href="/zulu">Z</a>
    <a href="/alpha">A</a>
    <a href="/mike">M</a>
    <a href="/alpha">A again</a>
    <a href="/bravo">B</a>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/zulu",
        "http://quotes.toscrape.com/alpha",
        "http://quotes.toscrape.com/mike",
        "http://quotes.toscrape.com/bravo",
    ]


def test_extract_links_deduplicates_canonical_duplicates() -> None:
    # Same target page spelled three different ways should collapse to one.
    html = """
    <a href="/page/1">First</a>
    <a href="/page/1/">Trailing slash</a>
    <a href="http://quotes.toscrape.com/page/1">Absolute form</a>
    <a href="/page/1#section">With fragment</a>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/page/1",
    ]


def test_extract_links_ignores_non_anchor_elements() -> None:
    html = """
    <link rel="stylesheet" href="/style.css">
    <img src="/logo.png">
    <area href="/map">
    <a href="/real">Real link</a>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/real",
    ]


def test_extract_links_ignores_anchors_without_href() -> None:
    html = """
    <a>No href</a>
    <a href="">Empty href</a>
    <a name="bookmark">Named anchor</a>
    <a href="/page/1">Real</a>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/page/1",
    ]


def test_extract_links_filters_non_http_schemes() -> None:
    html = """
    <a href="mailto:foo@example.com">Email</a>
    <a href="javascript:void(0)">JS</a>
    <a href="tel:+441234567890">Phone</a>
    <a href="ftp://quotes.toscrape.com/file">FTP</a>
    <a href="/page/1">Real</a>
    """
    assert extract_links(html, "http://quotes.toscrape.com/") == [
        "http://quotes.toscrape.com/page/1",
    ]


def test_extract_links_tolerates_malformed_html() -> None:
    # Missing close tags — lxml should recover and still surface both links.
    html = '<a href="/a">A<a href="/b">B'
    links = extract_links(html, "http://quotes.toscrape.com/")
    assert "http://quotes.toscrape.com/a" in links
    assert "http://quotes.toscrape.com/b" in links


@pytest.mark.parametrize("html", ["", "<html></html>", "<html><body></body></html>"])
def test_extract_links_empty_returns_empty_list(html: str) -> None:
    assert extract_links(html, "http://quotes.toscrape.com/") == []


# --- PoliteCrawler.fetch -----------------------------------------------------

URL = "http://quotes.toscrape.com/"


CrawlerFactory = Callable[..., PoliteCrawler]


@pytest.fixture
def crawler_factory() -> CrawlerFactory:
    """Build a :class:`PoliteCrawler` with an injected no-op or spy sleep.

    Pass ``sleep_calls=[]`` to capture back-off timings, or nothing for a
    default crawler whose sleep is a silent no-op so tests never actually
    block.
    """

    def _factory(
        *,
        delay: float = 6.0,
        sleep_calls: list[float] | None = None,
    ) -> PoliteCrawler:
        sleep: Callable[[float], None] = (
            sleep_calls.append if sleep_calls is not None else (lambda _s: None)
        )
        return PoliteCrawler(URL, delay=delay, sleep=sleep)

    return _factory


def test_fetch_success_returns_final_url_and_text(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(URL, text="<html>hello</html>", status_code=200)
    assert crawler_factory().fetch(URL) == (URL, "<html>hello</html>")


def test_fetch_sends_user_agent_header(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(URL, text="ok")
    crawler_factory().fetch(URL)
    assert requests_mock.last_request is not None
    assert "COMP3011-SearchEngineTool" in requests_mock.last_request.headers["User-Agent"]


def test_fetch_retries_500_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(
        URL,
        [
            {"status_code": 500},
            {"status_code": 200, "text": "recovered"},
        ],
    )
    assert crawler_factory().fetch(URL) == (URL, "recovered")


def test_fetch_gives_up_after_repeated_5xx(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(URL, [{"status_code": 500}, {"status_code": 500}])
    assert crawler_factory().fetch(URL) is None


@pytest.mark.parametrize("status", [401, 403, 404])
def test_fetch_skips_permanent_client_errors_without_retry(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, status: int
) -> None:
    requests_mock.get(URL, status_code=status)
    assert crawler_factory().fetch(URL) is None
    # Should not have retried — exactly one request.
    assert requests_mock.call_count == 1


def test_fetch_429_with_retry_after_backs_off_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    requests_mock.get(
        URL,
        [
            {"status_code": 429, "headers": {"Retry-After": "10"}},
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert crawler.fetch(URL) == (URL, "ok")
    # Retry-After (10) was larger than delay (6), so the bigger value wins.
    assert sleep_calls == [10.0]


def test_fetch_429_honours_delay_when_retry_after_is_smaller(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    requests_mock.get(
        URL,
        [
            {"status_code": 429, "headers": {"Retry-After": "1"}},
            {"status_code": 200, "text": "ok"},
        ],
    )
    crawler.fetch(URL)
    assert sleep_calls == [6.0]


def test_fetch_gives_up_after_repeated_429(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(URL, [{"status_code": 429}, {"status_code": 429}])
    assert crawler_factory().fetch(URL) is None


def test_fetch_retries_connection_error_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(
        URL,
        [
            {"exc": requests.ConnectionError("network down")},
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert crawler_factory().fetch(URL) == (URL, "ok")


def test_fetch_gives_up_after_repeated_timeout(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(URL, exc=requests.Timeout("slow"))
    assert crawler_factory().fetch(URL) is None


def test_fetch_follows_redirect_and_returns_final_url(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    target = "http://quotes.toscrape.com/page/2"
    requests_mock.get(URL, status_code=301, headers={"Location": target})
    requests_mock.get(target, text="final page", status_code=200)
    result = crawler_factory().fetch(URL)
    assert result is not None
    final_url, html = result
    assert final_url == target
    assert html == "final page"


def test_fetch_skips_unexpected_status_code(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # 418 is not permanent-skip, not retryable, not 429 — we log and skip.
    requests_mock.get(URL, status_code=418)
    assert crawler_factory().fetch(URL) is None
    assert requests_mock.call_count == 1
