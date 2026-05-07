"""Unit tests for the pure URL helpers in :mod:`src.crawler`."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message

import pytest
import requests
from requests_mock import Mocker

from src.crawler import PoliteCrawler, RobotsPolicy, canonicalise, extract_links, in_scope

#: The seed host that tests pretend to crawl. Used wherever the exact URL
#: string is *context* (the crawler's target, a base for relative-link
#: resolution, ...) rather than the subject of the test. ``canonicalise`` and
#: ``in_scope`` tests deliberately use hardcoded literals because the URL
#: *is* the thing under test.
BASE_URL = "http://quotes.toscrape.com/"


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
        "http://",  # valid scheme but no host
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
    assert extract_links(html, BASE_URL) == [
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
    assert extract_links(html, BASE_URL) == [
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
    assert extract_links(html, BASE_URL) == [
        "http://quotes.toscrape.com/page/1",
    ]


def test_extract_links_ignores_non_anchor_elements() -> None:
    html = """
    <link rel="stylesheet" href="/style.css">
    <img src="/logo.png">
    <area href="/map">
    <a href="/real">Real link</a>
    """
    assert extract_links(html, BASE_URL) == [
        "http://quotes.toscrape.com/real",
    ]


def test_extract_links_ignores_anchors_without_href() -> None:
    html = """
    <a>No href</a>
    <a href="">Empty href</a>
    <a name="bookmark">Named anchor</a>
    <a href="/page/1">Real</a>
    """
    assert extract_links(html, BASE_URL) == [
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
    assert extract_links(html, BASE_URL) == [
        "http://quotes.toscrape.com/page/1",
    ]


def test_extract_links_tolerates_malformed_html() -> None:
    # Missing close tags — lxml should recover and still surface both links.
    html = '<a href="/a">A<a href="/b">B'
    links = extract_links(html, BASE_URL)
    assert "http://quotes.toscrape.com/a" in links
    assert "http://quotes.toscrape.com/b" in links


@pytest.mark.parametrize("html", ["", "<html></html>", "<html><body></body></html>"])
def test_extract_links_empty_returns_empty_list(html: str) -> None:
    assert extract_links(html, BASE_URL) == []


# --- PoliteCrawler.fetch -----------------------------------------------------


CrawlerFactory = Callable[..., PoliteCrawler]


class FakeRobots(RobotsPolicy):
    def __init__(
        self,
        *,
        allowed: bool | dict[str, bool] = True,
        crawl_delay: int | float | None = None,
    ) -> None:
        self._allowed = allowed
        self._crawl_delay = crawl_delay
        self.checked_urls: list[str] = []

    def can_fetch(self, useragent: str, url: str) -> bool:
        del useragent
        self.checked_urls.append(url)
        if isinstance(self._allowed, dict):
            return self._allowed.get(url, True)
        return self._allowed

    def crawl_delay(self, useragent: str) -> int | float | None:
        del useragent
        return self._crawl_delay


class BrokenCanFetchRobots(FakeRobots):
    def can_fetch(self, useragent: str, url: str) -> bool:
        del useragent, url
        raise ValueError("bad robots rule")


class BrokenCrawlDelayRobots(FakeRobots):
    def crawl_delay(self, useragent: str) -> int | float | None:
        del useragent
        raise TypeError("bad crawl delay")


@pytest.fixture
def crawler_factory() -> CrawlerFactory:
    """Build a :class:`PoliteCrawler` with injected fakes for sleep and clock.

    Defaults: ``sleep`` is a silent no-op so tests never actually block, and
    ``clock`` returns ``0.0`` for every call so no real time appears to pass.
    That combination means politeness-window tests can assert on the exact
    arguments passed to ``sleep`` by setting ``sleep_calls=[]``.
    """

    def _factory(
        *,
        delay: float = 6.0,
        sleep_calls: list[float] | None = None,
        clock: Callable[[], float] | None = None,
        max_pages: int | None = None,
        start_url: str = BASE_URL,
        respect_robots: bool = False,
        robot_parser: FakeRobots | None = None,
    ) -> PoliteCrawler:
        sleep: Callable[[float], None] = (
            sleep_calls.append if sleep_calls is not None else (lambda _s: None)
        )
        return PoliteCrawler(
            start_url,
            delay=delay,
            max_pages=max_pages,
            respect_robots=respect_robots,
            robot_parser=robot_parser,
            sleep=sleep,
            clock=clock if clock is not None else (lambda: 0.0),
        )

    return _factory


def test_fetch_success_returns_final_url_and_text(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(BASE_URL, text="<html>hello</html>", status_code=200)
    assert crawler_factory().fetch(BASE_URL) == (BASE_URL, "<html>hello</html>")


def test_fetch_sends_user_agent_header(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(BASE_URL, text="ok")
    crawler_factory().fetch(BASE_URL)
    assert requests_mock.last_request is not None
    assert "COMP3011-SearchEngineTool" in requests_mock.last_request.headers["User-Agent"]


def test_fetch_retries_500_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(
        BASE_URL,
        [
            {"status_code": 500},
            {"status_code": 200, "text": "recovered"},
        ],
    )
    assert crawler_factory().fetch(BASE_URL) == (BASE_URL, "recovered")


def test_fetch_gives_up_after_repeated_5xx(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(BASE_URL, [{"status_code": 500}, {"status_code": 500}])
    assert crawler_factory().fetch(BASE_URL) is None


@pytest.mark.parametrize("status", [401, 403, 404])
def test_fetch_skips_permanent_client_errors_without_retry(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, status: int
) -> None:
    requests_mock.get(BASE_URL, status_code=status)
    assert crawler_factory().fetch(BASE_URL) is None
    # Should not have retried — exactly one request.
    assert requests_mock.call_count == 1


def test_fetch_429_with_retry_after_backs_off_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    requests_mock.get(
        BASE_URL,
        [
            {"status_code": 429, "headers": {"Retry-After": "10"}},
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert crawler.fetch(BASE_URL) == (BASE_URL, "ok")
    # Retry-After (10) was larger than delay (6), so the bigger value wins.
    assert sleep_calls == [10.0]


def test_fetch_429_honours_delay_when_retry_after_is_smaller(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    requests_mock.get(
        BASE_URL,
        [
            {"status_code": 429, "headers": {"Retry-After": "1"}},
            {"status_code": 200, "text": "ok"},
        ],
    )
    crawler.fetch(BASE_URL)
    assert sleep_calls == [6.0]


def test_fetch_429_with_http_date_retry_after_falls_back_to_delay(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # Retry-After can legally be an HTTP-date; we only parse delta-seconds,
    # so the unparseable value falls back to 0 and the delay wins via max().
    sleep_calls: list[float] = []
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    requests_mock.get(
        BASE_URL,
        [
            {
                "status_code": 429,
                "headers": {"Retry-After": "Thu, 01 Dec 2025 16:00:00 GMT"},
            },
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert crawler.fetch(BASE_URL) == (BASE_URL, "ok")
    assert sleep_calls == [6.0]  # delay wins because parsed Retry-After → 0


def test_fetch_gives_up_after_repeated_429(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(BASE_URL, [{"status_code": 429}, {"status_code": 429}])
    assert crawler_factory().fetch(BASE_URL) is None


def test_fetch_retries_connection_error_then_succeeds(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(
        BASE_URL,
        [
            {"exc": requests.ConnectionError("network down")},
            {"status_code": 200, "text": "ok"},
        ],
    )
    assert crawler_factory().fetch(BASE_URL) == (BASE_URL, "ok")


def test_fetch_gives_up_after_repeated_timeout(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    requests_mock.get(BASE_URL, exc=requests.Timeout("slow"))
    assert crawler_factory().fetch(BASE_URL) is None


def test_fetch_follows_redirect_and_returns_final_url(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    target = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, status_code=301, headers={"Location": target})
    requests_mock.get(target, text="final page", status_code=200)
    result = crawler_factory().fetch(BASE_URL)
    assert result is not None
    final_url, html = result
    assert final_url == target
    assert html == "final page"


def test_fetch_skips_unexpected_status_code(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # 418 is not permanent-skip, not retryable, not 429 — we log and skip.
    requests_mock.get(BASE_URL, status_code=418)
    assert crawler_factory().fetch(BASE_URL) is None
    assert requests_mock.call_count == 1


# --- PoliteCrawler.crawl -----------------------------------------------------


def _html_with_links(*hrefs: str) -> str:
    """Build a tiny HTML document whose body contains one ``<a>`` per href."""
    anchors = "\n".join(f'<a href="{h}">{h}</a>' for h in hrefs)
    return f"<html><body>{anchors}</body></html>"


def test_crawl_yields_single_page_and_does_not_sleep_before_first_fetch(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    requests_mock.get(BASE_URL, text=_html_with_links(), status_code=200)
    crawler = crawler_factory(sleep_calls=sleep_calls)
    results = list(crawler.crawl())
    assert results == [(BASE_URL, _html_with_links())]
    # The politeness window is a *gap between* fetches — never before the first.
    assert sleep_calls == []


def test_crawl_enforces_politeness_window_between_fetches(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    # Clock stuck at 0 ⇒ elapsed is always 0 ⇒ sleep gets the full delay each time.
    crawler = crawler_factory(delay=6.0, sleep_calls=sleep_calls)
    page2 = f"{BASE_URL}page/2"
    page3 = f"{BASE_URL}page/3"
    requests_mock.get(BASE_URL, text=_html_with_links(page2))
    requests_mock.get(page2, text=_html_with_links(page3))
    requests_mock.get(page3, text=_html_with_links())
    list(crawler.crawl())
    # 3 fetches ⇒ exactly 2 politeness waits (before fetches 2 and 3).
    assert sleep_calls == [6.0, 6.0]


def test_crawl_skips_sleep_when_enough_real_time_has_elapsed(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    # Each call to clock() advances by 10 s — more than the 6 s politeness
    # window, so the crawler should never need to sleep.
    ticks = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    crawler = crawler_factory(
        delay=6.0,
        sleep_calls=sleep_calls,
        clock=lambda: next(ticks),
    )
    page2 = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, text=_html_with_links(page2))
    requests_mock.get(page2, text=_html_with_links())
    list(crawler.crawl())
    assert sleep_calls == []


def test_crawl_traverses_breadth_first(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # Shape: seed → /a, /b ; /a → /c ; /b, /c are leaves.
    # BFS order: seed, /a, /b, /c.
    a = f"{BASE_URL}a"
    b = f"{BASE_URL}b"
    c = f"{BASE_URL}c"
    requests_mock.get(BASE_URL, text=_html_with_links(a, b))
    requests_mock.get(a, text=_html_with_links(c))
    requests_mock.get(b, text=_html_with_links())
    requests_mock.get(c, text=_html_with_links())
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [BASE_URL, a, b, c]


def test_crawl_filters_out_of_scope_links(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    internal = f"{BASE_URL}internal"
    external = "http://other.example.com/external"
    requests_mock.get(BASE_URL, text=_html_with_links(internal, external))
    requests_mock.get(internal, text=_html_with_links())
    # No mock is registered for *external* — if the crawler tried to fetch
    # it, requests_mock would raise NoMockAddress and this test would fail.
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [BASE_URL, internal]


def test_crawl_dedupes_visited_urls(requests_mock: Mocker, crawler_factory: CrawlerFactory) -> None:
    a = f"{BASE_URL}a"
    # /a links back to the seed *and* to itself; neither should be re-fetched.
    requests_mock.get(BASE_URL, text=_html_with_links(a))
    requests_mock.get(a, text=_html_with_links(BASE_URL, a))
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [BASE_URL, a]
    assert requests_mock.call_count == 2


def test_crawl_dedupes_url_enqueued_from_two_pages(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # Both /a and /b link to /c. /c enters the frontier twice; the second
    # pop must hit the `if url in visited: continue` guard, not re-fetch.
    a = f"{BASE_URL}a"
    b = f"{BASE_URL}b"
    c = f"{BASE_URL}c"
    requests_mock.get(BASE_URL, text=_html_with_links(a, b))
    requests_mock.get(a, text=_html_with_links(c))
    requests_mock.get(b, text=_html_with_links(c))
    requests_mock.get(c, text=_html_with_links())
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [BASE_URL, a, b, c]
    # /c fetched exactly once, not twice.
    assert requests_mock.call_count == 4


def test_crawl_respects_max_pages_cap(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    a = f"{BASE_URL}a"
    b = f"{BASE_URL}b"
    c = f"{BASE_URL}c"
    requests_mock.get(BASE_URL, text=_html_with_links(a, b, c))
    requests_mock.get(a, text=_html_with_links())
    requests_mock.get(b, text=_html_with_links())
    requests_mock.get(c, text=_html_with_links())
    crawler = crawler_factory(max_pages=2)
    results = [url for url, _ in crawler.crawl()]
    assert results == [BASE_URL, a]  # seed + one linked page, then stop


def test_crawl_skips_seed_disallowed_by_robots(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    robots = FakeRobots(allowed=False)
    requests_mock.get(BASE_URL, text=_html_with_links())
    crawler = crawler_factory(respect_robots=True, robot_parser=robots)

    with caplog.at_level(logging.DEBUG, logger="src.crawler"):
        assert list(crawler.crawl()) == []
    assert requests_mock.call_count == 0
    assert robots.checked_urls == [BASE_URL]
    assert "disallowed by robots.txt" in caplog.text


def test_crawl_skips_child_link_disallowed_by_robots(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    allowed = f"{BASE_URL}allowed"
    blocked = f"{BASE_URL}blocked"
    robots = FakeRobots(allowed={BASE_URL: True, allowed: True, blocked: False})
    requests_mock.get(BASE_URL, text=_html_with_links(allowed, blocked))
    requests_mock.get(allowed, text=_html_with_links())
    requests_mock.get(blocked, text=_html_with_links())
    crawler = crawler_factory(respect_robots=True, robot_parser=robots)

    results = [url for url, _ in crawler.crawl()]

    assert results == [BASE_URL, allowed]
    assert [request.url for request in requests_mock.request_history] == [BASE_URL, allowed]
    assert blocked in robots.checked_urls


def test_crawl_bypasses_robots_when_disabled(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    robots = FakeRobots(allowed=False)
    requests_mock.get(BASE_URL, text=_html_with_links())
    crawler = crawler_factory(respect_robots=False, robot_parser=robots)

    results = [url for url, _ in crawler.crawl()]

    assert results == [BASE_URL]
    assert robots.checked_urls == []


def test_crawl_continues_when_robots_can_fetch_errors(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    robots = BrokenCanFetchRobots()
    requests_mock.get(BASE_URL, text=_html_with_links())
    crawler = crawler_factory(respect_robots=True, robot_parser=robots)

    with caplog.at_level(logging.WARNING, logger="src.crawler"):
        results = [url for url, _ in crawler.crawl()]

    assert results == [BASE_URL]
    assert "Could not evaluate robots.txt rule" in caplog.text


def test_crawl_uses_robots_crawl_delay_when_it_is_larger_than_default(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    robots = FakeRobots(crawl_delay=10)
    page2 = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, text=_html_with_links(page2))
    requests_mock.get(page2, text=_html_with_links())
    crawler = crawler_factory(
        delay=6.0,
        sleep_calls=sleep_calls,
        respect_robots=True,
        robot_parser=robots,
    )

    list(crawler.crawl())

    assert sleep_calls == [10.0]


def test_crawl_ignores_robots_crawl_delay_below_coursework_minimum(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    sleep_calls: list[float] = []
    robots = FakeRobots(crawl_delay=1)
    page2 = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, text=_html_with_links(page2))
    requests_mock.get(page2, text=_html_with_links())
    crawler = crawler_factory(
        delay=6.0,
        sleep_calls=sleep_calls,
        respect_robots=True,
        robot_parser=robots,
    )

    list(crawler.crawl())

    assert sleep_calls == [6.0]


def test_crawl_uses_default_delay_when_robots_crawl_delay_errors(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    sleep_calls: list[float] = []
    robots = BrokenCrawlDelayRobots()
    page2 = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, text=_html_with_links(page2))
    requests_mock.get(page2, text=_html_with_links())
    crawler = crawler_factory(
        delay=6.0,
        sleep_calls=sleep_calls,
        respect_robots=True,
        robot_parser=robots,
    )

    with caplog.at_level(logging.WARNING, logger="src.crawler"):
        list(crawler.crawl())

    assert sleep_calls == [6.0]
    assert "Could not read robots.txt crawl delay" in caplog.text


def test_crawl_continues_when_robots_txt_is_missing(
    requests_mock: Mocker, crawler_factory: CrawlerFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_404(_url: str) -> None:
        raise urllib.error.HTTPError(
            f"{BASE_URL}robots.txt",
            404,
            "Not Found",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    requests_mock.get(BASE_URL, text=_html_with_links())
    crawler = crawler_factory(respect_robots=True)

    results = [url for url, _ in crawler.crawl()]

    assert results == [BASE_URL]


def test_crawl_continues_when_robots_txt_cannot_be_read(
    requests_mock: Mocker,
    crawler_factory: CrawlerFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def raise_os_error(_url: str) -> None:
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", raise_os_error)
    requests_mock.get(BASE_URL, text=_html_with_links())
    crawler = crawler_factory(respect_robots=True)

    with caplog.at_level(logging.WARNING, logger="src.crawler"):
        results = [url for url, _ in crawler.crawl()]

    assert results == [BASE_URL]
    assert "Could not read robots.txt" in caplog.text


def test_crawl_yields_nothing_when_start_url_is_uncrawlable(
    crawler_factory: CrawlerFactory,
) -> None:
    crawler = crawler_factory(start_url="mailto:foo@example.com")
    assert list(crawler.crawl()) == []


def test_crawl_yields_nothing_when_start_url_fails(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # Seed returns 500 on both attempts → fetch gives up → crawl yields nothing.
    requests_mock.get(BASE_URL, status_code=500)
    assert list(crawler_factory().crawl()) == []


def test_crawl_yields_final_url_after_redirect(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    target = f"{BASE_URL}page/2"
    requests_mock.get(BASE_URL, status_code=301, headers={"Location": target})
    requests_mock.get(target, text=_html_with_links(), status_code=200)
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [target]


def test_crawl_skips_page_when_redirect_leaves_scope(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # A link is in scope, but the server redirects it to an off-domain host.
    # The crawler should silently skip it and continue with other links.
    a = f"{BASE_URL}a"
    b = f"{BASE_URL}b"
    off_domain = "http://other.example.com/landing"
    requests_mock.get(BASE_URL, text=_html_with_links(a, b))
    requests_mock.get(a, status_code=301, headers={"Location": off_domain})
    requests_mock.get(off_domain, text="off domain", status_code=200)
    requests_mock.get(b, text=_html_with_links(), status_code=200)
    results = [url for url, _ in crawler_factory().crawl()]
    assert results == [BASE_URL, b]  # /a skipped, /b still yielded


def test_crawl_locks_redirect_to_seed_scheme(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # Regression test: quotes.toscrape.com sometimes redirects https:// to
    # http:// for author pages. A scheme downgrade must NOT leak into the
    # yielded URLs — every canonical URL stays on the seed's scheme so the
    # visited set and the indexer see one form per logical page.
    https_seed = "https://quotes.toscrape.com/"
    https_author = "https://quotes.toscrape.com/author/einstein"
    http_author = "http://quotes.toscrape.com/author/einstein"
    requests_mock.get(https_seed, text=_html_with_links("/author/einstein"))
    requests_mock.get(
        https_author,
        status_code=301,
        headers={"Location": http_author},
    )
    requests_mock.get(http_author, text=_html_with_links(), status_code=200)

    crawler = crawler_factory(start_url=https_seed)
    results = [url for url, _ in crawler.crawl()]
    assert results == [https_seed, https_author]


def test_crawl_rewrites_absolute_http_link_to_seed_https(
    requests_mock: Mocker, crawler_factory: CrawlerFactory
) -> None:
    # An HTML page can contain an absolute http:// link even when the seed
    # is https://. The crawler must rewrite the scheme before fetching so
    # the target is requested consistently with the rest of the crawl.
    https_seed = "https://quotes.toscrape.com/"
    https_about = "https://quotes.toscrape.com/about"
    requests_mock.get(
        https_seed,
        text=_html_with_links("http://quotes.toscrape.com/about"),
    )
    requests_mock.get(https_about, text=_html_with_links(), status_code=200)
    # If the crawler instead tried http://quotes.toscrape.com/about,
    # requests_mock would raise NoMockAddress and this test would fail.

    crawler = crawler_factory(start_url=https_seed)
    results = [url for url, _ in crawler.crawl()]
    assert results == [https_seed, https_about]
