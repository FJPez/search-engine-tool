"""Polite, depth-limited crawler for the COMP3011 search-engine-tool.

This module will eventually expose a ``PoliteCrawler`` class that yields
``(canonical_url, raw_html)`` pairs for every in-scope page reachable from a
seed URL, enforcing the brief's ≥6 s politeness window between requests.

The two-pass tokenization pipeline (identify markup, then extract words) lives
entirely in :mod:`src.indexer`; this module owns all network I/O and never
strips or interprets HTML beyond link extraction.

For now, the module exports only the pure URL helpers used by the crawler:
:func:`canonicalise` and :func:`in_scope`. The crawler class will follow in
later commits.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_DELAY = 6.0
DEFAULT_TIMEOUT = 10.0
DEFAULT_USER_AGENT = "COMP3011-SearchEngineTool/0.1 (educational)"

_PERMANENT_SKIP_STATUS = frozenset({401, 403, 404})
_RETRYABLE_STATUS = frozenset({408, 500, 501, 502, 503, 504})

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalise(url: str, base: str | None = None) -> str | None:
    """Return a canonical form of *url*, or ``None`` if it can't be crawled.

    Resolves *url* against *base* if given, lowercases the scheme and host,
    drops default ports and fragments, and strips trailing slashes from
    non-root paths so that ``/page/1/`` and ``/page/1`` canonicalise to the
    same URL. This prevents the crawler from fetching the same page twice
    under slightly different spellings.

    Returns ``None`` for empty or whitespace-only inputs and for non-HTTP(S)
    schemes such as ``mailto:``, ``javascript:``, ``tel:`` and ``ftp:``.
    """
    if url is None or not url.strip():
        return None

    stripped = url.strip()
    resolved = urljoin(base, stripped) if base is not None else stripped

    parsed = urlparse(resolved)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return None

    host = parsed.hostname  # urlparse lowercases the hostname for us
    if not host:
        return None

    netloc = host
    if parsed.port is not None and parsed.port != _DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def in_scope(url: str, allowed_netloc: str) -> bool:
    """Return ``True`` if *url*'s host equals *allowed_netloc* exactly.

    The comparison is case-insensitive but does **not** walk subdomains: a
    URL on ``www.quotes.toscrape.com`` is out of scope when *allowed_netloc*
    is ``quotes.toscrape.com``. The brief requires staying on the target
    site, and subdomain walking is an easy way to accidentally crawl the
    wrong host.
    """
    host = urlparse(url).hostname
    if host is None:
        return False
    return host == allowed_netloc.lower()


def extract_links(html: str, page_url: str) -> list[str]:
    """Return the deduplicated list of canonical crawlable URLs in *html*.

    Parses *html* with BeautifulSoup using the ``lxml`` backend, collects every
    ``<a>`` element that has an ``href`` attribute, resolves relative URLs
    against *page_url*, and drops anything :func:`canonicalise` rejects
    (``mailto:``, ``javascript:``, empty hrefs, ...). Order is preserved to
    match the order of first appearance in the document, which gives the
    breadth-first crawl a predictable traversal order.
    """
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    result: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href")
        if not href:
            continue
        canonical = canonicalise(str(href), base=page_url)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result


def _parse_retry_after(value: str | None) -> float:
    """Parse an HTTP ``Retry-After`` header as seconds. Return ``0`` on failure.

    Only the "delta-seconds" form is supported (``Retry-After: 30``); HTTP-date
    values are treated as unparseable and fall back to the crawler's own
    politeness delay. That's sufficient for the coursework scope.
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


class PoliteCrawler:
    """Breadth-first polite web crawler.

    Enforces a minimum delay between requests to a single host and yields
    ``(canonical_url, html)`` pairs for every in-scope page reachable from a
    seed URL. For testability, the class accepts injected ``sleep`` and
    ``clock`` callables so unit tests can assert on politeness timing without
    actually waiting.

    Only the network-I/O half (``_fetch``) is wired up in this commit; the
    BFS ``crawl`` loop lands in the next commit.
    """

    def __init__(
        self,
        start_url: str,
        *,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        max_pages: int | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._start_url = start_url
        self._delay = delay
        self._timeout = timeout
        self._max_pages = max_pages
        self._sleep = sleep
        self._clock = clock
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = user_agent

    def fetch(self, url: str) -> tuple[str, str] | None:
        """Fetch *url* and return ``(final_url, html)``, or ``None`` to skip.

        This is a low-level primitive. Most callers want :meth:`crawl`, which
        wraps ``fetch`` with the frontier, visited-set tracking and the
        politeness window; ``fetch`` is exposed for single-page use and for
        testing the retry policy in isolation.

        *final_url* is the URL after any redirects and may differ from *url*;
        callers should canonicalise it before checking their visited set.

        Returns ``None`` on any condition that means "skip this URL and move
        on": permanent client errors (401 / 403 / 404), retryable errors
        whose single retry also failed (408 / 429 / 5xx), network failures,
        and unexpected status codes.
        """
        for attempt in (1, 2):
            try:
                response = self._session.get(url, timeout=self._timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.warning("Network error fetching %s (attempt %d): %s", url, attempt, exc)
                if attempt == 2:
                    return None
                continue

            status = response.status_code
            if status == 200:
                return response.url, response.text

            if status in _PERMANENT_SKIP_STATUS:
                logger.info("Skipping %s: HTTP %d", url, status)
                return None

            if status == 429:
                if attempt == 2:
                    logger.warning("Giving up on %s after repeated HTTP 429", url)
                    return None
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                back_off = max(self._delay, retry_after)
                logger.info("HTTP 429 on %s; backing off %.1fs", url, back_off)
                self._sleep(back_off)
                continue

            if status in _RETRYABLE_STATUS:
                if attempt == 2:
                    logger.warning("Giving up on %s after HTTP %d on retry", url, status)
                    return None
                logger.info("HTTP %d on %s; retrying", status, url)
                continue

            logger.warning("Unexpected HTTP %d on %s; skipping", status, url)
            return None

        return None  # unreachable; keeps static checkers happy
