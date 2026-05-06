"""Polite, breadth-first web crawler for the COMP3011 search-engine-tool.

The module owns every piece of network I/O in the project. Everything
downstream — tokenisation, indexing, search — operates on the
``(canonical_url, raw_html)`` pairs this crawler produces.

Public API
----------

* :class:`PoliteCrawler` — the main entry point. Seeded with a start URL,
  it performs a breadth-first traversal over a single host and yields
  ``(canonical_url, html)`` for every successful in-scope fetch. The brief's
  ≥6 s politeness window is enforced inside the crawl loop using an injected
  ``sleep`` / ``clock`` pair, so unit tests can assert on timing without
  waiting on wall-clock. See the class docstring for the full retry and
  scope policy.

* :func:`canonicalise` — normalise a URL into a single canonical form
  (lowercased scheme and host, default ports dropped, fragments stripped,
  non-root trailing slashes collapsed). Returns ``None`` for anything that
  can't be crawled (``mailto:``, ``javascript:``, empty inputs, ...).

* :func:`in_scope` — exact-host scope check; no subdomain walking.

* :func:`extract_links` — parse HTML via BeautifulSoup + ``lxml`` and return
  the deduplicated, order-preserving list of canonical crawlable URLs.

Design constraints
------------------

* **HTML contract.** The crawler yields *raw* HTML strings. The two-pass
  tokenisation pipeline (identify markup, then extract words from useful
  regions) lives entirely in :mod:`src.indexer`. Keeping parsing decisions
  out of the crawler means indexer changes never force a re-crawl.

* **Politeness by default.** The default ``delay`` of 6 s matches the
  brief's hard requirement. The politeness window is measured from the end
  of one fetch to the start of the next, so the target server is guaranteed
  a full quiet interval between responses.

* **Testability via dependency injection.** ``sleep``, ``clock`` and
  ``session`` are all injectable constructor parameters. Production callers
  never pass them; tests always do.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator
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


def _force_scheme(url: str, scheme: str) -> str:
    """Return *url* with its scheme replaced by *scheme*.

    Used by :meth:`PoliteCrawler.crawl` to lock every canonical URL onto the
    seed's scheme. Sites such as ``quotes.toscrape.com`` sometimes redirect
    ``https://`` requests to ``http://`` (or vice versa), and absolute links
    inside HTML can disagree with the seed's scheme. Without this
    normalisation the visited set would see two entries for the same logical
    page and the yielded stream would contain mixed schemes.
    """
    parsed = urlparse(url)
    # Fragment is zeroed defensively — callers are expected to pass URLs that
    # have already been through canonicalise (which strips fragments), but
    # this avoids a subtle bug if that precondition is ever violated.
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


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

    The BFS ``crawl`` loop calls :meth:`fetch` for each URL it takes from the
    frontier, yields successful pages, and enqueues any in-scope links it
    discovers. The politeness window is enforced between fetches using the
    injected ``clock`` and ``sleep`` callables so tests can verify timing
    without wall-clock waits.
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
                logger.debug("Fetching %s (attempt %d)", url, attempt)
                response = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                logger.warning("Network error fetching %s (attempt %d): %s", url, attempt, exc)
                if attempt == 2:
                    return None
                continue

            status = response.status_code
            if status == 200:
                logger.debug("Fetched %s: HTTP 200", response.url)
                return response.url, response.text

            if status in _PERMANENT_SKIP_STATUS:
                logger.debug("Skipping %s: HTTP %d", url, status)
                return None

            if status == 429:
                if attempt == 2:
                    logger.warning("Giving up on %s after repeated HTTP 429", url)
                    return None
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                back_off = max(self._delay, retry_after)
                logger.debug("HTTP 429 on %s; backing off %.1fs", url, back_off)
                self._sleep(back_off)
                continue

            if status in _RETRYABLE_STATUS:
                if attempt == 2:
                    logger.warning("Giving up on %s after HTTP %d on retry", url, status)
                    return None
                logger.debug("HTTP %d on %s; retrying", status, url)
                continue

            logger.warning("Unexpected HTTP %d on %s; skipping", status, url)
            return None

        return None  # unreachable; keeps static checkers happy

    def crawl(self) -> Iterator[tuple[str, str]]:
        """Yield ``(canonical_url, html)`` for every in-scope reachable page.

        Performs a breadth-first traversal from the seed URL, enforcing a
        minimum ``delay`` seconds between successive fetches and skipping any
        URL whose host differs from the seed's host. Stops after
        ``max_pages`` successful yields if that limit was set.

        The politeness window is enforced from the *end* of one fetch to the
        *start* of the next — so 6 s of real quiet between the server sending
        its previous response and us sending the next request. This is the
        stricter of the two common readings of "6 s between requests" and
        keeps us well clear of the brief's threshold.
        """
        start_canonical = canonicalise(self._start_url)
        if start_canonical is None:
            logger.warning("Start URL %r is not crawlable; nothing to yield", self._start_url)
            return

        start_parsed = urlparse(start_canonical)
        allowed_host = start_parsed.hostname or ""
        start_scheme = start_parsed.scheme  # locked for the lifetime of this crawl
        visited: set[str] = set()
        frontier: deque[str] = deque([start_canonical])
        last_fetch_end: float | None = None
        yielded = 0

        while frontier:
            if self._max_pages is not None and yielded >= self._max_pages:
                return

            url = frontier.popleft()
            if url in visited:
                continue
            visited.add(url)

            if last_fetch_end is not None:
                elapsed = self._clock() - last_fetch_end
                if elapsed < self._delay:
                    wait = self._delay - elapsed
                    logger.debug("Waiting %.1fs before fetching %s", wait, url)
                    self._sleep(wait)

            result = self.fetch(url)
            last_fetch_end = self._clock()
            if result is None:
                continue

            final_url, html = result
            canonical_final = canonicalise(final_url)
            if canonical_final is None or not in_scope(canonical_final, allowed_host):
                continue
            # Lock onto the seed's scheme so a redirect that downgrades
            # https→http (or upgrades http→https) doesn't leak a second
            # canonical form for the same logical page.
            canonical_final = _force_scheme(canonical_final, start_scheme)
            visited.add(canonical_final)

            yield canonical_final, html
            yielded += 1

            links = extract_links(html, canonical_final)
            logger.debug("Discovered %d crawlable links on %s", len(links), canonical_final)
            for raw_link in links:
                if not in_scope(raw_link, allowed_host):
                    continue
                link = _force_scheme(raw_link, start_scheme)
                if link in visited:
                    continue
                frontier.append(link)
