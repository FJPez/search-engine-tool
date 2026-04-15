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

from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

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
