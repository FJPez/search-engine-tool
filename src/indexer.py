"""Inverted-index construction for the COMP3011 search-engine-tool.

This module consumes the crawler's ``(canonical_url, raw_html)`` stream and
produces an on-disk inverted index. Tokenisation, field extraction and the
index data structure all live here so that future changes to parsing never
require a re-crawl.

The first two increments expose :func:`tokenise` and :func:`extract_fields`;
the rest of the public surface (``Posting``, ``Document``, ``InvertedIndex``,
``build_from_crawler``) lands in subsequent commits.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# Tags whose text content is never meaningful for search. Removing them up
# front keeps the body stream clean of JS payloads, CSS rules and <noscript>
# fallback markup that would otherwise inflate positions and pollute tokens.
_NON_CONTENT_TAGS = ("script", "style", "noscript")

# Matches any run of non-word characters (``\W`` is ``[^A-Za-z0-9_]`` plus the
# unicode letter classes when ``re.UNICODE`` is active — which it is by
# default on ``str`` patterns in Python 3). Used as the token *separator*, so
# we split on punctuation, whitespace, newlines and stray symbols uniformly.
_TOKEN_SPLIT = re.compile(r"\W+", re.UNICODE)

# Apostrophe-like characters are stripped *before* the ``\W+`` split so that
# contractions collapse into a single token (``can't`` -> ``cant``) rather
# than splitting into ``can`` + ``t``. Both the ASCII apostrophe and the
# curly right-single-quotation-mark are included because real-world HTML
# uses either.
_APOSTROPHE_STRIP = str.maketrans("", "", "'\u2019")


def tokenise(text: str) -> list[str]:
    """Return the list of index tokens extracted from *text*.

    Lowercases, strips apostrophes so ``can't`` collapses to ``cant``, then
    splits on any run of non-alphanumeric characters. Empty strings produced
    by leading/trailing separators are filtered out.

    The same function is used for both indexing and query parsing, so query
    tokens and index tokens round-trip identically — a symmetry that any
    future search layer depends on.
    """
    if not text:
        return []
    normalised = text.lower().translate(_APOSTROPHE_STRIP)
    return [token for token in _TOKEN_SPLIT.split(normalised) if token]


def extract_fields(html: str) -> dict[str, str]:
    """Return ``{'title': ..., 'body': ...}`` extracted from *html*.

    This is the first pass of the two-pass tokenisation pipeline (NOTES.md
    L11 p.6): use BeautifulSoup + lxml to identify markup, then hand the
    resulting text to :func:`tokenise`. ``<script>``, ``<style>`` and
    ``<noscript>`` subtrees are discarded before extraction — their text
    never carries indexable content.

    The title is the ``<title>`` tag's string (empty string if absent or
    self-closing). The body is ``soup.get_text(separator=' ')`` so adjacent
    inline elements don't glue together (``<p>hello<b>world</b></p>`` comes
    back as ``"hello world"``, not ``"helloworld"``).
    """
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(_NON_CONTENT_TAGS):
        tag.decompose()

    title_tag = soup.title
    title = title_tag.get_text(strip=True) if title_tag is not None else ""

    # ``<title>`` is inside ``<head>`` and ``soup.get_text`` would otherwise
    # include it in the body — drop it so the two streams don't overlap.
    if title_tag is not None:
        title_tag.decompose()

    body = soup.get_text(separator=" ")
    return {"title": title, "body": body}
