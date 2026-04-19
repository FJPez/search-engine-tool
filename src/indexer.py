"""Inverted-index construction for the COMP3011 search-engine-tool.

This module consumes the crawler's ``(canonical_url, raw_html)`` stream and
produces an on-disk inverted index. Tokenisation, field extraction and the
index data structure all live here so that future changes to parsing never
require a re-crawl.

The first increment exposes only :func:`tokenise`; the rest of the public
surface (``extract_fields``, ``Posting``, ``Document``, ``InvertedIndex``,
``build_from_crawler``) lands in subsequent commits.
"""

from __future__ import annotations

import re

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
