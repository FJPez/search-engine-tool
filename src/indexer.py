"""Inverted-index construction for the COMP3011 search-engine-tool.

This module consumes the crawler's ``(canonical_url, raw_html)`` stream and
produces an on-disk inverted index. Tokenisation, field extraction and the
index data structure all live here so that future changes to parsing never
require a re-crawl.

Public API so far: :func:`tokenise`, :func:`extract_fields`, :class:`Posting`,
:class:`Document`, :class:`InvertedIndex` (with ``save`` / ``load``). The
:func:`build_from_crawler` convenience lands in a follow-up commit.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

# Bumped whenever the on-disk JSON shape changes in a way that older loaders
# can't handle. Loading a file with a different version is a hard error: it
# is safer to refuse than to silently misinterpret stored fields.
_INDEX_VERSION = 1

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


@dataclass(frozen=True, slots=True)
class Posting:
    """One inverted-index entry: a term's occurrences inside a single doc.

    ``positions`` are token indices into that document's concatenated
    ``title + body`` stream, 0-indexed. ``count`` is redundant with
    ``len(positions)`` but cached so query code can read it without a
    ``len`` call and so a position-less variant could be swapped in later
    without breaking the field shape.
    """

    doc_id: int
    count: int
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Document:
    """An entry in the document catalogue (NOTES.md L12 p.9).

    ``fields`` maps a field name (``"title"``, ``"body"``) to its half-open
    extent ``(start, end)`` inside the document's token stream. Extents are
    recorded once per document rather than per posting so positions in the
    posting list stay a single flat list.
    """

    doc_id: int
    url: str
    fields: dict[str, tuple[int, int]] = field(default_factory=dict)


class InvertedIndex:
    """In-memory inverted index with a document catalogue.

    Exposes the minimum surface the search layer needs: :meth:`add_document`
    to ingest a crawled page, :meth:`lookup` to fetch a term's posting list
    (sorted by ``doc_id``, so the later two-pointer intersection in
    ``search.py`` just works — NOTES.md L13 p.12), :meth:`field_of` to
    recover the field containing a given token position, and the
    :attr:`documents` catalogue for turning ``doc_id`` back into a URL.
    """

    def __init__(self) -> None:
        self._documents: dict[int, Document] = {}
        self._url_to_id: dict[str, int] = {}
        self._postings: dict[str, list[Posting]] = {}
        self._next_id: int = 0

    def add_document(self, url: str, html: str) -> int:
        """Tokenise *html*, assign a new ``doc_id``, update postings.

        Idempotent on URL: re-adding the same URL returns the existing
        ``doc_id`` without re-indexing. That guards against the crawler
        yielding the same canonical URL twice (redirect edge cases).
        """
        if url in self._url_to_id:
            return self._url_to_id[url]

        fields = extract_fields(html)
        title_tokens = tokenise(fields["title"])
        body_tokens = tokenise(fields["body"])

        title_end = len(title_tokens)
        body_end = title_end + len(body_tokens)
        extents: dict[str, tuple[int, int]] = {
            "title": (0, title_end),
            "body": (title_end, body_end),
        }

        doc_id = self._next_id
        self._next_id += 1

        self._documents[doc_id] = Document(doc_id=doc_id, url=url, fields=extents)
        self._url_to_id[url] = doc_id

        # Build per-term position lists in one pass over the combined stream,
        # then materialise the Posting. ``defaultdict(list)`` keeps the inner
        # append loop tight and avoids a membership check per token.
        positions_by_term: dict[str, list[int]] = defaultdict(list)
        for position, token in enumerate(title_tokens):
            positions_by_term[token].append(position)
        for offset, token in enumerate(body_tokens):
            positions_by_term[token].append(title_end + offset)

        for term, positions in positions_by_term.items():
            posting = Posting(doc_id=doc_id, count=len(positions), positions=tuple(positions))
            # Documents are added in strictly increasing ``doc_id`` order, so
            # appending preserves the by-``doc_id`` sort invariant the search
            # layer relies on without an explicit resort.
            self._postings.setdefault(term, []).append(posting)

        return doc_id

    def lookup(self, word: str) -> list[Posting]:
        """Return the posting list for *word*, or ``[]`` if absent.

        The query is lowercased to keep lookups case-insensitive (brief rule)
        without forcing callers to normalise. Apostrophes aren't stripped
        here because the indexed tokens already have them removed — a query
        like ``"can't"`` should go through :func:`tokenise` first, which
        yields ``["cant"]``; the raw string passed in here is treated as a
        single already-tokenised term.
        """
        return self._postings.get(word.lower(), [])

    def field_of(self, doc_id: int, position: int) -> str | None:
        """Return the field name containing *position* in *doc_id*.

        ``None`` if *doc_id* is unknown or *position* lies outside every
        field extent. Used by future ranking to weight title hits.
        """
        document = self._documents.get(doc_id)
        if document is None:
            return None
        for name, (start, end) in document.fields.items():
            if start <= position < end:
                return name
        return None

    @property
    def documents(self) -> dict[int, Document]:
        """Read-only view of the ``doc_id → Document`` catalogue."""
        # Copy shields the internal dict from mutation by callers. Cheap at
        # the scale this coursework operates on (tens of pages).
        return dict(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return word.lower() in self._postings

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the index to *path* as JSON.

        JSON is the deliberate choice over pickle: the compiled index file
        is a graded submission artefact, JSON is human-inspectable
        (``less data/index.json``) and safe to load. Parent directories are
        created on demand so callers can ``idx.save("data/index.json")``
        without first ensuring ``data/`` exists.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "documents": {
                # JSON object keys are always strings; we cast doc_ids back
                # to int on load. ``fields`` extents go to JSON as 2-element
                # lists and rehydrate to tuples in ``load``.
                str(doc.doc_id): {
                    "url": doc.url,
                    "fields": {name: list(extent) for name, extent in doc.fields.items()},
                }
                for doc in self._documents.values()
            },
            "vocabulary": {
                term: [
                    {
                        "doc_id": posting.doc_id,
                        "count": posting.count,
                        "positions": list(posting.positions),
                    }
                    for posting in postings
                ]
                for term, postings in self._postings.items()
            },
        }
        # ``ensure_ascii=False`` keeps the file readable for unicode tokens
        # like ``café``; the file is written UTF-8 explicitly to match.
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> InvertedIndex:
        """Deserialise an index previously written by :meth:`save`.

        Raises :class:`FileNotFoundError` if *path* doesn't exist and
        :class:`ValueError` for any structural problem (malformed JSON,
        missing keys, unsupported version). Refusing to limp along on a
        corrupt index is intentional — silently producing wrong query
        results would be worse than a clear failure at load time.
        """
        source = Path(path)
        try:
            with source.open(encoding="utf-8") as handle:
                payload: Any = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Index file {source} is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Index file {source} must contain a JSON object")

        version = payload.get("version")
        if version != _INDEX_VERSION:
            raise ValueError(
                f"Index file {source} has version {version!r}; expected {_INDEX_VERSION}"
            )

        documents_raw = payload.get("documents")
        vocabulary_raw = payload.get("vocabulary")
        if not isinstance(documents_raw, dict) or not isinstance(vocabulary_raw, dict):
            raise ValueError(f"Index file {source} is missing 'documents' or 'vocabulary'")

        index = cls()
        # Documents first so ``_next_id`` and ``_url_to_id`` are seeded
        # before postings reference them. Any missing key on a document or
        # posting record is treated as structural corruption: silently
        # defaulting (e.g. ``fields`` -> ``{}``) would let ``field_of``
        # return ``None`` for every hit in that document, masking the
        # corruption rather than surfacing it.
        try:
            for raw_id, raw_doc in documents_raw.items():
                doc_id = int(raw_id)
                if not isinstance(raw_doc, dict):
                    raise ValueError(f"document {raw_id} is not an object")
                url = raw_doc["url"]
                fields_obj = raw_doc["fields"]
                if not isinstance(fields_obj, dict):
                    raise ValueError(f"document {raw_id} 'fields' must be an object")
                extents: dict[str, tuple[int, int]] = {}
                for name, extent in fields_obj.items():
                    # Reject anything that isn't a 2-element ``[start, end]``
                    # list up front so a one-item extent raises a documented
                    # ValueError rather than IndexError.
                    if not isinstance(extent, list) or len(extent) != 2:
                        raise ValueError(
                            f"document {raw_id} field {name!r} extent must be a 2-element list"
                        )
                    extents[name] = (int(extent[0]), int(extent[1]))
                # Every v1 document has both extents (see ``add_document``).
                # A subset is structural corruption — the same kind that
                # leads to ``field_of`` silently returning ``None``.
                missing = {"title", "body"} - extents.keys()
                if missing:
                    raise ValueError(
                        f"document {raw_id} is missing field extents: {sorted(missing)}"
                    )
                index._documents[doc_id] = Document(doc_id=doc_id, url=url, fields=extents)
                index._url_to_id[url] = doc_id

            if index._documents:
                index._next_id = max(index._documents) + 1

            for term, postings in vocabulary_raw.items():
                parsed: list[Posting] = []
                for p in postings:
                    posting_doc_id = int(p["doc_id"])
                    # A posting that references a doc we never loaded is
                    # corrupt: ``lookup`` would return hits with no URL in
                    # the catalogue, so the search layer can't display them.
                    if posting_doc_id not in index._documents:
                        raise ValueError(
                            f"vocabulary entry {term!r} references unknown doc_id {posting_doc_id}"
                        )
                    positions = tuple(int(pos) for pos in p["positions"])
                    count = int(p["count"])
                    # ``count`` is meant to be cached ``len(positions)``;
                    # disagreement is corruption that would later skew any
                    # term-frequency computation built on top of postings.
                    if count != len(positions):
                        raise ValueError(
                            f"vocabulary entry {term!r} for doc_id "
                            f"{posting_doc_id} has count {count} but "
                            f"{len(positions)} positions"
                        )
                    parsed.append(Posting(doc_id=posting_doc_id, count=count, positions=positions))
                index._postings[term] = parsed
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError(f"Index file {source} is structurally invalid: {exc}") from exc

        return index
