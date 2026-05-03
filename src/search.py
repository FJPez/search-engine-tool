"""Query layer over the inverted index.

Public so far: :func:`find` (multi-word conjunctive lookup, the brief's
``find`` command) and :func:`print_entry` (single-word inverted-index
entry formatting, the brief's ``print`` command).
"""

from __future__ import annotations

from src.indexer import InvertedIndex, Posting, tokenise


def _intersect_postings(lists: list[list[Posting]]) -> list[int]:
    """Return the doc_ids present in every posting list.

    N-ary two-pointer intersection (NOTES.md L13 p.12). Each input list
    is required to be sorted by ``doc_id`` ascending — an
    :class:`InvertedIndex` invariant — so the algorithm can advance each
    pointer past doc_ids smaller than the current maximum and stop the
    moment any list is exhausted. Output is sorted ascending.

    A list with no postings short-circuits to ``[]``: a query term with
    no postings means the conjunctive result is empty regardless of the
    other lists. Callers must pass at least one list; ``find()`` already
    handles the empty-query case before this helper is called.

    Lists are walked in length-ascending order so the rarest term drives
    the advance — the rare+common term optimisation (NOTES.md L13 p.10).
    Sorting is stable, which doesn't matter for correctness here but
    keeps debugging output predictable.
    """
    if any(not lst for lst in lists):
        return []

    # Copy + in-place sort: ``sorted(..., key=len)`` widens to ``list[Sized]``
    # under the type checker, while ``list.sort`` preserves ``list[Posting]``.
    sorted_lists = list(lists)
    sorted_lists.sort(key=len)
    pointers = [0] * len(sorted_lists)
    result: list[int] = []

    while True:
        current = [sorted_lists[i][pointers[i]].doc_id for i in range(len(sorted_lists))]
        max_id = max(current)

        if min(current) == max_id:
            # All pointers reference the same doc_id — emit and advance all.
            result.append(max_id)
            for i in range(len(pointers)):
                pointers[i] += 1
                if pointers[i] >= len(sorted_lists[i]):
                    return result
            continue

        # Otherwise advance every pointer whose current doc_id is below
        # the max so the next iteration tests a larger candidate.
        for i in range(len(pointers)):
            while (
                pointers[i] < len(sorted_lists[i]) and sorted_lists[i][pointers[i]].doc_id < max_id
            ):
                pointers[i] += 1
            if pointers[i] >= len(sorted_lists[i]):
                return result


def find(index: InvertedIndex, query: str) -> list[str]:
    """Return URLs of documents containing every word in *query*.

    *query* is a raw string from the CLI ("good friends", "Indifference").
    It is tokenised through :func:`src.indexer.tokenise` so the query
    splits and normalises identically to the way the index was built —
    same lowercasing, same apostrophe stripping. That symmetry is what
    makes ``"can't"`` find documents that contain the indexed token
    ``cant`` without the caller doing anything special.

    The conjunctive (AND) semantics match the brief's ``find`` command
    (NOTES.md L13 p.10): a returned URL must contain *all* terms. Empty
    queries, whitespace-only queries, and queries that tokenise to
    nothing (only punctuation) return ``[]``. Any term whose posting
    list is empty short-circuits the whole result to ``[]`` because the
    intersection cannot be non-empty.

    Results are ordered by ``doc_id`` ascending — the same order the
    crawler emitted the pages, which is the most predictable thing to
    hand to the CLI's ``print``.
    """
    tokens = tokenise(query)
    if not tokens:
        return []

    posting_lists = [index.lookup(token) for token in tokens]
    matched_doc_ids = _intersect_postings(posting_lists)

    documents = index.documents
    return [documents[doc_id].url for doc_id in matched_doc_ids]


def print_entry(index: InvertedIndex, word: str) -> str:
    """Return the formatted inverted-index entry for *word*.

    The brief's ``print`` command prints "the inverted index for a
    particular word". The CLI hands this function the user's argument
    and pipes the return value through ``print()``.

    *word* is run through :func:`src.indexer.tokenise` so the case-fold
    and apostrophe-stripping happen identically to indexing time —
    typing ``Good`` looks up ``good``; typing ``can't`` looks up
    ``cant``. The header line shows the canonical (tokenised) form so
    the user can see what was actually matched.

    The return shape is always a string, even for absent words — the
    CLI can wrap the call in ``print(...)`` for both hits and misses
    without a None-check. Format:

        good (3 documents)
          0 http://quotes.toscrape.com/        count=2  positions=[1, 7]
          2 http://quotes.toscrape.com/page/2  count=1  positions=[3]

    A word that tokenises to nothing (punctuation only, empty string)
    falls back to ``"<word> (0 documents)"`` using the original input
    so the user still gets a recognisable echo of what they typed.
    """
    tokens = tokenise(word)
    if not tokens:
        return f"{word} (0 documents)"

    # ``print`` is documented as "for a particular word" (singular). If a
    # caller hands us a multi-word string we look up the first token and
    # ignore the rest — the CLI dispatcher should be using the right
    # argument shape, but this is the defensive read.
    term = tokens[0]
    postings = index.lookup(term)

    header = f"{term} ({len(postings)} documents)"
    if not postings:
        return header

    documents = index.documents
    lines = [header]
    for posting in postings:
        url = documents[posting.doc_id].url
        lines.append(
            f"  {posting.doc_id} {url}  count={posting.count}  positions={list(posting.positions)}"
        )
    return "\n".join(lines)
