"""Query layer over the inverted index.

The module powers two of the brief's four CLI commands by reading from a
built :class:`src.indexer.InvertedIndex` — it never mutates the index and
never touches the network. Both public functions are pure: ``find`` for
multi-word conjunctive lookup and ``print_entry`` for single-word
formatted output.

Public API
----------

* :func:`find` — the brief's ``find <w1> [w2 ...]`` command. Tokenises
  the query through :func:`src.indexer.tokenise` so the same case-fold
  and apostrophe-stripping happen at query time as at indexing time
  (so ``can't`` matches indexed ``cant`` with no caller normalisation).
  Returns the URLs of documents containing **all** query terms, sorted
  by ``doc_id`` ascending. Empty / whitespace / punctuation-only
  queries return ``[]``.

* :func:`print_entry` — the brief's ``print <word>`` command. Returns a
  formatted multi-line string ready for the CLI to print. The return
  shape is always a string (never ``None``), so the CLI can pipe both
  hits and misses through ``print()`` without a special path.

* :func:`_intersect_postings` — module-private helper that powers
  :func:`find`. N-ary two-pointer intersection over posting lists,
  walked smallest-first.

Algorithm choices
-----------------

* **Conjunctive (AND) semantics for `find`** match the brief's
  requirement and are the lecture-taught default for web search
  (NOTES.md L13 p.10). A returned URL must contain *every* query term.

* **N-ary two-pointer intersection** (NOTES.md L13 p.12). Each posting
  list is sorted by ``doc_id`` ascending — an :class:`InvertedIndex`
  invariant — so we keep one pointer per list, find the maximum
  current ``doc_id``, advance every pointer below it, emit when all
  pointers agree, and stop the moment any list is exhausted.
  Complexity is O(sum of list lengths).

* **Document-at-a-time evaluation** (NOTES.md L13 pp.4-9). Walks all
  posting lists in parallel with constant memory rather than
  accumulating partial scores per term. Cleaner to explain in the
  video walkthrough and the right pick at the quotes.toscrape.com
  scale where a priority queue isn't needed.

* **Sort-by-length rare+common optimisation** (NOTES.md L13 p.10).
  Posting lists are walked smallest-first so a query like
  ``"fish locomotion"`` skips most of the long list's postings.
  Externally observable behaviour is the same regardless of list
  order; the sort only changes the wall-clock cost.

* **Tokenisation symmetry**. Both ``find`` and ``print_entry`` route
  the user's input through :func:`src.indexer.tokenise` — the same
  function the indexer uses. Skipping that step is the easiest way
  to introduce query/index drift; pinning it explicitly makes the
  symmetry impossible to break by accident.

Out of scope
------------

* **Ranking** — TF-IDF, PageRank, anchor-text scoring. The brief asks
  for AND semantics; ranking is the 80-100 stretch (NOTES.md L10).
  Positions and counts are stored in the index so a future ranking
  layer can be built on top without rebuilding.

* **Skip pointers / list skipping** (NOTES.md L13 pp.13-17).
  Irrelevant at our corpus scale. Worth mentioning in the video as a
  "what I'd do next" talking point.

* **Term-at-a-time evaluation**. The alternative to document-at-a-time
  (NOTES.md L13 pp.4-9). Reads each posting list once but needs an
  accumulator hash table; the trade-off doesn't pay off at this scale.

* **Phrase queries**. Positions are recorded by the indexer but search
  treats queries as bag-of-words AND. Phrase support is a v2 feature
  the position data already enables.

* **Disjunctive (OR) queries**. Not in the brief.

* **Index mutation**. Search reads, never writes.
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
