"""Query layer over the inverted index.

The first increment exposes only :func:`_intersect_postings`, the
algorithmic core that the public ``find`` and ``print_entry`` functions
will sit on top of in subsequent commits.
"""

from __future__ import annotations

from src.indexer import Posting


def _intersect_postings(lists: list[list[Posting]]) -> list[int]:
    """Return the doc_ids present in every posting list.

    N-ary two-pointer intersection (NOTES.md L13 p.12). Each input list
    is required to be sorted by ``doc_id`` ascending — an
    :class:`InvertedIndex` invariant — so the algorithm can advance each
    pointer past doc_ids smaller than the current maximum and stop the
    moment any list is exhausted. Output is sorted ascending.

    Empty input or any empty list short-circuits to ``[]``: a query term
    with no postings means the conjunctive result is empty regardless of
    the other lists.

    Lists are walked in length-ascending order so the rarest term drives
    the advance — the rare+common term optimisation (NOTES.md L13 p.10).
    Sorting is stable, which doesn't matter for correctness here but
    keeps debugging output predictable.
    """
    if not lists or any(not lst for lst in lists):
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
