"""Unit tests for :mod:`src.search`."""

from __future__ import annotations

import pytest

from src.indexer import Posting
from src.search import _intersect_postings


def _postings(*doc_ids: int) -> list[Posting]:
    """Build a posting list from doc_ids; count and positions are placeholders.

    The intersection algorithm only inspects ``Posting.doc_id``, so the
    other fields can be anything sensible. Building lists this way keeps
    each test's intent (which doc_ids appear in which list) at the top of
    the call.
    """
    return [Posting(doc_id=d, count=1, positions=(0,)) for d in doc_ids]


# --------------------------------------------------------------------------
# _intersect_postings
# --------------------------------------------------------------------------


def test_intersect_empty_input_returns_empty() -> None:
    assert _intersect_postings([]) == []


def test_intersect_any_empty_list_short_circuits() -> None:
    # A single empty list means the conjunctive result is empty regardless
    # of the other lists' contents — there are no docs in *every* list.
    assert _intersect_postings([_postings(1, 2, 3), []]) == []
    assert _intersect_postings([[], _postings(1, 2, 3)]) == []
    assert _intersect_postings([[], []]) == []


def test_intersect_single_list_passthrough() -> None:
    # With one input list, every doc_id in it is "in every list".
    assert _intersect_postings([_postings(1, 3, 5)]) == [1, 3, 5]


def test_intersect_two_lists_classic_case() -> None:
    # Lecture's textbook example (NOTES.md L13 p.12): walk both lists,
    # emit doc_ids present in both.
    a = _postings(1, 2, 4, 7, 9)
    b = _postings(2, 4, 5, 9, 10)
    assert _intersect_postings([a, b]) == [2, 4, 9]


def test_intersect_three_lists() -> None:
    # N-ary case — generalises the two-pointer algorithm.
    a = _postings(1, 2, 3, 4, 5)
    b = _postings(2, 4, 5, 6)
    c = _postings(2, 5, 7)
    assert _intersect_postings([a, b, c]) == [2, 5]


def test_intersect_disjoint_lists() -> None:
    a = _postings(1, 2, 3)
    b = _postings(4, 5, 6)
    assert _intersect_postings([a, b]) == []


def test_intersect_identical_lists() -> None:
    a = _postings(1, 2, 3)
    b = _postings(1, 2, 3)
    assert _intersect_postings([a, b]) == [1, 2, 3]


def test_intersect_one_subset_of_other() -> None:
    a = _postings(2, 4)
    b = _postings(1, 2, 3, 4, 5)
    assert _intersect_postings([a, b]) == [2, 4]


def test_intersect_output_sorted_ascending() -> None:
    # The sort-by-length reorder happens internally, but the OUTPUT must
    # always be sorted ascending — search.py callers depend on that for
    # deterministic find() result ordering.
    a = _postings(5, 2, 9)  # not actually sorted — but real InvertedIndex
    b = _postings(2, 5, 7, 9)  # always feeds sorted lists; this just guards
    # against accidentally sorting the output by something else
    result = _intersect_postings([sorted(a, key=lambda p: p.doc_id), b])
    assert result == sorted(result)


@pytest.mark.parametrize("rare_first", [True, False])
def test_intersect_invariant_to_input_order(rare_first: bool) -> None:
    # Whether the rare term is passed first or last, the algorithm
    # produces the same intersection — the sort-by-length reorder
    # (NOTES.md L13 p.10) is a perf hint, not a correctness gate.
    rare = _postings(7, 42)
    common = _postings(*range(50))
    lists = [rare, common] if rare_first else [common, rare]
    assert _intersect_postings(lists) == [7, 42]


def test_intersect_handles_ten_term_query() -> None:
    # Stress: ten lists, intersection is one doc_id.
    lists = [_postings(*range(i, 100)) for i in range(10)]
    # Doc ids 9..99 are in every list (each list starts higher than the last)
    assert _intersect_postings(lists) == list(range(9, 100))


def test_intersect_does_not_mutate_inputs() -> None:
    # The internal sort-by-length must not reorder the caller's lists in
    # place — InvertedIndex hands out references to its real posting
    # lists, and reordering them would break lookup() guarantees.
    a = _postings(5, 10)
    b = _postings(1, 2, 3, 5, 10)
    a_before = list(a)
    b_before = list(b)
    _intersect_postings([b, a])  # bigger first, so sort would reorder
    assert a == a_before
    assert b == b_before
