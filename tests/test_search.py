"""Unit tests for :mod:`src.search`."""

from __future__ import annotations

import pytest

from src.indexer import InvertedIndex, Posting
from src.search import _intersect_postings, find
from tests.conftest import HtmlFactory


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


@pytest.mark.parametrize("rare_first", [True, False])
def test_intersect_invariant_to_input_order(rare_first: bool) -> None:
    # Whether the rare term is passed first or last, the algorithm
    # produces the same intersection — the sort-by-length reorder
    # (NOTES.md L13 p.10) is a perf hint, not a correctness gate.
    # Awkward to assert through find() since callers don't normally
    # control list order, hence a direct test.
    rare = _postings(7, 42)
    common = _postings(*range(50))
    lists = [rare, common] if rare_first else [common, rare]
    assert _intersect_postings(lists) == [7, 42]


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


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


@pytest.fixture
def three_doc_index(html: HtmlFactory) -> InvertedIndex:
    """Three documents used by most find() tests.

    doc 0: title "One" + body "good good friends"
    doc 1: title "Two" + body "good things"
    doc 2: title "Three" + body "friends only"

    Token frequencies that matter:
      "good"    in {0, 1}
      "friends" in {0, 2}
      "things"  in {1}
    """
    return InvertedIndex.from_pages(
        [
            ("http://q/1", html(title="One", body="good good friends")),
            ("http://q/2", html(title="Two", body="good things")),
            ("http://q/3", html(title="Three", body="friends only")),
        ]
    )


@pytest.mark.parametrize("query", ["", "   ", "\t\n  "])
def test_find_empty_or_whitespace_query_returns_empty(
    three_doc_index: InvertedIndex, query: str
) -> None:
    # Brief lists empty queries as an edge case; we treat them as "no docs
    # match" rather than raising so the CLI doesn't need a special path.
    assert find(three_doc_index, query) == []


def test_find_query_of_only_punctuation_returns_empty(three_doc_index: InvertedIndex) -> None:
    # "..." tokenises to [] — same shape as empty query, same result.
    assert find(three_doc_index, "...???!!!") == []


def test_find_single_word_hit_returns_all_matching_urls(three_doc_index: InvertedIndex) -> None:
    assert find(three_doc_index, "good") == ["http://q/1", "http://q/2"]


def test_find_single_word_miss_returns_empty(three_doc_index: InvertedIndex) -> None:
    assert find(three_doc_index, "missing") == []


def test_find_two_word_intersection(three_doc_index: InvertedIndex) -> None:
    # "good" ∩ "friends" = {0} — only doc 0 contains both.
    assert find(three_doc_index, "good friends") == ["http://q/1"]


def test_find_one_term_absent_short_circuits(three_doc_index: InvertedIndex) -> None:
    # If any term has no postings, the whole result is empty regardless
    # of how common the others are.
    assert find(three_doc_index, "good missing") == []


def test_find_two_terms_with_no_co_occurrence_returns_empty(html: HtmlFactory) -> None:
    # Both terms exist on the site, but no single document contains
    # both. Different code path from the absent-term short-circuit
    # above (where one term has zero postings) — this exercises the
    # full intersection loop reaching the end of one list.
    idx = InvertedIndex.from_pages(
        [
            ("http://q/1", html(body="good morning")),
            ("http://q/2", html(body="bad evening")),
        ]
    )
    assert find(idx, "good evening") == []


def test_find_is_case_insensitive(three_doc_index: InvertedIndex) -> None:
    # "Good Friends" must hit the same docs as "good friends" because
    # tokenise() lowercases at both index and query time.
    assert find(three_doc_index, "Good Friends") == find(three_doc_index, "good friends")
    assert find(three_doc_index, "GOOD") == find(three_doc_index, "good")


def test_find_apostrophe_round_trip(html: HtmlFactory) -> None:
    # "can't" tokenises to ["cant"] at index time AND at query time, so
    # the contraction in the query matches the stripped form indexed in
    # the body. This is the symmetry property tokenise() advertises.
    idx = InvertedIndex.from_pages(
        [
            ("http://q/1", html(body="I can't see why")),
            ("http://q/2", html(body="just text")),
        ]
    )
    assert find(idx, "can't") == ["http://q/1"]
    assert find(idx, "cant") == ["http://q/1"]


def test_find_results_ordered_by_doc_id_ascending(html: HtmlFactory) -> None:
    # Crawler emits pages in BFS order, indexer assigns doc_ids in that
    # same order, so URL ordering = crawl ordering. Worth pinning.
    idx = InvertedIndex.from_pages(
        [
            ("http://q/a", html(body="hit")),
            ("http://q/b", html(body="miss")),
            ("http://q/c", html(body="hit")),
            ("http://q/d", html(body="hit")),
        ]
    )
    assert find(idx, "hit") == ["http://q/a", "http://q/c", "http://q/d"]


def test_find_three_term_query(html: HtmlFactory) -> None:
    # Three terms all in doc 0, none of which are in any single other doc
    # together — exercises the N-ary path through find().
    idx = InvertedIndex.from_pages(
        [
            ("http://q/1", html(body="alpha beta gamma delta")),
            ("http://q/2", html(body="alpha beta")),
            ("http://q/3", html(body="gamma delta")),
        ]
    )
    assert find(idx, "alpha beta gamma") == ["http://q/1"]


def test_find_rare_common_term_pair(html: HtmlFactory) -> None:
    # NOTES.md L13 p.10 calls out the rare+common case as the demo win
    # for conjunctive processing — the rare term keeps the result tight.
    idx = InvertedIndex.from_pages(
        [
            ("http://q/1", html(body="the cat sat")),
            ("http://q/2", html(body="the dog ran")),
            ("http://q/3", html(body="the cat ran")),
            ("http://q/4", html(body="the bird flew")),
        ]
    )
    # "the" is in every doc; "cat" is rare. Result: docs containing both.
    assert find(idx, "the cat") == ["http://q/1", "http://q/3"]


def test_find_on_empty_index_returns_empty() -> None:
    assert find(InvertedIndex(), "anything") == []


def test_find_repeated_term_in_query_is_idempotent(three_doc_index: InvertedIndex) -> None:
    # "good good" tokenises to ["good", "good"]; intersecting the same
    # posting list with itself gives the same list back. Behaviourally
    # equivalent to the single-term query.
    assert find(three_doc_index, "good good") == find(three_doc_index, "good")
