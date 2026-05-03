"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.indexer import InvertedIndex

SingleDocIndexFactory = Callable[..., InvertedIndex]


@pytest.fixture
def single_doc_index() -> SingleDocIndexFactory:
    """Factory: build a one-doc index whose URL is a placeholder.

    Most indexer tests don't care which URL a document has — only what the
    title and body tokenise to. This fixture absorbs the boilerplate so
    those tests show their inputs at a glance:

        idx = single_doc_index(body="Hello World")
        idx = single_doc_index(title="T", body="good good friends")

    Multi-doc setups stay inline; the URLs are usually meaningful there.
    """

    def _build(*, title: str = "", body: str = "") -> InvertedIndex:
        title_tag = f"<title>{title}</title>" if title else ""
        html = f"<!doctype html><html><head>{title_tag}</head><body>{body}</body></html>"
        idx = InvertedIndex()
        idx.add_document("http://p/", html)
        return idx

    return _build
