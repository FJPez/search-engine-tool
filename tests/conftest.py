"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.indexer import InvertedIndex

HtmlFactory = Callable[..., str]
SingleDocIndexFactory = Callable[..., InvertedIndex]


@pytest.fixture
def html() -> HtmlFactory:
    """Factory: build a minimal HTML document with optional title and body.

    Used wherever a test wants to feed real HTML to ``extract_fields`` /
    ``add_document`` / ``InvertedIndex.from_pages`` without spelling out
    the boilerplate ``<!doctype html><html>...`` wrapper:

        html(title="Quotes", body="<p>Hello world</p>")
        html(body="good good friends")

    Both kwargs default to ``""`` so missing pieces are omitted from the
    output rather than rendered as empty tags.
    """

    def _build(*, title: str = "", body: str = "") -> str:
        title_tag = f"<title>{title}</title>" if title else ""
        return f"<!doctype html><html><head>{title_tag}</head><body>{body}</body></html>"

    return _build


@pytest.fixture
def single_doc_index(html: HtmlFactory) -> SingleDocIndexFactory:
    """Factory: build a one-doc index whose URL is a placeholder.

    Most indexer tests don't care which URL a document has — only what the
    title and body tokenise to. This fixture absorbs the boilerplate so
    those tests show their inputs at a glance:

        idx = single_doc_index(body="Hello World")
        idx = single_doc_index(title="T", body="good good friends")

    Multi-doc setups stay inline; the URLs are usually meaningful there.
    """

    def _build(*, title: str = "", body: str = "") -> InvertedIndex:
        idx = InvertedIndex()
        idx.add_document("http://p/", html(title=title, body=body))
        return idx

    return _build
