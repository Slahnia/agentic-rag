"""Web search source backed by DuckDuckGo (no API key required)."""

import logging

from ddgs import DDGS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 4) -> list[Document]:
    """Search the web and return the snippets as Documents.

    Returns an empty list on network errors so the graph can still
    produce an honest "I don't know" answer instead of crashing.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception:
        logger.warning("Web search failed for query: %s", query, exc_info=True)
        return []
    return [
        Document(
            page_content=item.get("body", ""),
            metadata={"source": item.get("href", ""), "title": item.get("title", "")},
        )
        for item in results
        if item.get("body")
    ]
