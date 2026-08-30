"""Semantic Scholar integration for Scholar.

Uses the free, keyless Semantic Scholar Academic Graph API to find
papers related to the currently loaded document. No API key required
— unauthenticated requests are rate-limited to roughly 1 request per
second, which is more than sufficient for this feature's usage pattern
(one lookup per "Find Related Papers" click).

API docs: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_FIELDS = "title,abstract,year,authors,url,externalIds,citationCount"

_REQUEST_TIMEOUT_SECONDS = 15

# Keyless requests are rate-limited by Semantic Scholar to ~1/sec.
# A tiny client-side delay avoids tripping that limit on rapid re-clicks.
_MIN_SECONDS_BETWEEN_REQUESTS = 1.1
_last_request_time = 0.0


class SemanticScholarError(Exception):
    """Raised when the Semantic Scholar API request fails."""


@dataclass
class RelatedPaper:
    """A single related-paper search result."""

    title: str
    abstract: str
    year: int | None
    authors: list[str]
    url: str
    citation_count: int | None
    doi: str | None


def _respect_rate_limit() -> None:
    """Sleep briefly if the last request was too recent."""
    global _last_request_time

    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

    _last_request_time = time.monotonic()


def _build_query_from_paper(title: str, abstract: str) -> str:
    """Build a search query from the loaded paper's title/abstract.

    Semantic Scholar's relevance search works best with a focused
    query rather than a full abstract dump, so this uses the title
    plus the first sentence or so of the abstract as a topic hint.
    """
    query = title.strip()

    if abstract:
        first_sentence = abstract.strip().split(". ")[0]
        if first_sentence and first_sentence.lower() not in query.lower():
            query = f"{query} {first_sentence}"

    # Semantic Scholar's search endpoint works better with shorter,
    # keyword-dense queries than with a full sentence-length string.
    return query[:300]


def find_related_papers(
    title: str,
    abstract: str = "",
    limit: int = 6,
    exclude_title: str | None = None,
) -> list[RelatedPaper]:
    """Search Semantic Scholar for papers related to the loaded document.

    Args:
        title: Title of the currently loaded paper (from its metadata
            or the first heading found during extraction).
        abstract: Abstract text, if available, used to sharpen the
            search query.
        limit: Max number of related papers to return.
        exclude_title: If provided, results with a near-identical
            title are filtered out (avoids "related papers" just
            returning the paper you already uploaded).

    Returns:
        A list of RelatedPaper results, best-effort ordered by
        Semantic Scholar's relevance ranking.

    Raises:
        SemanticScholarError: If the request fails, times out, or the
            API returns an error status.
    """
    if not title or not title.strip():
        raise SemanticScholarError(
            "No paper title available to search from."
        )

    query = _build_query_from_paper(title, abstract)

    _respect_rate_limit()

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": query,
                "limit": min(limit + 3, 20),  # pad in case of self-match filtering
                "fields": _FIELDS,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise SemanticScholarError(
            "Semantic Scholar took too long to respond. Please try again."
        ) from None
    except requests.exceptions.RequestException as exc:
        raise SemanticScholarError(
            "Could not reach Semantic Scholar. Check your connection."
        ) from exc

    if response.status_code == 429:
        raise SemanticScholarError(
            "Semantic Scholar's rate limit was hit. Please wait a moment "
            "and try again."
        )

    if not response.ok:
        raise SemanticScholarError(
            f"Semantic Scholar returned an error (status {response.status_code})."
        )

    try:
        payload = response.json()
    except ValueError:
        raise SemanticScholarError(
            "Semantic Scholar returned an unreadable response."
        ) from None

    raw_papers = payload.get("data", []) or []

    exclude_normalized = (
        exclude_title.strip().lower() if exclude_title else None
    )

    results: list[RelatedPaper] = []

    for raw in raw_papers:
        paper_title = (raw.get("title") or "").strip()

        if not paper_title:
            continue

        if exclude_normalized and paper_title.strip().lower() == exclude_normalized:
            continue  # skip the paper the user already uploaded

        authors = [
            a.get("name", "")
            for a in (raw.get("authors") or [])
            if a.get("name")
        ]

        external_ids = raw.get("externalIds") or {}

        results.append(
            RelatedPaper(
                title=paper_title,
                abstract=(raw.get("abstract") or "").strip(),
                year=raw.get("year"),
                authors=authors,
                url=raw.get("url") or "",
                citation_count=raw.get("citationCount"),
                doi=external_ids.get("DOI"),
            )
        )

        if len(results) >= limit:
            break

    return results