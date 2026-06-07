"""Source registry.

Keeping sources behind a tiny registry (rather than importing one provider
directly in the server) is what makes the "add another legal corpus" step —
and the future Chinese-corpus rerun — a config change, not a rewrite.
"""
from __future__ import annotations

import os

from .arxiv import ArxivSource
from .base import PaperSource
from .openalex import OpenAlexSource
from .semanticscholar import SemanticScholarSource

_SOURCES: dict[str, PaperSource] = {
    "arxiv": ArxivSource(),
    "semanticscholar": SemanticScholarSource(
        api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    ),
    "openalex": OpenAlexSource(mailto=os.getenv("OPENALEX_MAILTO")),
}

# Friendly aliases that resolve to a canonical source key.
_ALIASES = {
    "s2": "semanticscholar",
    "semantic-scholar": "semanticscholar",
    "semantic_scholar": "semanticscholar",
    "semanticscholar.org": "semanticscholar",
    "oa": "openalex",
    "open-alex": "openalex",
    "openalex.org": "openalex",
}

DEFAULT_SOURCE = "arxiv"


def get_source(name: str | None = None) -> PaperSource:
    key = (name or DEFAULT_SOURCE).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _SOURCES:
        raise ValueError(
            f"unknown source {key!r}; available: {', '.join(sorted(_SOURCES))}"
        )
    return _SOURCES[key]


def list_sources() -> list[str]:
    return sorted(_SOURCES)


__all__ = ["PaperSource", "get_source", "list_sources", "DEFAULT_SOURCE"]
