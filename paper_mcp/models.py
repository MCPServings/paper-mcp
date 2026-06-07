"""Normalized, source-agnostic data model.

Every source (arXiv now; PubMed / OpenAlex / OA repositories later) maps its
native record into `Paper`, so downstream tools — and the future retrieval,
extraction and synthesis layers — never depend on a single provider's schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Paper:
    id: str  # canonical id within the source, e.g. arXiv "2401.01234v1"
    source: str  # "arxiv", "pubmed", ...
    title: str
    authors: list[str] = field(default_factory=list)
    summary: str = ""  # abstract
    published: str = ""  # ISO date string
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    url: str = ""  # landing page
    pdf_url: str = ""
    doi: str = ""
    comment: str = ""
    journal_ref: str = ""
    # Source-specific metrics that don't fit the common schema
    # (e.g. citationCount, hIndex, matchScore). Empty for sources without them.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
