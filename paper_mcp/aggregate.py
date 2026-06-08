"""Cross-source result fusion for the aggregated ``search_all`` tool.

Pure functions only — they take the per-source ranked ``Paper`` lists the
server has already fetched and merge them into one de-duplicated, re-ranked
list. No network, no I/O, so the ranking/merge logic is unit-testable on its
own (mirrors how ``mcp_safety`` keeps its rules pure).

De-duplication
    Two records are the same work when they share a normalized DOI or, lacking
    a DOI, a normalized title. arXiv preprints often have no DOI yet, so the
    title fallback is what links an arXiv hit to its Semantic Scholar /
    OpenAlex counterparts.

Ranking
    Reciprocal Rank Fusion (RRF) over each source's own relevance order — the
    standard way to combine ranked lists whose raw scores are not comparable.
    A paper surfaced by several corpora accumulates rank contributions from
    each and therefore rises above one found by a single corpus. Citation
    count breaks ties only.
"""
from __future__ import annotations

import re

from .models import Paper

# Standard RRF damping constant; larger flattens the contribution of top ranks.
RRF_K = 60

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DOI_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def _norm_title(title: str) -> str:
    """Lowercase and collapse every run of non-alphanumerics to one space."""
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


def _norm_doi(doi: str) -> str:
    return _DOI_PREFIX.sub("", (doi or "").strip().lower())


def _citation_count(p: Paper) -> int | None:
    cc = p.extra.get("citationCount") if p.extra else None
    return cc if isinstance(cc, int) else None


def _richer(a: Paper, b: Paper) -> Paper:
    """Pick the better representative of a duplicate pair.

    Prefer the record with the longer abstract, then one carrying a PDF link,
    then one carrying a DOI. This keeps the merged hit as informative as the
    richest single source allowed.
    """
    if len(a.summary) != len(b.summary):
        return a if len(a.summary) > len(b.summary) else b
    if bool(a.pdf_url) != bool(b.pdf_url):
        return a if a.pdf_url else b
    if bool(a.doi) != bool(b.doi):
        return a if a.doi else b
    return a


def fuse(results_by_source: dict[str, list[Paper]]) -> list[dict]:
    """Merge per-source ranked ``Paper`` lists into one fused, ranked list.

    Args:
        results_by_source: ``{canonical_source_name: [Paper, ...]}`` where each
            list is in that source's own relevance order (rank 0 = best).

    Returns:
        A list of merged-group dicts, best first. Each has:
        ``rep`` (the richest ``Paper``), ``sources`` (sorted names that found
        it), ``ids`` (``{source: native_id}`` for follow-up calls),
        ``citationCount`` (max seen, or ``None``), ``agreement`` (number of
        sources), and ``score`` (RRF, rounded).
    """
    groups: list[dict] = []
    by_doi: dict[str, dict] = {}
    by_title: dict[str, dict] = {}

    for source, papers in results_by_source.items():
        for rank, p in enumerate(papers):
            doi = _norm_doi(p.doi) or None
            title = _norm_title(p.title) or None
            group = None
            if doi is not None:
                group = by_doi.get(doi)
            if group is None and title is not None:
                group = by_title.get(title)
            if group is None:
                group = {
                    "rep": p,
                    "sources": set(),
                    "ids": {},
                    "citations": None,
                    "rrf": 0.0,
                }
                groups.append(group)

            group["rep"] = _richer(group["rep"], p)
            group["sources"].add(source)
            group["ids"].setdefault(source, p.id)
            group["rrf"] += 1.0 / (RRF_K + rank)
            cc = _citation_count(p)
            if cc is not None:
                group["citations"] = cc if group["citations"] is None else max(group["citations"], cc)

            # Register both keys so a later record sharing either one joins this
            # group (links a DOI-bearing record to a DOI-less same-title one).
            if doi is not None:
                by_doi.setdefault(doi, group)
            if title is not None:
                by_title.setdefault(title, group)

    groups.sort(key=lambda g: (g["rrf"], g["citations"] or 0), reverse=True)

    out: list[dict] = []
    for g in groups:
        out.append(
            {
                "rep": g["rep"],
                "sources": sorted(g["sources"]),
                "ids": g["ids"],
                "citationCount": g["citations"],
                "agreement": len(g["sources"]),
                "score": round(g["rrf"], 5),
            }
        )
    return out
