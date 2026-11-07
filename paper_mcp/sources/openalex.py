"""OpenAlex source — a thin client over the OpenAlex REST API.

OpenAlex (https://openalex.org, by OurResearch) is the free CC0 successor to
Microsoft Academic Graph: ~316M works across every field, with citation graph,
authors, institutions, topics and influence metrics. The data is CC0 (public
domain), so wrapping it carries no licensing strings.

We use the *polite pool* (a ``mailto`` query param, no API key needed) and the
same retry/backoff discipline as the arXiv source. Abstracts come back as an
inverted index, which we reconstruct to plain text.

``OPENALEX_BASE`` points the client at an OpenAlex-compatible mirror instead of
api.openalex.org. The mirror rate-limits per end user, so the caller identity is
forwarded as ``X-Real-IP``; a daily-quota 429 is reported to the user as is.
"""
from __future__ import annotations

import asyncio
import os

import httpx

from ..models import Paper
from .base import CALLER

_OFFICIAL = "https://api.openalex.org"
_BASE = (os.getenv("OPENALEX_BASE") or _OFFICIAL).rstrip("/")
_MIRROR = _BASE != _OFFICIAL
_USER_AGENT = "paper-mcp/0.1 (+https://latex-tools.online/mcp)"
_RETRY_STATUS = {429, 500, 502, 503, 504}

# Field selection keeps payloads small (does not change quota cost).
_WORK_FIELDS = (
    "id,doi,title,display_name,publication_year,publication_date,language,"
    "type,cited_by_count,referenced_works_count,fwci,is_retracted,"
    "open_access,primary_location,authorships,topics,abstract_inverted_index"
)
_AUTHOR_FIELDS = (
    "id,display_name,orcid,works_count,cited_by_count,summary_stats,"
    "last_known_institutions,affiliations"
)
_INST_FIELDS = (
    "id,display_name,ror,country_code,type,works_count,cited_by_count,homepage_url"
)

# OpenAlex sort keys we expose via the generic sort_by argument.
_SORT_MAP = {
    "relevance": "relevance_score:desc",
    "newest": "publication_date:desc",
    "cited": "cited_by_count:desc",
}

_OPENALEX_ARXIV_SOURCE = "S4306400194"


def _short_id(oid: str) -> str:
    """OpenAlex ids are full URLs; keep the short form (e.g. W2741809807)."""
    return (oid or "").rstrip("/").rsplit("/", 1)[-1]


def _reconstruct_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlexSource:
    name = "openalex"
    label = "OpenAlex"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout: float = 25.0,
        max_retries: int = 4,
    ) -> None:
        # The polite pool just needs a contact address; no key, no secret.
        self._mailto = (mailto or os.getenv("OPENALEX_MAILTO")
                        or "api@latex-tools.online").strip()
        self._timeout = timeout
        self._max_retries = max_retries
        self._base = _BASE
        # Only a mirror (an address we were configured with) gets the caller's identity.
        self._forward_caller = _MIRROR

    # ---- PaperSource protocol -------------------------------------------
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        params = {
            "search": query.strip(),
            "per_page": max(1, min(max_results, 100)),
            "page": max(1, start // max(1, max_results) + 1),
            "select": _WORK_FIELDS,
        }
        if sort_by in _SORT_MAP and sort_by != "relevance":
            params["sort"] = _SORT_MAP[sort_by]
        data = await self._get("/works", params)
        return [self._to_paper(w) for w in (data.get("results") or [])]

    async def fetch(self, paper_id: str) -> Paper | None:
        ident = self._normalize_work_id(paper_id)
        try:
            data = await self._get(f"/works/{ident}", {"select": _WORK_FIELDS})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return self._to_paper(data) if data else None

    async def search_by_author(
        self, author: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]:
        name = author.strip()
        if not name:
            raise ValueError("author must not be empty")
        res = await self._get(
            "/authors",
            {"search": name, "per_page": 1, "select": "id,display_name"},
        )
        items = res.get("results") or []
        if not items:
            return []
        aid = _short_id(items[0]["id"])
        works = await self._get(
            "/works",
            {
                "filter": f"author.id:{aid}",
                "sort": "publication_date:desc",
                "per_page": max(1, min(max_results, 100)),
                "page": max(1, start // max(1, max_results) + 1),
                "select": _WORK_FIELDS,
            },
        )
        return [self._to_paper(w) for w in (works.get("results") or [])]

    async def recent(
        self, category: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]:
        # `category` is treated as an OpenAlex topic/concept search term.
        cat = category.strip()
        if not cat:
            raise ValueError("category must not be empty")
        data = await self._get(
            "/works",
            {
                "search": cat,
                "sort": "publication_date:desc",
                "per_page": max(1, min(max_results, 100)),
                "page": max(1, start // max(1, max_results) + 1),
                "select": _WORK_FIELDS,
            },
        )
        return [self._to_paper(w) for w in (data.get("results") or [])]

    def categories(self) -> list[dict]:
        """OpenAlex uses ~4500 topics rather than a fixed code list.

        We return a small curated set of common topic search terms; any text
        works as a `recent`/`search` topic. Use `list_openalex_topics` (tool)
        for the live, searchable topic taxonomy.
        """
        return [
            {"id": t, "name": t} for t in (
                "machine learning", "deep learning", "natural language processing",
                "computer vision", "reinforcement learning", "large language models",
                "graph neural networks", "information retrieval", "robotics",
                "computational biology", "quantum computing", "materials science",
            )
        ]

    # ---- OpenAlex-specific capabilities ---------------------------------
    async def get_work(self, work_id: str) -> dict:
        ident = self._normalize_work_id(work_id)
        data = await self._get(f"/works/{ident}", {"select": _WORK_FIELDS})
        return self._work_dict(data)

    async def work_citations(
        self, work_id: str, *, max_results: int = 10, start: int = 0
    ) -> dict:
        ident = self._normalize_work_id(work_id)
        data = await self._get(
            "/works",
            {
                "filter": f"cites:{ident}",
                "sort": "cited_by_count:desc",
                "per_page": max(1, min(max_results, 100)),
                "page": max(1, start // max(1, max_results) + 1),
                "select": _WORK_FIELDS,
            },
        )
        return {
            "total": (data.get("meta") or {}).get("count"),
            "results": [self._work_dict(w) for w in (data.get("results") or [])],
        }

    async def work_references(self, work_id: str, *, max_results: int = 25) -> dict:
        ident = self._normalize_work_id(work_id)
        work = await self._get(
            f"/works/{ident}", {"select": "referenced_works,referenced_works_count"}
        )
        refs = (work.get("referenced_works") or [])[:max(1, min(max_results, 100))]
        if not refs:
            return {"total": work.get("referenced_works_count", 0), "results": []}
        ids = "|".join(_short_id(r) for r in refs)
        data = await self._get(
            "/works",
            {"filter": f"ids.openalex:{ids}", "per_page": len(refs),
             "select": _WORK_FIELDS},
        )
        return {
            "total": work.get("referenced_works_count", len(refs)),
            "results": [self._work_dict(w) for w in (data.get("results") or [])],
        }

    async def search_authors(
        self, query: str, *, max_results: int = 10, start: int = 0
    ) -> dict:
        data = await self._get(
            "/authors",
            {
                "search": query.strip(),
                "per_page": max(1, min(max_results, 100)),
                "page": max(1, start // max(1, max_results) + 1),
                "select": _AUTHOR_FIELDS,
            },
        )
        return {
            "total": (data.get("meta") or {}).get("count"),
            "results": [self._author_dict(a) for a in (data.get("results") or [])],
        }

    async def search_institutions(
        self, query: str, *, max_results: int = 10
    ) -> dict:
        data = await self._get(
            "/institutions",
            {"search": query.strip(),
             "per_page": max(1, min(max_results, 100)),
             "select": _INST_FIELDS},
        )
        return {
            "total": (data.get("meta") or {}).get("count"),
            "results": [self._inst_dict(i) for i in (data.get("results") or [])],
        }

    async def search_filtered(
        self,
        query: str = "",
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        is_oa: bool | None = None,
        min_citations: int | None = None,
        institution_id: str | None = None,
        sort_by: str = "relevance",
        max_results: int = 25,
    ) -> dict:
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if is_oa is not None:
            filters.append(f"is_oa:{'true' if is_oa else 'false'}")
        if min_citations:
            filters.append(f"cited_by_count:>{min_citations - 1}")
        if institution_id:
            filters.append(f"authorships.institutions.id:{_short_id(institution_id)}")
        params = {
            "per_page": max(1, min(max_results, 100)),
            "select": _WORK_FIELDS,
        }
        if query.strip():
            params["search"] = query.strip()
        if filters:
            params["filter"] = ",".join(filters)
        if sort_by in _SORT_MAP and sort_by != "relevance":
            params["sort"] = _SORT_MAP[sort_by]
        data = await self._get("/works", params)
        return {
            "total": (data.get("meta") or {}).get("count"),
            "results": [self._work_dict(w) for w in (data.get("results") or [])],
        }

    async def trends(self, query: str, *, group_by: str = "publication_year") -> dict:
        """Aggregate counts without downloading rows (group_by)."""
        params = {"group_by": group_by, "per_page": 1}
        if query.strip():
            params["search"] = query.strip()
        data = await self._get("/works", params)
        return {
            "total": (data.get("meta") or {}).get("count"),
            "group_by": group_by,
            "groups": [
                {"key": g.get("key_display_name") or g.get("key"),
                 "count": g.get("count")}
                for g in (data.get("group_by") or [])
            ],
        }

    async def list_topics(self, query: str, *, max_results: int = 15) -> dict:
        data = await self._get(
            "/topics",
            {"search": query.strip(), "per_page": max(1, min(max_results, 50)),
             "select": "id,display_name,description,works_count"},
        )
        return {
            "total": (data.get("meta") or {}).get("count"),
            "results": [
                {"id": _short_id(t.get("id")), "name": t.get("display_name"),
                 "works_count": t.get("works_count"),
                 "description": (t.get("description") or "")[:160]}
                for t in (data.get("results") or [])
            ],
        }

    # ---- internals -------------------------------------------------------
    def _normalize_work_id(self, raw: str) -> str:
        v = raw.strip()
        if v.lower().startswith("http"):
            return _short_id(v)
        if v.upper().startswith("W") and v[1:].isdigit():
            return v.upper()
        # arXiv / DOI passthrough using OpenAlex's id resolver syntax.
        if "/" in v or v.lower().startswith("10."):
            return f"doi:{v}" if not v.lower().startswith("doi:") else v
        if v.replace(".", "").isdigit() and "." in v:  # bare arXiv id
            return f"arxiv:{v}"
        return v

    async def _get(self, path: str, params: dict) -> dict:
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        if self._mailto:
            clean["mailto"] = self._mailto
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self._forward_caller and CALLER.get():
            headers["X-Real-IP"] = CALLER.get()
        last_exc: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            for attempt in range(self._max_retries):
                backoff = min(2.0 * (attempt + 1), 10.0)
                try:
                    resp = await client.get(f"{self._base}{path}", params=clean)
                except httpx.TransportError as exc:
                    last_exc = exc
                    await asyncio.sleep(backoff)
                    continue
                if resp.status_code == 429:
                    # Per-user quota of an OpenAlex-compatible service: {"limit": "per_second"|"daily", ...}.
                    try:
                        info = resp.json()
                    except ValueError:
                        info = {}
                    which = info.get("limit") if isinstance(info, dict) else None
                    if which == "per_second" and attempt < self._max_retries - 1:
                        await asyncio.sleep(min(2.0, float(info.get("retry_after") or 1.0)) + 0.05)
                        continue
                    if which in ("per_second", "daily"):
                        raise httpx.HTTPError(info.get("message") or "rate limit exceeded, try again later")
                if resp.status_code in _RETRY_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"{self.label} returned {resp.status_code}",
                        request=resp.request, response=resp)
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json() if resp.content else {}
        assert last_exc is not None
        raise last_exc

    def _to_paper(self, w: dict) -> Paper:
        if not w:
            return Paper(id="", source=self.name, title="")
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        oa = w.get("open_access") or {}
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (w.get("authorships") or [])
            if a.get("author", {}).get("display_name")
        ]
        topics = [t.get("display_name", "") for t in (w.get("topics") or []) if t.get("display_name")]
        extra = {}
        for k_out, k_in in (("citationCount", "cited_by_count"),
                            ("referenceCount", "referenced_works_count")):
            if w.get(k_in) is not None:
                extra[k_out] = w[k_in]
        if w.get("fwci") is not None:
            extra["fwci"] = w["fwci"]
        if src.get("display_name"):
            extra["venue"] = src["display_name"]
        return Paper(
            id=_short_id(w.get("id")),
            source=self.name,
            title=w.get("title") or w.get("display_name") or "",
            authors=authors,
            summary=_reconstruct_abstract(w.get("abstract_inverted_index")),
            published=w.get("publication_date") or (str(w.get("publication_year")) if w.get("publication_year") else ""),
            updated="",
            categories=topics,
            url=w.get("id") or "",
            pdf_url=(oa.get("oa_url") or loc.get("pdf_url") or ""),
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            extra=extra,
        )

    def _work_dict(self, w: dict) -> dict:
        p = self._to_paper(w)
        d = {
            "id": p.id, "title": p.title, "year": w.get("publication_year"),
            "authors": p.authors[:10] + (["et al."] if len(p.authors) > 10 else []),
            "citationCount": w.get("cited_by_count"),
            "fwci": w.get("fwci"),
            "is_oa": (w.get("open_access") or {}).get("is_oa"),
            "topics": p.categories[:5],
            "url": p.url,
        }
        if p.doi:
            d["doi"] = p.doi
        if p.pdf_url:
            d["pdf_url"] = p.pdf_url
        if p.summary:
            d["abstract_preview"] = p.summary[:320].rstrip() + ("…" if len(p.summary) > 320 else "")
        return {k: v for k, v in d.items() if v is not None}

    def _author_dict(self, a: dict) -> dict:
        stats = a.get("summary_stats") or {}
        insts = a.get("last_known_institutions") or []
        d = {
            "id": _short_id(a.get("id")),
            "name": a.get("display_name"),
            "orcid": (a.get("orcid") or "").replace("https://orcid.org/", "") or None,
            "works_count": a.get("works_count"),
            "cited_by_count": a.get("cited_by_count"),
            "h_index": stats.get("h_index"),
            "i10_index": stats.get("i10_index"),
            "institutions": [i.get("display_name") for i in insts if i.get("display_name")],
        }
        return {k: v for k, v in d.items() if v is not None}

    def _inst_dict(self, i: dict) -> dict:
        d = {
            "id": _short_id(i.get("id")),
            "name": i.get("display_name"),
            "country": i.get("country_code"),
            "type": i.get("type"),
            "works_count": i.get("works_count"),
            "cited_by_count": i.get("cited_by_count"),
            "ror": (i.get("ror") or "").replace("https://ror.org/", "") or None,
            "homepage": i.get("homepage_url"),
        }
        return {k: v for k, v in d.items() if v is not None}
