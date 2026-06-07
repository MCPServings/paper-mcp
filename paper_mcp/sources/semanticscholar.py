"""Semantic Scholar source — a thin client over the S2 public APIs.

Wraps the full ~20-endpoint S2 surface (Academic Graph + Recommendations +
Datasets) the same way `arxiv.py` wraps the arXiv Atom API: a pure
pass-through with polite retry/backoff and an `x-api-key` header, normalizing
paper records into `Paper` where the generic tools need it, and returning
compact dicts for the S2-specific capabilities (citations, references,
authors, recommendations, snippets, datasets) that have no arXiv equivalent.
"""
from __future__ import annotations

import asyncio
import os
import re

import httpx

from ..models import Paper

_GRAPH = "https://api.semanticscholar.org/graph/v1"
_REC = "https://api.semanticscholar.org/recommendations/v1"
_DATA = "https://api.semanticscholar.org/datasets/v1"

_USER_AGENT = "paper-mcp/0.1 (+https://latex-tools.online/mcp)"
_RETRY_STATUS = {429, 500, 502, 503, 504}

# S2 returns only paperId+title unless fields are requested explicitly.
_PAPER_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,publicationDate,"
    "citationCount,referenceCount,influentialCitationCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,publicationTypes,authors"
)
_SEARCH_FIELDS = (
    "paperId,externalIds,url,title,abstract,year,publicationDate,venue,"
    "citationCount,influentialCitationCount,openAccessPdf,fieldsOfStudy,authors"
)
_AUTHOR_FIELDS = (
    "authorId,name,affiliations,homepage,paperCount,citationCount,hIndex,url"
)
# For citations/references: contexts/intents/isInfluential are edge fields;
# the rest apply to the nested citing/cited paper.
_EDGE_FIELDS = (
    "contexts,intents,isInfluential,title,year,authors,venue,"
    "citationCount,externalIds,url,abstract,openAccessPdf"
)

# S2's controlled fieldsOfStudy vocabulary (used for categories()/recent()).
_FIELDS_OF_STUDY = (
    "Computer Science", "Medicine", "Chemistry", "Biology",
    "Materials Science", "Physics", "Geology", "Psychology", "Art",
    "History", "Geography", "Sociology", "Business", "Political Science",
    "Economics", "Philosophy", "Mathematics", "Engineering",
    "Environmental Science", "Agricultural and Food Sciences", "Education",
    "Law", "Linguistics",
)

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_KNOWN_PREFIXES = (
    "CorpusId:", "DOI:", "ARXIV:", "MAG:", "ACL:", "PMID:", "PMCID:", "URL:",
)


def _norm_paper_id(pid: str) -> str:
    """Accept bare arXiv ids for convenience; pass through known S2 id forms."""
    p = pid.strip()
    if any(p.upper().startswith(k.upper()) for k in _KNOWN_PREFIXES):
        return p
    if _ARXIV_ID_RE.match(p):
        return f"ARXIV:{p}"
    return p


class SemanticScholarSource:
    name = "semanticscholar"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 25.0,
        max_retries: int = 4,
    ) -> None:
        self._api_key = (api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    # ---- PaperSource protocol (plugs into the generic tools) -------------
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        data = await self._get(
            f"{_GRAPH}/paper/search",
            {
                "query": query.strip(),
                "offset": max(0, start),
                "limit": max(1, min(max_results, 100)),
                "fields": _SEARCH_FIELDS,
            },
        )
        papers = [self._to_paper(p) for p in (data.get("data") or [])]
        if sort_by in ("newest", "updated"):
            papers.sort(key=lambda p: p.published, reverse=True)
        return papers

    async def fetch(self, paper_id: str) -> Paper | None:
        try:
            data = await self._get(
                f"{_GRAPH}/paper/{_norm_paper_id(paper_id)}",
                {"fields": _PAPER_FIELDS},
            )
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
            f"{_GRAPH}/author/search",
            {"query": name, "fields": "authorId,name,paperCount", "limit": 1},
        )
        items = res.get("data") or []
        if not items:
            return []
        aid = items[0]["authorId"]
        papers = await self._get(
            f"{_GRAPH}/author/{aid}/papers",
            {
                "fields": _SEARCH_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 100)),
            },
        )
        out = [self._to_paper(p) for p in (papers.get("data") or [])]
        out.sort(key=lambda p: p.published, reverse=True)
        return out

    async def recent(
        self, category: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]:
        # S2 search is query-centric; approximate "recent in a field" by
        # querying the field name, filtering fieldsOfStudy, sorting by date.
        cat = category.strip()
        if not cat:
            raise ValueError("category must not be empty")
        data = await self._get(
            f"{_GRAPH}/paper/search",
            {
                "query": cat,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 100)),
                "fields": _SEARCH_FIELDS,
                "fieldsOfStudy": cat,
            },
        )
        out = [self._to_paper(p) for p in (data.get("data") or [])]
        out.sort(key=lambda p: p.published, reverse=True)
        return out

    def categories(self) -> list[dict]:
        """S2 fields of study, usable as the `fieldsOfStudy` filter."""
        return [{"id": f, "name": f} for f in _FIELDS_OF_STUDY]

    # ---- S2-specific capabilities (return raw S2 JSON) -------------------
    async def paper_citations(
        self, paper_id: str, *, max_results: int = 10, start: int = 0
    ) -> dict:
        return await self._get(
            f"{_GRAPH}/paper/{_norm_paper_id(paper_id)}/citations",
            {
                "fields": _EDGE_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 1000)),
            },
        )

    async def paper_references(
        self, paper_id: str, *, max_results: int = 10, start: int = 0
    ) -> dict:
        return await self._get(
            f"{_GRAPH}/paper/{_norm_paper_id(paper_id)}/references",
            {
                "fields": _EDGE_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 1000)),
            },
        )

    async def paper_authors(
        self, paper_id: str, *, max_results: int = 100, start: int = 0
    ) -> dict:
        return await self._get(
            f"{_GRAPH}/paper/{_norm_paper_id(paper_id)}/authors",
            {
                "fields": _AUTHOR_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 1000)),
            },
        )

    async def match_title(self, title: str) -> dict:
        return await self._get(
            f"{_GRAPH}/paper/search/match",
            {"query": title.strip(), "fields": _SEARCH_FIELDS},
        )

    async def autocomplete(self, query: str) -> dict:
        return await self._get(
            f"{_GRAPH}/paper/autocomplete", {"query": query.strip()}
        )

    async def search_bulk(
        self,
        query: str,
        *,
        sort: str | None = None,
        fields_of_study: str | None = None,
        year: str | None = None,
        venue: str | None = None,
        publication_types: str | None = None,
        open_access_pdf: bool = False,
        token: str | None = None,
        max_results: int = 100,
    ) -> dict:
        params: dict = {
            "query": query.strip(),
            "fields": _SEARCH_FIELDS,
            "limit": max(1, min(max_results, 1000)),
            "sort": sort,
            "fieldsOfStudy": fields_of_study,
            "year": year,
            "venue": venue,
            "publicationTypes": publication_types,
            "token": token,
        }
        if open_access_pdf:
            params["openAccessPdf"] = ""
        return await self._get(f"{_GRAPH}/paper/search/bulk", params)

    async def papers_batch(self, ids: list[str], *, fields: str | None = None) -> list:
        data = await self._post(
            f"{_GRAPH}/paper/batch",
            {"fields": fields or _PAPER_FIELDS},
            {"ids": [_norm_paper_id(i) for i in ids]},
        )
        return data if isinstance(data, list) else []

    async def author_search(
        self, query: str, *, max_results: int = 10, start: int = 0
    ) -> dict:
        return await self._get(
            f"{_GRAPH}/author/search",
            {
                "query": query.strip(),
                "fields": _AUTHOR_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 100)),
            },
        )

    async def author(self, author_id: str) -> dict:
        return await self._get(
            f"{_GRAPH}/author/{author_id.strip()}", {"fields": _AUTHOR_FIELDS}
        )

    async def author_papers(
        self, author_id: str, *, max_results: int = 20, start: int = 0
    ) -> dict:
        return await self._get(
            f"{_GRAPH}/author/{author_id.strip()}/papers",
            {
                "fields": _SEARCH_FIELDS,
                "offset": max(0, start),
                "limit": max(1, min(max_results, 1000)),
            },
        )

    async def authors_batch(self, ids: list[str]) -> list:
        data = await self._post(
            f"{_GRAPH}/author/batch",
            {"fields": _AUTHOR_FIELDS},
            {"ids": [i.strip() for i in ids]},
        )
        return data if isinstance(data, list) else []

    async def snippet_search(self, query: str, *, max_results: int = 10) -> dict:
        return await self._get(
            f"{_GRAPH}/snippet/search",
            {"query": query.strip(), "limit": max(1, min(max_results, 1000))},
        )

    async def recommend_for_paper(
        self, paper_id: str, *, max_results: int = 10, pool: str = "recent"
    ) -> dict:
        return await self._get(
            f"{_REC}/papers/forpaper/{_norm_paper_id(paper_id)}",
            {
                "fields": _SEARCH_FIELDS,
                "limit": max(1, min(max_results, 500)),
                "from": pool,
            },
        )

    async def recommend_from_examples(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        *,
        max_results: int = 10,
    ) -> dict:
        return await self._post(
            f"{_REC}/papers",
            {"fields": _SEARCH_FIELDS, "limit": max(1, min(max_results, 500))},
            {
                "positivePaperIds": [_norm_paper_id(i) for i in positive_ids],
                "negativePaperIds": [_norm_paper_id(i) for i in (negative_ids or [])],
            },
        )

    async def dataset_releases(self) -> list:
        data = await self._get(f"{_DATA}/release", {})
        return data if isinstance(data, list) else []

    async def dataset_release(self, release_id: str = "latest") -> dict:
        return await self._get(f"{_DATA}/release/{release_id.strip()}", {})

    async def dataset_download(
        self, dataset_name: str, *, release_id: str = "latest"
    ) -> dict:
        return await self._get(
            f"{_DATA}/release/{release_id.strip()}/dataset/{dataset_name.strip()}",
            {},
        )

    async def dataset_diffs(
        self, dataset_name: str, *, start_release: str, end_release: str = "latest"
    ) -> dict:
        return await self._get(
            f"{_DATA}/diffs/{start_release.strip()}/to/"
            f"{end_release.strip()}/{dataset_name.strip()}",
            {},
        )

    # ---- internals -------------------------------------------------------
    def _headers(self) -> dict:
        h = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    async def _request(self, method: str, url: str, *, params=None, json_body=None):
        last_exc: Exception | None = None
        clean = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=self._headers()
        ) as client:
            for attempt in range(self._max_retries):
                backoff = min(2.0 * (attempt + 1), 10.0)
                try:
                    resp = await client.request(
                        method, url, params=clean, json=json_body
                    )
                except httpx.TransportError as exc:
                    last_exc = exc
                    await asyncio.sleep(backoff)
                    continue
                if resp.status_code in _RETRY_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"S2 returned {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                if not resp.content:
                    return {}
                return resp.json()
        assert last_exc is not None
        raise last_exc

    async def _get(self, url: str, params: dict):
        return await self._request("GET", url, params=params)

    async def _post(self, url: str, params: dict, json_body):
        return await self._request("POST", url, params=params, json_body=json_body)

    def _to_paper(self, raw: dict) -> Paper:
        if not raw:
            return Paper(id="", source=self.name, title="")
        ext = raw.get("externalIds") or {}
        authors = [a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")]
        oa = raw.get("openAccessPdf") or {}
        pub = raw.get("publicationDate") or (
            str(raw["year"]) if raw.get("year") else ""
        )
        extra: dict = {}
        for k in ("citationCount", "influentialCitationCount", "referenceCount"):
            if raw.get(k) is not None:
                extra[k] = raw[k]
        if raw.get("matchScore") is not None:
            extra["matchScore"] = raw["matchScore"]
        return Paper(
            id=raw.get("paperId") or ext.get("ArXiv") or ext.get("DOI") or "",
            source=self.name,
            title=raw.get("title") or "",
            authors=authors,
            summary=raw.get("abstract") or "",
            published=pub,
            updated="",
            categories=raw.get("fieldsOfStudy") or [],
            url=raw.get("url") or "",
            pdf_url=oa.get("url") or "",
            doi=ext.get("DOI") or "",
            extra=extra,
        )
