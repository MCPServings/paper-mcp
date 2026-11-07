"""Yongleji source (永乐集, https://yongleji.com) — about 465M scholarly works.

The Yongleji API follows the OpenAlex REST conventions (works, filters, W ids),
so this source reuses the OpenAlex client. Differences: relevance search uses
its semantic search, which understands questions in Chinese or English (the
``newest`` / ``cited`` orders use keyword search); the paper URL is the
publisher landing page or the DOI; no ``mailto`` is sent.

Yongleji allows each user 100 calls a day and one per second. ``YONGLEJI_BASE``
(default https://yongleji.com/v1) points the client at another address of the
same API; only then is the caller identity forwarded as ``X-Real-IP``, so the
limit applies per end user. A daily-quota 429 is reported to the user as is.
"""
from __future__ import annotations

import os

from ..models import Paper
from .openalex import _WORK_FIELDS, OpenAlexSource

_PUBLIC = "https://yongleji.com/v1"
_SEMANTIC_MAX = 200  # semantic results reachable through paging
_KEYWORD_SORTS = {"newest": "publication_date:desc", "cited": "cited_by_count:desc"}


class YonglejiSource(OpenAlexSource):
    name = "yongleji"
    label = "Yongleji"

    def __init__(
        self,
        *,
        base: str | None = None,
        timeout: float = 25.0,
        max_retries: int = 4,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        override = (base or os.getenv("YONGLEJI_BASE") or "").strip().rstrip("/")
        self._base = override or _PUBLIC
        self._forward_caller = bool(override)
        self._mailto = ""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        q = query.strip()
        keyword = sort_by in _KEYWORD_SORTS
        per_page = max(1, min(max_results, 100 if keyword else 50))
        page = max(1, start // per_page + 1)
        params = {"per_page": per_page, "page": page, "select": _WORK_FIELDS}
        if keyword:
            params.update({"search": q, "sort": _KEYWORD_SORTS[sort_by]})
        elif page * per_page > _SEMANTIC_MAX:
            return []
        else:
            params["search.semantic"] = q
        data = await self._get("/works", params)
        return [self._to_paper(w) for w in (data.get("results") or [])]

    def _to_paper(self, w: dict) -> Paper:
        p = super()._to_paper(w)
        if w:
            loc = w.get("primary_location") or {}
            p.url = loc.get("landing_page_url") or (f"https://doi.org/{p.doi}" if p.doi else "")
        return p
