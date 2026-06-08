"""FastMCP server exposing paper search/fetch over streamable HTTP.

Runs standalone at 127.0.0.1:$PAPER_MCP_PORT with the MCP endpoint at
$PAPER_MCP_PATH (default ``/mcp``), to be reverse-proxied as
``https://latex-tools.online/mcp``.

Tools (baseline):
  * ``search_papers`` — query a source, get back normalized hits.
  * ``get_paper``     — fetch one record (full abstract + links) by id.
  * ``list_sources``  — enumerate available corpora.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from collections import defaultdict, deque

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import __version__
from .aggregate import fuse
from .models import Paper
from .sources import DEFAULT_SOURCE, get_source, list_sources
from .sources import recognize

HOST = os.getenv("PAPER_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("PAPER_MCP_PORT", "9400"))
PATH = os.getenv("PAPER_MCP_PATH", "/mcp")

_INSTRUCTIONS = (
    "Academic paper search for AI agents, served at latex-tools.online/mcp.\n\n"
    "Three corpora are available via the `source` argument: `arxiv` (default), "
    "`semanticscholar` (alias `s2`), and `openalex` (alias `oa`).\n\n"
    "Generic, source-agnostic tools:\n"
    "  1. `search_papers(query=..., source='arxiv', max_results=10, "
    "sort_by='relevance')` to find papers. For arxiv, `query` accepts plain "
    "text or field syntax like `ti:`, `au:`, `cat:cs.CL`, `abs:` + AND/OR.\n"
    "  1b. `search_all(query=..., max_results=10)` searches ALL three corpora "
    "at once, de-duplicates the same work across them (by DOI/title) and "
    "re-ranks with Reciprocal Rank Fusion; each hit carries `sources` (who "
    "found it) and an `ids` map you can hand to get_paper/read_paper. Use it "
    "as the default broad search; use `search_papers` when you want one "
    "specific corpus or arxiv field syntax.\n"
    "  2. `get_paper(paper_id=..., source='arxiv')` for one paper's full "
    "record. For s2, id accepts S2 id / `DOI:` / `ARXIV:` / `CorpusId:`.\n"
    "  3. `search_by_author(author=..., source='arxiv')` newest first.\n"
    "  4. `list_recent(category='cs.CL', source='arxiv')` latest in a "
    "category (arxiv code, or an S2 field of study).\n"
    "  5. `list_categories(source='arxiv')` category codes.\n"
    "  6. `read_paper(paper_id=..., format='markdown')` for the FULL text "
    "(arXiv): 'markdown' renders the body with formulas as $LaTeX$, 'html' "
    "returns the raw LaTeXML page, 'latex' returns the original manuscript "
    "source.\n"
    "  7. `list_paper_sources()` available corpora.\n\n"
    "Image → LaTeX (turn a formula or table image back into LaTeX, e.g. a "
    "figure cropped from a paper; no vision model needed on your side):\n"
    "  • `recognize_formula(image_url=... or image_base64=...)` → LaTeX\n"
    "  • `recognize_table(image_url=... or image_base64=...)` → LaTeX tabular\n"
    "  • `list_ocr_models()` available OCR models\n\n"
    "Semantic Scholar capabilities (the full S2 API surface — citation "
    "graph, authors, recommendations, full-text snippets, bulk datasets):\n"
    "  • `get_paper_citations` / `get_paper_references` / `get_paper_authors`\n"
    "  • `match_paper_title` (exact title) / `autocomplete_papers`\n"
    "  • `search_papers_bulk` (≤1000, sortable, token paging) / "
    "`get_papers_batch`\n"
    "  • `search_authors` / `get_author` / `get_author_papers` / "
    "`get_authors_batch`\n"
    "  • `search_snippets` (search inside paper full text)\n"
    "  • `recommend_papers_for_paper` / `recommend_papers_from_examples`\n"
    "  • `list_dataset_releases` / `get_dataset_release` / "
    "`get_dataset_download_links` / `get_dataset_diffs`\n\n"
    "OpenAlex capabilities (free CC0 all-field corpus, 316M works — citation "
    "graph, authors with h-index, institutions, topics, influence metrics):\n"
    "  • `get_openalex_work` / `get_openalex_citations` / "
    "`get_openalex_references`\n"
    "  • `search_openalex_authors` / `search_openalex_institutions`\n"
    "  • `search_openalex_works` (filters: year range, open-access, "
    "min-citations, institution)\n"
    "  • `get_openalex_trends` (publication-trend analytics) / "
    "`list_openalex_topics`\n\n"
    "Results are normalized across sources, so the same fields apply no "
    "matter which corpus is queried."
)


_TS = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp = FastMCP(
    name="paper-search",
    instructions=_INSTRUCTIONS,
    host=HOST,
    port=PORT,
    streamable_http_path=PATH,
    transport_security=_TS,
)

# Per-IP rate limit for the MCP endpoint. MCP sessions are chatty (initialize +
# tools/list + many tools/call, each a separate JSON-RPC POST), so the hourly
# budget is generous: enough for several deep agent sessions, while blocking
# scripted hammering of the shared inference / upstream backends.
MCP_MAX_PER_HOUR = int(os.getenv("MCP_MAX_PER_HOUR", "300"))
MCP_RATE_WINDOW_SEC = 3600.0


class _RateLimitMiddleware:
    """Pure-ASGI per-IP rate limit for the MCP endpoint.

    Counts JSON-RPC POSTs per client IP in a sliding hourly window and rejects
    with HTTP 429 once the limit is exceeded. Implemented as pure ASGI (not
    Starlette BaseHTTPMiddleware) so it never buffers the streamable-HTTP / SSE
    response body.
    """

    def __init__(self, app, *, max_per_hour: int, window: float = MCP_RATE_WINDOW_SEC):
        self.app = app
        self.max_per_hour = max_per_hour
        self.window = window
        self._lock = threading.Lock()
        self._buckets: dict = defaultdict(lambda: deque(maxlen=max_per_hour * 2 + 32))

    @staticmethod
    def _client_ip(scope) -> str:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        xri = headers.get("x-real-ip")
        if xri:
            return xri.strip()
        xff = headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "0.0.0.0"

    def _allow(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._buckets[ip]
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.max_per_hour:
                return False
            dq.append(now)
            return True

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        if not self._allow(self._client_ip(scope)):
            body = (b'{"jsonrpc":"2.0","error":{"code":-32029,'
                    b'"message":"rate limit exceeded"},"id":null}')
            await send({"type": "http.response.start", "status": 429,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"retry-after", b"3600")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)

# arXiv field prefixes; if the query already uses one we pass it through as-is.
_FIELD_RE = re.compile(r"\b(ti|au|abs|co|jr|cat|rn|id|all):", re.IGNORECASE)
_ABSTRACT_PREVIEW = 320

# Corpora fused by `search_all` when the caller doesn't name specific ones.
_AGG_SOURCES = ("arxiv", "semanticscholar", "openalex")


def _build_query(query: str, source: str = DEFAULT_SOURCE) -> str:
    q = query.strip()
    if not q:
        raise ValueError("query must not be empty")
    # The `all:` prefix and field syntax (ti:/au:/cat:) are arXiv-specific; for
    # other corpora (openalex, s2) we pass the plain text straight through.
    if source not in ("arxiv",):
        return q
    if _FIELD_RE.search(q):
        return q
    return f"all:{q}"


def _hit(p: Paper) -> dict:
    """Compact form for search hits (abstract truncated to save tokens)."""
    summary = p.summary
    if len(summary) > _ABSTRACT_PREVIEW:
        summary = summary[:_ABSTRACT_PREVIEW].rstrip() + "…"
    hit = {
        "id": p.id,
        "source": p.source,
        "title": p.title,
        "authors": p.authors[:8] + (["et al."] if len(p.authors) > 8 else []),
        "published": p.published[:10],
        "categories": p.categories,
        "url": p.url,
        "abstract_preview": summary,
    }
    if p.extra:
        hit.update(p.extra)
    return hit


@mcp.tool(description="Search academic papers. Returns normalized hits with a "
          "short abstract preview; call get_paper for the full record.")
async def search_papers(
    query: str,
    source: str = DEFAULT_SOURCE,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "relevance",
) -> dict:
    """Search a paper corpus.

    Args:
        query: Plain text, or arXiv field syntax (ti:/au:/cat:/abs: + AND/OR).
        source: Corpus to search (currently: arxiv).
        max_results: 1–50.
        start: Offset for pagination.
        sort_by: relevance | newest | updated.
    """
    src = get_source(source)
    try:
        papers = await src.search(
            _build_query(query, source),
            max_results=max_results,
            start=start,
            sort_by=sort_by,
        )
    except httpx.HTTPError as exc:
        return {"query": query, "source": source, "error": f"upstream error: {exc}"}
    return {
        "query": query,
        "source": source,
        "sort_by": sort_by,
        "count": len(papers),
        "results": [_hit(p) for p in papers],
    }


def _merged_hit(group: dict) -> dict:
    """Render one fused group (rep Paper + cross-source meta) as a hit."""
    p: Paper = group["rep"]
    summary = p.summary
    if len(summary) > _ABSTRACT_PREVIEW:
        summary = summary[:_ABSTRACT_PREVIEW].rstrip() + "…"
    hit = {
        "title": p.title,
        "authors": p.authors[:8] + (["et al."] if len(p.authors) > 8 else []),
        "published": p.published[:10],
        "categories": p.categories,
        "url": p.url,
        "abstract_preview": summary,
        "sources": group["sources"],
        "ids": group["ids"],
        "agreement": group["agreement"],
        "score": group["score"],
    }
    if p.doi:
        hit["doi"] = p.doi
    if group["citationCount"] is not None:
        hit["citationCount"] = group["citationCount"]
    return hit


@mcp.tool(description="Aggregated search across arXiv, Semantic Scholar and "
          "OpenAlex at once. Fans out concurrently, de-duplicates the same "
          "work across corpora (by DOI or title) and re-ranks with Reciprocal "
          "Rank Fusion, so papers found by several sources rank highest. Each "
          "hit lists which `sources` found it and an `ids` map "
          "({source: id}) you can pass to get_paper / read_paper / the "
          "citation tools. Prefer this over search_papers for a broad lookup.")
async def search_all(
    query: str,
    max_results: int = 10,
    sources: str = "arxiv,semanticscholar,openalex",
    per_source: int = 0,
) -> dict:
    """Search several corpora at once and return one fused, de-duplicated list.

    Args:
        query: Plain text. Field syntax (ti:/au:/cat:) only affects arXiv.
        max_results: 1–50 merged results to return.
        sources: Comma/space separated corpora to fuse (names or aliases:
            arxiv, semanticscholar/s2, openalex/oa). Defaults to all three.
        per_source: How many raw hits to pull from each corpus before fusing.
            0 (default) uses max(max_results, 10) to give the ranker material.
    """
    max_results = max(1, min(max_results, 50))
    wanted = [s for s in re.split(r"[,\s]+", sources.strip()) if s] or list(_AGG_SOURCES)
    fetch_n = per_source if per_source > 0 else max(max_results, 10)

    async def _one(name: str):
        try:
            src = get_source(name)
        except ValueError as exc:
            return name, exc
        try:
            papers = await src.search(
                _build_query(query, src.name), max_results=fetch_n, sort_by="relevance"
            )
            return src.name, papers
        except (httpx.HTTPError, ValueError) as exc:
            return src.name, exc

    pairs = await asyncio.gather(*(_one(s) for s in wanted))

    results: dict[str, list[Paper]] = {}
    errors: dict[str, str] = {}
    for name, res in pairs:
        if isinstance(res, Exception):
            errors[name] = f"{type(res).__name__}: {res}"
        else:
            results[name] = res

    if not results:
        return {"query": query, "sources_queried": wanted,
                "errors": errors, "count": 0, "results": []}

    fused = fuse(results)
    return {
        "query": query,
        "sources_queried": sorted(results),
        "errors": errors,
        "total_merged": len(fused),
        "count": min(len(fused), max_results),
        "results": [_merged_hit(g) for g in fused[:max_results]],
    }


@mcp.tool(description="Fetch one paper by id, with full abstract and PDF link.")
async def get_paper(paper_id: str, source: str = DEFAULT_SOURCE) -> dict:
    """Fetch a single paper's full record by its source id (e.g. 2401.01234)."""
    src = get_source(source)
    try:
        paper = await src.fetch(paper_id)
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "source": source, "error": f"upstream error: {exc}"}
    if paper is None:
        return {"paper_id": paper_id, "source": source, "error": "not found"}
    return paper.to_dict()


@mcp.tool(description="Find papers by a specific author, newest first.")
async def search_by_author(
    author: str, source: str = DEFAULT_SOURCE, max_results: int = 10, start: int = 0
) -> dict:
    """Search papers by author name (e.g. 'Yann LeCun'), sorted newest first."""
    src = get_source(source)
    try:
        papers = await src.search_by_author(
            author, max_results=max_results, start=start
        )
    except httpx.HTTPError as exc:
        return {"author": author, "source": source, "error": f"upstream error: {exc}"}
    return {
        "author": author,
        "source": source,
        "count": len(papers),
        "results": [_hit(p) for p in papers],
    }


@mcp.tool(description="List the latest papers in a subject category, newest first.")
async def list_recent(
    category: str, source: str = DEFAULT_SOURCE, max_results: int = 10, start: int = 0
) -> dict:
    """Latest papers in a category code (e.g. 'cs.CL'); see list_categories."""
    src = get_source(source)
    try:
        papers = await src.recent(category, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"category": category, "source": source, "error": f"upstream error: {exc}"}
    return {
        "category": category,
        "source": source,
        "count": len(papers),
        "results": [_hit(p) for p in papers],
    }


@mcp.tool(description="List common subject category codes for filtering/recent.")
def list_categories(source: str = DEFAULT_SOURCE) -> dict:
    """Common category codes usable in search (cat:CODE) and list_recent."""
    src = get_source(source)
    return {"source": source, "categories": src.categories()}


@mcp.tool(description="Read a paper's full text. format='markdown' (default, "
          "body with formulas as $LaTeX$), 'html' (raw LaTeXML HTML), or "
          "'latex' (the original LaTeX manuscript from the e-print source). "
          "arXiv only; id like 2401.01234.")
async def read_paper(
    paper_id: str, format: str = "markdown", source: str = "arxiv"
) -> dict:
    """Fetch the full text of a paper (not just the abstract).

    Args:
        paper_id: Source id, e.g. arXiv '1706.03762'.
        format: 'markdown' | 'html' | 'latex'.
        source: Corpus (currently only 'arxiv' supports full text).
    """
    src = get_source(source)
    reader = getattr(src, "read_paper", None)
    if reader is None:
        return {"paper_id": paper_id, "source": source,
                "error": f"source '{source}' does not support full-text reading"}
    try:
        return await reader(paper_id, fmt=format)
    except ValueError as exc:
        return {"paper_id": paper_id, "source": source, "error": str(exc)}
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "source": source, "error": f"upstream error: {exc}"}


@mcp.tool(description="List available paper corpora.")
def list_paper_sources() -> dict:
    return {"sources": list_sources(), "default": DEFAULT_SOURCE, "version": __version__}


# ---------------------------------------------------------------------------
# Formula / table image recognition (free companion to the paper pipeline).
# When an agent has a cropped equation or table image — e.g. a figure from a
# paper it is reading — these turn the raster back into LaTeX without needing
# its own vision model. Backed by the co-located recognize service.
# Image input follows the de-facto MCP OCR convention: image_url OR image_base64.
# ---------------------------------------------------------------------------

@mcp.tool(description="Recognize a math formula from an image and return LaTeX. "
          "Provide image_url (downloaded server-side) OR image_base64. model: "
          "deepseek-ocr (default), paddleocr-vl, or texify. Returns "
          "{latex, model, elapsed_ms}.")
async def recognize_formula(
    image_url: str = "", image_base64: str = "", model: str = "deepseek-ocr"
) -> dict:
    """Image → LaTeX for a single formula. Give image_url or image_base64."""
    try:
        return await recognize.recognize_formula(
            image_url=image_url or None,
            image_base64=image_base64 or None,
            model=model,
        )
    except recognize.RecognizeError as exc:
        return {"error": str(exc)}


@mcp.tool(description="Recognize a table from an image and return LaTeX tabular "
          "code. Provide image_url OR image_base64. model: deepseek-ocr "
          "(default), paddleocr-vl, or texify. Returns {latex, model, "
          "elapsed_ms}.")
async def recognize_table(
    image_url: str = "", image_base64: str = "", model: str = "deepseek-ocr"
) -> dict:
    """Image → LaTeX tabular for a table. Give image_url or image_base64."""
    try:
        return await recognize.recognize_table(
            image_url=image_url or None,
            image_base64=image_base64 or None,
            model=model,
        )
    except recognize.RecognizeError as exc:
        return {"error": str(exc)}


@mcp.tool(description="List the OCR models available for recognize_formula / "
          "recognize_table.")
def list_ocr_models() -> dict:
    return {"models": recognize.list_models(), "default": "deepseek-ocr"}


# ---------------------------------------------------------------------------
# Semantic Scholar capabilities (the full S2 API surface). These wrap the
# Academic Graph / Recommendations / Datasets endpoints that arXiv has no
# equivalent for. They all hit the registered `semanticscholar` source.
# ---------------------------------------------------------------------------

def _s2():
    return get_source("semanticscholar")


def _s2_paper(raw: dict) -> dict:
    """Compact a raw S2 paper object, abstract truncated to save tokens."""
    if not raw:
        return {}
    ext = raw.get("externalIds") or {}
    oa = raw.get("openAccessPdf") or {}
    authors = [a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")]
    out = {
        "paperId": raw.get("paperId"),
        "title": raw.get("title"),
        "year": raw.get("year"),
        "venue": raw.get("venue"),
        "authors": authors[:10] + (["et al."] if len(authors) > 10 else []),
        "citationCount": raw.get("citationCount"),
        "influentialCitationCount": raw.get("influentialCitationCount"),
        "fieldsOfStudy": raw.get("fieldsOfStudy"),
        "url": raw.get("url"),
    }
    if ext.get("DOI"):
        out["doi"] = ext["DOI"]
    if ext.get("ArXiv"):
        out["arxiv"] = ext["ArXiv"]
    if oa.get("url"):
        out["openAccessPdf"] = oa["url"]
    if raw.get("matchScore") is not None:
        out["matchScore"] = raw["matchScore"]
    abstract = raw.get("abstract")
    if abstract:
        out["abstract_preview"] = (
            abstract[:_ABSTRACT_PREVIEW].rstrip()
            + ("…" if len(abstract) > _ABSTRACT_PREVIEW else "")
        )
    return {k: v for k, v in out.items() if v is not None}


def _s2_author(raw: dict) -> dict:
    if not raw:
        return {}
    out = {
        "authorId": raw.get("authorId"),
        "name": raw.get("name"),
        "affiliations": raw.get("affiliations"),
        "homepage": raw.get("homepage"),
        "paperCount": raw.get("paperCount"),
        "citationCount": raw.get("citationCount"),
        "hIndex": raw.get("hIndex"),
        "url": raw.get("url"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _edge_row(raw: dict, paper_key: str) -> dict:
    """Flatten a citation/reference edge into one compact row."""
    row = _s2_paper(raw.get(paper_key) or {})
    if raw.get("isInfluential"):
        row["isInfluential"] = True
    if raw.get("intents"):
        row["intents"] = raw["intents"]
    return row


@mcp.tool(description="Semantic Scholar: papers that CITE this one (forward "
          "citation graph). id accepts S2 id / DOI: / ARXIV: / CorpusId:.")
async def get_paper_citations(paper_id: str, max_results: int = 10, start: int = 0) -> dict:
    try:
        data = await _s2().paper_citations(paper_id, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "error": f"upstream error: {exc}"}
    rows = [_edge_row(it, "citingPaper") for it in (data.get("data") or [])]
    return {"paper_id": paper_id, "count": len(rows), "next": data.get("next"),
            "citations": rows}


@mcp.tool(description="Semantic Scholar: papers this one REFERENCES (its "
          "bibliography). id accepts S2 id / DOI: / ARXIV: / CorpusId:.")
async def get_paper_references(paper_id: str, max_results: int = 10, start: int = 0) -> dict:
    try:
        data = await _s2().paper_references(paper_id, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "error": f"upstream error: {exc}"}
    rows = [_edge_row(it, "citedPaper") for it in (data.get("data") or [])]
    return {"paper_id": paper_id, "count": len(rows), "next": data.get("next"),
            "references": rows}


@mcp.tool(description="Semantic Scholar: the authors of a paper (with "
          "h-index, paper/citation counts).")
async def get_paper_authors(paper_id: str, max_results: int = 100, start: int = 0) -> dict:
    try:
        data = await _s2().paper_authors(paper_id, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "error": f"upstream error: {exc}"}
    rows = [_s2_author(a) for a in (data.get("data") or [])]
    return {"paper_id": paper_id, "count": len(rows), "authors": rows}


@mcp.tool(description="Semantic Scholar: find the single paper whose title "
          "best matches the given text (exact-match lookup).")
async def match_paper_title(title: str) -> dict:
    try:
        data = await _s2().match_title(title)
    except httpx.HTTPError as exc:
        return {"title": title, "error": f"upstream error: {exc}"}
    items = data.get("data") or []
    return {"title": title, "match": _s2_paper(items[0]) if items else None}


@mcp.tool(description="Semantic Scholar: autocomplete paper titles for a "
          "partial query (fast type-ahead).")
async def autocomplete_papers(query: str) -> dict:
    try:
        data = await _s2().autocomplete(query)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    return {"query": query, "matches": data.get("matches") or []}


@mcp.tool(description="Semantic Scholar: bulk paper search (up to 1000 hits, "
          "sortable e.g. 'citationCount:desc' or 'publicationDate:desc', with "
          "a continuation token). Filters: fields_of_study, year (e.g. "
          "'2020-2024'), venue, publication_types, open_access_pdf.")
async def search_papers_bulk(
    query: str,
    sort: str = "",
    fields_of_study: str = "",
    year: str = "",
    venue: str = "",
    publication_types: str = "",
    open_access_pdf: bool = False,
    token: str = "",
    max_results: int = 100,
) -> dict:
    try:
        data = await _s2().search_bulk(
            query,
            sort=sort or None,
            fields_of_study=fields_of_study or None,
            year=year or None,
            venue=venue or None,
            publication_types=publication_types or None,
            open_access_pdf=open_access_pdf,
            token=token or None,
            max_results=max_results,
        )
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    rows = [_s2_paper(p) for p in (data.get("data") or [])]
    return {"query": query, "total": data.get("total"), "token": data.get("token"),
            "count": len(rows), "results": rows}


@mcp.tool(description="Semantic Scholar: fetch many papers at once by id "
          "(S2/DOI:/ARXIV:/CorpusId:), up to ~500 per call.")
async def get_papers_batch(ids: list[str]) -> dict:
    try:
        data = await _s2().papers_batch(ids)
    except httpx.HTTPError as exc:
        return {"error": f"upstream error: {exc}"}
    rows = [_s2_paper(p) for p in data if p]
    return {"count": len(rows), "papers": rows}


@mcp.tool(description="Semantic Scholar: search for authors by name; returns "
          "profiles with h-index and paper/citation counts.")
async def search_authors(query: str, max_results: int = 10, start: int = 0) -> dict:
    try:
        data = await _s2().author_search(query, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    rows = [_s2_author(a) for a in (data.get("data") or [])]
    return {"query": query, "total": data.get("total"), "count": len(rows),
            "authors": rows}


@mcp.tool(description="Semantic Scholar: a single author's profile by id.")
async def get_author(author_id: str) -> dict:
    try:
        data = await _s2().author(author_id)
    except httpx.HTTPError as exc:
        return {"author_id": author_id, "error": f"upstream error: {exc}"}
    return _s2_author(data) or {"author_id": author_id, "error": "not found"}


@mcp.tool(description="Semantic Scholar: all papers by a given author id, "
          "newest first.")
async def get_author_papers(author_id: str, max_results: int = 20, start: int = 0) -> dict:
    try:
        data = await _s2().author_papers(author_id, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"author_id": author_id, "error": f"upstream error: {exc}"}
    rows = [_s2_paper(p) for p in (data.get("data") or [])]
    rows.sort(key=lambda r: r.get("year") or 0, reverse=True)
    return {"author_id": author_id, "count": len(rows), "next": data.get("next"),
            "papers": rows}


@mcp.tool(description="Semantic Scholar: fetch many authors at once by id.")
async def get_authors_batch(ids: list[str]) -> dict:
    try:
        data = await _s2().authors_batch(ids)
    except httpx.HTTPError as exc:
        return {"error": f"upstream error: {exc}"}
    rows = [_s2_author(a) for a in data if a]
    return {"count": len(rows), "authors": rows}


@mcp.tool(description="Semantic Scholar: search INSIDE paper full text and "
          "return matching text snippets (not just titles/abstracts).")
async def search_snippets(query: str, max_results: int = 10) -> dict:
    try:
        data = await _s2().snippet_search(query, max_results=max_results)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    out = []
    for it in (data.get("data") or []):
        snip = it.get("snippet") or {}
        paper = it.get("paper") or {}
        out.append({
            "score": it.get("score"),
            "text": snip.get("text"),
            "section": (snip.get("snippetKind") or snip.get("section")),
            "paperId": paper.get("corpusId") or paper.get("paperId"),
            "title": paper.get("title"),
        })
    return {"query": query, "count": len(out), "snippets": out}


@mcp.tool(description="Semantic Scholar: recommend papers similar to one paper. "
          "pool='recent' (last open corpus) or 'all-cs' (all of CS). If the "
          "'recent' pool yields nothing (common for older papers), it "
          "automatically retries the 'all-cs' pool.")
async def recommend_papers_for_paper(
    paper_id: str, max_results: int = 10, pool: str = "recent"
) -> dict:
    try:
        data = await _s2().recommend_for_paper(paper_id, max_results=max_results, pool=pool)
        rows = [_s2_paper(p) for p in (data.get("recommendedPapers") or [])]
        pool_used = pool
        if not rows and pool != "all-cs":
            data = await _s2().recommend_for_paper(
                paper_id, max_results=max_results, pool="all-cs"
            )
            rows = [_s2_paper(p) for p in (data.get("recommendedPapers") or [])]
            pool_used = "all-cs"
    except httpx.HTTPError as exc:
        return {"paper_id": paper_id, "error": f"upstream error: {exc}"}
    return {"paper_id": paper_id, "pool_used": pool_used, "count": len(rows),
            "recommendations": rows}


@mcp.tool(description="Semantic Scholar: recommend papers from positive (and "
          "optional negative) example paper ids.")
async def recommend_papers_from_examples(
    positive_ids: list[str], negative_ids: list[str] | None = None, max_results: int = 10
) -> dict:
    try:
        data = await _s2().recommend_from_examples(
            positive_ids, negative_ids, max_results=max_results
        )
    except httpx.HTTPError as exc:
        return {"error": f"upstream error: {exc}"}
    rows = [_s2_paper(p) for p in (data.get("recommendedPapers") or [])]
    return {"count": len(rows), "recommendations": rows}


@mcp.tool(description="Semantic Scholar Datasets: list all available release "
          "ids (dated snapshots of the full corpus).")
async def list_dataset_releases() -> dict:
    try:
        releases = await _s2().dataset_releases()
    except httpx.HTTPError as exc:
        return {"error": f"upstream error: {exc}"}
    return {"count": len(releases), "latest": releases[-1] if releases else None,
            "releases": releases[-30:]}


@mcp.tool(description="Semantic Scholar Datasets: which datasets a release "
          "contains (papers, abstracts, citations, embeddings, s2orc, tldrs…). "
          "release_id defaults to 'latest'.")
async def get_dataset_release(release_id: str = "latest") -> dict:
    try:
        data = await _s2().dataset_release(release_id)
    except httpx.HTTPError as exc:
        return {"release_id": release_id, "error": f"upstream error: {exc}"}
    datasets = [
        {"name": d.get("name"), "description": (d.get("description") or "")[:200]}
        for d in (data.get("datasets") or [])
    ]
    return {"release_id": data.get("release_id") or release_id,
            "count": len(datasets), "datasets": datasets}


@mcp.tool(description="Semantic Scholar Datasets: get download links (presigned "
          "URLs) for one dataset in a release. Needs the API key.")
async def get_dataset_download_links(dataset_name: str, release_id: str = "latest") -> dict:
    try:
        data = await _s2().dataset_download(dataset_name, release_id=release_id)
    except httpx.HTTPError as exc:
        return {"dataset": dataset_name, "error": f"upstream error: {exc}"}
    files = data.get("files") or []
    return {"release_id": release_id, "dataset": data.get("name") or dataset_name,
            "description": data.get("description"), "file_count": len(files),
            "files": files[:20]}


@mcp.tool(description="Semantic Scholar Datasets: incremental diff (added/"
          "updated/deleted) for a dataset between two releases. Needs the key.")
async def get_dataset_diffs(
    dataset_name: str, start_release: str, end_release: str = "latest"
) -> dict:
    try:
        data = await _s2().dataset_diffs(
            dataset_name, start_release=start_release, end_release=end_release
        )
    except httpx.HTTPError as exc:
        return {"dataset": dataset_name, "error": f"upstream error: {exc}"}
    return data


# ---------------------------------------------------------------------------
# OpenAlex capabilities (the free CC0 successor to Microsoft Academic Graph:
# ~316M works across every field, citation graph, authors, institutions,
# topics and influence metrics). No API key — uses the polite pool.
# ---------------------------------------------------------------------------

def _oa():
    return get_source("openalex")


@mcp.tool(description="OpenAlex: fetch one work's full record (316M-work, "
          "all-field corpus). id accepts OpenAlex Wxxxx, a DOI, or an arXiv id.")
async def get_openalex_work(work_id: str) -> dict:
    try:
        return await _oa().get_work(work_id)
    except httpx.HTTPError as exc:
        return {"work_id": work_id, "error": f"upstream error: {exc}"}


@mcp.tool(description="OpenAlex: papers that CITE this work (forward citation "
          "graph), most-cited first.")
async def get_openalex_citations(work_id: str, max_results: int = 10, start: int = 0) -> dict:
    try:
        d = await _oa().work_citations(work_id, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"work_id": work_id, "error": f"upstream error: {exc}"}
    return {"work_id": work_id, "total": d.get("total"),
            "count": len(d.get("results") or []), "citations": d.get("results")}


@mcp.tool(description="OpenAlex: the works this one REFERENCES (its bibliography).")
async def get_openalex_references(work_id: str, max_results: int = 25) -> dict:
    try:
        d = await _oa().work_references(work_id, max_results=max_results)
    except httpx.HTTPError as exc:
        return {"work_id": work_id, "error": f"upstream error: {exc}"}
    return {"work_id": work_id, "total": d.get("total"),
            "count": len(d.get("results") or []), "references": d.get("results")}


@mcp.tool(description="OpenAlex: search authors; returns profiles with h-index, "
          "i10-index, works/citation counts and institutions.")
async def search_openalex_authors(query: str, max_results: int = 10, start: int = 0) -> dict:
    try:
        d = await _oa().search_authors(query, max_results=max_results, start=start)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    return {"query": query, "total": d.get("total"),
            "count": len(d.get("results") or []), "authors": d.get("results")}


@mcp.tool(description="OpenAlex: search institutions (universities, labs) with "
          "ROR id, country, works/citation counts.")
async def search_openalex_institutions(query: str, max_results: int = 10) -> dict:
    try:
        d = await _oa().search_institutions(query, max_results=max_results)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    return {"query": query, "total": d.get("total"),
            "count": len(d.get("results") or []), "institutions": d.get("results")}


@mcp.tool(description="OpenAlex: advanced filtered work search. Filters: "
          "from_year, to_year, is_oa (open access only), min_citations, "
          "institution_id. sort_by: relevance|newest|cited.")
async def search_openalex_works(
    query: str = "",
    from_year: int = 0,
    to_year: int = 0,
    is_oa: bool = False,
    min_citations: int = 0,
    institution_id: str = "",
    sort_by: str = "relevance",
    max_results: int = 25,
) -> dict:
    try:
        d = await _oa().search_filtered(
            query,
            from_year=from_year or None,
            to_year=to_year or None,
            is_oa=True if is_oa else None,
            min_citations=min_citations or None,
            institution_id=institution_id or None,
            sort_by=sort_by,
            max_results=max_results,
        )
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}
    return {"query": query, "total": d.get("total"),
            "count": len(d.get("results") or []), "results": d.get("results")}


@mcp.tool(description="OpenAlex: publication-trend analytics for a query — "
          "counts grouped by year (default), or by 'institutions.id', "
          "'authorships.author.id', 'open_access.is_oa', 'type', 'language'. "
          "Returns aggregate counts only (cheap, no rows).")
async def get_openalex_trends(query: str, group_by: str = "publication_year") -> dict:
    try:
        return await _oa().trends(query, group_by=group_by)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}


@mcp.tool(description="OpenAlex: search the topic taxonomy (~4500 topics) to "
          "find the right subject term for filtering or recent-work queries.")
async def list_openalex_topics(query: str, max_results: int = 15) -> dict:
    try:
        return await _oa().list_topics(query, max_results=max_results)
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"upstream error: {exc}"}


def main() -> None:
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(_RateLimitMiddleware, max_per_hour=MCP_MAX_PER_HOUR)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
