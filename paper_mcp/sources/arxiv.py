"""arXiv source — a thin client over the public arXiv Atom API.

This is a pure pass-through wrapper (no corpus of our own, no index): it
forwards queries to https://export.arxiv.org/api/query and normalizes the
Atom feed into `Paper`. It is the baseline skeleton the value layers plug
into later, not a moat in itself.
"""
from __future__ import annotations

import asyncio
import html as _htmllib
import io
import re
import tarfile
import xml.etree.ElementTree as ET

import httpx

from ..models import Paper

ARXIV_API = "https://export.arxiv.org/api/query"
# arXiv asks API clients to identify themselves and to rate-limit politely.
_USER_AGENT = "paper-mcp/0.1 (+https://latex-tools.online/mcp)"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# Full-text sources. The LaTeXML HTML keeps the original LaTeX of every formula
# in the <math alttext="..."> attribute, so we recover `$...$` losslessly.
_HTML_URL = "https://arxiv.org/html/{id}"
_AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{id}"
_EPRINT_URL = "https://arxiv.org/e-print/{id}"
_MAX_SOURCE_BYTES = 80 * 1024 * 1024  # cap e-print downloads


def _strip_tags(fragment: str) -> str:
    """Drop remaining HTML tags and unescape entities to plain text."""
    # Drop scripts/styles and LaTeXML tag/error spans entirely.
    fragment = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S)
    text = re.sub(r"<[^>]+>", "", fragment)
    return _htmllib.unescape(text)

_SORT_MAP = {
    "relevance": "relevance",
    "newest": "submittedDate",
    "updated": "lastUpdatedDate",
}

# Agent-relevant subset of the arXiv taxonomy (any valid code works in cat:).
_CATEGORIES = (
    {"id": "cs.AI", "name": "Artificial Intelligence"},
    {"id": "cs.CL", "name": "Computation and Language (NLP)"},
    {"id": "cs.LG", "name": "Machine Learning"},
    {"id": "cs.CV", "name": "Computer Vision and Pattern Recognition"},
    {"id": "cs.NE", "name": "Neural and Evolutionary Computing"},
    {"id": "cs.RO", "name": "Robotics"},
    {"id": "cs.IR", "name": "Information Retrieval"},
    {"id": "cs.CR", "name": "Cryptography and Security"},
    {"id": "cs.DC", "name": "Distributed, Parallel, and Cluster Computing"},
    {"id": "cs.DS", "name": "Data Structures and Algorithms"},
    {"id": "cs.SE", "name": "Software Engineering"},
    {"id": "cs.HC", "name": "Human-Computer Interaction"},
    {"id": "cs.SY", "name": "Systems and Control"},
    {"id": "stat.ML", "name": "Machine Learning (Statistics)"},
    {"id": "stat.ME", "name": "Methodology (Statistics)"},
    {"id": "eess.AS", "name": "Audio and Speech Processing"},
    {"id": "eess.IV", "name": "Image and Video Processing"},
    {"id": "eess.SP", "name": "Signal Processing"},
    {"id": "math.OC", "name": "Optimization and Control"},
    {"id": "math.NA", "name": "Numerical Analysis"},
    {"id": "math.PR", "name": "Probability"},
    {"id": "math.ST", "name": "Statistics Theory"},
    {"id": "q-bio.QM", "name": "Quantitative Methods (Biology)"},
    {"id": "q-bio.NC", "name": "Neurons and Cognition"},
    {"id": "q-fin.CP", "name": "Computational Finance"},
    {"id": "q-fin.TR", "name": "Trading and Market Microstructure"},
    {"id": "econ.EM", "name": "Econometrics"},
    {"id": "physics.comp-ph", "name": "Computational Physics"},
)



class ArxivSource:
    name = "arxiv"

    def __init__(
        self,
        *,
        base_url: str = ARXIV_API,
        timeout: float = 20.0,
        max_retries: int = 4,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        params = {
            "search_query": query,
            "start": max(0, start),
            "max_results": max(1, min(max_results, 50)),
            "sortBy": _SORT_MAP.get(sort_by, "relevance"),
            "sortOrder": "descending",
        }
        return self._parse_feed(await self._get(params))

    async def fetch(self, paper_id: str) -> Paper | None:
        clean = paper_id.strip().rsplit("/", 1)[-1]
        params = {"id_list": clean, "max_results": 1}
        papers = self._parse_feed(await self._get(params))
        return papers[0] if papers else None

    async def search_by_author(
        self, author: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]:
        name = author.strip()
        if not name:
            raise ValueError("author must not be empty")
        # arXiv au: matches author names; quote multi-word names as a phrase.
        term = f'au:"{name}"' if " " in name else f"au:{name}"
        return await self.search(
            term, max_results=max_results, start=start, sort_by="newest"
        )

    async def recent(
        self, category: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]:
        cat = category.strip()
        if not cat:
            raise ValueError("category must not be empty")
        return await self.search(
            f"cat:{cat}", max_results=max_results, start=start, sort_by="newest"
        )

    def categories(self) -> list[dict]:
        """Common arXiv category codes for use in `cat:` queries.

        Not exhaustive — any valid arXiv category code works in search; this
        is the agent-relevant subset across CS/ML/EE/math/quant/bio.
        """
        return list(_CATEGORIES)


    async def read_paper(self, paper_id: str, *, fmt: str = "markdown") -> dict:
        """Return a paper's full text.

        fmt:
          * ``markdown`` (default) — readable body with formulas as ``$LaTeX$``,
            parsed from arXiv's LaTeXML HTML (ar5iv fallback).
          * ``html`` — the raw LaTeXML HTML page.
          * ``latex`` — the original LaTeX source (largest ``.tex`` from the
            e-print tarball), i.e. the author's manuscript.
        """
        pid = paper_id.strip().rsplit("/", 1)[-1]
        if not pid:
            raise ValueError("paper_id must not be empty")
        fmt = (fmt or "markdown").strip().lower()
        if fmt not in ("markdown", "html", "latex"):
            raise ValueError("fmt must be 'markdown', 'html' or 'latex'")

        if fmt == "latex":
            name, text = await self._fetch_latex_source(pid)
            return {"id": pid, "format": "latex", "source_file": name,
                    "length": len(text), "content": text}

        html, src = await self._fetch_html(pid)
        if html is None:
            return {"id": pid, "format": fmt, "error": "no HTML full text "
                    "available for this paper (older papers may be scan-only)"}
        if fmt == "html":
            return {"id": pid, "format": "html", "source": src,
                    "length": len(html), "content": html}
        md, title = self._html_to_markdown(html)
        return {"id": pid, "format": "markdown", "source": src,
                "title": title, "length": len(md), "content": md}

    async def _fetch_html(self, pid: str) -> tuple[str | None, str | None]:
        """LaTeXML HTML from arxiv.org/html, falling back to ar5iv."""
        for url, tag in ((_HTML_URL, "arxiv-html"), (_AR5IV_URL, "ar5iv")):
            try:
                text = await self._fetch_text(url.format(id=pid))
            except httpx.HTTPError:
                continue
            if text and "<math" in text or (text and "ltx_p" in text):
                return text, tag
            if text and len(text) > 2000:
                return text, tag
        return None, None

    async def _fetch_latex_source(self, pid: str) -> tuple[str, str]:
        """Download the e-print tarball, return (name, text) of the main .tex."""
        raw = await self._fetch_bytes(_EPRINT_URL.format(id=pid))
        if raw[:2] == b"\x1f\x8b":  # gzip — usually a tar.gz
            try:
                tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
            except tarfile.TarError:
                # single gzipped .tex (not a tar)
                import gzip
                return f"{pid}.tex", gzip.decompress(raw).decode("utf-8", "ignore")
            texs = [m for m in tf.getmembers()
                    if m.isfile() and m.name.endswith(".tex")]
            if not texs:
                raise ValueError("no .tex file in e-print source")
            # Heuristic: the main file is the one with \documentclass, else the
            # largest .tex.
            best = None
            for m in texs:
                f = tf.extractfile(m)
                body = f.read().decode("utf-8", "ignore") if f else ""
                if "\\documentclass" in body:
                    return m.name, body
                if best is None or len(body) > len(best[1]):
                    best = (m.name, body)
            return best
        # not gzip: maybe a bare .tex
        return f"{pid}.tex", raw.decode("utf-8", "ignore")

    @staticmethod
    def _html_to_markdown(html: str) -> tuple[str, str]:
        """Convert arXiv LaTeXML HTML to markdown, formulas as ``$LaTeX$``.

        Relies on the LaTeXML invariant that every <math> carries the original
        TeX in its ``alttext`` attribute; display math (ltx_equation) becomes a
        ``$$...$$`` block, inline math becomes ``$...$``.
        """
        title = ""
        mt = re.search(r'<h1[^>]*ltx_title_document[^>]*>(.*?)</h1>', html, re.S)
        if not mt:
            mt = re.search(r'<h1[^>]*ltx_title[^>]*>(.*?)</h1>', html, re.S)
        if mt:
            title = re.sub(r"\s+", " ", _strip_tags(mt.group(1))).strip()

        # Body: prefer the <article>, else the whole doc.
        ma = re.search(r"<article\b[^>]*>(.*?)</article>", html, re.S)
        body = ma.group(1) if ma else html
        # Drop the document title heading; it is returned separately.
        body = re.sub(r'<h1[^>]*ltx_title_document[^>]*>.*?</h1>', "", body, flags=re.S)

        # Replace <math ... alttext="TeX"> with $TeX$ (or $$ $$ for display).
        def _math(m: re.Match) -> str:
            tag = m.group(0)
            alt = re.search(r'alttext="([^"]*)"', tag)
            tex = _htmllib.unescape(alt.group(1)).strip() if alt else ""
            if not tex:
                return ""
            display = 'display="block"' in tag or "ltx_equation" in tag
            return f"\n$$\n{tex}\n$$\n" if display else f"${tex}$"
        body = re.sub(r"<math\b.*?</math>", _math, body, flags=re.S)

        # Headings.
        for lvl, cls in ((2, "ltx_title_section"), (3, "ltx_title_subsection")):
            body = re.sub(
                rf'<h\d[^>]*{cls}[^>]*>(.*?)</h\d>',
                lambda m, p="#" * lvl: f"\n\n{p} {_strip_tags(m.group(1)).strip()}\n\n",
                body, flags=re.S)

        # Paragraphs and list items → text blocks.
        body = re.sub(r'<li\b[^>]*>(.*?)</li>',
                      lambda m: f"\n- {_strip_tags(m.group(1)).strip()}", body, flags=re.S)
        body = re.sub(r'<(p|div)\b[^>]*class="[^"]*ltx_(p|para)[^"]*"[^>]*>(.*?)</\1>',
                      lambda m: f"\n\n{_strip_tags(m.group(3)).strip()}\n\n", body, flags=re.S)

        text = _strip_tags(body)
        # Collapse whitespace-only lines and runs of blank lines.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, title


    async def _fetch_text(self, url: str) -> str:
        """GET a URL as text with the same retry/backoff policy as the API."""
        headers = {"User-Agent": _USER_AGENT}
        last_exc: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            for attempt in range(self._max_retries):
                try:
                    resp = await client.get(url)
                except httpx.TransportError as exc:
                    last_exc = exc
                    await asyncio.sleep(min(3.0 * (attempt + 1), 12.0))
                    continue
                if resp.status_code == 404:
                    return ""
                if resp.status_code in _RETRY_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"arXiv returned {resp.status_code}",
                        request=resp.request, response=resp)
                    await asyncio.sleep(min(3.0 * (attempt + 1), 12.0))
                    continue
                resp.raise_for_status()
                return resp.text
        assert last_exc is not None
        raise last_exc

    async def _fetch_bytes(self, url: str) -> bytes:
        """GET a URL as bytes (for the e-print tarball), capped in size."""
        headers = {"User-Agent": _USER_AGENT}
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            for attempt in range(self._max_retries):
                try:
                    resp = await client.get(url)
                except httpx.TransportError:
                    await asyncio.sleep(min(3.0 * (attempt + 1), 12.0))
                    continue
                if resp.status_code in _RETRY_STATUS:
                    await asyncio.sleep(min(3.0 * (attempt + 1), 12.0))
                    continue
                resp.raise_for_status()
                data = resp.content
                if len(data) > _MAX_SOURCE_BYTES:
                    raise ValueError("e-print source exceeds size cap")
                return data
        raise httpx.HTTPError("failed to download e-print source")


    async def _get(self, params: dict) -> str:
        headers = {"User-Agent": _USER_AGENT}
        last_exc: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True, headers=headers
        ) as client:
            for attempt in range(self._max_retries):
                backoff = min(3.0 * (attempt + 1), 12.0)
                try:
                    resp = await client.get(self._base_url, params=params)
                except httpx.TransportError as exc:  # timeouts, conn resets
                    last_exc = exc
                    await asyncio.sleep(backoff)
                    continue
                if resp.status_code in _RETRY_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"arXiv returned {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.text
        assert last_exc is not None
        raise last_exc

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        return [self._parse_entry(e) for e in root.findall(f"{_ATOM}entry")]

    def _parse_entry(self, entry: ET.Element) -> Paper:
        def text(tag: str) -> str:
            el = entry.find(tag)
            return " ".join(el.text.split()) if el is not None and el.text else ""

        raw_id = text(f"{_ATOM}id")  # http://arxiv.org/abs/2401.01234v1
        arxiv_id = raw_id.rsplit("/", 1)[-1]

        authors = [
            " ".join(name.text.split())
            for a in entry.findall(f"{_ATOM}author")
            if (name := a.find(f"{_ATOM}name")) is not None and name.text
        ]
        categories = [
            c.get("term", "")
            for c in entry.findall(f"{_ATOM}category")
            if c.get("term")
        ]
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return Paper(
            id=arxiv_id,
            source="arxiv",
            title=text(f"{_ATOM}title"),
            authors=authors,
            summary=text(f"{_ATOM}summary"),
            published=text(f"{_ATOM}published"),
            updated=text(f"{_ATOM}updated"),
            categories=categories,
            url=raw_id,
            pdf_url=pdf_url,
            doi=text(f"{_ARXIV}doi"),
            comment=text(f"{_ARXIV}comment"),
            journal_ref=text(f"{_ARXIV}journal_ref"),
        )
