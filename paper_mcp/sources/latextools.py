"""LaTeX-tools companion services — lint & PDF extraction.

Thin MCP clients over two services that already run on the latex-tools host and
back the latex-tools.online web tools (/lint, /pdf):

  * **lint**  — LaTeX linter on the recognize service (127.0.0.1:8502/api/lint):
    POST {code} -> {errors, fixed_code, summary_*}. Synchronous.
  * **pdf-extract** — MinerU-backed PDF -> structured text (127.0.0.1:8504),
    an async job API (submit -> poll -> fetch). We hide the task lifecycle and
    return the finished result in one call, so an agent just gets text back.

Both are free companions to the paper pipeline; the heavy lifting lives in the
co-located services, here we only marshal requests. URL inputs are downloaded
server-side with SSRF guards (mirrors recognize.py) so an agent cannot reach the
host's private network through us.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import urllib.parse

import httpx

_LINT_URL = os.getenv("LINT_URL", "http://127.0.0.1:8502/api")
_PDF_URL = os.getenv("PDF_EXTRACT_URL", "http://127.0.0.1:8504")
_USER_AGENT = "paper-mcp/0.5 (+https://latex-tools.online/mcp)"

_MAX_PDF_BYTES = 50 * 1024 * 1024  # cap a downloaded PDF
_PDF_POLL_INTERVAL = 2.0
_PDF_POLL_TIMEOUT = 300.0  # MinerU can be slow on big PDFs


class LatexToolsError(Exception):
    """Upstream lint / pdf-extract failure, surfaced as a tool error."""


# --- SSRF guard (same policy as recognize.py) --------------------------------

def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


async def _download(url: str, cap: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LatexToolsError("url must be http(s) with a host")
    if not await asyncio.get_running_loop().run_in_executor(
            None, _is_public_host, parsed.hostname):
        raise LatexToolsError("url host is not a public address")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True,
                                 headers={"User-Agent": _USER_AGENT}) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise LatexToolsError(f"download failed: HTTP {resp.status_code}")
        data = resp.content
        if len(data) > cap:
            raise LatexToolsError(f"file exceeds {cap // (1024*1024)} MB cap")
        return data


# --- lint --------------------------------------------------------------------

async def lint_latex(code: str) -> dict:
    """Lint a LaTeX snippet; return errors + an auto-fixed version.

    Returns ``{errors, fixed_code, summary_en, summary_zh, elapsed_ms}``.
    """
    if not (code or "").strip():
        raise LatexToolsError("code must not be empty")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{_LINT_URL}/lint", json={"code": code})
        except httpx.HTTPError as exc:
            raise LatexToolsError(f"lint service unreachable: {exc}") from exc
    if resp.status_code != 200:
        try:
            msg = resp.json().get("detail") or resp.text
        except Exception:
            msg = resp.text
        raise LatexToolsError((msg or f"HTTP {resp.status_code}")[:300])
    out = resp.json()
    return {
        "errors": out.get("errors", []),
        "fixed_code": out.get("fixed_code"),
        "summary_en": out.get("summary_en"),
        "summary_zh": out.get("summary_zh"),
        "elapsed_ms": out.get("elapsed_ms"),
    }


# --- pdf extraction (async job, collapsed into one call) ---------------------

async def _pdf_submit(client: httpx.AsyncClient, pdf: bytes,
                      formula: bool, table: bool) -> dict:
    files = {"file": ("document.pdf", pdf, "application/pdf")}
    data = {"formula": str(bool(formula)).lower(), "table": str(bool(table)).lower()}
    resp = await client.post(f"{_PDF_URL}/extract", files=files, data=data)
    if resp.status_code == 429:
        raise LatexToolsError("pdf-extract rate limit exceeded")
    if resp.status_code == 413:
        raise LatexToolsError("PDF too large for the service")
    if resp.status_code != 200:
        try:
            msg = resp.json().get("detail") or resp.text
        except Exception:
            msg = resp.text
        raise LatexToolsError((msg or f"HTTP {resp.status_code}")[:300])
    return resp.json()


async def extract_pdf(
    pdf_url: str | None = None,
    pdf_base64: str | None = None,
    formula: bool = True,
    table: bool = True,
) -> dict:
    """Extract a PDF to clean Markdown/LaTeX text via MinerU (submit + poll).

    Provide ``pdf_url`` (downloaded server-side, SSRF-guarded) or ``pdf_base64``.
    ``formula`` / ``table`` toggle math / table reconstruction. Submits the job
    and polls to completion, returning ``{task_id, cached, content, chars}``
    where ``content`` is the UTF-8 text extraction. Note: MinerU is GPU-heavy,
    so a fresh (uncached) large PDF can take minutes.
    """
    import base64 as _b64
    if pdf_base64:
        try:
            pdf = _b64.b64decode(pdf_base64, validate=True)
        except Exception as exc:
            raise LatexToolsError("pdf_base64 is not valid base64") from exc
    elif pdf_url:
        pdf = await _download(pdf_url, _MAX_PDF_BYTES)
    else:
        raise LatexToolsError("provide pdf_url or pdf_base64")
    if len(pdf) > _MAX_PDF_BYTES:
        raise LatexToolsError(f"PDF exceeds {_MAX_PDF_BYTES // (1024*1024)} MB cap")

    async with httpx.AsyncClient(timeout=90.0) as client:
        sub = await _pdf_submit(client, pdf, formula, table)
        task_id = sub.get("task_id")
        if not task_id:
            raise LatexToolsError("pdf-extract did not return a task_id")

        # Poll until the job leaves the running/pending state.
        deadline = asyncio.get_running_loop().time() + _PDF_POLL_TIMEOUT
        status = sub.get("status") or "pending"
        while status in ("pending", "running", "processing", "queued"):
            if asyncio.get_running_loop().time() > deadline:
                raise LatexToolsError(f"pdf-extract timed out (task {task_id})")
            await asyncio.sleep(_PDF_POLL_INTERVAL)
            try:
                r = await client.get(f"{_PDF_URL}/status/{task_id}")
                status = (r.json() or {}).get("status", status) if r.status_code == 200 else status
            except httpx.HTTPError:
                pass  # transient; keep polling until the deadline

        if status not in ("done", "success", "completed", "finished"):
            raise LatexToolsError(f"pdf-extract failed (task {task_id}, status {status})")

        # /result is a zip attachment (markdown + images); /latex is the clean
        # UTF-8 text extraction. Agents want text, so read /latex.
        res = await client.get(f"{_PDF_URL}/latex/{task_id}")
        if res.status_code != 200:
            raise LatexToolsError(f"result fetch failed: HTTP {res.status_code}")
        content = res.text

        out = {
            "task_id": task_id,
            "cached": bool(sub.get("cached")),
            "content": content,
            "chars": len(content),
        }
        return out
