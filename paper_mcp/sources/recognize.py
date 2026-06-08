"""Formula / table image recognition — a thin client over the latex-tools
``/api/recognize`` service that already runs on the same host.

This is a free companion capability for the paper pipeline: once an agent has
a figure or a cropped equation (e.g. from a paper it is reading) it can turn
the raster image back into LaTeX without needing its own vision model. The
heavy lifting (PaddleOCR-VL / DeepSeek-OCR / texify) lives in the existing
service on 127.0.0.1; here we only marshal the image and call it.

Image input mirrors the de-facto MCP OCR convention (see axel-belfort/ocr-extract
and peers): accept either a URL or base64. For a URL we download server-side
and forward the bytes — with SSRF guards so an agent cannot use us to reach the
host's private network.
"""
from __future__ import annotations

import asyncio
import base64
import ipaddress
import os
import socket
import urllib.parse

import httpx

# The recognize service is co-located; talk to it over loopback.
_RECOGNIZE_URL = os.getenv("RECOGNIZE_URL", "http://127.0.0.1:8502/api")
_USER_AGENT = "paper-mcp/0.2 (+https://latex-tools.online/mcp)"

# Models the upstream advertises via /api/health.
_MODELS = ("deepseek-ocr", "paddleocr-vl", "texify")
_DEFAULT_MODEL = "deepseek-ocr"

# Cap a downloaded image so a hostile URL can't exhaust memory.
_MAX_IMAGE_BYTES = 12 * 1024 * 1024


class RecognizeError(Exception):
    """A recognition request failed in a way worth showing the caller."""


def _is_public_host(host: str) -> bool:
    """Resolve host and reject loopback/private/link-local/reserved targets.

    Same posture as the directory crawler's SSRF guard: an image_url must point
    at a genuinely public address, so we cannot be used to probe 127.0.0.1,
    169.254.169.254 (cloud metadata) or the intranet.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return False
    return True


async def _download_image(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RecognizeError("image_url must be http(s)")
    if not parsed.hostname or not _is_public_host(parsed.hostname):
        raise RecognizeError("image_url host is not a public address")
    async with httpx.AsyncClient(
        timeout=25.0, follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        # Stream so we can stop early once we exceed the byte cap.
        try:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise RecognizeError(
                        f"could not fetch image_url (HTTP {resp.status_code})")
                ctype = resp.headers.get("content-type", "")
                if ctype and not ctype.startswith(("image/", "application/octet-stream")):
                    raise RecognizeError(f"image_url is not an image ({ctype})")
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_IMAGE_BYTES:
                        raise RecognizeError("image exceeds 12 MB limit")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise RecognizeError(f"could not fetch image_url: {exc}") from exc
    return b"".join(chunks)


def _norm_model(model: str | None) -> str:
    m = (model or _DEFAULT_MODEL).strip().lower()
    return m if m in _MODELS else _DEFAULT_MODEL


async def _recognize(
    path: str, image_bytes: bytes, model: str
) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(
                f"{_RECOGNIZE_URL}/{path}",
                params={"model": _norm_model(model)},
                data={"base64_image": b64},
            )
        except httpx.HTTPError as exc:
            raise RecognizeError(f"recognize service unreachable: {exc}") from exc
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error") or resp.text
        except Exception:
            msg = resp.text
        raise RecognizeError(msg[:300] or f"HTTP {resp.status_code}")
    return resp.json()


async def _resolve_image(image_url: str | None, image_base64: str | None) -> bytes:
    if image_base64:
        try:
            return base64.b64decode(image_base64, validate=True)
        except Exception as exc:
            raise RecognizeError("image_base64 is not valid base64") from exc
    if image_url:
        return await _download_image(image_url)
    raise RecognizeError("provide image_url or image_base64")


async def recognize_formula(
    image_url: str | None = None,
    image_base64: str | None = None,
    model: str | None = None,
) -> dict:
    """Recognize a math formula image and return LaTeX.

    Returns ``{latex, model, elapsed_ms}`` on success.
    """
    img = await _resolve_image(image_url, image_base64)
    out = await _recognize("recognize", img, model or _DEFAULT_MODEL)
    return {
        "latex": out.get("latex", ""),
        "model": out.get("model"),
        "elapsed_ms": out.get("elapsed_ms"),
    }


async def recognize_table(
    image_url: str | None = None,
    image_base64: str | None = None,
    model: str | None = None,
) -> dict:
    """Recognize a table image and return LaTeX ``tabular`` code."""
    img = await _resolve_image(image_url, image_base64)
    out = await _recognize("recognize-table", img, model or _DEFAULT_MODEL)
    return {
        "latex": out.get("latex", ""),
        "model": out.get("model"),
        "elapsed_ms": out.get("elapsed_ms"),
    }


def list_models() -> list[str]:
    return list(_MODELS)
