"""Log the MCP client name each connection reports on ``initialize``.

Only ``params.clientInfo.name`` and ``.version`` (and the requested protocol
revision) are logged, trimmed and stripped of control characters. No IP
address, headers, arguments or credentials are recorded here; the web server's
access log already holds request metadata, and the timestamp is enough to
match the two when an operator needs to.

Pure ASGI: the request body is passed through unchanged as it arrives, only a
bounded copy (``MAX_PEEK`` bytes) is inspected, and the response is never
touched or buffered, so streamable-HTTP / SSE keeps working.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("paper_mcp.clients")

MAX_PEEK = 64 * 1024
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


def _clean(value, limit: int = 80) -> str:
    if not isinstance(value, str):
        return "-"
    value = _CTRL.sub(" ", value).strip()
    return value[:limit] or "-"


def client_info(body: bytes) -> tuple[str, str, str] | None:
    """(name, version, protocolVersion) of an initialize request, else None."""
    if b"initialize" not in body:
        return None
    try:
        msg = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    for item in msg if isinstance(msg, list) else [msg]:
        if not isinstance(item, dict) or item.get("method") != "initialize":
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        return (_clean(info.get("name")), _clean(info.get("version"), 40),
                _clean(params.get("protocolVersion"), 20))
    return None


class ClientInfoLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        buf = bytearray()
        done = False

        async def peek():
            nonlocal done
            message = await receive()
            if not done and message.get("type") == "http.request":
                if len(buf) < MAX_PEEK:
                    buf.extend(message.get("body", b"")[: MAX_PEEK - len(buf)])
                if not message.get("more_body", False):
                    done = True
                    try:
                        found = client_info(bytes(buf)) if len(buf) < MAX_PEEK else None
                    except Exception:  # noqa: BLE001 - logging must never break a request
                        found = None
                    if found:
                        log.info("mcp client: name=%s version=%s protocol=%s", *found)
            return message

        await self.app(scope, peek, send)
