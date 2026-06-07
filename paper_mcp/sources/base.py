"""The swappable source contract.

A `PaperSource` knows how to search a corpus and fetch a single record,
returning normalized `Paper` objects. The MCP server talks only to this
interface, never to a provider's raw API — so adding a source (or, later,
putting a semantic-retrieval brain in front of one) needs no server change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Paper


@runtime_checkable
class PaperSource(Protocol):
    name: str

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[Paper]: ...

    async def fetch(self, paper_id: str) -> Paper | None: ...

    async def search_by_author(
        self, author: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]: ...

    async def recent(
        self, category: str, *, max_results: int = 10, start: int = 0
    ) -> list[Paper]: ...

    def categories(self) -> list[dict]: ...
