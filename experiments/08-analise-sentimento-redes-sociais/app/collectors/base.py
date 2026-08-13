from __future__ import annotations

from typing import Protocol

from app.models import Comment, Post


class SocialMediaCollector(Protocol):
    """Contrato para APIs oficiais ou collectors autorizados por plataforma."""

    async def supports(self, url: str) -> bool: ...
    async def get_post(self, url: str) -> Post: ...
    async def get_comments(self, url: str, max_comments: int) -> list[Comment]: ...
