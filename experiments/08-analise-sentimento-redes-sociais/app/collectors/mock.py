from __future__ import annotations

import json
from pathlib import Path

from app.models import Comment, Post


class MockSocialMediaCollector:
    """Collector explícito de fixture: permite validar o pipeline sem rede."""

    def __init__(self, fixture_path: str | Path) -> None:
        self.path = Path(fixture_path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    async def supports(self, url: str) -> bool:
        return url == self.data["post"]["url"]

    async def get_post(self, url: str) -> Post:
        if not await self.supports(url):
            raise ValueError(f"A fixture não contém a publicação: {url}")
        return Post.model_validate(self.data["post"])

    async def get_comments(self, url: str, max_comments: int) -> list[Comment]:
        post = await self.get_post(url)
        return [
            Comment.model_validate({**item, "post_id": post.id, "source_url": post.url, "source_platform": post.platform})
            for item in self.data["comments"][:max_comments]
        ]
