from __future__ import annotations

import asyncio
import hashlib

from app.aggregation import aggregate, summarize
from app.agents.providers import SentimentProvider
from app.collectors.base import SocialMediaCollector
from app.config import Settings
from app.models import Comment, CommentAnalysis, PostResult
from app.preprocessing import preprocess


class AnalysisPipeline:
    def __init__(self, collector: SocialMediaCollector, provider: SentimentProvider, settings: Settings) -> None:
        self.collector, self.provider, self.settings = collector, provider, settings
        self.cache: dict[str, CommentAnalysis] = {}

    async def run(self, url: str, max_comments: int = 1_000) -> PostResult:
        post = await self.collector.get_post(url)
        raw_comments = await self.collector.get_comments(url, max_comments)
        comments, discarded = preprocess(raw_comments, self.settings.anonymize_authors)
        semaphore = asyncio.Semaphore(self.settings.max_llm_concurrency)

        async def one(comment: Comment) -> CommentAnalysis:
            cache_key = hashlib.sha256(f"{comment.text}|{type(self.provider).__name__}".encode()).hexdigest()
            if cache_key not in self.cache:
                async with semaphore:
                    analysis = await self.provider.analyze(comment, f"{post.title or ''}\n{post.text}")
                    analysis.needs_review = analysis.model_confidence < self.settings.low_confidence_threshold
                    self.cache[cache_key] = analysis
            return self.cache[cache_key].model_copy(update={"comment_id": comment.id})

        analyses = list(await asyncio.gather(*(one(comment) for comment in comments)))
        distribution, avg_sentiment, avg_intensity, emotions, topics, matrix = aggregate(analyses)
        return PostResult(post=post, comments_collected=len(raw_comments), comments_analyzed=len(analyses), comments_discarded=discarded, sentiment_distribution=distribution, average_sentiment=avg_sentiment, average_emotional_intensity=avg_intensity, dominant_emotions=emotions, top_topics=topics, topic_sentiment_matrix=matrix, summary=summarize(distribution, sum(not item.insufficient_context for item in analyses), topics, emotions), analyses=analyses)
