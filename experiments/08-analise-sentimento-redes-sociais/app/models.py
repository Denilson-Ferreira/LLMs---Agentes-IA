from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["positive", "negative", "neutral", "mixed", "unknown"]
EmotionLabel = Literal[
    "joy", "enthusiasm", "satisfaction", "admiration", "gratitude", "anger",
    "frustration", "disappointment", "sadness", "fear", "concern", "disgust",
    "surprise", "confusion", "irony", "humor", "neutral", "other", "unknown",
]


class Post(BaseModel):
    id: str
    url: str
    platform: str = "mock"
    text: str = ""
    title: str | None = None
    published_at: datetime | None = None


class Comment(BaseModel):
    id: str
    post_id: str
    text: str
    author_id: str | None = None
    author_display_name: str | None = None
    created_at: datetime | None = None
    likes: int = 0
    replies_count: int = 0
    parent_comment_id: str | None = None
    source_platform: str = "mock"
    source_url: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class EmotionScore(BaseModel):
    label: EmotionLabel
    score: float = Field(ge=0, le=1)


class EmotionMetric(BaseModel):
    """Percentual de uma emoção na agregação da publicação."""
    label: EmotionLabel
    percentage: float = Field(ge=0, le=100)


class Sentiment(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)


class Sarcasm(BaseModel):
    detected: bool
    confidence: float = Field(ge=0, le=1)


class Toxicity(BaseModel):
    detected: bool
    score: float = Field(ge=0, le=1)


class CommentAnalysis(BaseModel):
    comment_id: str
    sentiment: Sentiment
    emotions: list[EmotionScore]
    dominant_emotion: EmotionLabel
    intensity: float = Field(ge=0, le=1)
    sarcasm: Sarcasm
    topics: list[str] = Field(default_factory=list)
    stance: Literal["supportive", "critical", "informational", "unclear"] = "unclear"
    toxicity: Toxicity
    insufficient_context: bool = False
    explanation: str = Field(max_length=400)
    model_confidence: float = Field(ge=0, le=1)
    needs_review: bool = False


class TopicMetrics(BaseModel):
    topic: str
    comments: int
    negative_percentage: float
    distribution: dict[SentimentLabel, float]


class PostResult(BaseModel):
    post: Post
    comments_collected: int
    comments_analyzed: int
    comments_discarded: int
    sentiment_distribution: dict[SentimentLabel, float]
    average_sentiment: float
    average_emotional_intensity: float
    dominant_emotions: list[EmotionMetric]
    top_topics: list[TopicMetrics]
    topic_sentiment_matrix: dict[str, dict[SentimentLabel, float]]
    summary: str
    analyses: list[CommentAnalysis]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
