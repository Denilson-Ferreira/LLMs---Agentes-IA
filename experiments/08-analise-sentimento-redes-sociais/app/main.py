from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.providers import GroqSentimentProvider, HeuristicSentimentProvider
from app.collectors.mock import MockSocialMediaCollector
from app.config import settings
from app.models import PostResult
from app.services import AnalysisPipeline

app = FastAPI(title="Social Sentiment Agent", version="0.1.0")
results: dict[str, PostResult] = {}


class AnalysisRequest(BaseModel):
    fixture: str
    provider: str = "groq"
    max_comments: int = 1_000


def provider_for(name: str):
    if name == "heuristic":
        return HeuristicSentimentProvider()
    if name != "groq":
        raise ValueError("provider deve ser 'groq' ou 'heuristic'.")
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY não configurada; use provider=heuristic para a demonstração local.")
    return GroqSentimentProvider(settings.groq_api_key, settings.groq_model)


def get_result(analysis_id: str) -> PostResult:
    if analysis_id not in results:
        raise HTTPException(status_code=404, detail="Análise não encontrada (o armazenamento é temporário).")
    return results[analysis_id]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analysis")
async def analyze(request: AnalysisRequest):
    try:
        collector = MockSocialMediaCollector(request.fixture)
        post = await collector.get_post(collector.data["post"]["url"])
        result = await AnalysisPipeline(collector, provider_for(request.provider), settings).run(post.url, request.max_comments)
        analysis_id = str(uuid4())
        results[analysis_id] = result
        return {"analysis_id": analysis_id, "status": "completed"}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str) -> PostResult:
    return get_result(analysis_id)


@app.get("/analysis/{analysis_id}/comments")
async def get_comments(analysis_id: str):
    return get_result(analysis_id).analyses


@app.get("/analysis/{analysis_id}/summary")
async def get_summary(analysis_id: str):
    result = get_result(analysis_id)
    return {"summary": result.summary, "sentiment_distribution": result.sentiment_distribution}


@app.get("/analysis/{analysis_id}/topics")
async def get_topics(analysis_id: str):
    result = get_result(analysis_id)
    return {"top_topics": result.top_topics, "topic_sentiment_matrix": result.topic_sentiment_matrix}


@app.post("/posts/collect")
async def collect(request: AnalysisRequest):
    """Demonstra o contrato de coleta através da fixture mock, sem acesso externo."""
    try:
        collector = MockSocialMediaCollector(request.fixture)
        post = await collector.get_post(collector.data["post"]["url"])
        comments = await collector.get_comments(post.url, request.max_comments)
        return {"post": post, "comments_collected": len(comments), "collector": "mock"}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
