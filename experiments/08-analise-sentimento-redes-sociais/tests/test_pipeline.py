from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents.providers import HeuristicSentimentProvider
from app.collectors.mock import MockSocialMediaCollector
from app.config import Settings
from app.preprocessing import normalize_text
from app.services import AnalysisPipeline
from app.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "post.json"


def test_normalization_preserves_emojis_and_negation() -> None:
    assert normalize_text("  Não   gostei 😍 ") == "Não gostei 😍"


def test_mock_pipeline_calculates_statistics() -> None:
    async def run():
        collector = MockSocialMediaCollector(FIXTURE)
        post = await collector.get_post("https://mock.social/posts/001")
        return await AnalysisPipeline(collector, HeuristicSentimentProvider(), Settings()).run(post.url)

    result = asyncio.run(run())
    assert result.comments_collected == 8
    assert result.comments_discarded == 1
    assert result.comments_analyzed == 7
    assert sum(result.sentiment_distribution.values()) == 100.0
    assert any(item.sarcasm.detected for item in result.analyses)
    assert any(item.insufficient_context for item in result.analyses)
    assert "amostra coletada" in result.summary


def test_api_returns_saved_analysis() -> None:
    client = TestClient(app)
    response = client.post("/analysis", json={"fixture": str(FIXTURE), "provider": "heuristic"})
    assert response.status_code == 200
    analysis_id = response.json()["analysis_id"]
    assert client.get(f"/analysis/{analysis_id}/summary").status_code == 200
    assert client.get(f"/analysis/{analysis_id}/topics").status_code == 200
