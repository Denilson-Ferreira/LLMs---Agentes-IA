from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.agents.providers import GroqSentimentProvider, HeuristicSentimentProvider
from app.collectors.mock import MockSocialMediaCollector
from app.config import settings
from app.services import AnalysisPipeline


def build_provider(name: str):
    if name == "heuristic":
        return HeuristicSentimentProvider()
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY não configurada. Use --provider heuristic para a demonstração local.")
    return GroqSentimentProvider(settings.groq_api_key, settings.groq_model)


async def command_analyze(args: argparse.Namespace) -> None:
    collector = MockSocialMediaCollector(args.fixture)
    post = await collector.get_post(collector.data["post"]["url"])
    result = await AnalysisPipeline(collector, build_provider(args.provider), settings).run(post.url, args.max_comments)
    text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Resultado salvo em {args.output}")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de reações expressas em comentários")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--fixture", required=True, help="Arquivo JSON para MockSocialMediaCollector")
    analyze.add_argument("--provider", choices=("groq", "heuristic"), default="groq")
    analyze.add_argument("--max-comments", type=int, default=1_000)
    analyze.add_argument("--output")
    args = parser.parse_args()
    if args.command == "analyze":
        asyncio.run(command_analyze(args))


if __name__ == "__main__":
    main()
