from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    batch_size: int = int(os.getenv("LLM_BATCH_SIZE", "20"))
    max_llm_concurrency: int = int(os.getenv("MAX_LLM_CONCURRENCY", "5"))
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.55"))
    anonymize_authors: bool = os.getenv("ANONYMIZE_AUTHORS", "true").lower() == "true"


settings = Settings()
