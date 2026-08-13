from __future__ import annotations

import asyncio
import json
import re
from typing import Protocol

from groq import Groq

from app.agents.prompts import SENTIMENT_SYSTEM_PROMPT
from app.models import Comment, CommentAnalysis, EmotionScore, Sarcasm, Sentiment, Toxicity


class SentimentProvider(Protocol):
    async def analyze(self, comment: Comment, post_context: str) -> CommentAnalysis: ...


class HeuristicSentimentProvider:
    """Modo local explícito para testes e demonstração; não substitui uma LLM."""
    positive = {"adorei", "perfeito", "amei", "ótimo", "otimo", "excelente", "bonito", "incrível", "incrivel", "👏", "😍"}
    negative = {"porcaria", "péssima", "pessima", "horrível", "horrivel", "não gostei", "nao gostei", "parou", "ruim", "absurdo", "decepcion"}
    topic_terms = {"preço": "price", "preco": "price", "design": "design", "atualização": "update", "atualizacao": "update", "funcionar": "reliability", "produto": "product"}

    async def analyze(self, comment: Comment, post_context: str) -> CommentAnalysis:
        text = comment.text.casefold()
        insufficient = len(re.sub(r"[^\w]", "", text)) < 4 or text in {"kkkk", "kkkkkk", "sim", "???"}
        sarcasm = any(marker in text for marker in ("só ", "so ", "🙄")) and any(word in text for word in ("excelente", "ótimo", "otimo", "parou"))
        pos = sum(token in text for token in self.positive)
        neg = sum(token in text for token in self.negative)
        if insufficient:
            label, score, emotion, stance, explanation = "unknown", 0.0, "unknown", "unclear", "Contexto linguístico insuficiente para uma classificação confiável."
        elif sarcasm or neg > pos:
            label, score, emotion, stance, explanation = "negative", -0.78, "frustration", "critical", "O texto expressa crítica ou insatisfação; a classificação descreve o comentário, não a pessoa."
        elif pos > neg:
            label, score, emotion, stance, explanation = "positive", 0.76, "enthusiasm", "supportive", "O texto expressa aprovação ou entusiasmo pela publicação."
        elif pos and neg:
            label, score, emotion, stance, explanation = "mixed", -0.1, "confusion", "unclear", "O comentário contém sinais positivos e negativos."
        else:
            label, score, emotion, stance, explanation = "neutral", 0.0, "neutral", "informational", "O texto é predominantemente informativo ou sem polaridade clara."
        topics = sorted({topic for term, topic in self.topic_terms.items() if term in text})
        intensity = 0.15 if insufficient else (0.82 if sarcasm or abs(score) > 0.7 else 0.4)
        return CommentAnalysis(comment_id=comment.id, sentiment=Sentiment(label=label, score=score, confidence=0.72), emotions=[EmotionScore(label=emotion, score=intensity)], dominant_emotion=emotion, intensity=intensity, sarcasm=Sarcasm(detected=sarcasm, confidence=0.75), topics=topics, stance=stance, toxicity=Toxicity(detected=False, score=0.05), insufficient_context=insufficient, explanation=explanation, model_confidence=0.72)


class GroqSentimentProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Groq(api_key=api_key)
        self.model = model

    async def analyze(self, comment: Comment, post_context: str) -> CommentAnalysis:
        schema = CommentAnalysis.model_json_schema()
        user_prompt = f"POST CONTEXT:\n{post_context}\n\nCOMMENT ID: {comment.id}\nCOMMENT:\n{comment.text}\n\nJSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
        response = await asyncio.to_thread(self.client.chat.completions.create, model=self.model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": SENTIMENT_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
        payload = json.loads(response.choices[0].message.content or "{}")
        payload["comment_id"] = comment.id
        return CommentAnalysis.model_validate(payload)
