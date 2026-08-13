from __future__ import annotations

from collections import Counter, defaultdict

from app.models import CommentAnalysis, EmotionMetric, SentimentLabel, TopicMetrics

LABELS: tuple[SentimentLabel, ...] = ("positive", "negative", "neutral", "mixed", "unknown")


def aggregate(analyses: list[CommentAnalysis]) -> tuple[dict[SentimentLabel, float], float, float, list[EmotionMetric], list[TopicMetrics], dict[str, dict[SentimentLabel, float]]]:
    valid = [item for item in analyses if not item.insufficient_context]
    base = valid or analyses
    counts = Counter(item.sentiment.label for item in base)
    total = len(base) or 1
    distribution = {label: round(counts[label] / total * 100, 1) for label in LABELS}
    avg_sentiment = round(sum(item.sentiment.score for item in base) / total, 3)
    avg_intensity = round(sum(item.intensity for item in base) / total, 3)
    emotion_counts = Counter(item.dominant_emotion for item in base)
    emotions = [EmotionMetric(label=label, percentage=round(count / total * 100, 1)) for label, count in emotion_counts.most_common(5)]
    topic_items: dict[str, list[CommentAnalysis]] = defaultdict(list)
    for item in base:
        for topic in item.topics:
            topic_items[topic].append(item)
    matrix: dict[str, dict[SentimentLabel, float]] = {}
    metrics: list[TopicMetrics] = []
    for topic, items in topic_items.items():
        item_total = len(items)
        item_counts = Counter(item.sentiment.label for item in items)
        row = {label: round(item_counts[label] / item_total * 100, 1) for label in LABELS}
        matrix[topic] = row
        metrics.append(TopicMetrics(topic=topic, comments=item_total, negative_percentage=row["negative"], distribution=row))
    metrics.sort(key=lambda item: item.comments, reverse=True)
    return distribution, avg_sentiment, avg_intensity, emotions, metrics, matrix


def summarize(distribution: dict[SentimentLabel, float], total: int, topics: list[TopicMetrics], emotions: list[EmotionMetric]) -> str:
    main = max((label for label in ("positive", "negative", "neutral", "mixed")), key=lambda label: distribution[label])
    emotion = emotions[0].label if emotions else "unknown"
    topic_part = f" O tópico mais recorrente foi {topics[0].topic}." if topics else " Não houve tópico recorrente identificado."
    return f"Nos {total} comentários analisáveis, o tom predominante foi {main} ({distribution[main]:.1f}%). A emoção expressa mais frequente foi {emotion}.{topic_part} A síntese descreve apenas a amostra coletada."
