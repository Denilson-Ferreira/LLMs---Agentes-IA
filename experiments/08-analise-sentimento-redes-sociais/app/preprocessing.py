from __future__ import annotations

import hashlib
import re
import unicodedata

from app.models import Comment


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    return re.sub(r"\s+", " ", text)


def is_spam(text: str) -> bool:
    return len(re.findall(r"https?://", text)) > 2 or len(text) > 1_500


def preprocess(comments: list[Comment], anonymize_authors: bool) -> tuple[list[Comment], int]:
    """Normaliza e remove duplicatas exatas, sem apagar emojis ou pontuação."""
    seen: set[str] = set()
    cleaned: list[Comment] = []
    discarded = 0
    for comment in comments:
        text = normalize_text(comment.text)
        fingerprint = hashlib.sha256(text.casefold().encode()).hexdigest()
        if not text or fingerprint in seen or is_spam(text):
            discarded += 1
            continue
        seen.add(fingerprint)
        author_id = comment.author_id
        if anonymize_authors and author_id:
            author_id = hashlib.sha256(author_id.encode()).hexdigest()[:12]
        cleaned.append(comment.model_copy(update={"text": text, "author_id": author_id, "author_display_name": None if anonymize_authors else comment.author_display_name}))
    return cleaned, discarded
