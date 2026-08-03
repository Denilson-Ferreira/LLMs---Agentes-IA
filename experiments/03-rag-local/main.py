"""RAG local simplificado usando similaridade de cosseno.

Demonstra apenas a etapa de recuperação. Não chama um LLM.
"""
from __future__ import annotations

from collections import Counter
from math import sqrt
import re

DOCUMENTS = {
    "doc_cve": "CVE é o identificador de uma vulnerabilidade conhecida. Ela indica risco potencial e deve ser avaliada e corrigida.",
    "doc_ioc": "IOC é um indício técnico, como IP, domínio ou hash, que merece investigação. Um indicador isolado não confirma comprometimento.",
    "doc_rag": "RAG recupera trechos relevantes de fontes externas antes da geração da resposta.",
    "doc_memoria": "Memória preserva fatos e experiências para uso posterior pelo agente.",
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-záàâãéêíóôõúç0-9]+", text.lower())


def cosine(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in common)
    denominator = sqrt(sum(v*v for v in a.values())) * sqrt(sum(v*v for v in b.values()))
    return numerator / denominator if denominator else 0.0


def retrieve(query: str, top_k: int = 2) -> list[tuple[str, float, str]]:
    q = Counter(tokens(query))
    ranked = []
    for name, text in DOCUMENTS.items():
        score = cosine(q, Counter(tokens(text)))
        ranked.append((name, score, text))
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_k]


def run() -> None:
    query = "Um IOC quer dizer que eu já fui atacado?"
    print("\n[RAG local]")
    print("Consulta:", query)
    for name, score, text in retrieve(query):
        print(f"- {name} | score={score:.3f} | {text}")


if __name__ == "__main__":
    run()
