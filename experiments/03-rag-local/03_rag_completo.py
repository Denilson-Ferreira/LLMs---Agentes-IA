# %% [markdown]
# # Experimento 03 — Retrieval Augmented Generation (RAG)
#
# **RAG = buscar antes de responder.** A LLM não precisa ter o conhecimento em
# seus parâmetros: o sistema recupera documentos externos, monta um contexto e só
# então pede uma resposta.
#
# ```text
# Pergunta → Retrieval → Documentos relevantes → Contexto → LLM → Resposta
# ```

# %% [markdown]
# ## Sem RAG e com RAG
#
# ```text
# Sem RAG: Usuário → LLM → conhecimento/contexto já disponível na LLM
# Com RAG: Usuário → Retriever → base → contexto relevante → LLM → resposta
# ```
#
# Como a TechService Brasil é fictícia, a comparação evidencia por que dados
# privados ou recentes precisam ser recuperados antes da geração.

# %% [markdown]
# ## Instalação
#
# ```bash
# pip install -r requirements_03_rag.txt
# ```
#
# ```python
# # %pip install -U langchain langchain-groq langchain-text-splitters rank-bm25 python-dotenv pandas numpy scikit-learn ipykernel
# ```

# %% Variáveis de ambiente
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = next(
    (path for path in (BASE_DIR, *BASE_DIR.parents) if (path / "experiments").is_dir()),
    BASE_DIR,
)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)


def validate_configuration() -> None:
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "cole_sua_chave_aqui":
        raise EnvironmentError(
            "GROQ_API_KEY ausente. Configure a chave no .env da raiz ou no .env local."
        )


validate_configuration()
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").removeprefix("groq/")
EMBEDDING_DIMS = 384

# %% Imports
import re
import time
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# %% Configuração da LLM
# GPT-5 mini usa controle de esforço de raciocínio; temperature não é enviada.
try:
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        timeout=60,
        max_retries=2,
    )
except Exception as error:
    raise RuntimeError(
        f"Falha ao configurar a LLM: {type(error).__name__}: {error}"
    ) from error

# %% Modelo de embeddings
class LocalHashEmbeddings(Embeddings):
    """Vetores locais, determinísticos e sem dependência de uma API externa."""

    def __init__(self, dims: int = EMBEDDING_DIMS) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dims
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


try:
    embeddings = LocalHashEmbeddings()
except Exception as error:
    raise RuntimeError(
        f"Falha ao configurar embeddings: {type(error).__name__}: {error}"
    ) from error

if __name__ == "__main__":
    try:
        sample_vector = embeddings.embed_query("Qual o prazo de atendimento?")
        print("Dimensões do embedding:", len(sample_vector))
        print("Primeiros valores:", sample_vector[:5])
    except Exception as error:
        raise RuntimeError(
            f"Falha ao gerar embedding de demonstração: {type(error).__name__}: {error}"
        ) from error

# %% Carregamento dos documentos
DATA_DIR = BASE_DIR / "data"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Pasta data inexistente: {data_dir}")
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"Nenhum documento .txt encontrado em {data_dir}")
    loaded = []
    for path in paths:
        text = re.sub(r"\n{3,}", "\n\n", path.read_text(encoding="utf-8")).strip()
        if not text:
            raise ValueError(f"Documento vazio: {path.name}")
        loaded.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "category": path.stem},
            )
        )
    return loaded


documents: list[Document] = []
if __name__ == "__main__":
    documents = load_documents()
    for document in documents:
        print(document.metadata, "caracteres=", len(document.page_content))

# %% Chunking
def split_documents(
    source_documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 80
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split = splitter.split_documents(source_documents)
    for index, chunk in enumerate(split):
        chunk.metadata = {**chunk.metadata, "chunk_id": f"chunk-{index:03d}"}
    return split


chunks: list[Document] = []
if __name__ == "__main__":
    chunks = split_documents(documents)
    print("Documentos originais:", len(documents), "| Chunks:", len(chunks))
    for chunk in chunks[:4]:
        print(chunk.metadata, "tamanho=", len(chunk.page_content), "\n", chunk.page_content[:180])

# %% Experimento de chunking
if __name__ == "__main__":
    chunking_experiment = []
    for tested_size in (200, 500, 1000):
        tested_chunks = split_documents(documents, chunk_size=tested_size, chunk_overlap=min(80, tested_size // 4))
        chunking_experiment.append(
            {
                "chunk_size": tested_size,
                "quantidade": len(tested_chunks),
                "tamanho_médio": round(np.mean([len(c.page_content) for c in tested_chunks]), 1),
            }
        )
    print(pd.DataFrame(chunking_experiment).to_string(index=False))

# %% Indexação em vector database local
@dataclass
class VectorRecord:
    document: Document
    vector: np.ndarray


class LocalVectorDatabase:
    """Banco vetorial didático com vetores, metadata e similaridade cosseno."""

    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    def add(self, source_chunks: list[Document], vectors: list[list[float]]) -> None:
        if len(source_chunks) != len(vectors):
            raise ValueError("Quantidade de chunks e embeddings não corresponde.")
        self.records = [
            VectorRecord(document=chunk, vector=np.asarray(vector, dtype=float))
            for chunk, vector in zip(source_chunks, vectors)
        ]

    def search(self, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        if not self.records:
            raise RuntimeError("Vector database vazio. Execute a indexação primeiro.")
        matrix = np.vstack([record.vector for record in self.records])
        scores = cosine_similarity(np.asarray(query_vector).reshape(1, -1), matrix)[0]
        order = np.argsort(scores)[::-1][:k]
        return [
            {
                "document": self.records[index].document,
                "semantic_score": float(scores[index]),
            }
            for index in order
        ]


vector_db = LocalVectorDatabase()
if __name__ == "__main__":
    try:
        chunk_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
        vector_db.add(chunks, chunk_vectors)
    except Exception as error:
        raise RuntimeError(
            f"Falha no embedding/indexação: {type(error).__name__}: {error}"
        ) from error
    print("Chunks indexados:", len(vector_db.records))

# %% Semantic search
def semantic_search(query: str, k: int = 4) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("A consulta não pode estar vazia.")
    try:
        query_vector = embeddings.embed_query(query)
        return vector_db.search(query_vector, k)
    except Exception as error:
        if isinstance(error, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(
            f"Falha na busca semântica: {type(error).__name__}: {error}"
        ) from error


if __name__ == "__main__":
    for item in semantic_search("Qual o tempo de atendimento para clientes Enterprise?"):
        print(item["document"].metadata, "score=", round(item["semantic_score"], 4))

# %% Keyword search com BM25
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", text.lower())


bm25: BM25Okapi | None = None


def prepare_bm25() -> None:
    global bm25
    if not chunks:
        raise RuntimeError("Não há chunks para preparar o índice BM25.")
    bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])


def keyword_search(query: str, k: int = 4) -> list[dict[str, Any]]:
    if bm25 is None:
        raise RuntimeError("Índice BM25 vazio. Execute prepare_bm25() primeiro.")
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return [
        {"document": chunks[index], "keyword_score": float(scores[index])}
        for index in order
    ]


if __name__ == "__main__":
    prepare_bm25()
    for item in keyword_search("prazo suporte Enterprise"):
        print(item["document"].metadata, "BM25=", round(item["keyword_score"], 4))

# %% Comparação semantic x keyword
if __name__ == "__main__":
    comparison_rows = []
    comparison_query = "Qual o prazo do suporte Enterprise?"
    for method, results, score_name in (
        ("semantic", semantic_search(comparison_query), "semantic_score"),
        ("keyword", keyword_search(comparison_query), "keyword_score"),
    ):
        for item in results:
            comparison_rows.append(
                {
                    "método": method,
                    "documento": item["document"].metadata["source"],
                    "chunk": item["document"].metadata["chunk_id"],
                    "score": round(item[score_name], 4),
                }
            )
    print(pd.DataFrame(comparison_rows).to_string(index=False))

# %% Hybrid search
def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if np.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def hybrid_search(query: str, k: int = 5, alpha: float = 0.7) -> list[dict[str, Any]]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha deve estar entre 0 e 1.")
    semantic = semantic_search(query, k=len(chunks))
    keyword = keyword_search(query, k=len(chunks))
    semantic_by_id = {item["document"].metadata["chunk_id"]: item for item in semantic}
    keyword_by_id = {item["document"].metadata["chunk_id"]: item for item in keyword}
    semantic_norm = normalize_scores([item["semantic_score"] for item in semantic])
    keyword_norm = normalize_scores([item["keyword_score"] for item in keyword])
    semantic_normalized = {
        item["document"].metadata["chunk_id"]: score for item, score in zip(semantic, semantic_norm)
    }
    keyword_normalized = {
        item["document"].metadata["chunk_id"]: score for item, score in zip(keyword, keyword_norm)
    }
    combined = []
    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        combined.append(
            {
                "document": chunk,
                "semantic_score": semantic_by_id[chunk_id]["semantic_score"],
                "keyword_score": keyword_by_id[chunk_id]["keyword_score"],
                "hybrid_score": alpha * semantic_normalized[chunk_id]
                + (1 - alpha) * keyword_normalized[chunk_id],
            }
        )
    return sorted(combined, key=lambda item: item["hybrid_score"], reverse=True)[:k]


if __name__ == "__main__":
    print([(r["document"].metadata["source"], round(r["hybrid_score"], 3)) for r in hybrid_search("prazo Enterprise")])

# %% Query parsing
class ParsedQuery(BaseModel):
    intent: str = Field(description="Intenção principal em poucas palavras")
    keywords: list[str] = Field(description="Termos importantes para retrieval")
    product: str | None = Field(default=None, description="Produto ou plano citado")
    filters: dict[str, str] = Field(default_factory=dict)


query_parser = llm.with_structured_output(ParsedQuery)


def parse_query(query: str) -> ParsedQuery:
    try:
        result = query_parser.invoke(
            [
                SystemMessage(content="Analise consultas para busca documental. Não responda à pergunta."),
                HumanMessage(content=query),
            ]
        )
        return result if isinstance(result, ParsedQuery) else ParsedQuery.model_validate(result)
    except ValidationError as error:
        raise RuntimeError(f"Query parsing estruturado inválido: {error}") from error
    except Exception as error:
        raise RuntimeError(
            f"Falha no query parsing: {type(error).__name__}: {error}"
        ) from error


if __name__ == "__main__":
    print(parse_query("Qual o prazo do suporte Enterprise?").model_dump())

# %% Retriever central
def retrieve(query: str, k: int = 5) -> dict[str, Any]:
    parsed = parse_query(query)
    search_query = " ".join([query, *parsed.keywords, parsed.product or ""]).strip()
    candidates = hybrid_search(search_query, k=max(k, 5))
    if parsed.product:
        filtered = [
            item for item in candidates if parsed.product.lower() in item["document"].page_content.lower()
        ]
        if filtered:
            candidates = filtered
    if not candidates:
        raise RuntimeError("Retrieval não retornou resultados.")
    return {"parsed_query": parsed, "candidates": candidates[:k]}

# %% Reranking local
def rerank(query: str, candidates: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("Não há candidatos para reranking.")
    query_terms = set(tokenize(query))
    reranked = []
    for item in candidates:
        document_terms = set(tokenize(item["document"].page_content))
        lexical_overlap = len(query_terms & document_terms) / max(len(query_terms), 1)
        reranked_item = dict(item)
        reranked_item["rerank_score"] = 0.8 * item["hybrid_score"] + 0.2 * lexical_overlap
        reranked.append(reranked_item)
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_n]


if __name__ == "__main__":
    before = retrieve("Qual o prazo Enterprise?", k=5)["candidates"]
    after = rerank("Qual o prazo Enterprise?", before, top_n=3)
    print("Antes:", [(x["document"].metadata["chunk_id"], round(x["hybrid_score"], 3)) for x in before])
    print("Depois:", [(x["document"].metadata["chunk_id"], round(x["rerank_score"], 3)) for x in after])

# %% Top-K
if __name__ == "__main__":
    top_k_rows = []
    for tested_k in (1, 3, 5):
        result = retrieve("cancelamento do plano", k=tested_k)
        top_k_rows.append({"k": tested_k, "chunks": [x["document"].metadata["chunk_id"] for x in result["candidates"]]})
    print(pd.DataFrame(top_k_rows).to_string(index=False))

# %% Construção do contexto
def build_context(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        raise ValueError("Contexto vazio: nenhum documento recuperado.")
    sections = []
    for item in candidates:
        document = item["document"]
        sections.append(
            f"[Fonte: {document.metadata['source']} | Chunk: {document.metadata['chunk_id']}]\n{document.page_content}"
        )
    context = "\n\n".join(sections).strip()
    if not context:
        raise ValueError("O contexto construído ficou vazio.")
    return context

# %% Augmented prompt
RAG_SYSTEM_PROMPT = """
Você é um assistente que responde usando apenas o contexto fornecido.
Se a informação não estiver no contexto, diga exatamente:
"Não encontrei essa informação na base de conhecimento fornecida."
Não invente dados. Responda em português e apresente as fontes utilizadas.
""".strip()


def build_augmented_prompt(question: str, context: str) -> str:
    return f"CONTEXTO:\n{context}\n\nPERGUNTA:\n{question}"

# %% Pipeline RAG
def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) for block in content
            if isinstance(block, dict) and block.get("text")
        ).strip()
    return ""


def rag_answer(question: str, top_n: int = 3) -> dict[str, Any]:
    total_start = time.perf_counter()
    retrieval_start = time.perf_counter()
    retrieval = retrieve(question, k=5)
    retrieval_seconds = time.perf_counter() - retrieval_start
    rerank_start = time.perf_counter()
    selected = rerank(question, retrieval["candidates"], top_n=top_n)
    rerank_seconds = time.perf_counter() - rerank_start
    context = build_context(selected)
    llm_start = time.perf_counter()
    try:
        response = llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=build_augmented_prompt(question, context)),
            ]
        )
    except Exception as error:
        raise RuntimeError(f"Erro da LLM: {type(error).__name__}: {error}") from error
    answer = message_text(response)
    if not answer:
        raise RuntimeError("A LLM retornou uma resposta vazia.")
    llm_seconds = time.perf_counter() - llm_start
    sources = sorted({item["document"].metadata["source"] for item in selected})
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": selected,
        "parsed_query": retrieval["parsed_query"].model_dump(),
        "context": context,
        "latency": {
            "retrieval": retrieval_seconds,
            "reranking": rerank_seconds,
            "llm": llm_seconds,
            "total": time.perf_counter() - total_start,
        },
    }

# %% Primeiro teste completo
if __name__ == "__main__":
    first_result = rag_answer("Qual é o prazo de atendimento para clientes Enterprise?")
    print("Pergunta:", first_result["question"])
    print("Resposta:", first_result["answer"])
    print("Fontes:", first_result["sources"])
    print("Chunks:", [x["document"].metadata["chunk_id"] for x in first_result["retrieved_chunks"]])

# %% Teste sem informação
if __name__ == "__main__":
    missing_result = rag_answer("Quem é o atual CEO da TechService Brasil?")
    print(missing_result["answer"])

# %% LLM sem RAG x LLM com RAG
def answer_without_rag(question: str) -> str:
    try:
        response = llm.invoke([HumanMessage(content=question)])
    except Exception as error:
        raise RuntimeError(f"Erro da LLM sem RAG: {type(error).__name__}: {error}") from error
    answer = message_text(response)
    if not answer:
        raise RuntimeError("Resposta sem RAG vazia.")
    return answer


if __name__ == "__main__":
    comparison_question = "Qual o prazo do suporte Enterprise da TechService Brasil?"
    print("SEM RAG\n", answer_without_rag(comparison_question))
    print("\nCOM RAG\n", rag_answer(comparison_question)["answer"])

# %% Avaliação do retrieval
EVALUATION_QUESTIONS = [
    {"question": "Qual o prazo Enterprise?", "expected_source": "politicas.txt"},
    {"question": "Quanto custa o Plano Basic?", "expected_source": "produtos.txt"},
    {"question": "Onde fica a empresa?", "expected_source": "documento_empresa.txt"},
    {"question": "Qual a antecedência para cancelar?", "expected_source": "politicas.txt"},
]


def evaluate_retrieval(k: int = 3) -> pd.DataFrame:
    rows = []
    for case in EVALUATION_QUESTIONS:
        results = retrieve(case["question"], k=k)["candidates"]
        sources = [item["document"].metadata["source"] for item in results]
        relevant_ranks = [index + 1 for index, source in enumerate(sources) if source == case["expected_source"]]
        hit = int(bool(relevant_ranks))
        rows.append(
            {
                "question": case["question"],
                "expected_source": case["expected_source"],
                f"hit_rate@{k}": hit,
                f"precision@{k}": hit / max(len(sources), 1),
                f"recall@{k}": hit,
                "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    retrieval_evaluation = evaluate_retrieval()
    print(retrieval_evaluation.to_string(index=False))
    print("Médias:\n", retrieval_evaluation.select_dtypes(include="number").mean())

# %% Avaliação da resposta
def evaluate_answer(result: dict[str, Any]) -> dict[str, float]:
    answer_terms = set(tokenize(result["answer"]))
    question_terms = set(tokenize(result["question"]))
    context_terms = set(tokenize(result["context"]))
    relevance = len(answer_terms & question_terms) / max(len(question_terms), 1)
    groundedness = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    known_sources = {chunk.metadata["source"] for chunk in chunks}
    source_correctness = float(all(source in known_sources for source in result["sources"]))
    return {
        "answer_relevance": round(relevance, 3),
        "groundedness": round(groundedness, 3),
        "source_correctness": source_correctness,
    }


if __name__ == "__main__":
    print(evaluate_answer(first_result))

# %% Latência
if __name__ == "__main__":
    latency_result = rag_answer("Qual o valor do Plano Professional?")
    latency_table = pd.DataFrame(
        [{"etapa": stage, "segundos": round(seconds, 4)} for stage, seconds in latency_result["latency"].items()]
    )
    print(latency_table.to_string(index=False))
    print("Produção equilibra qualidade, latência e custo.")

# %% Debug do pipeline e testes acadêmicos
LAB_TESTS = [
    ("encontrada diretamente", "Quanto custa o Plano Basic?"),
    ("semanticamente semelhante", "Em quanto tempo clientes grandes recebem retorno?"),
    ("dependente de palavra-chave", "Enterprise 2 horas úteis"),
    ("beneficia busca híbrida", "prazo atendimento plano Enterprise"),
    ("sem resposta", "Quem é o CEO atual?"),
    ("comparação", "Qual o prazo Enterprise?"),
]


def debug_rag(question: str) -> dict[str, Any]:
    print("1. Query original:\n", question)
    retrieval = retrieve(question, k=5)
    print("\n2. Query interpretada:\n", retrieval["parsed_query"].model_dump())
    print("\n3 e 4. Chunks recuperados e scores:")
    for item in retrieval["candidates"]:
        print(item["document"].metadata, {k: round(v, 4) for k, v in item.items() if k.endswith("score")})
    selected = rerank(question, retrieval["candidates"], top_n=3)
    print("\n5. Reranking:", [(x["document"].metadata["chunk_id"], round(x["rerank_score"], 4)) for x in selected])
    context = build_context(selected)
    print("\n6. Contexto entregue à LLM:\n", context)
    result = rag_answer(question)
    print("\n7. Resposta final:\n", result["answer"])
    return result


if __name__ == "__main__":
    debug_rag("Qual o prazo de atendimento do Plano Enterprise?")

# %% [markdown]
# ## Diagrama final
#
# ```text
# DOCUMENTOS → CHUNKING → EMBEDDINGS → VECTOR DB
#                                      ↑
# USUÁRIO → QUERY → QUERY PARSING → HYBRID RETRIEVAL → RERANKING → TOP-K
# → CONTEXT → AUGMENTED PROMPT → LLM → RESPOSTA → FONTES
# ```

# %% [markdown]
# ## Explicação para apresentação
#
# - **Retrieval:** busca conhecimento relevante.
# - **Augmented:** acrescenta o conhecimento recuperado ao prompt.
# - **Generation:** a LLM usa o contexto para produzir a resposta.
#
# **RAG = Retrieval + Augmentation + Generation.**
#
# **RAG busca antes de responder.**

# %% [markdown]
# ## RAG x fine-tuning
#
# - **RAG:** fornece conhecimento no momento da consulta.
# - **Fine-tuning:** modifica comportamento/parâmetros por treinamento.
#
# RAG é especialmente útil para informações privadas, recentes, específicas do
# domínio ou frequentemente atualizadas.

# %% [markdown]
# ## RAG x MCP
#
# ```text
# LLM
#  ├── RAG ──► documentos / vector database
#  └── MCP ──► APIs / ferramentas / sistemas
# ```
#
# RAG recupera conhecimento; MCP padroniza o acesso a ferramentas e sistemas.
# Eles podem trabalhar juntos.

# %% [markdown]
# ## RAG em um agente
#
# ```text
# Agente → Preciso de conhecimento? → Retriever RAG → Base → Contexto → Agente
# ```
#
# O RAG pode ser uma capacidade utilizada por um agente maior quando a tarefa
# exigir conhecimento externo, privado ou atualizado.
