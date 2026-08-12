# %% [markdown]
# # Experimento 02 — Memória de Longo Prazo com LangGraph
#
# Este experimento mostra um assistente de triagem de e-mails com três tipos de
# memória de longo prazo:
#
# - **Semântica:** fatos e preferências — o que o agente sabe.
# - **Episódica:** experiências anteriores — o que aconteceu antes.
# - **Procedural:** regras — como o agente deve agir.
# - **Long-term memory:** informação armazenada fora da LLM e entre threads.
# - **Short-term memory:** estado da thread atual, salvo pelo checkpointer.
#
# A LLM raciocina. O LangGraph orquestra e recupera memórias armazenadas
# externamente. A LLM não guarda essas memórias sozinha.

# %% [markdown]
# ## Instalação
#
# ```bash
# pip install -r requirements.txt
# ```
#
# Instalação opcional dentro do notebook:
#
# ```python
# # %pip install -U langgraph langchain langchain-groq python-dotenv pydantic ipykernel
# ```

# %% Variáveis de ambiente
import os
from pathlib import Path

from dotenv import load_dotenv

# No script, usa sua própria pasta; no notebook, usa a pasta aberta no VS Code.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = next(
    (path for path in (BASE_DIR, *BASE_DIR.parents) if (path / "experiments").is_dir()),
    BASE_DIR,
)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)


def validate_configuration() -> None:
    """Interrompe com uma mensagem clara quando a chave não foi configurada."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        raise EnvironmentError(
            "GROQ_API_KEY ausente. Configure a chave no .env da raiz ou preencha "
            "uma chave válida antes de executar."
        )


validate_configuration()
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").removeprefix("groq/")
EMBEDDING_DIMS = 384
print(f"Configuração carregada. Modelo selecionado: {MODEL_NAME}")

# %% Imports
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Literal
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

# %% LLM, embeddings e memória
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
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=os.environ["GROQ_API_KEY"],
        timeout=60,
        max_retries=2,
    )
    embeddings = LocalHashEmbeddings()
except Exception as error:
    raise RuntimeError(
        "Não foi possível configurar a Groq e os embeddings locais. Verifique a chave, os modelos e "
        f"as dependências. Detalhe: {type(error).__name__}: {error}"
    ) from error

# InMemorySaver = memória de curto prazo/estado da thread atual.
checkpointer = InMemorySaver()

# InMemoryStore = memória de longo prazo compartilhada entre threads.
# O encoder local gera vetores de 384 dimensões para busca por similaridade lexical.
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": EMBEDDING_DIMS,
        "fields": ["text"],
    }
)

# %% Contexto do usuário
@dataclass
class Context:
    """Contexto imutável da execução; user_id identifica o dono das memórias."""

    user_id: str

# %% Estado do grafo
class EmailState(TypedDict, total=False):
    """Estado de curto prazo salvo separadamente para cada thread_id."""

    email_text: str
    result: dict[str, Any]
    memories_used: list[str]

# %% Resposta estruturada
class TriageDecision(BaseModel):
    """Formato obrigatório da decisão produzida pela LLM."""

    action: Literal["respond", "notify", "ignore"] = Field(
        description="Ação escolhida para o e-mail."
    )
    reason: str = Field(description="Explicação objetiva da decisão.")
    draft: str | None = Field(
        default=None,
        description="Rascunho em português quando a ação for respond.",
    )


structured_llm = llm.with_structured_output(TriageDecision)

# %% Namespaces de memória
def semantic_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "semantic")


def episodic_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "episodic")


def procedural_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "procedural")

# %% Memória semântica
def add_semantic_memory(user_id: str, fact: str) -> str:
    """Armazena um fato ou preferência: aquilo que o agente sabe."""
    if not fact.strip():
        raise ValueError("A memória semântica não pode estar vazia.")

    memory_id = str(uuid4())
    try:
        store.put(
            semantic_namespace(user_id),
            memory_id,
            {"type": "semantic", "text": fact.strip()},
        )
    except Exception as error:
        raise RuntimeError(
            "Falha ao gerar o embedding ou salvar a memória semântica. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error
    return memory_id

# %% Memória episódica
def add_episodic_memory(
    user_id: str,
    email: str,
    action: Literal["respond", "notify", "ignore"],
    learning: str,
) -> str:
    """Armazena um caso anterior: aquilo que aconteceu antes."""
    if not email.strip() or not learning.strip():
        raise ValueError("E-mail e aprendizado do episódio são obrigatórios.")

    memory_id = str(uuid4())
    text = f"E-mail: {email.strip()}\nAção: {action}\nAprendizado: {learning.strip()}"
    try:
        store.put(
            episodic_namespace(user_id),
            memory_id,
            {
                "type": "episodic",
                "email": email.strip(),
                "action": action,
                "learning": learning.strip(),
                "text": text,
            },
        )
    except Exception as error:
        raise RuntimeError(
            "Falha ao gerar o embedding ou salvar a memória episódica. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error
    return memory_id

# %% Memória procedural
def set_procedural_memory(user_id: str, rules: str) -> None:
    """Substitui as regras atuais: a maneira como o agente deve agir."""
    if not rules.strip():
        raise ValueError("As regras procedurais não podem estar vazias.")

    try:
        store.put(
            procedural_namespace(user_id),
            "current",
            {"type": "procedural", "rules": rules.strip(), "text": rules.strip()},
            index=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Falha ao salvar a memória procedural. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error

# %% Criar memórias iniciais
USER_ID = "usuario-demo"

INITIAL_PROCEDURAL_RULES = """
1. Responda automaticamente apenas a solicitações simples.
2. Assuntos financeiros devem ser encaminhados para um humano.
3. Mensagens apenas de agradecimento podem ser ignoradas.
4. Nunca invente informações.
5. Responda em português.
""".strip()


def create_initial_memories() -> None:
    """Prepara dados de demonstração para o usuário acadêmico."""
    add_semantic_memory(USER_ID, "O usuário prefere respostas profissionais e objetivas.")
    add_semantic_memory(USER_ID, "Assuntos financeiros são considerados prioridade alta.")

    add_episodic_memory(
        USER_ID,
        "Fui cobrado duas vezes pela mesma compra.",
        "notify",
        "Cobrança duplicada deve ser revisada por um humano.",
    )
    add_episodic_memory(
        USER_ID,
        "Obrigado pelo atendimento, o problema já foi resolvido.",
        "ignore",
        "Mensagens somente de agradecimento não exigem nova resposta.",
    )
    set_procedural_memory(USER_ID, INITIAL_PROCEDURAL_RULES)


if __name__ == "__main__":
    create_initial_memories()
    print("Memórias iniciais criadas.")

# %% Recuperar memórias antes de responder
def search_relevant_memories(
    memory_store: Any,
    namespace: tuple[str, str],
    query: str,
    limit: int = 3,
) -> list[Any]:
    """Executa busca semântica e torna falhas de embedding compreensíveis."""
    try:
        return list(memory_store.search(namespace, query=query, limit=limit))
    except Exception as error:
        raise RuntimeError(
            "Falha ao gerar embeddings ou buscar memórias semelhantes. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error


def triage_email(state: EmailState, runtime: Runtime[Context]) -> EmailState:
    """Recupera as três memórias, monta o contexto e solicita uma decisão."""
    email_text = state.get("email_text", "").strip()
    if not email_text:
        raise ValueError("O texto do e-mail não pode estar vazio.")
    if runtime.store is None:
        raise RuntimeError("Store indisponível: compile o grafo com store=store.")

    user_id = runtime.context.user_id
    semantic_items = search_relevant_memories(
        runtime.store, semantic_namespace(user_id), email_text
    )
    episodic_items = search_relevant_memories(
        runtime.store, episodic_namespace(user_id), email_text
    )
    procedural_item = runtime.store.get(procedural_namespace(user_id), "current")

    semantic_texts = [item.value["text"] for item in semantic_items]
    episodic_texts = [item.value["text"] for item in episodic_items]
    procedural_rules = (
        procedural_item.value["rules"]
        if procedural_item
        else "Nenhuma regra procedural armazenada. Não invente informações."
    )

    # Store vazio é tratado explicitamente e aparece no contexto enviado à LLM.
    semantic_context = "\n".join(f"- {text}" for text in semantic_texts) or "- Nenhuma memória semântica encontrada."
    episodic_context = "\n\n".join(episodic_texts) or "Nenhuma memória episódica encontrada."
    prompt = build_triage_prompt(
        procedural_rules,
        semantic_context,
        episodic_context,
        email_text,
    )

    try:
        raw_decision = structured_llm.invoke(
            [SystemMessage(content=TRIAGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
    except ValidationError as error:
        raise RuntimeError(
            f"A LLM retornou uma resposta estruturada inválida: {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(
            "Falha ao chamar a LLM. Verifique chave, conexão, modelo e limites. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error

    try:
        decision = (
            raw_decision
            if isinstance(raw_decision, TriageDecision)
            else TriageDecision.model_validate(raw_decision)
        )
    except ValidationError as error:
        raise RuntimeError(
            f"Resposta estruturada incompatível com TriageDecision: {error}"
        ) from error

    memories_used = [
        *(f"SEMÂNTICA: {text}" for text in semantic_texts),
        *(f"EPISÓDICA: {text}" for text in episodic_texts),
        f"PROCEDURAL: {procedural_rules}",
    ]
    return {"result": decision.model_dump(), "memories_used": memories_used}

# %% Prompt do agente
TRIAGE_SYSTEM_PROMPT = """
Você é um assistente de triagem de e-mails. Use as memórias apenas como contexto.
Não copie cegamente decisões anteriores: analise o caso atual. Não invente fatos.
Responda em português e obedeça ao formato estruturado solicitado.
""".strip()


def build_triage_prompt(
    procedural_rules: str,
    semantic_context: str,
    episodic_context: str,
    email_text: str,
) -> str:
    return f"""
## Instruções procedurais
{procedural_rules}

## Memórias semânticas relevantes
{semantic_context}

## Experiências anteriores relevantes
{episodic_context}

## Novo e-mail
{email_text}

Decida entre respond, notify ou ignore. Explique brevemente o motivo. Inclua um
draft somente quando a ação for respond.
""".strip()

# %% Construção do LangGraph
# O foco deste experimento é memória; não há Tool Calling.
builder = StateGraph(EmailState, context_schema=Context)
builder.add_node("triage_email", triage_email)
builder.add_edge(START, "triage_email")
builder.add_edge("triage_email", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

# %% Visualização do grafo
if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

# %% Primeiro teste
def run_triage(email_text: str, thread_id: str, user_id: str = USER_ID) -> EmailState:
    """Executa uma thread com o mesmo Store de longo prazo."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"email_text": email_text},
        config=config,
        context=Context(user_id=user_id),
    )


def show_result(title: str, result: EmailState) -> None:
    decision = result["result"]
    print(f"\n{title}")
    print(f"Decisão: {decision['action']}")
    print(f"Motivo: {decision['reason']}")
    print(f"Draft: {decision.get('draft') or '(não necessário)'}")
    print("Memórias utilizadas:")
    for memory in result.get("memories_used", []):
        print(f"- {memory}")


if __name__ == "__main__":
    email_1 = """Olá. Percebi que o valor da minha assinatura foi cobrado duas
vezes no cartão. Podem verificar?"""
    result_1 = run_triage(email_1, "thread-001")
    show_result("TESTE 1 — episódio semelhante", result_1)

# %% Nova thread, mesmo usuário
# thread_id diferente = nova conversa e novo estado de curto prazo.
# user_id igual = as mesmas memórias de longo prazo continuam disponíveis.
if __name__ == "__main__":
    email_2 = "Obrigado, consegui resolver a solicitação anterior."
    result_2 = run_triage(email_2, "thread-002")
    show_result("TESTE 2 — nova thread, mesmo usuário", result_2)

# %% Demonstrar memória semântica nova
if __name__ == "__main__":
    add_semantic_memory(
        USER_ID,
        "O usuário prefere que solicitações simples de redefinição de senha "
        "sejam respondidas automaticamente.",
    )
    result_3 = run_triage(
        "Esqueci minha senha. Podem me orientar a redefini-la?",
        "thread-003",
    )
    show_result("TESTE 3 — nova memória semântica", result_3)

# %% Demonstrar memória episódica nova
if __name__ == "__main__":
    add_episodic_memory(
        USER_ID,
        "Esqueci minha senha.",
        "respond",
        "Pedidos simples de redefinição de senha podem receber orientação automática.",
    )
    result_4 = run_triage(
        "Não consigo entrar na conta porque esqueci minha senha.",
        "thread-004",
    )
    show_result("TESTE 4 — nova memória episódica", result_4)

# %% Memória procedural por feedback humano
def message_text(message: Any) -> str:
    """Extrai texto de uma mensagem sem depender de métodos depreciados."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("text")
        ).strip()
    return ""


def update_procedural_from_feedback(user_id: str, feedback: str) -> str:
    """Usa feedback humano para reescrever e salvar as regras de comportamento."""
    if not feedback.strip():
        raise ValueError("O feedback humano não pode estar vazio.")

    current = store.get(procedural_namespace(user_id), "current")
    current_rules = current.value["rules"] if current else INITIAL_PROCEDURAL_RULES
    prompt = f"""
Reescreva as instruções procedurais abaixo incorporando integralmente o feedback
humano. Preserve regras que não entrarem em conflito. Retorne somente a lista de
regras atualizada, em português.

INSTRUÇÕES ATUAIS:
{current_rules}

FEEDBACK HUMANO:
{feedback}
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content="Você atualiza políticas operacionais com precisão."),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as error:
        raise RuntimeError(
            "Falha ao chamar a LLM para atualizar a memória procedural. "
            f"Detalhe: {type(error).__name__}: {error}"
        ) from error

    updated_rules = message_text(response)
    if not updated_rules:
        raise RuntimeError("A LLM retornou regras procedurais vazias.")

    # Mantém o feedback literal como trilha de auditoria e regra obrigatória.
    updated_rules = f"{updated_rules}\n\nRegra adicionada por feedback humano:\n{feedback.strip()}"
    set_procedural_memory(user_id, updated_rules)
    return updated_rules


if __name__ == "__main__":
    HUMAN_FEEDBACK = """Nunca responda automaticamente a mensagens envolvendo
cobranças, pagamentos, estornos ou valores financeiros. Sempre classifique como notify."""
    updated_rules = update_procedural_from_feedback(USER_ID, HUMAN_FEEDBACK)
    print("\nREGRAS ATUALIZADAS POR FEEDBACK HUMANO\n", updated_rules)

# %% Testar após feedback
if __name__ == "__main__":
    result_5 = run_triage(
        "Minha fatura veio com um valor diferente do contratado. Vocês conseguem corrigir?",
        "thread-005",
    )
    show_result("TESTE 5 — após feedback humano", result_5)
    if result_5["result"]["action"] != "notify":
        print("ATENÇÃO: a decisão não foi notify; revise a saída e o modelo configurado.")

# %% Inspecionar a memória
def inspect_namespace(title: str, namespace: tuple[str, str]) -> None:
    print(f"\n{title}")
    items = list(store.search(namespace, limit=100))
    if not items:
        print("(Store vazio para este namespace)")
        return
    for item in items:
        print(f"- chave={item.key}: {item.value}")


if __name__ == "__main__":
    inspect_namespace("MEMÓRIA SEMÂNTICA", semantic_namespace(USER_ID))
    inspect_namespace("MEMÓRIA EPISÓDICA", episodic_namespace(USER_ID))
    inspect_namespace("MEMÓRIA PROCEDURAL", procedural_namespace(USER_ID))

# %% [markdown]
# ## Explicação acadêmica
#
# ### Memória semântica
#
# Guarda fatos e preferências. Responde: **O que o agente sabe?**
#
# ### Memória episódica
#
# Guarda experiências anteriores, ações e aprendizados. Responde:
# **O que aconteceu antes em casos parecidos?**
#
# ### Memória procedural
#
# Guarda regras e comportamento. Responde: **Como o agente deve executar a tarefa?**
#
# A memória fica fora da LLM. A LLM recebe somente o contexto recuperado, e o
# LangGraph controla quando recuperar e usar essas memórias.

# %% [markdown]
# ## Curto prazo x longo prazo
#
# ```text
# Thread 001 ─┐
# Thread 002 ─┼── user_id ──► Long-Term Store
# Thread 003 ─┘                    │
#                    ┌─────────────┼─────────────┐
#                    ▼             ▼             ▼
#                Semântica      Episódica    Procedural
# ```
#
# - **checkpointer:** salva o estado da thread, identificado por `thread_id`.
# - **Store:** guarda memórias de longo prazo organizadas por `user_id` e namespace.
# - Threads diferentes podem recuperar as mesmas memórias quando usam o mesmo usuário.
