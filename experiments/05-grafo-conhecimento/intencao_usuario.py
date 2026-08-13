# %% [markdown]
# # Experimento 05.1 — Understanding User Intent
#
# **Interpretando a intenção do usuário antes da construção de um Knowledge Graph**
#
# Um Knowledge Graph combina entidades, relacionamentos e propriedades. Antes de
# decidir como modelar `Cliente → Pedido → Produto` ou `Cliente → Avaliação → Produto`,
# precisamos compreender quais perguntas o usuário realmente quer responder.

# %% [markdown]
# ## O que é um Knowledge Graph?
#
# “João comprou Notebook” pode ser representado como
# `(João)-[:COMPROU]->(Notebook)`. “João avaliou Notebook” pode virar
# `(João)-[:AVALIOU]->(Notebook)`.
#
# - Nó: entidade.
# - Relacionamento: ligação entre entidades.
# - Propriedade: característica de um nó ou relacionamento.

# %% [markdown]
# ## Por que entender a intenção?
#
# A mesma base pode gerar grafos diferentes. “Analisar vendas por cliente” prioriza
# Cliente, Pedido e Produto. “Entender o sentimento sobre produtos” pode priorizar
# Cliente, Produto, Avaliação e Sentimento.
#
# **Dados não definem sozinhos o Knowledge Graph. A intenção ajuda a definir o modelo.**

# %% [markdown]
# ## Instalação
#
# ```bash
# pip install -r requirements.txt
# ```
#
# Opcional no notebook:
#
# ```python
# # %pip install -U google-adk google-genai python-dotenv pydantic pandas ipykernel
# ```
#
# Neo4j ainda não é usado e, por isso, não é uma dependência desta etapa.

# %%
# Configuração
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def locate_project_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    current = Path.cwd().resolve()
    candidates = [current, current / "experiments" / "05-grafo-conhecimento"]
    candidates.extend(parent / "experiments" / "05-grafo-conhecimento" for parent in current.parents)
    for candidate in candidates:
        if (candidate / "requirements.txt").is_file():
            return candidate
    return current


PROJECT_DIR = locate_project_dir()
REPO_ROOT = next(
    (path for path in (PROJECT_DIR, *PROJECT_DIR.parents) if (path / "experiments").is_dir()),
    PROJECT_DIR,
)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(PROJECT_DIR / ".env", override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip().removeprefix("groq/")
RUN_LIVE_DEMOS = os.getenv("RUN_LIVE_DEMOS", "false").lower() == "true"


def validate_configuration(*, show_messages: bool = True) -> bool:
    problems: list[str] = []
    if not MODEL:
        problems.append("GROQ_MODEL ausente.")
    if not GROQ_API_KEY or GROQ_API_KEY == "cole_sua_chave_aqui":
        problems.append("GROQ_API_KEY ausente ou ainda com o valor de exemplo.")

    if show_messages:
        if problems:
            print("Configuração incompleta:")
            for problem in problems:
                print(f"- {problem}")
            print("Copie .env.example para .env e configure o provedor escolhido.")
        else:
            print("Groq configurada via LiteLLM no Google ADK.")
            print(f"Modelo selecionado: {MODEL}")
    return not problems


CONFIG_OK = validate_configuration(show_messages=__name__ == "__main__")

# %%
# Imports atuais do Google ADK 2.x e das bibliotecas auxiliares
import asyncio
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

import pandas as pd
from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# %%
# Modelo estruturado da intenção
class UserIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, description="Objetivo principal do Knowledge Graph.")
    domain: str = Field(min_length=1, description="Domínio do problema, por exemplo vendas ou pesquisa.")
    business_questions: list[str] = Field(default_factory=list)
    candidate_entities: list[str] = Field(default_factory=list)
    candidate_relationships: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    ready_for_next_step: bool
    confidence: float = Field(
        ge=0,
        le=1,
        description="Indicador didático entre 0 e 1; não é probabilidade cientificamente calibrada.",
    )

# %%
# System prompt do Intent Agent
INTENT_SYSTEM_PROMPT = """Você é o agente que compreende a intenção de uma pessoa que deseja construir um Knowledge Graph.

Sua função NÃO é construir o grafo, escolher arquivos definitivamente, criar um schema definitivo, executar Cypher ou acessar dados externos. Transforme somente a solicitação recebida em uma especificação estruturada para os próximos agentes.

Identifique: objetivo, domínio, perguntas de negócio, possíveis entidades, possíveis relacionamentos, requisitos de dados, restrições, ambiguidades e perguntas de esclarecimento.

Regras:
1. Não invente requisitos que o usuário não declarou.
2. Entidades e relacionamentos são candidatos, nunca decisões finais de schema.
3. Não crie dados inexistentes nem alegue ter inspecionado arquivos.
4. Não execute nem proponha como definitiva qualquer query Cypher.
5. Não presuma o significado de termos ambíguos.
6. Se faltarem objetivo, perguntas de negócio, domínio ou dados relevantes, registre ambiguidades, gere perguntas de esclarecimento e defina ready_for_next_step=false.
7. Use confidence apenas como indicador didático de completude da interpretação, não como probabilidade calibrada.
8. Responda em português e obedeça rigorosamente ao schema de saída.
"""

# %%
# Criação do Intent Agent com JSON validado pela aplicação
def create_intent_agent(model: str = MODEL) -> Agent:
    if not model.strip():
        raise ValueError("Modelo inválido: GROQ_MODEL está vazio.")
    schema = json.dumps(UserIntent.model_json_schema(), ensure_ascii=False)
    instruction = f"""{INTENT_SYSTEM_PROMPT}

Retorne SOMENTE um objeto JSON válido, sem Markdown e sem comentários. O objeto
deve obedecer exatamente a este JSON Schema:
{schema}
""".strip()
    return Agent(
        name="intent_agent",
        model=LiteLlm(model=f"groq/{model.removeprefix('groq/')}"),
        description=(
            "Analisa solicitações de usuários e produz uma representação estruturada "
            "da intenção para planejamento de Knowledge Graphs."
        ),
        instruction=instruction,
        output_key="current_intent",
    )


intent_agent = create_intent_agent()

# %%
# Runner, sessão e função pública de execução
APP_NAME = "knowledge_graph_intent_lab"
USER_ID = "academic_user"
SESSION_SERVICE = InMemorySessionService()
INTENT_RUNNER = Runner(
    app_name=APP_NAME,
    agent=intent_agent,
    session_service=SESSION_SERVICE,
    auto_create_session=True,
)
MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("GROQ_MIN_REQUEST_INTERVAL", "12"))
_last_groq_request_at = 0.0


def _wait_for_groq_rate_window() -> None:
    """Espaça chamadas consecutivas para respeitar limites gratuitos de TPM."""
    global _last_groq_request_at
    remaining = MIN_REQUEST_INTERVAL_SECONDS - (
        time.monotonic() - _last_groq_request_at
    )
    if remaining > 0:
        time.sleep(remaining)
    _last_groq_request_at = time.monotonic()


def describe_groq_error(error: Exception) -> str:
    name = type(error).__name__.lower()
    detail = str(error).strip() or "sem detalhe retornado pelo serviço"
    normalized = detail.lower()
    if "api key" in normalized or "unauth" in name or "authentication" in normalized:
        return f"Erro de autenticação na API Groq: {detail}"
    if "resourceexhausted" in name or "quota" in normalized or "429" in normalized:
        return f"Erro de quota da API Groq: {detail}"
    if "notfound" in name or "not found" in normalized or "invalid model" in normalized:
        return f"Modelo inválido ou indisponível ('{MODEL}'): {detail}"
    if "permission" in normalized or "forbidden" in name:
        return f"Erro de permissão no provedor Groq: {detail}"
    return f"Erro da API Groq via Google ADK ({type(error).__name__}): {detail}"


def _event_payload(event: Any) -> UserIntent | dict[str, Any] | str | None:
    output = getattr(event, "output", None)
    if isinstance(output, (UserIntent, dict, str)):
        return output
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content else None
    texts = [part.text for part in (parts or []) if getattr(part, "text", None)]
    return "\n".join(texts).strip() or None


def _validate_intent_payload(payload: UserIntent | dict[str, Any] | str) -> UserIntent:
    if isinstance(payload, UserIntent):
        return payload
    try:
        if isinstance(payload, dict):
            data = dict(payload)
        else:
            text = payload.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1]).strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("A resposta não contém um objeto JSON.")
            data = json.loads(text[start : end + 1])
        for field_name in ("goal", "domain"):
            if not str(data.get(field_name, "")).strip():
                data[field_name] = "não informado"
        return UserIntent.model_validate(data)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"Structured output inválido ou falha Pydantic: {error}") from error


def _run_coroutine_in_worker(coroutine: Any) -> Any:
    """Executa operações assíncronas de sessão sem conflitar com o loop do Jupyter."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def _persist_intent_metadata(session_id: str, intent: UserIntent) -> None:
    session = await SESSION_SERVICE.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session is None:
        return
    event = Event(
        author="intent_agent",
        actions=EventActions(
            state_delta={
                "user_domain": intent.domain,
                "ready_for_next_step": intent.ready_for_next_step,
            }
        ),
    )
    await SESSION_SERVICE.append_event(session, event)


def analyze_user_intent(
    user_request: str,
    *,
    session_id: str | None = None,
    clarification_answers: list[str] | None = None,
    runner: Any | None = None,
) -> UserIntent:
    """Converte linguagem natural em UserIntent e esconde Runner/Session do restante da aplicação."""
    clean_request = user_request.strip()
    if not clean_request:
        raise ValueError("Intenção vazia: informe uma solicitação em linguagem natural.")
    if runner is None and not CONFIG_OK:
        raise RuntimeError(
            "Configuração da API ausente. Defina GROQ_API_KEY no .env da raiz."
        )

    active_runner = runner or INTENT_RUNNER
    active_session_id = session_id or f"intent-{uuid.uuid4().hex}"
    message = types.Content(role="user", parts=[types.Part.from_text(text=clean_request)])
    payload: UserIntent | dict[str, Any] | str | None = None

    for attempt in range(3):
        _wait_for_groq_rate_window()
        try:
            events: Iterable[Any] = active_runner.run(
                user_id=USER_ID,
                session_id=active_session_id,
                new_message=message,
                state_delta={
                    "last_user_request": clean_request,
                    "clarifications": clarification_answers or [],
                },
            )
            for event in events:
                if getattr(event, "error_code", None) or getattr(event, "error_message", None):
                    raise RuntimeError(
                        f"{getattr(event, 'error_code', 'erro')}: "
                        f"{getattr(event, 'error_message', 'falha sem detalhe')}"
                    )
                if event.is_final_response():
                    payload = _event_payload(event)
            break
        except Exception as error:
            detail = str(error)
            rate_limited = "429" in detail or "rate limit" in detail.lower()
            if rate_limited and attempt < 2:
                match = re.search(r"try again in ([0-9.]+)s", detail, re.IGNORECASE)
                time.sleep(float(match.group(1)) + 1 if match else 15)
                continue
            raise RuntimeError(describe_groq_error(error)) from error

    if payload is None:
        raise ValueError("Resposta vazia: o Intent Agent não retornou conteúdo estruturado.")
    intent = _validate_intent_payload(payload)
    if active_runner is INTENT_RUNNER:
        _run_coroutine_in_worker(_persist_intent_metadata(active_session_id, intent))
    return intent

# %%
# Primeiro teste — executado somente quando autorizado
FIRST_REQUEST = """Quero construir um grafo que me permita descobrir quais clientes
compraram quais produtos, quais pedidos foram realizados e quais avaliações esses
clientes fizeram sobre os produtos."""

first_intent: UserIntent | None = None
if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    first_intent = analyze_user_intent(FIRST_REQUEST, session_id="first-intent-demo")
    print(first_intent.model_dump_json(indent=2))
elif __name__ == "__main__":
    print("Primeiro teste não executado: RUN_LIVE_DEMOS=false ou configuração incompleta.")

# %%
# Visualização didática da intenção
def display_intent(intent: UserIntent) -> pd.DataFrame:
    def lines(values: list[str]) -> str:
        return "\n".join(f"• {value}" for value in values) or "—"

    rows = [
        ("OBJETIVO", intent.goal),
        ("DOMÍNIO", intent.domain),
        ("PERGUNTAS DE NEGÓCIO", lines(intent.business_questions)),
        ("ENTIDADES CANDIDATAS", lines(intent.candidate_entities)),
        ("RELACIONAMENTOS CANDIDATOS", lines(intent.candidate_relationships)),
        ("DADOS NECESSÁRIOS", lines(intent.data_requirements)),
        ("RESTRIÇÕES", lines(intent.constraints)),
        ("AMBIGUIDADES", lines(intent.ambiguities)),
        ("PERGUNTAS PARA O USUÁRIO", lines(intent.clarification_questions)),
        ("PRONTO PARA PRÓXIMA ETAPA?", "Sim" if intent.ready_for_next_step else "Não"),
        ("CONFIANÇA DIDÁTICA", f"{intent.confidence:.2f}"),
    ]
    frame = pd.DataFrame(rows, columns=["Campo", "Interpretação"])
    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__" and first_intent is not None:
    display_intent(first_intent)

# %%
# Intenção incompleta: o modelo deve pedir esclarecimentos
INCOMPLETE_REQUEST = "Quero criar um grafo dos meus dados para descobrir coisas interessantes."

if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    incomplete_intent = analyze_user_intent(INCOMPLETE_REQUEST, session_id="incomplete-demo")
    display_intent(incomplete_intent)

# %%
# Clarification loop: combina a resposta humana e reanalisa a intenção
def merge_clarification(
    original_request: str,
    clarification_answer: str,
    *,
    session_id: str | None = None,
) -> UserIntent:
    if not clarification_answer.strip():
        raise ValueError("A resposta de esclarecimento não pode estar vazia.")
    combined_request = (
        "SOLICITAÇÃO ORIGINAL:\n"
        f"{original_request.strip()}\n\n"
        "ESCLARECIMENTO POSTERIOR DO USUÁRIO:\n"
        f"{clarification_answer.strip()}"
    )
    return analyze_user_intent(
        combined_request,
        session_id=session_id,
        clarification_answers=[clarification_answer.strip()],
    )


if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    clarification_session = "clarification-demo"
    before = analyze_user_intent("Quero analisar meus clientes.", session_id=clarification_session)
    after = merge_clarification(
        "Quero analisar meus clientes.",
        "Quero saber quais produtos cada cliente compra e quais recebem avaliações negativas.",
        session_id=clarification_session,
    )
    print("ANTES DA CLARIFICAÇÃO")
    display_intent(before)
    print("DEPOIS DA CLARIFICAÇÃO")
    display_intent(after)

# %% [markdown]
# ## Intenção não é schema
#
# A intenção “Cliente compra Produto” não obriga o schema final a ser
# `(:Cliente)-[:COMPRA]->(:Produto)`. O Schema Agent pode concluir que a estrutura
# adequada é `(:Cliente)-[:REALIZOU]->(:Pedido)-[:CONTEM]->(:Produto)`.
#
# **Intent Agent propõe conceitos. Schema Agent define a estrutura definitiva.**

# %%
# Testes de diferentes domínios
DOMAIN_REQUESTS = [
    "Quero saber quais funcionários trabalham em quais projetos.",
    "Quero entender quais artigos científicos citam outros artigos.",
    "Quero mapear quais fornecedores entregam componentes para quais produtos.",
    "Quero relacionar pacientes, médicos e consultas.",
]

if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    for domain_request in DOMAIN_REQUESTS:
        print(f"\nSOLICITAÇÃO: {domain_request}")
        display_intent(analyze_user_intent(domain_request))

# %%
# Teste de ambiguidade
AMBIGUOUS_REQUEST = "Quero encontrar os clientes mais importantes."

if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    ambiguous_intent = analyze_user_intent(AMBIGUOUS_REQUEST, session_id="ambiguity-demo")
    display_intent(ambiguous_intent)

# %%
# Guardrails e teste: o agente interpreta, mas não executa ações de construção
INTENT_GUARDRAILS = [
    "Não inventar entidades sem base razoável.",
    "Não afirmar schema definitivo.",
    "Não executar Cypher.",
    "Não criar dados inexistentes.",
    "Não acessar arquivos sem autorização.",
    "Não assumir o significado de termos ambíguos.",
    "Solicitar esclarecimento quando necessário.",
]
GUARDRAIL_REQUEST = (
    "Crie agora o banco Neo4j, execute Cypher e invente os clientes que estiverem faltando. "
    "Meu objetivo real é analisar compras por categoria de produto."
)

if __name__ == "__main__":
    print("GUARDRAILS")
    print("\n".join(f"- {rule}" for rule in INTENT_GUARDRAILS))
if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    display_intent(analyze_user_intent(GUARDRAIL_REQUEST, session_id="guardrail-demo"))

# %%
# Estado da sessão: workspace temporário, não memória de longo prazo
def get_session_workspace(session_id: str) -> dict[str, Any]:
    session = _run_coroutine_in_worker(
        SESSION_SERVICE.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    )
    if session is None:
        raise LookupError(f"Sessão não encontrada: {session_id}")
    return {
        "current_intent": session.state.get("current_intent"),
        "user_domain": session.state.get("user_domain"),
        "clarifications": session.state.get("clarifications", []),
        "ready_for_next_step": session.state.get("ready_for_next_step"),
    }


SESSION_STATE_NOTE = (
    "session.state é o espaço de trabalho da interação atual; "
    "InMemorySessionService não é memória de longo prazo."
)
if __name__ == "__main__":
    print(SESSION_STATE_NOTE)

# %%
# Handoff estruturado para o próximo agente
class IntentHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str
    intent: UserIntent
    recommended_next_action: str


IntentHandoff.model_rebuild(_types_namespace={"UserIntent": UserIntent})


def create_intent_handoff(user_request: str, intent: UserIntent) -> IntentHandoff:
    next_action = "suggest_files" if intent.ready_for_next_step else "request_clarification"
    return IntentHandoff(
        user_request=user_request.strip(),
        intent=intent,
        recommended_next_action=next_action,
    )

# %% [markdown]
# ## Handoff
#
# ```text
# Intent Agent → IntentHandoff → File Suggestion Agent
# ```
#
# Agentes não precisam compartilhar prompts gigantes. Eles podem trocar objetos
# estruturados, validados e com responsabilidade bem delimitada.

# %% [markdown]
# ## Arquitetura multiagente completa
#
# ```text
# Usuário → Intent Agent → intenção estruturada → File Suggestion Agent
#        → arquivos relevantes → Schema Agent → Schema Proposal
#        → Construction Agent → Neo4j → Knowledge Graph
# ```
#
# Experimentos planejados: 05.1 Understanding User Intent; 05.2 File Suggestions;
# 05.3 Schema Proposal; 05.4 Knowledge Graph Construction. Este notebook implementa
# apenas o primeiro agente.

# %% [markdown]
# ## Preparação para Neo4j
#
# Fluxo futuro: `Intenção → Schema → Cypher → Neo4j → Knowledge Graph`.
#
# Exemplo exclusivamente ilustrativo, não executado:
#
# ```cypher
# CREATE (:Cliente {id: "C001"})
# CREATE (:Produto {id: "P001"})
# MATCH (c:Cliente {id: "C001"})
# MATCH (p:Produto {id: "P001"})
# CREATE (c)-[:COMPROU]->(p)
# ```
#
# Cypher será responsabilidade de experimentos posteriores.

# %% [markdown]
# ## Knowledge Graph × RAG tradicional
#
# RAG tradicional: `Documento → Chunk → Embedding → Vector DB`.
#
# Knowledge Graph explicita relações, por exemplo:
# `Cliente → COMPROU → Produto`, `Cliente → ESCREVEU → Avaliação` e
# `Avaliação → SOBRE → Produto`.
#
# As abordagens podem trabalhar juntas; Knowledge Graph não substitui necessariamente
# um banco vetorial.

# %%
# Debug didático
def debug_intent_analysis(request: str, intent: UserIntent) -> dict[str, Any]:
    handoff = create_intent_handoff(request, intent)
    debug_data = {
        "1_solicitacao_original": request,
        "2_objetivo": intent.goal,
        "3_dominio": intent.domain,
        "4_entidades_candidatas": intent.candidate_entities,
        "5_relacionamentos_candidatos": intent.candidate_relationships,
        "6_ambiguidades": intent.ambiguities,
        "7_perguntas_esclarecimento": intent.clarification_questions,
        "8_confianca_didatica": intent.confidence,
        "9_decisao_handoff": handoff.recommended_next_action,
    }
    print(json.dumps(debug_data, ensure_ascii=False, indent=2))
    return debug_data

# %%
# Avaliação simples e didática (não é uma métrica científica)
TEST_CASES = [
    {
        "request": "Quero relacionar clientes, pedidos e produtos para analisar compras.",
        "expected_entities": ["cliente", "pedido", "produto"],
        "expect_ambiguity": False,
    },
    {
        "request": "Quero encontrar os clientes mais importantes.",
        "expected_entities": ["cliente"],
        "expect_ambiguity": True,
    },
    {
        "request": "Quero criar um grafo para descobrir coisas interessantes.",
        "expected_entities": [],
        "expect_ambiguity": True,
    },
]


def evaluate_intent_cases(
    test_cases: list[dict[str, Any]],
    analyzer: Callable[[str], UserIntent] = analyze_user_intent,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for case in test_cases:
        intent = analyzer(case["request"])
        actual_entities = " ".join(intent.candidate_entities).lower()
        entities_ok = all(entity.lower() in actual_entities for entity in case["expected_entities"])
        ambiguity_detected = bool(intent.ambiguities or intent.clarification_questions)
        results.append(
            {
                "request": case["request"],
                "entities_ok": entities_ok,
                "ambiguity_ok": ambiguity_detected == case["expect_ambiguity"],
                "ready_for_next_step": intent.ready_for_next_step,
                "has_clarification_questions": bool(intent.clarification_questions),
            }
        )
    frame = pd.DataFrame(results)
    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__" and RUN_LIVE_DEMOS and CONFIG_OK:
    evaluate_intent_cases(TEST_CASES)

# %% [markdown]
# ## O que aprendi neste experimento
#
# 1. Um Knowledge Graph começa com uma pergunta de negócio.
# 2. A intenção deve ser entendida antes da modelagem.
# 3. Entidades e relacionamentos começam como candidatos.
# 4. Ambiguidades devem gerar perguntas de esclarecimento.
# 5. O Intent Agent não assume responsabilidades do Schema Agent.
# 6. Saídas estruturadas facilitam a comunicação entre agentes.
# 7. O Google ADK coordena agentes especializados e sessões.
# 8. O resultado do Intent Agent alimentará os próximos agentes.
#
# **Antes de construir o grafo, o sistema precisa entender qual problema o usuário
# realmente quer resolver.**
#
# O Intent Agent não constrói o grafo: ele traduz a intenção humana em uma
# especificação estruturada para os próximos agentes.
