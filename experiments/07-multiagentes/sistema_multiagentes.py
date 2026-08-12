"""Experimento 07.1: pesquisa, escrita e revisão com uma Crew sequencial.

Implementação acadêmica original inspirada apenas nos conceitos de sistemas
multiagentes. Importar este módulo não realiza chamadas pagas; a execução real
acontece apenas ao executar o arquivo ou a célula 16 do notebook com uma chave.
"""

# %% [markdown] — Célula 1
# # Experimento 07.1 — Sistema Multiagente com CrewAI
# ## Agentes colaborando para pesquisar, escrever e revisar um artigo
#
# Uma LLM única poderia receber “Escreva um artigo sobre IA”. Aqui o problema é
# decomposto em três responsabilidades: pesquisar/planejar → escrever → revisar.

# %% [markdown] — Célula 2
# ## LLM única × multiagentes
#
# **LLM única:** Usuário → prompt amplo → LLM → resultado.
#
# **Multiagentes:** Usuário → problema → Planejador → Redator → Editor → resultado.
#
# Multiagentes não implica modelos diferentes: os agentes podem compartilhar o
# mesmo modelo e se especializar por papel, objetivo, instruções, ferramentas e tarefa.

# %% [markdown] — Célula 3
# ## Instalação
#
# ```bash
# pip install -r requirements.txt
# ```
#
# O extra `crewai[tools]` já está nos requisitos porque a célula 29 implementa,
# de forma opcional, `SerperDevTool`.

# %% — Célula 4: configuração
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _project_dir() -> Path:
    """Localiza a pasta do experimento no script e no kernel do notebook."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    candidates = [Path.cwd(), Path.cwd() / "experiments" / "07-multiagentes"]
    return next(
        (path for path in candidates if (path / "requirements.txt").exists()),
        Path.cwd(),
    )


PROJECT_DIR = _project_dir()
REPO_ROOT = next((path for path in (PROJECT_DIR, *PROJECT_DIR.parents) if (path / "experiments").is_dir()), PROJECT_DIR)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(PROJECT_DIR / ".env", override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip().removeprefix("groq/") or "llama-3.3-70b-versatile"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
PLACEHOLDER_KEYS = {"", "cole_sua_chave_aqui", "sua_chave_aqui"}


def has_groq_key() -> bool:
    return GROQ_API_KEY.lower() not in PLACEHOLDER_KEYS


print("Configuração carregada.")
print(f"Modelo: {GROQ_MODEL}")
print(f"GROQ_API_KEY: {'configurada' if has_groq_key() else 'ausente'}")

# %% — Célula 5: imports e auxiliares
import time
from typing import Any, Tuple

import pandas as pd
from crewai import Agent, Crew, LLM, Process, Task, TaskOutput
from IPython.display import Markdown, display
from groq import Groq, GroqError


def create_llm() -> LLM:
    if not has_groq_key():
        raise RuntimeError(
            "GROQ_API_KEY ausente. Configure a chave no .env da raiz."
        )
    if not GROQ_MODEL:
        raise ValueError("Modelo inválido: GROQ_MODEL está vazio.")
    return LLM(model=f"groq/{GROQ_MODEL}", api_key=GROQ_API_KEY)


def optional_search_tools() -> list[Any]:
    """Retorna Serper somente quando sua chave opcional foi configurada."""
    if not SERPER_API_KEY:
        return []
    try:
        from crewai_tools import SerperDevTool
    except ImportError as exc:
        raise RuntimeError(
            "SERPER_API_KEY existe, mas crewai-tools não está instalado. "
            "Reinstale requirements.txt."
        ) from exc
    return [SerperDevTool(n_results=5)]


def article_quality_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """Guardrail simples: bloqueia apenas uma saída final vazia ou muito curta."""
    raw = getattr(output, "raw", "")
    if not isinstance(raw, str) or len(raw.strip()) < 200:
        return False, "A versão final precisa ser um artigo Markdown substancial."
    return True, raw.strip()


def raw_text(value: Any) -> str:
    raw = getattr(value, "raw", value)
    return "" if raw is None else str(raw).strip()


llm = create_llm() if has_groq_key() else None
if llm is None:
    print("Crew não montada ainda: configure GROQ_API_KEY para criar os agentes.")

# %% — Célula 6: tema
TOPIC = "Como sistemas multiagentes podem melhorar processos empresariais"
print(f"TEMA DO ARTIGO:\n{TOPIC}")

# %% — Célula 7: planejador/pesquisador
def create_planner(model: LLM, use_web_search: bool = False) -> Agent:
    tools = optional_search_tools() if use_web_search else []
    return Agent(
        role="Planejador e Pesquisador de Conteúdo",
        goal=(
            "Analisar o tema, identificar os pontos mais importantes e criar um "
            "plano completo para um artigo didático e tecnicamente correto."
        ),
        backstory=(
            "Você é especialista em pesquisa, planejamento editorial e inteligência "
            "artificial. Identifica conceitos essenciais, organiza uma linha de "
            "raciocínio e prepara o trabalho para o redator. Não inventa fontes."
        ),
        llm=model,
        tools=tools,
        verbose=True,
        allow_delegation=False,
    )


planner = create_planner(llm, bool(SERPER_API_KEY)) if llm else None

# %% — Célula 8: redator
def create_writer(model: LLM) -> Agent:
    return Agent(
        role="Redator Técnico",
        goal=(
            "Transformar o planejamento e a pesquisa em um artigo claro, didático "
            "e bem estruturado."
        ),
        backstory=(
            "Você é um redator especializado em tecnologia e inteligência artificial. "
            "Recebe o trabalho de outro especialista e o transforma em texto "
            "compreensível, coerente e envolvente, sem acrescentar fatos não sustentados."
        ),
        llm=model,
        verbose=True,
        allow_delegation=False,
    )


writer = create_writer(llm) if llm else None

# %% — Célula 9: editor
def create_editor(model: LLM) -> Agent:
    return Agent(
        role="Editor Técnico",
        goal=(
            "Revisar o artigo, eliminar inconsistências e melhorar clareza, estrutura "
            "e precisão técnica."
        ),
        backstory=(
            "Você é um editor experiente em conteúdo técnico. Verifica se o artigo é "
            "claro, coerente, profissional e adequado ao público, sem alterar fatos "
            "apenas para melhorar o estilo."
        ),
        llm=model,
        verbose=True,
        allow_delegation=False,
    )


editor = create_editor(llm) if llm else None

# %% — Célula 10: visualização dos agentes
AGENTS_TABLE = pd.DataFrame(
    [
        {
            "AGENTE": "Planner",
            "ROLE": "Planejador/Pesquisador",
            "GOAL": "Criar plano",
            "RESPONSABILIDADE": "Pesquisa e estrutura",
        },
        {
            "AGENTE": "Writer",
            "ROLE": "Redator Técnico",
            "GOAL": "Escrever artigo",
            "RESPONSABILIDADE": "Produção de conteúdo",
        },
        {
            "AGENTE": "Editor",
            "ROLE": "Editor Técnico",
            "GOAL": "Revisar",
            "RESPONSABILIDADE": "Qualidade final",
        },
    ]
)
display(AGENTS_TABLE)

# %% — Célula 11: task de planejamento
def create_planning_task(planner_agent: Agent) -> Task:
    return Task(
        description="""
Analise o tema: {topic}

Produza um plano detalhado para um artigo. Identifique objetivo, público-alvo,
principais conceitos, estrutura de seções, argumentos, exemplos, cuidados técnicos
e conclusão sugerida. Não invente fontes específicas; sinalize incertezas.
""".strip(),
        expected_output=(
            "Um plano estruturado com: 1. título sugerido; 2. público; 3. objetivo; "
            "4. tópicos; 5. ordem das seções; 6. argumentos; 7. exemplos; 8. conclusão."
        ),
        agent=planner_agent,
    )


planning_task = create_planning_task(planner) if planner else None

# %% — Célula 12: task de redação e contexto
def create_writing_task(writer_agent: Agent, plan_task: Task) -> Task:
    return Task(
        description="""
Use o plano do agente planejador para escrever o artigo completo sobre {topic}.
Inclua introdução, explicação dos conceitos, exemplos, subtítulos e conclusão.
Se uma afirmação técnica não estiver sustentada pelo plano, sinalize a limitação
em vez de apresentá-la como fato.
""".strip(),
        expected_output="Artigo completo em Markdown, didático e bem estruturado.",
        agent=writer_agent,
        context=[plan_task],
    )


writing_task = (
    create_writing_task(writer, planning_task) if writer and planning_task else None
)

# %% — Célula 13: task de edição e contexto
def create_editing_task(editor_agent: Agent, draft_task: Task) -> Task:
    return Task(
        description="""
Revise o artigo sobre {topic} produzido pelo redator. Verifique clareza, gramática,
coerência, precisão técnica, repetições, organização, título, subtítulos e conclusão.
Faça as correções necessárias, sem mudar o tema principal nem alterar fatos somente
para melhorar o estilo.
""".strip(),
        expected_output="Versão final revisada do artigo em Markdown.",
        agent=editor_agent,
        context=[draft_task],
        guardrail=article_quality_guardrail,
        guardrail_max_retries=1,
    )


editing_task = (
    create_editing_task(editor, writing_task) if editor and writing_task else None
)

# %% — Célula 14: Crew sequencial
def build_article_crew(use_web_search: bool | None = None) -> tuple[Crew, list[Agent], list[Task]]:
    model = create_llm()
    search_enabled = bool(SERPER_API_KEY) if use_web_search is None else use_web_search
    agents = [
        create_planner(model, search_enabled),
        create_writer(model),
        create_editor(model),
    ]
    tasks = [
        create_planning_task(agents[0]),
        None,
        None,
    ]
    tasks[1] = create_writing_task(agents[1], tasks[0])
    tasks[2] = create_editing_task(agents[2], tasks[1])

    if not all(isinstance(agent, Agent) for agent in agents):
        raise TypeError("Agent inválido: todos os itens precisam ser instâncias de Agent.")
    if not all(isinstance(task, Task) for task in tasks):
        raise TypeError("Task inválida: todos os itens precisam ser instâncias de Task.")

    crew_instance = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    return crew_instance, agents, tasks


crew = (
    Crew(
        agents=[planner, writer, editor],
        tasks=[planning_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )
    if all([planner, writer, editor, planning_task, writing_task, editing_task])
    else None
)

# %% [markdown] — Célula 15
# ## Processo sequencial
#
# `Process.sequential` executa Task 1 → Task 2 → Task 3. Neste projeto:
# Planejamento → Redação → Edição. O output do planejador entra explicitamente no
# contexto do redator; o draft do redator entra no contexto do editor.

# %% — Célula 16: executar a Crew
def run_crew(topic: str, crew_instance: Crew | None = None) -> tuple[Any, float]:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Tema inválido: forneça um texto não vazio.")
    active_crew = crew_instance or build_article_crew()[0]
    started = time.perf_counter()
    try:
        crew_result = active_crew.kickoff(inputs={"topic": topic.strip()})
    except GroqError as exc:
        raise RuntimeError(
            f"Erro da API Groq. Verifique chave, acesso e modelo {GROQ_MODEL!r}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Erro do CrewAI durante a execução: {exc}") from exc
    elapsed = time.perf_counter() - started
    if not raw_text(crew_result):
        raise RuntimeError("Output vazio: a Crew terminou sem produzir artigo final.")
    return crew_result, elapsed


result: Any | None = None
crew_elapsed: float | None = None
if __name__ == "__main__":
    if has_groq_key():
        result, crew_elapsed = run_crew(TOPIC, crew)
    else:
        print("Execução real ignorada: GROQ_API_KEY ausente no arquivo .env.")

# %% — Célula 17: resultado final
if result is not None:
    display(Markdown("# ARTIGO FINAL\n\n" + raw_text(result)))
else:
    print("ARTIGO FINAL: execute primeiro a célula 16 com uma chave válida.")

# %% — Célula 18: resultados intermediários
def get_task_outputs(crew_result: Any) -> list[Any]:
    outputs = getattr(crew_result, "tasks_output", None)
    if outputs is None:
        raise AttributeError("O resultado do CrewAI não possui tasks_output.")
    return list(outputs)


if result is not None:
    labels = ["PLANO DO PESQUISADOR", "PRIMEIRO ARTIGO", "ARTIGO EDITADO"]
    for label, task_output in zip(labels, get_task_outputs(result), strict=False):
        display(Markdown(f"## {label}\n\n{raw_text(task_output)}"))
else:
    print("Resultados intermediários ainda não existem; nenhum conteúdo foi inventado.")

# %% — Célula 19: debug visual do fluxo
FLOW = """INPUT: Tema
  |
  v
AGENTE 1: Planner -> OUTPUT 1: Plano
  |
  v
AGENTE 2: Writer  -> OUTPUT 2: Draft
  |
  v
AGENTE 3: Editor  -> OUTPUT 3: Final"""
print(FLOW)
if result is not None:
    for index, task_output in enumerate(get_task_outputs(result), start=1):
        print(
            {
                "task": index,
                "agent": getattr(task_output, "agent", "não informado"),
                "output": raw_text(task_output),
            }
        )

# %% [markdown] — Célula 20
# ## Por que usar vários agentes?
#
# Um único modelo poderia fazer tudo. O objetivo é separar responsabilidades:
# Planner → estratégia; Writer → escrita; Editor → avaliação e melhoria.
# Especialização + divisão de tarefas + contexto entre etapas = workflow multiagente.

# %% [markdown] — Célula 21
# ## `role`
#
# Responde “Quem é esse agente?”. Exemplo: **Editor Técnico**.

# %% [markdown] — Célula 22
# ## `goal`
#
# Responde “O que esse agente precisa alcançar?”. Exemplo: garantir a qualidade e
# a precisão do artigo.

# %% [markdown] — Célula 23
# ## `backstory`
#
# Contextualiza experiência, especialização e forma de agir. Não é memória real:
# é parte das instruções usadas para caracterizar o agente.

# %% [markdown] — Célula 24
# ## `Task`
#
# Define o que deve ser feito, por quem e qual saída é esperada.
# `Agent` = quem executa; `Task` = o trabalho executado.

# %% [markdown] — Célula 25
# ## `expected_output`
#
# Explicita formato e qualidade esperados, por exemplo: artigo Markdown com
# introdução, seções, exemplos e conclusão.

# %% [markdown] — Célula 26
# ## `Crew`
#
# É a equipe: Planner + Writer + Editor. Também reúne as Tasks e o Process que
# coordena a execução.

# %% [markdown] — Célula 27
# ## `Process`
#
# **Sequential:** A → B → C. **Hierarchical:** um manager distribui trabalho entre
# agentes. Este primeiro experimento implementa somente `Process.sequential`.

# %% [markdown] — Célula 28
# ## Contexto entre Tasks
#
# O Writer recebe título, objetivo, estrutura e argumentos do Planner. Depois, o
# Editor recebe o artigo do Writer. Colaboração não exige simultaneidade: pode ocorrer
# pela passagem estruturada de contexto entre tarefas.

# %% — Célula 29: pesquisa web opcional
print("PESQUISA WEB OPCIONAL")
if SERPER_API_KEY:
    print("SERPER_API_KEY configurada: a próxima Crew usará Serper apenas no Planner.")
else:
    print("SERPER_API_KEY ausente: o fluxo principal continua somente com a LLM.")
# Para criar explicitamente com pesquisa: web_crew, _, _ = build_article_crew(True)

# %% [markdown] — Célula 30
# ## Tool por agente
#
# Nem todo agente precisa de todas as ferramentas. Neste exemplo opcional, somente o
# Planner recebe Web Search; Writer e Editor não recebem tools. Isso aplica o princípio
# de menor privilégio.

# %% — Célula 31: comparação opcional com uma única LLM
def single_llm_article(topic: str) -> str:
    """Realiza uma chamada Chat Completions direta, somente quando invocada."""
    if not has_groq_key():
        raise RuntimeError("GROQ_API_KEY ausente para a comparação com uma única LLM.")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Tema inválido para single_llm_article().")
    try:
        response = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content":"Escreva em português, com clareza técnica e sem inventar fontes."},
                {"role":"user","content":f"Escreva um artigo completo sobre {topic.strip()}."},
            ],
        )
    except GroqError as exc:
        raise RuntimeError(f"Erro da API na comparação com uma única LLM: {exc}") from exc
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Output vazio na comparação com uma única LLM.")
    return text


# Chamada opcional e paga: single_result = single_llm_article(TOPIC)

# %% — Célula 32: latência
if crew_elapsed is None:
    print("Tempo total da Crew: ainda não medido. Execute a célula 16.")
else:
    print(f"Tempo total da Crew: {crew_elapsed:.2f} segundos")
print("Não são exibidos tempos por Task porque este exemplo não os mede.")

# %% [markdown] — Célula 33
# ## Custo conceitual
#
# Três agentes não significam uma única chamada. Mais agentes podem aumentar
# especialização e controle, mas também custo, latência e complexidade. O número real
# de chamadas depende do fluxo, das ferramentas, de tentativas e de guardrails.

# %% — Célula 34: guardrails e qualidade
print("Guardrail ativo na Editing Task:")
print("- rejeita saída vazia ou muito curta;")
print("- permite uma nova tentativa;")
print("- não substitui avaliação factual ou revisão humana.")

# %% — Célula 35: salvar outputs
def save_outputs(crew_result: Any, output_dir: Path | None = None) -> list[Path]:
    outputs = get_task_outputs(crew_result)
    if len(outputs) < 3:
        raise RuntimeError("Falha ao salvar: eram esperados três outputs de Task.")
    destination = output_dir or PROJECT_DIR / "output"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        files = [
            destination / "plano.md",
            destination / "draft.md",
            destination / "artigo_final.md",
        ]
        for path, content in zip(files, outputs, strict=True):
            text = raw_text(content)
            if not text:
                raise RuntimeError(f"Output vazio para {path.name}.")
            path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar artigo em {destination}: {exc}") from exc
    return files


if result is not None:
    print("Arquivos salvos:", *save_outputs(result), sep="\n- ")
else:
    print("Nada foi salvo: execute a Crew primeiro.")

# %% — Célula 36: segundo tema opcional
SECOND_TOPIC = "Como RAG melhora respostas de modelos de linguagem"
print(f"Segundo tema preparado (não executado): {SECOND_TOPIC}")
# Chamada opcional e paga: second_result, second_elapsed = run_crew(SECOND_TOPIC)

# %% [markdown] — Célula 37
# ## Reutilização
#
# Mesmos papéis + mesmas Tasks + novo input = novo trabalho. As descrições usam
# `{topic}`, portanto o sistema não está fixo em um único artigo.

# %% [markdown] — Célula 38
# ## Agente × LLM
#
# LLM = modelo de linguagem. Agente = LLM + role + goal + backstory/instruções +
# tools + Task + contexto. CrewAI não é a LLM; é o framework de orquestração.

# %% [markdown] — Célula 39
# ## CrewAI × LangGraph
#
# CrewAI pensa naturalmente em equipes, papéis, tarefas e colaboração. LangGraph
# pensa naturalmente em estado, nós, arestas e controle fino do fluxo. Não são
# concorrentes absolutos: atendem necessidades e níveis de controle diferentes.

# %% [markdown] — Célula 40
# ## Multiagentes × A2A
#
# CrewAI organiza agentes em uma aplicação/equipe. A2A padroniza a comunicação entre
# agentes ou sistemas independentes: `CrewAI App A → A2A → LangGraph App B`.

# %% [markdown] — Célula 41
# ## Human in the loop
#
# Uma aprovação humana pode entrar entre Writer e Editor:
# Planner → Writer → **REVISÃO HUMANA** → Editor. Este exemplo não bloqueia o notebook
# esperando interação; a etapa é uma opção de desenho para cenários de maior risco.

# %% — Célula 42: teste de responsabilidades
def responsibility_evaluation(crew_result: Any | None) -> pd.DataFrame:
    expected = ["Plano estruturado", "Artigo em Markdown", "Versão revisada", "Final existe"]
    if crew_result is None:
        actual = ["Não executado"] * 4
        status = ["NÃO AVALIADO"] * 4
    else:
        outputs = get_task_outputs(crew_result)
        texts = [raw_text(item) for item in outputs]
        checks = [len(texts) > i and bool(texts[i]) for i in range(3)]
        checks.append(bool(raw_text(crew_result)))
        actual = ["Presente" if item else "Ausente" for item in checks]
        status = ["OK" if item else "FALHOU" for item in checks]
    return pd.DataFrame(
        {
            "etapa": ["Planner", "Writer", "Editor", "Resultado final"],
            "esperado": expected,
            "resultado": actual,
            "status": status,
        }
    )


display(responsibility_evaluation(result))

# %% — Célula 43: debug completo
def debug_crew_run(crew_result: Any | None, topic: str = TOPIC) -> None:
    print("1. Tema\n", topic)
    print("\n2. Agents\n", AGENTS_TABLE[["AGENTE", "ROLE"]].to_string(index=False))
    print("\n3. Tasks\n Planejamento -> Redação -> Edição")
    if crew_result is None:
        print("\n4–7. Outputs\n Não executados; nenhum resultado foi inventado.")
        return
    outputs = get_task_outputs(crew_result)
    labels = ["4. Planner output", "5. Writer output", "6. Editor output"]
    for label, item in zip(labels, outputs, strict=False):
        print(f"\n{label}\n{raw_text(item)}")
    print(f"\n7. Final output\n{raw_text(crew_result)}")


debug_crew_run(result)

# %% [markdown] — Célula 44
# ## Diagrama final
#
# ```text
# USER → TOPIC → CrewAI
#                    ↓
# Planner Agent → PLAN OUTPUT
#                    ↓
# Writer Agent  → DRAFT OUTPUT
#                    ↓
# Editor Agent  → FINAL ARTICLE
# ```

# %% [markdown] — Célula 45
# ## O que aprendi
#
# 1. Problemas complexos podem ser decompostos em tarefas menores.
# 2. Cada agente pode ter responsabilidade especializada.
# 3. `role` define quem o agente é; `goal`, o que deve atingir.
# 4. `backstory` contextualiza o comportamento, sem ser memória real.
# 5. `Task` define trabalho; `expected_output`, o resultado esperado.
# 6. `Crew` organiza agentes e tarefas; `Process` controla a ordem.
# 7. No processo sequencial, a saída de uma tarefa alimenta a próxima.
# 8. Especialização pode melhorar controle, mas aumenta custo e complexidade.
# 9. CrewAI orquestra a equipe; o raciocínio textual vem da LLM configurada.
#
# **Frase de apresentação:** em vez de pedir a uma única LLM para pesquisar,
# planejar, escrever e revisar ao mesmo tempo, dividimos as responsabilidades entre
# especialistas. O resultado de um agente vira contexto para o próximo.
