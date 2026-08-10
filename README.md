# Estudos de LLMs e Agentes de IA

Portfólio de estudos com teoria e experimentos executáveis sobre **LLMs, agentes de IA, LangGraph, memória, RAG, protocolos entre agentes, grafos de conhecimento e CrewAI**. O contexto aplicado é Cyber Threat Intelligence (CTI), com um experimento adicional de utilidades públicas do Brasil.

> Este repositório contém resumos autorais e implementações próprias. Ele não reproduz materiais proprietários dos cursos. Dados marcados como didáticos são fictícios; integrações identificadas como reais consultam APIs externas e informam suas fontes.

## O que este repositório demonstra

- a diferença entre LLM, ferramenta e agente;
- estado, nós, arestas e tomada de decisão no LangGraph;
- o ciclo ReAct sem expor raciocínio privado do modelo;
- memória, RAG, protocolos e grafos em versões conceituais;
- agentes reais usando GroqCloud e APIs públicas;
- divisão de responsabilidades em um sistema multiagente com CrewAI.

## Trilha dos sete cursos

| # | Tema estudado | Experimento principal |
|---|---|---|
| 1 | AI Agents in LangGraph | Grafo conceitual e agente ReAct real de CEP/cotação |
| 2 | Long-Term Agentic Memory | Memória didática persistida em JSON |
| 3 | Retrieval-Augmented Generation (RAG) | Recuperação local por similaridade textual |
| 4 | Agent Communication Protocol (ACP) | Mensagens padronizadas entre agentes |
| 5 | Agentic Knowledge Graph | Entidades, relações e exportação DOT |
| 6 | LLMs as Operating Systems | Gerenciamento da janela de contexto |
| 7 | Multi-Agent Systems with CrewAI | Pipeline local e equipe CrewAI real em notebook |

Os resumos teóricos estão em [`courses/`](courses/) e os códigos em [`experiments/`](experiments/).

## Destaques práticos reais

### 1. Agente de Utilidades do Brasil

Exemplo principal de LangGraph + ReAct. A LLM decide quando chamar `consultar_cep` e `consultar_cotacao`; o Python consulta ViaCEP e AwesomeAPI e devolve os resultados como `ToolMessage`.

```text
pergunta → LLM → tool call → API real → ToolMessage → LLM → resposta
```

O terminal exibe os eventos observáveis, e o projeto também gera uma visualização interativa do grafo. Consulte o [README do experimento](experiments/01-langgraph-conceitual/agente-utilidades-brasil/README.md).

### 2. Agente de CVEs

Agente LangGraph com GroqCloud que consulta a API pública do NVD e organiza uma análise em português. O modelo escolhe a ferramenta; a consulta HTTP é executada pelo programa.

### 3. Equipe CrewAI

Notebook com planejador, redator e editor executados sequencialmente usando GroqCloud. Consulte o [README do experimento](experiments/07-multiagentes/README.md).

## Preparar o ambiente

Recomendado: Python 3.11 ou superior. O ambiente atual foi validado com Python 3.13.

No PowerShell, dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite somente o arquivo `.env` e informe sua chave:

```dotenv
GROQ_API_KEY=sua_chave_groq_aqui
GROQ_MODEL=llama-3.1-8b-instant
```

O `.env`, a `.venv`, caches e resultados gerados estão no `.gitignore` e não devem ser enviados ao GitHub.

## Executar

Todos os sete exemplos conceituais, sem chamadas externas:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Agente real de utilidades do Brasil:

```powershell
cd experiments\01-langgraph-conceitual\agente-utilidades-brasil
run.cmd
```

Visualização do grafo do agente:

```powershell
cd experiments\01-langgraph-conceitual\agente-utilidades-brasil
ver-grafo.cmd
```

Agente real de CVEs:

```powershell
.\.venv\Scripts\python.exe Biblioteca\agente_cve_langgraph_groq.py
```

Equipe CrewAI: abra `experiments/07-multiagentes/equipe_artigo_crewai.ipynb` no VS Code e execute as células em ordem.

## Estrutura

```text
llms-agentes-ia/
├── .github/workflows/       # Testes automáticos do GitHub Actions
├── Biblioteca/              # Agentes com integrações reais
├── courses/                 # Teoria organizada por curso
├── docs/                    # Glossário, CTI, publicação e roteiro
├── experiments/             # Sete grupos de experimentos
├── outputs/                 # Saídas locais geradas (ignoradas)
├── sample_data/             # Entrada didática identificada como fictícia
├── tests/                   # Testes automatizados
├── .env.example             # Modelo seguro de configuração
├── PROJECT_STATUS.md        # Estado real do portfólio
├── requirements.txt         # Dependências dos exemplos reais
└── run_all.py               # Executa a suíte conceitual
```

## Ordem sugerida para apresentar

1. Abra este README e explique a trilha dos sete cursos.
2. Execute `python run_all.py` para mostrar os conceitos sem depender da internet.
3. Abra `agent.py` do Agente de Utilidades e mostre `StateGraph`, `ToolNode` e `tools_condition`.
4. Execute `ver-grafo.cmd` para mostrar visualmente os nós e arestas.
5. Execute `run.cmd` com uma pergunta que use CEP e cotação ao mesmo tempo.
6. Mostre o agente de CVEs e, por último, o notebook multiagente.

Há um texto de apoio pronto em [`docs/roteiro-apresentacao.md`](docs/roteiro-apresentacao.md).

## Segurança e limites

- Nenhuma chave de API deve aparecer em código, notebook, commit ou captura de tela.
- As APIs externas podem ficar indisponíveis ou alterar seus limites.
- Os exemplos de CTI são educacionais e não constituem uma plataforma de segurança pronta para produção.
- Resultados produzidos por LLM devem ser verificados antes de decisões reais.

## Cursos de referência

1. [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
2. [Long-Term Agentic Memory with LangGraph](https://www.deeplearning.ai/short-courses/long-term-agentic-memory-with-langgraph/)
3. [Retrieval-Augmented Generation (RAG)](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/)
4. [ACP: Agent Communication Protocol](https://www.deeplearning.ai/short-courses/acp-agent-communication-protocol/)
5. [Agentic Knowledge Graph Construction](https://www.deeplearning.ai/short-courses/agentic-knowledge-graph-construction/)
6. [LLMs as Operating Systems: Agent Memory](https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory/)
7. [Multi-AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
