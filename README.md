# Estudos de LLMs e Agentes de IA

Repositório de estudos sobre **agentes de IA, LLMs e Cyber Threat Intelligence (CTI)**, organizado a partir de sete cursos indicados para preparação ao projeto **aplicações em CTI (Agregador) — Fase 1**.

> **Transparência:** este repositório reúne anotações do estudo guiado e experimentos didáticos autorais. Ele não reproduz o conteúdo integral nem os materiais proprietários dos cursos. Os experimentos foram simplificados para funcionar localmente sem chave de API e sem dependências externas.

## Objetivo

Demonstrar, de forma organizada e reproduzível:

- o que foi aprendido em cada curso;
- como os conceitos se conectam a CTI;
- pequenos experimentos executáveis;
- a evolução de uma arquitetura conceitual para o aplicações em CTI.

## Os 7 cursos

| # | Curso | Tema central | Experimento |
|---|---|---|---|
| 1 | AI Agents in LangGraph | Fluxos, estado, nós, arestas e decisões | Máquina de estados para análise de ameaça |
| 2 | Long-Term Agentic Memory with LangGraph | Memória episódica, semântica e persistência | Memória em JSON |
| 3 | Retrieval-Augmented Generation (RAG) | Busca antes da resposta | RAG local por similaridade textual |
| 4 | ACP: Agent Communication Protocol | Comunicação padronizada entre agentes | Mensagens de tarefa em JSON |
| 5 | Agentic Knowledge Graph Construction | Entidades e relações | Grafo de CTI e exportação DOT |
| 6 | LLMs as Operating Systems: Agent Memory | Gestão de contexto e memória externa | Gerenciador de contexto |
| 7 | Multi-AI Agent Systems with CrewAI | Papéis, tarefas e cooperação | Equipe multiagente simulada |

## Como executar

Os experimentos iniciais usam apenas a biblioteca padrão do Python e podem ser executados no Python 3.14 já instalado.

```bash
python --version
python run_all.py
```

Também é possível executar um experimento isolado:

```bash
python experiments/03-rag-local/main.py
```

## Estrutura

```text
llms-agentes-ia/
├── courses/                  # Resumo de cada curso
├── docs/                     # Glossário, contexto e roteiro de apresentação
├── experiments/              # Experimentos didáticos executáveis
├── sample_data/              # Relatório fictício para testes
├── tests/                    # Testes básicos
├── presentation/             # Apresentação em PPTX para importar no Google Slides
├── PROJECT_STATUS.md         # O que já foi feito e próximos passos
└── run_all.py                # Executa todos os experimentos
```

## Relação com CTI

A arquitetura estudada pode ser resumida assim:

```text
Fontes de CTI
   ↓
RAG recupera documentos
   ↓
LLM interpreta o conteúdo
   ↓
Agentes executam tarefas e usam ferramentas
   ↓
LangGraph controla o fluxo
   ↓
Memória preserva fatos e experiências
   ↓
Grafo conecta ameaças, malwares, CVEs e IOCs
   ↓
ACP padroniza a comunicação
   ↓
CrewAI organiza agentes especializados
   ↓
Analista humano valida a inteligência produzida
```

## Próximos passos técnicos

Depois da instalação do Python 3.13:

1. substituir a máquina de estados didática por LangGraph real;
2. integrar um LLM por API ou modelo local;
3. usar um banco vetorial no RAG;
4. integrar fontes autorizadas de CTI;
5. experimentar CrewAI;
6. adicionar avaliação humana e guardrails.

## Cursos originais

1. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/
2. https://www.deeplearning.ai/short-courses/long-term-agentic-memory-with-langgraph/
3. https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/
4. https://www.deeplearning.ai/short-courses/acp-agent-communication-protocol/
5. https://www.deeplearning.ai/short-courses/agentic-knowledge-graph-construction/
6. https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory/
7. https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/
