# LLMs e Agentes de IA — exemplos executáveis

Repositório organizado com projetos Python e notebooks que executam fluxos reais
de LLMs e agentes usando **Groq**, **LangGraph**, **Google ADK**, **RAG** e
**CrewAI**.

Os protótipos conceituais duplicados foram removidos. Os arquivos em `data/` são
entradas controladas de demonstração necessárias para executar RAG, QA, grafo de
conhecimento e memória; as integrações, chamadas de modelo e ferramentas são reais.

## Estrutura

```text
LLMs Agentes IA/
├── experiments/
│   ├── 01-langgraph/
│   │   ├── agente-utilidades-brasil/  # Groq + ViaCEP + AwesomeAPI
│   │   ├── agente-busca-tavily/       # Groq + Tavily + notebook
│   │   └── agente-cve-nvd/            # Groq + API pública do NVD
│   ├── 02-memoria-longo-prazo/        # LangGraph + memória + notebook
│   ├── 03-rag-local/                   # RAG híbrido + Groq + notebook
│   ├── 04-protocolo-a2a/              # QA Agent preparado para A2A
│   ├── 05-grafo-conhecimento/          # Google ADK + Groq/LiteLLM
│   ├── 06-gerenciamento-contexto/      # Memória editável + tool calling
│   └── 07-multiagentes/                # CrewAI + Groq
├── tests/                              # testes do agente NVD
├── docs/                               # instruções auxiliares
├── .env.example
├── requirements.txt
└── README.md
```

Cada projeto contém seu próprio `README.md`, `requirements.txt` e, quando
aplicável, um notebook equivalente ao script Python.

## Configuração da Groq

Crie o ambiente principal e copie o modelo seguro de configuração:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Preencha somente o `.env` da raiz:

```dotenv
GROQ_API_KEY=sua_chave_groq
GROQ_MODEL=llama-3.1-8b-instant
```

O `.env` e os ambientes virtuais são ignorados pelo Git.

## Projetos

| Curso | Projeto executável | Serviço principal |
|---|---|---|
| 1 | Agente de utilidades do Brasil | Groq, ViaCEP e AwesomeAPI |
| 1 | Agente de busca web | Groq e Tavily |
| 1 | Agente de CVEs | Groq e NVD |
| 2 | Memória de longo prazo | Groq e LangGraph |
| 3 | RAG local híbrido | Groq, BM25 e vetores locais |
| 4 | QA Agent para A2A | Groq |
| 5 | Interpretação para grafo de conhecimento | Google ADK, LiteLLM e Groq |
| 6 | Memória editável | Groq e tool calling |
| 7 | Sistema multiagente | CrewAI, LiteLLM e Groq |

## Execução

Leia o README da pasta desejada. Exemplos:

```powershell
# Agente com APIs públicas brasileiras
cd experiments\01-langgraph\agente-utilidades-brasil
.\run.ps1

# RAG local
cd ..\..\03-rag-local
python rag_local.py

# Sistema multiagente
cd ..\07-multiagentes
python sistema_multiagentes.py
```

Nos cursos 4, 5 e 6, `RUN_LIVE_DEMOS=false` evita chamadas remotas ao executar
todas as células. Altere para `true` somente quando quiser executar as
demonstrações reais.

O agente de busca web também exige `TAVILY_API_KEY`; a pesquisa opcional do
CrewAI exige `SERPER_API_KEY`.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Segurança

- Nunca coloque chaves em scripts, notebooks ou commits.
- Revise respostas de LLM antes de usá-las em decisões reais.
- APIs externas podem ter limites, indisponibilidade e cobrança própria.
- Arquivos `.env.example` contêm apenas placeholders seguros.
