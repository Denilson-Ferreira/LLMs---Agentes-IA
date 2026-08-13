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
│   │   └── agente-busca-tavily/       # Groq + Tavily + notebook
│   ├── 02-memoria-longo-prazo/        # LangGraph + memória + notebook
│   ├── 03-rag-local/                   # RAG híbrido + Groq + notebook
│   ├── 04-protocolo-a2a/              # QA Agent preparado para A2A
│   ├── 05-grafo-conhecimento/          # Google ADK + Groq/LiteLLM
│   ├── 06-gerenciamento-contexto/      # Memória editável + tool calling
│   ├── 07-multiagentes/                # CrewAI + Groq
│   └── 08-analise-sentimento-redes-sociais/ # FastAPI + Groq + análise de comentários
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
| 2 | Memória de longo prazo | Groq e LangGraph |
| 3 | RAG local híbrido | Groq, BM25 e vetores locais |
| 4 | QA Agent para A2A | Groq |
| 5 | Interpretação para grafo de conhecimento | Google ADK, LiteLLM e Groq |
| 6 | Memória editável | Groq e tool calling |
| 7 | Sistema multiagente | CrewAI, LiteLLM e Groq |
| 8 | Análise de sentimento em redes sociais | FastAPI, Groq e Pydantic |

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

## Resultados dos notebooks

Todos os notebooks foram executados integralmente em **12 de agosto de 2026**.
As saídas estão incorporadas nos próprios arquivos `.ipynb` e são renderizadas
pelo GitHub logo abaixo de cada célula. Células de importação, configuração e
definição de funções podem terminar sem saída visual, mas também possuem contador
de execução salvo.

| Curso | Notebook com as saídas | Células de código executadas | Células com saída | Saídas salvas | Erros |
|---|---|---:|---:|---:|---:|
| 1 | [Agente de busca Tavily](experiments/01-langgraph/agente-busca-tavily/agente_busca_tavily.ipynb) | 11/11 | 4 | 5 | 0 |
| 2 | [Memória de longo prazo](experiments/02-memoria-longo-prazo/memoria_longo_prazo.ipynb) | 22/22 | 10 | 10 | 0 |
| 3 | [RAG local híbrido](experiments/03-rag-local/rag_local.ipynb) | 26/26 | 19 | 22 | 0 |
| 4 | [QA Agent com Groq](experiments/04-protocolo-a2a/agente_qa_groq.ipynb) | 16/16 | 10 | 18 | 0 |
| 5 | [Interpretação para grafo de conhecimento](experiments/05-grafo-conhecimento/intencao_usuario.ipynb) | 17/17 | 10 | 15 | 0 |
| 6 | [Memória editável](experiments/06-gerenciamento-contexto/memoria_editavel.ipynb) | 28/28 | 6 | 8 | 0 |
| 7 | [Sistema multiagente](experiments/07-multiagentes/sistema_multiagentes.ipynb) | 24/24 | 15 | 53 | 0 |
| **Total** | **7 notebooks** | **144/144** | **74** | **131** | **0** |

### O que foi produzido

- **Curso 1:** o grafo foi compilado e visualizado, e a resposta da Groq foi
  gerada pelo caminho de fallback. A busca web não foi acionada porque
  `TAVILY_API_KEY` não está configurada com uma chave real.
- **Curso 2:** o agente classificou cobrança duplicada como `notify`, agradecimento
  como `ignore`, redefinição de senha como `respond` e preservou o feedback humano
  na memória procedural.
- **Curso 3:** seis chunks foram indexados; a pergunta sobre o atendimento
  Enterprise retornou prazo de até duas horas úteis e citou `politicas.txt` e
  `produtos.txt`. Na avaliação de recuperação, `hit_rate@3` e `recall@3` foram
  `1.0`, com `MRR` de `0.875`.
- **Curso 4:** o agente respondeu perguntas sobre a apólice com evidências,
  identificou a franquia de `R$ 2.500,00` e recusou a instrução adversarial que
  tentava alterá-la para `R$ 100,00`.
- **Curso 5:** solicitações foram convertidas em intenções estruturadas, com
  entidades, relacionamentos, ambiguidades e perguntas de esclarecimento; os três
  casos de avaliação passaram nas verificações esperadas.
- **Curso 6:** os blocos de memória foram exibidos e atualizados, a preferência
  “prática antes de teoria” foi persistida e uma memória arquivada sobre A2A foi
  recuperada.
- **Curso 7:** Planner, Writer e Editor executaram a Crew sequencial, produziram
  plano, rascunho e artigo final, e todas as quatro verificações de resultado
  terminaram com status `OK`.

As respostas de modelos generativos podem variar em uma nova execução. Os números
acima descrevem exatamente as saídas que estão salvas nesta versão dos notebooks.

## Segurança

- Nunca coloque chaves em scripts, notebooks ou commits.
- Revise respostas de LLM antes de usá-las em decisões reais.
- APIs externas podem ter limites, indisponibilidade e cobrança própria.
- Arquivos `.env.example` contêm apenas placeholders seguros.
