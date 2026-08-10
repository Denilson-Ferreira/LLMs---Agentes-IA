# Experimentos

Cada pasta corresponde a um tema da trilha. Os arquivos `main.py` formam uma suíte conceitual, pequena e determinística; os exemplos identificados como reais usam frameworks, LLM e/ou APIs externas.

| Pasta | Conceito observado | Tipo |
|---|---|---|
| `01-langgraph-conceitual` | estado, nós, arestas, roteamento e ReAct | conceitual + agente real |
| `02-memoria-longo-prazo` | registro e recuperação de memória | conceitual |
| `03-rag-local` | recuperação antes da resposta | conceitual |
| `04-acp-protocolo` | mensagens estruturadas entre agentes | conceitual |
| `05-grafo-conhecimento` | entidades e relações de CTI | conceitual |
| `06-gerenciamento-contexto` | memória de trabalho e resumo | conceitual |
| `07-multiagentes` | papéis e fluxo sequencial | conceitual + CrewAI real |

## Executar a suíte conceitual

Na raiz do repositório:

```powershell
python run_all.py
```

Esses sete scripts usam apenas a biblioteca padrão. O relatório de entrada e os endereços de rede são explicitamente reservados para demonstração; eles não fazem varredura nem atuam em sistemas reais.

## Executar o LangGraph real

```powershell
cd experiments\01-langgraph-conceitual\agente-utilidades-brasil
run.cmd
```

Esse projeto usa uma LLM via API, duas ferramentas reais e permite visualizar o grafo. Veja as instruções completas no [README próprio](01-langgraph-conceitual/agente-utilidades-brasil/README.md).

## Executar o CrewAI real

Abra `07-multiagentes/equipe_artigo_crewai.ipynb` no VS Code, selecione o ambiente virtual do repositório e execute as células em ordem. A chave Groq deve estar apenas no `.env` da raiz.

> Integrações reais dependem de internet, credenciais válidas e limites do provedor. Elas podem gerar consumo na conta configurada.
