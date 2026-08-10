# 01 — LangGraph conceitual

Esta pasta reúne dois níveis do mesmo estudo:

- `main.py`: demonstração conceitual mínima de estado, nós, arestas e fluxo.
- `agente-utilidades-brasil/`: experimento real com LangGraph, ReAct, tool calling,
  Groq/Llama e APIs públicas do ViaCEP e AwesomeAPI.

## Executar o agente real

No PowerShell ou no CMD, a partir desta pasta:

```bat
.\agente-utilidades-brasil\run.cmd
```

Ou entre na subpasta e execute:

```bat
cd .\agente-utilidades-brasil
run.cmd
```

O arquivo `.env` local contém a configuração do provedor e é ignorado pelo Git.
Nunca publique esse arquivo.

Para abrir o diagrama visual do StateGraph:

```bat
.\agente-utilidades-brasil\ver-grafo.cmd
```
