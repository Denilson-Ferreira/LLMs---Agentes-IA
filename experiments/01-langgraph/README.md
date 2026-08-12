# 01 — Agentes com LangGraph

Duas implementações executáveis e independentes:

| Projeto | Ferramentas reais | Notebook |
|---|---|---|
| [`agente-utilidades-brasil`](./agente-utilidades-brasil/) | ViaCEP e AwesomeAPI | não |
| [`agente-busca-tavily`](./agente-busca-tavily/) | Tavily Search | sim |

Todos usam Groq como provedora da LLM e carregam `GROQ_API_KEY` do `.env` da
raiz. Consulte o README de cada subpasta para instalar e executar.
