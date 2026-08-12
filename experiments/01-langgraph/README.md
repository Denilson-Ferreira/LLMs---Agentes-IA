# 01 — Agentes com LangGraph

Três implementações executáveis e independentes:

| Projeto | Ferramentas reais | Notebook |
|---|---|---|
| [`agente-utilidades-brasil`](./agente-utilidades-brasil/) | ViaCEP e AwesomeAPI | não |
| [`agente-busca-tavily`](./agente-busca-tavily/) | Tavily Search | sim |
| [`agente-cve-nvd`](./agente-cve-nvd/) | API pública do NVD | não |

Todos usam Groq como provedora da LLM e carregam `GROQ_API_KEY` do `.env` da
raiz. Consulte o README de cada subpasta para instalar e executar.
