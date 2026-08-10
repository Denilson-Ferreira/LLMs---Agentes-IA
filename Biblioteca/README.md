# Biblioteca de agentes reais

Esta pasta reúne implementações que usam LLMs e fontes externas reais.

## Agente de CVEs

`agente_cve_langgraph_groq.py` implementa manualmente um ciclo ReAct com LangGraph:

```text
usuário → modelo → consultar_nvd → API pública do NVD → modelo → resposta
```

Para executar a partir da raiz do repositório:

```powershell
.\.venv\Scripts\python.exe Biblioteca\agente_cve_langgraph_groq.py
```

Requisitos:

- dependências de `requirements.txt` instaladas;
- `GROQ_API_KEY` configurada no `.env` da raiz;
- conexão com a internet.

A chave nunca deve ser adicionada ao código ou ao Git.
