# Experimento 07 — Multiagentes

Esta pasta contém dois exemplos complementares:

- `main.py`: simulação local e determinística de uma equipe de agentes, sem chamadas externas;
- `equipe_artigo_crewai.ipynb`: equipe CrewAI real com planejador, redator e editor executados em sequência.

## Executar o notebook

1. Ative o ambiente virtual do projeto.
2. Defina `GROQ_API_KEY` no arquivo `.env` da raiz.
3. Abra `equipe_artigo_crewai.ipynb` no VS Code ou Jupyter e execute todas as células.

Escolha o modelo com `GROQ_MODEL`; por exemplo, `GROQ_MODEL=llama-3.1-8b-instant`. O prefixo `groq/` exigido pelo LiteLLM é acrescentado automaticamente.

O exemplo remove, por meio do hook público `before_llm_call`, marcadores internos de cache que a API Groq não aceita.

A Crew usa `max_rpm=1` e limita cada resposta a 1.200 tokens para permanecer dentro dos limites da camada gratuita da Groq. Por isso, a execução completa leva alguns minutos.

Os dados locais do CrewAI são gravados em `.crewai-data/` na raiz do projeto e ficam fora do controle de versão.

> A execução do notebook faz chamadas à API do provedor configurado e pode gerar custos.
