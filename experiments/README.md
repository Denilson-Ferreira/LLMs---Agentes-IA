# Projetos executáveis

Esta pasta contém somente implementações completas, scripts Python, notebooks e
arquivos de apoio necessários para execução.

| Pasta | Implementação |
|---|---|
| `01-langgraph` | dois agentes LangGraph com Groq e ferramentas externas reais |
| `02-memoria-longo-prazo` | agente de e-mail com memória semântica, episódica e procedural |
| `03-rag-local` | pipeline RAG híbrido com BM25, vetores locais e Groq |
| `04-protocolo-a2a` | QA Agent com interface preparada para integração A2A |
| `05-grafo-conhecimento` | interpretação estruturada de intenção com ADK e Groq |
| `06-gerenciamento-contexto` | memória editável e tool calling com Groq |
| `07-multiagentes` | planejador, redator e editor com CrewAI e Groq |

Os diretórios `data/` contêm entradas pequenas e controladas para os projetos que
precisam de documentos locais. Eles não são protótipos separados.

Para instalar tudo de uma vez, use o `requirements.txt` da raiz. Para isolar um
curso, use o `requirements.txt` da própria pasta.
