# Retrieval-Augmented Generation (RAG)

## O que aprendi

- preparar documentos para recuperação;
- dividir conteúdo em trechos;
- representar a consulta e encontrar trechos relacionados;
- fornecer os trechos recuperados ao modelo;
- separar recuperação de geração.

## Relação com CTI

RAG permite consultar relatórios, boletins e documentação atualizada antes de produzir uma análise. A qualidade depende das fontes, da recuperação e da forma como a resposta usa as evidências.

## Experimento

`experiments/03-rag-local/main.py`

Executa recuperação local por similaridade de palavras, sem LLM e sem banco vetorial.

## Principal conclusão

**RAG não é memória: é consulta pontual a uma base externa para fundamentar a resposta atual.**

## Link do curso

Veja a lista de cursos no README principal.
