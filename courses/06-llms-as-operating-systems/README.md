# LLMs as Operating Systems: Agent Memory

## O que aprendi

- a janela de contexto é limitada;
- contexto funciona como memória de trabalho;
- informações podem ser resumidas, arquivadas e recuperadas;
- memória central e memória de arquivo têm papéis diferentes;
- o agente precisa administrar o que está disponível no momento.

## Relação com CTI

Uma investigação pode ter muitos documentos. O agente deve manter no contexto apenas o necessário para a etapa atual e buscar o restante quando precisar.

## Experimento

`experiments/06-gerenciamento-contexto/main.py`

Simula um orçamento de contexto e uma memória de arquivo pesquisável.

## Principal conclusão

**Gerenciar contexto é escolher o que o LLM precisa ver agora, sem perder acesso ao restante.**

## Link do curso

Veja a lista de cursos no README principal.
