# ACP: Agent Communication Protocol

## O que aprendi

- agentes precisam de um formato comum de mensagens;
- uma mensagem pode conter remetente, destinatário, tarefa, status e resultado;
- falhas e estados de execução precisam ser explícitos;
- a comunicação deve permitir rastreabilidade.

## Relação com CTI

Um agente coordenador pode solicitar a outro agente que verifique indicadores e receber um resultado estruturado para continuar a análise.

## Experimento

`experiments/04-acp-protocolo/main.py`

Cria mensagens JSON entre um coordenador e um agente de enriquecimento.

## Principal conclusão

**ACP organiza a conversa entre agentes; não substitui a lógica de cada agente.**

## Link do curso

Veja a lista de cursos no README principal.
