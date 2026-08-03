# AI Agents in LangGraph

## O que aprendi

- diferença entre chatbot e agente;
- representação do fluxo como grafo;
- nós, arestas, decisões condicionais e estado;
- uso de ferramentas externas;
- importância de interrupções e aprovação humana.

## Relação com CTI

Um fluxo pode receber um relatório, extrair indicadores, consultar fontes, avaliar risco e gerar um resumo. Ações críticas, como bloqueio ou isolamento, devem poder parar para revisão humana.

## Experimento

`experiments/01-langgraph-conceitual/main.py`

Implementa uma máquina de estados didática para representar o fluxo sem depender do framework real.

## Principal conclusão

**O LLM interpreta e decide; o grafo controla e torna o processo observável.**

## Link do curso

Veja a lista de cursos no README principal.
