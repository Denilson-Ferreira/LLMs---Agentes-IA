# Roteiro de apresentação ao professor

Tempo sugerido: 10 a 15 minutos.

## 1. Abertura

“Este repositório registra minha evolução em sete temas de agentes de IA. Para cada tema organizei a teoria e um experimento executável. Também evoluí alguns conceitos para integrações reais com LangGraph, GroqCloud, CrewAI e APIs públicas.”

## 2. Visão geral da teoria

Use a tabela do README principal e resuma:

- **LLM:** interpreta a solicitação e gera linguagem;
- **Tool:** função que o modelo pode solicitar;
- **Agente:** a LLM recebe ferramentas e decide como avançar;
- **LangGraph:** organiza estado, nós, arestas e ciclos;
- **memória e RAG:** preservam ou recuperam contexto relevante;
- **multiagentes:** dividem uma tarefa entre papéis especializados.

## 3. Demonstração conceitual

No terminal do VS Code, na raiz do repositório:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Explique que esses exemplos são determinísticos. Eles permitem estudar cada arquitetura sem custo, chave ou indisponibilidade de rede.

## 4. Demonstração principal: LangGraph real

Abra `experiments/01-langgraph-conceitual/agente-utilidades-brasil/agent.py` e aponte:

1. `MessagesState`, que guarda as mensagens;
2. o node `agent`, que chama a LLM;
3. o `ToolNode`, que executa ferramentas;
4. `tools_condition`, que escolhe entre ferramenta e fim;
5. a aresta `tools → agent`, que devolve a observação para a LLM.

Mostre o desenho:

```powershell
cd experiments\01-langgraph-conceitual\agente-utilidades-brasil
ver-grafo.cmd
```

Depois execute:

```powershell
run.cmd
```

Pergunta sugerida:

> Onde fica o CEP 01001-000 e quanto está o dólar hoje?

Enquanto o programa roda, destaque `TOOL CALL`, argumentos, `TOOL RESULT` e resposta final. Não é chain-of-thought: são eventos observáveis do sistema.

## 5. O momento em que vira um agente

“A LLM não chama a internet diretamente. Ela decide que precisa de uma ferramenta e gera uma chamada estruturada. O Python executa a requisição real, converte o resultado em `ToolMessage` e devolve essa observação à LLM. Essa tomada de decisão seguida de ação e observação caracteriza o agente.”

## 6. Outros exemplos reais

- Mostre `Biblioteca/agente_cve_langgraph_groq.py`: Groq decide consultar uma CVE na API do NVD.
- Mostre `experiments/07-multiagentes/equipe_artigo_crewai.ipynb`: planejador, redator e editor executam tarefas sequenciais.

Se o tempo for curto, apenas apresente o código desses dois projetos e mantenha a execução ao vivo no Agente de Utilidades.

## 7. Encerramento

“Minha principal conclusão é que uma LLM isolada gera texto; um agente combina decisão, ferramentas, estado e controle de fluxo. Em aplicações reais, os resultados continuam precisando de fontes verificáveis, tratamento de falhas, proteção das credenciais e supervisão humana.”

## Checklist antes da aula

- [ ] Abrir o repositório no VS Code
- [ ] Confirmar que o `.env` existe localmente e não aparece no Git
- [ ] Executar os testes
- [ ] Testar `run.cmd` com CEP e cotação
- [ ] Testar `ver-grafo.cmd`
- [ ] Fechar terminais que possam mostrar a chave
- [ ] Confirmar que a internet está disponível
