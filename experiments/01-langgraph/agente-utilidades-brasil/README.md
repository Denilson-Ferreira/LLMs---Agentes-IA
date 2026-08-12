# Agente de Utilidades do Brasil

Projeto didático e funcional para estudar **LLM + LangGraph + ReAct + Tool Calling**.
Ele não usa respostas prontas para CEP ou cotação: as ferramentas fazem requisições HTTP
reais ao [ViaCEP](https://viacep.com.br/) e à
[AwesomeAPI](https://docs.awesomeapi.com.br/api-de-moedas).

O projeto foi validado com Python 3.13. Ele requer Python 3.11 ou superior.

## Visão geral

O usuário escreve uma pergunta em linguagem natural. A LLM configurada recebe a
descrição das ferramentas disponíveis e decide se precisa chamar alguma delas. O LangGraph
controla o ciclo entre o modelo e as ferramentas até existir uma resposta final.

O programa mostra apenas eventos observáveis:

1. mensagem do usuário;
2. ferramenta escolhida pelo modelo;
3. argumentos gerados pelo modelo;
4. resultado real retornado pela API;
5. resposta final do agente.

O programa **não mostra chain-of-thought** nem raciocínio privado do modelo.

## Estrutura do projeto

```text
agente-utilidades-brasil/
├── .env.example
├── .gitignore
├── README.md
├── agent.py
├── main.py
├── requirements.txt
├── run.cmd
├── run.ps1
├── ver-grafo.cmd
├── visualizar_grafo.py
└── tools.py
```

- `main.py`: interface de terminal, streaming didático dos eventos e perguntas de teste.
- `agent.py`: configura ChatGroq ou ChatXAI e constrói o grafo ReAct manual com LangGraph.
- `tools.py`: contém as duas ferramentas que chamam APIs reais.
- `requirements.txt`: dependências atuais, sem fixação em versões antigas.
- `.env.example`: modelo seguro para configurar a chave e o nome do modelo.
- `.gitignore`: impede o versionamento da chave, ambiente virtual e caches Python.
- `README.md`: explicação conceitual e instruções de uso.

## 1. O que é LLM

LLM significa *Large Language Model*, ou modelo de linguagem de grande escala. A LLM
interpreta texto, identifica a intenção do usuário e produz linguagem natural.

Neste projeto, a LLM pode ser **Llama 3.1 8B na Groq** ou **Grok na xAI**. Ela recebe a
pergunta, o prompt de sistema, o histórico de mensagens e a descrição das ferramentas. A LLM
pode responder diretamente ou produzir uma
solicitação estruturada de ferramenta (*tool call*).

É importante não confundir:

- **Grok** é a família de modelos da xAI;
- **Groq** é o provedor de inferência usado com `llama-3.1-8b-instant` nesta instalação.

## 2. O que é Tool

Uma Tool é uma função que o modelo pode solicitar que o programa execute. A descrição e o
esquema de argumentos da função são apresentados à LLM por meio de `bind_tools()`.

```text
consultar_cep
    ↓
requisição HTTPS real ao ViaCEP

consultar_cotacao
    ↓
requisição HTTPS real à AwesomeAPI
```

`consultar_cep` valida oito dígitos, aceita hífen/espaços e consulta:

```text
https://viacep.com.br/ws/{CEP}/json/
```

`consultar_cotacao` normaliza `USD`, `EUR` ou `BTC` para um par contra BRL e consulta:

```text
https://economia.awesomeapi.com.br/json/last/{PAR}
```

As duas funções usam `requests`, `timeout=10`, `raise_for_status()` e tratamento específico
para timeout, conexão, status HTTP e outros erros de requisição.

## 3. O que é agente

Neste projeto, a LLM não está apenas respondendo perguntas. Ela recebe ferramentas e pode
tomar uma decisão:

```text
Preciso chamar alguma ferramenta?
            │
            ├── não → produzir resposta final
            │
            └── sim
                 ↓
            selecionar ferramenta
                 ↓
            definir argumentos
                 ↓
            Python executa a função
                 ↓
            resultado volta à LLM
                 ↓
            LLM continua
```

Não existem regras como `if "dólar" in pergunta`. A decisão vem do tool calling da LLM.

## 4. O que é ReAct

ReAct combina raciocínio e ação em ciclos:

```text
Reason
  ↓
Act
  ↓
Observe
  ↓
Reason
  ↓
Answer
```

`Reason` não significa que o programa deve expor raciocínio privado ou chain-of-thought.
Para estudar o comportamento, observamos externamente:

```text
pergunta
  ↓
tool call
  ↓
tool result
  ↓
resposta
```

Uma pergunta pode provocar mais de um ciclo `AGENT → TOOLS → AGENT`.

## 5. O que é LangGraph

LangGraph é o framework que controla o fluxo. O projeto usa a API de grafo explicitamente,
sem esconder a lógica em um agente pronto:

- `MessagesState`: estado compartilhado baseado em mensagens;
- `StateGraph`: construtor do grafo;
- `ToolNode`: nó que executa as ferramentas solicitadas;
- `tools_condition`: roteamento que verifica `AIMessage.tool_calls`;
- `START` e `END`: entrada e término do fluxo;
- edges: conexões fixas e condicionais entre os nós.

```text
START
  |
  v
AGENT
  |
  v
Tool necessária?
 /             \
SIM             NÃO
 |               |
 v               v
TOOLS            END
 |
 v
AGENT
```

A aresta `TOOLS → AGENT` é essencial. Depois de executar a função, o `ToolNode` transforma o
resultado em `ToolMessage`. Essa mensagem volta ao estado, e a LLM precisa lê-la para redigir
a resposta ou decidir por uma nova ferramenta.

O estado pode conter:

- `HumanMessage`: pergunta do usuário;
- `AIMessage`: resposta da LLM ou pedido de ferramenta;
- `ToolMessage`: observação produzida pela ferramenta.

## O momento em que isso vira um agente

Considere a pergunta:

> Quanto está o dólar?

Sem Tool, a LLM poderia responder apenas com conhecimento aprendido durante o treinamento.
Isso não garante uma cotação atual.

Com este agente:

```text
LLM analisa a solicitação
  ↓
percebe que existe consultar_cotacao
  ↓
gera um tool call com o argumento USD
  ↓
Python chama a AwesomeAPI pela internet
  ↓
a API retorna a cotação real
  ↓
o resultado vira ToolMessage
  ↓
o ToolMessage volta à LLM
  ↓
LLM formula a resposta
```

A distinção fundamental é:

- a LLM **decide** que precisa da ferramenta e produz nome/argumentos;
- o programa Python **executa** a ferramenta;
- o resultado real volta para a LLM.

A LLM não executa `requests.get()` diretamente. O código Python é responsável pelo acesso à
rede, timeout, validação e tratamento de falhas.

## State, Nodes, Edges e tomada de decisão

O `MessagesState` é a memória de trabalho da execução. Ele não é uma memória persistente nem
um banco de dados. Em uma sessão do chat, o histórico fica apenas em memória e é descartado
quando o processo termina.

O grafo possui dois nodes:

- `agent`: chama a LLM com o prompt, histórico e schemas das tools;
- `tools`: executa todos os tool calls presentes na última `AIMessage`.

As edges são:

```text
START → agent
agent → tools       quando existem tool_calls
agent → END         quando não existem tool_calls
tools → agent       sempre após executar as ferramentas
```

Na pergunta combinada sobre CEP e dólar, a LLM pode solicitar as duas ferramentas na mesma
resposta. O `ToolNode` aceita múltiplas chamadas, e `parallel_tool_calls=True` permite que o
modelo produza tool calls paralelos quando considerar adequado.

## Instalação

Entre na pasta do projeto:

```powershell
cd "C:\Users\denil\Downloads\LLMS - agentes IA\LLMs Agentes IA\experiments\01-langgraph\agente-utilidades-brasil"
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

### Ativar no Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Se a política do PowerShell bloquear a ativação, você ainda pode executar diretamente com
`.venv\Scripts\python.exe`, como mostrado mais abaixo.

### Ativar no Windows CMD

```bat
.venv\Scripts\activate.bat
```

### Ativar no Linux/macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Configuração da LLM

Copie `.env.example` para `.env`.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Windows CMD:

```bat
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Para usar sua chave Groq, edite `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=MINHA_CHAVE_GROQ
GROQ_MODEL=llama-3.1-8b-instant
```

O modelo `llama-3.1-8b-instant` suporta tool calling na Groq. Para usar xAI/Grok em vez disso,
configure `LLM_PROVIDER=xai`, `XAI_API_KEY` e `XAI_MODEL`. As APIs podem gerar cobrança
conforme o provedor, modelo e uso da conta.

## Como executar

Nesta cópia organizada, o caminho mais simples no PowerShell ou CMD é:

```bat
run.cmd
```

Você também pode dar duplo clique em `run.cmd` no Explorador de Arquivos. O lançador reutiliza
automaticamente o `.venv` existente na raiz do repositório.

Se quiser usar o script PowerShell, execute com bypass apenas para esse processo:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Também é possível executar manualmente.

Com o ambiente ativado:

```bash
python main.py
```

Sem ativar o ambiente, no Windows PowerShell:

```powershell
.venv\Scripts\python.exe main.py
```

O terminal mostrará:

```text
==================================================
AGENTE DE UTILIDADES

Digite sua pergunta. Digite "sair" para encerrar.

Você:
```

## Visualizar o grafo

Dê duplo clique em `ver-grafo.cmd` ou execute:

```bat
ver-grafo.cmd
```

O comando consulta o próprio `StateGraph` compilado, gera `grafo_langgraph.mmd` e
`grafo_langgraph.html`, e abre o diagrama no navegador. O HTML usa Mermaid para desenhar os
nós e as arestas. A visualização mostra o fluxo observável e não expõe chain-of-thought.

## Testes reais pelo agente

Para executar as quatro perguntas didáticas e depois abrir o chat:

```bash
python main.py --testes
```

Esse comando chama o provedor real da LLM e as APIs públicas a cada teste. Ele não usa mocks.

Para executar somente uma pergunta e encerrar:

```bash
python main.py --pergunta "Qual o endereço do CEP 01001-000?"
python main.py --pergunta "Quanto está o dólar em reais?"
python main.py --pergunta "Quanto está o euro em reais?"
python main.py --pergunta "Onde fica o CEP 01001-000 e quanto está o dólar hoje?"
```

No teste combinado, o comportamento esperado é:

```text
USUÁRIO
  ↓
LLM escolhe consultar_cep e consultar_cotacao
  ↓
ToolNode executa ViaCEP e AwesomeAPI
  ↓
dois ToolMessages retornam ao estado
  ↓
LLM produz a resposta final
```

## Testar somente as ferramentas reais

Esses comandos não usam a LLM e não geram custo de modelo. Eles ainda acessam a internet de
verdade:

```powershell
python -c "from tools import consultar_cep; print(consultar_cep.invoke({'cep':'01001-000'}))"
python -c "from tools import consultar_cotacao; print(consultar_cotacao.invoke({'moeda':'USD'}))"
```

## Streaming e debug didático

`main.py` usa `grafo.stream(..., stream_mode="updates")`. Cada atualização de node é
inspecionada:

- `AIMessage.tool_calls` gera a seção `TOOL CALL`;
- `ToolMessage` gera a seção `TOOL RESULT`;
- uma `AIMessage` sem tool calls gera `RESPOSTA DO AGENTE`.

Exemplo de formato:

```text
==================================================
TOOL CALL

Ferramenta: consultar_cep

Argumentos:
{
  "cep": "01001-000"
}

==================================================
TOOL RESULT

Ferramenta: consultar_cep

CEP: 01001-000
...
```

Esses são eventos públicos do protocolo de tool calling, não pensamentos internos.

## Segurança

- Nunca coloque `GROQ_API_KEY` ou `XAI_API_KEY` no código.
- Nunca envie ou publique seu arquivo `.env`.
- `.env` está no `.gitignore`.
- O programa nunca imprime a chave.
- `.venv/` e `__pycache__/` também estão ignorados.
- Revogue a chave imediatamente caso ela seja exposta.

## APIs e comportamento real

- O ViaCEP exige oito dígitos e sinaliza `erro` para um CEP válido no formato, mas
  inexistente. O código aceita tanto o booleano `true` documentado quanto o texto `"true"`
  observado na API durante os testes.
- A AwesomeAPI retorna campos como `bid`, `ask`, `high`, `low`, `pctChange` e `create_date`.
- Segundo a documentação da AwesomeAPI, consultas sem chave própria podem usar cache de até
  um minuto. Ainda são respostas reais do serviço, não dados fixos no projeto.
- Indisponibilidade, timeout e erros HTTP são mostrados explicitamente ao usuário.

## Compatibilidade verificada

Na criação deste exemplo, os imports foram executados com:

```text
langgraph       1.2.10
langchain       1.3.14
langchain-xai   1.3.0
langchain-groq  1.1.3
python-dotenv   1.2.2
requests        2.34.2
```

O `requirements.txt` não fixa versões antigas; uma instalação futura poderá selecionar
versões mais novas. As APIs usadas foram verificadas na versão acima:

```python
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_xai import ChatXAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
```

`create_react_agent` está deprecated na versão verificada e não é usado. O fluxo foi montado
manualmente com `StateGraph`.

## Referências oficiais

- [LangGraph: visão geral](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain: ToolNode e tools_condition](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain: integração ChatXAI](https://docs.langchain.com/oss/python/integrations/chat/xai)
- [LangChain: integração ChatGroq](https://docs.langchain.com/oss/python/integrations/chat/groq)
- [Groq: llama-3.1-8b-instant](https://console.groq.com/docs/model/llama-3.1-8b-instant)
- [xAI: function calling](https://docs.x.ai/developers/tools/function-calling)
- [xAI: modelos disponíveis](https://docs.x.ai/developers/models)
- [ViaCEP](https://viacep.com.br/)
- [AwesomeAPI: API de moedas](https://docs.awesomeapi.com.br/api-de-moedas)

## Fora do escopo desta aula

Este projeto não implementa RAG, embeddings, banco vetorial, CrewAI, memória persistente,
FastAPI, frontend, Docker, banco de dados, MCP ou multi-agent. O foco é somente:

```text
LLM + LangGraph + ReAct + Tools reais + APIs reais
```
