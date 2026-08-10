"""CLI didática para observar um agente ReAct executando ferramentas reais."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage

from agent import ErroConfiguracao, ErroModelo, criar_grafo


SEPARADOR = "=" * 50
PERGUNTAS_DE_TESTE = (
    "Qual o endereço do CEP 01001-000?",
    "Quanto está o dólar em reais?",
    "Quanto está o euro em reais?",
    "Onde fica o CEP 01001-000 e quanto está o dólar hoje?",
)


def titulo(nome: str) -> None:
    print(f"\n{SEPARADOR}\n{nome}\n")


def texto_mensagem(mensagem: BaseMessage) -> str:
    """Extrai somente texto público de uma mensagem do LangChain."""
    conteudo = mensagem.content
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        partes: list[str] = []
        for bloco in conteudo:
            if isinstance(bloco, str):
                partes.append(bloco)
            elif isinstance(bloco, dict) and isinstance(bloco.get("text"), str):
                partes.append(bloco["text"])
        return "\n".join(partes)
    return str(conteudo)


def _lista_mensagens(atualizacao: dict[str, Any]) -> list[BaseMessage]:
    mensagens = atualizacao.get("messages", [])
    if isinstance(mensagens, BaseMessage):
        return [mensagens]
    return list(mensagens)


def executar_pergunta(
    grafo: Any,
    pergunta: str,
    historico: list[BaseMessage] | None = None,
) -> tuple[list[BaseMessage], str]:
    """Executa o grafo e mostra apenas eventos observáveis do ReAct."""
    historico_atual = list(historico or [])
    mensagem_usuario = HumanMessage(content=pergunta)
    entrada = [*historico_atual, mensagem_usuario]
    historico_atual.append(mensagem_usuario)

    titulo("USUÁRIO")
    print(pergunta)

    nomes_por_chamada: dict[str, str] = {}
    resposta_final = ""

    for evento in grafo.stream(
        {"messages": entrada},
        config={"recursion_limit": 12},
        stream_mode="updates",
    ):
        for nome_node, atualizacao in evento.items():
            if not isinstance(atualizacao, dict):
                continue

            novas_mensagens = _lista_mensagens(atualizacao)
            historico_atual.extend(novas_mensagens)

            if nome_node == "agent":
                for mensagem in novas_mensagens:
                    if not isinstance(mensagem, AIMessage):
                        continue
                    if mensagem.tool_calls:
                        for chamada in mensagem.tool_calls:
                            nome = chamada.get("name", "ferramenta_desconhecida")
                            identificador = chamada.get("id", "")
                            if identificador:
                                nomes_por_chamada[identificador] = nome
                            titulo("TOOL CALL")
                            print(f"Ferramenta: {nome}\n")
                            print("Argumentos:")
                            print(
                                json.dumps(
                                    chamada.get("args", {}),
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            )
                    else:
                        resposta_final = texto_mensagem(mensagem).strip()

            elif nome_node == "tools":
                for mensagem in novas_mensagens:
                    if not isinstance(mensagem, ToolMessage):
                        continue
                    nome = mensagem.name or nomes_por_chamada.get(
                        mensagem.tool_call_id, "ferramenta"
                    )
                    titulo("TOOL RESULT")
                    print(f"Ferramenta: {nome}\n")
                    print(texto_mensagem(mensagem))

    titulo("RESPOSTA DO AGENTE")
    print(resposta_final or "O agente terminou sem produzir uma resposta textual.")
    return historico_atual, resposta_final


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agente ReAct de utilidades do Brasil com Grok e LangGraph."
    )
    parser.add_argument(
        "--testes",
        action="store_true",
        help="Executa as quatro perguntas didáticas antes de abrir o chat.",
    )
    parser.add_argument(
        "--pergunta",
        help="Executa uma única pergunta e encerra, útil para testes automatizados.",
    )
    return parser


def chat_interativo(grafo: Any) -> None:
    titulo("AGENTE DE UTILIDADES")
    print('Digite sua pergunta. Digite "sair" para encerrar.\n')
    historico: list[BaseMessage] = []

    while True:
        try:
            pergunta = input("Você: ").strip()
        except EOFError:
            print("\nEntrada encerrada.")
            return

        if pergunta.lower() == "sair":
            print("Até logo!")
            return
        if not pergunta:
            print("Digite uma pergunta ou 'sair'.")
            continue

        historico, _ = executar_pergunta(grafo, pergunta, historico)


def main() -> int:
    args = criar_parser().parse_args()
    try:
        grafo = criar_grafo()

        if args.pergunta:
            executar_pergunta(grafo, args.pergunta)
            return 0

        if args.testes:
            for numero, pergunta in enumerate(PERGUNTAS_DE_TESTE, start=1):
                titulo(f"TESTE REAL {numero}")
                executar_pergunta(grafo, pergunta)

        chat_interativo(grafo)
        return 0
    except (ErroConfiguracao, ErroModelo) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nExecução cancelada.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
