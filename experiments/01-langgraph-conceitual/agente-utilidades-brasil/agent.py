"""Grafo ReAct manual do agente de utilidades do Brasil."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import consultar_cep, consultar_cotacao


SYSTEM_PROMPT = """Você é um agente de utilidades do Brasil.

Você possui ferramentas externas que acessam dados reais.

Sempre que a pergunta depender de CEP ou cotação de moeda, utilize a ferramenta apropriada.
Nunca invente dados que podem ser obtidos pelas ferramentas.
Para endereços brasileiros, utilize consultar_cep.
Para cotação de moedas, utilize consultar_cotacao.

Depois de receber o resultado da ferramenta, explique o resultado ao usuário de forma simples
e clara. Se a ferramenta retornar erro ou informação indisponível, informe isso explicitamente.
Você pode utilizar mais de uma ferramenta para responder uma pergunta quando necessário.

Não revele raciocínio interno ou chain-of-thought. Forneça apenas a resposta útil ao usuário.
"""

FERRAMENTAS = [consultar_cep, consultar_cotacao]
VALORES_DE_EXEMPLO = {
    "sua_chave_xai_aqui",
    "sua_chave_groq_aqui",
    "MINHA_CHAVE",
}


class ErroConfiguracao(RuntimeError):
    """Erro de configuração que deve ser mostrado sem traceback."""


class ErroModelo(RuntimeError):
    """Erro amigável ocorrido ao chamar o provedor da LLM."""


def carregar_configuracao() -> tuple[str, str, str]:
    """Carrega provedor, chave e modelo sem jamais imprimir a credencial."""
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    provedor = os.getenv("LLM_PROVIDER", "").strip().lower()
    chave_xai = os.getenv("XAI_API_KEY", "").strip()
    chave_groq = os.getenv("GROQ_API_KEY", "").strip()

    # Compatibilidade com um .env antigo no qual uma chave gsk_ foi colocada
    # por engano em XAI_API_KEY. A chave é reutilizada sem ser exibida.
    if not chave_groq and chave_xai.startswith("gsk_"):
        chave_groq = chave_xai

    if not provedor:
        provedor = "groq" if chave_groq else "xai"

    if provedor == "groq":
        modelo = (
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
            or "llama-3.1-8b-instant"
        )
        if not chave_groq or chave_groq in VALORES_DE_EXEMPLO:
            raise ErroConfiguracao(
                "GROQ_API_KEY não encontrada. Configure a chave Groq no arquivo .env."
            )
        return provedor, chave_groq, modelo

    if provedor == "xai":
        modelo = os.getenv("XAI_MODEL", "grok-4").strip() or "grok-4"
        if not chave_xai or chave_xai in VALORES_DE_EXEMPLO:
            raise ErroConfiguracao(
                "XAI_API_KEY não encontrada. Configure a chave xAI no arquivo .env."
            )
        if chave_xai.startswith("gsk_"):
            raise ErroConfiguracao(
                "LLM_PROVIDER=xai, mas a chave configurada pertence à Groq. "
                "Use LLM_PROVIDER=groq."
            )
        return provedor, chave_xai, modelo

    raise ErroConfiguracao("LLM_PROVIDER deve ser 'groq' ou 'xai'.")


def _converter_erro_modelo(
    erro: Exception, provedor: str, modelo: str
) -> ErroModelo:
    texto = str(erro)
    texto_minusculo = texto.lower()
    nome_erro = type(erro).__name__.lower()

    if "authentication" in nome_erro or "api key" in texto_minusculo or "401" in texto:
        return ErroModelo(
            f"A {provedor} rejeitou a credencial. Verifique a API key no arquivo .env."
        )

    indicadores_modelo = ("model", "modelo")
    indicadores_indisponivel = (
        "not found",
        "does not exist",
        "not available",
        "permission",
        "access",
        "404",
    )
    if any(item in texto_minusculo for item in indicadores_modelo) and any(
        item in texto_minusculo for item in indicadores_indisponivel
    ):
        return ErroModelo(
            f"O modelo '{modelo}' não está disponível na {provedor} para esta conta. "
            f"Altere {'GROQ_MODEL' if provedor == 'groq' else 'XAI_MODEL'} no .env."
        )

    return ErroModelo(
        f"Não foi possível consultar {provedor}/{modelo}: {type(erro).__name__}."
    )


def criar_grafo() -> Any:
    """Monta e compila explicitamente o ciclo AGENT -> TOOLS -> AGENT."""
    provedor, chave, nome_modelo = carregar_configuracao()
    if provedor == "groq":
        model = ChatGroq(
            model=nome_modelo,
            temperature=0,
            api_key=chave,
            timeout=60,
            max_retries=2,
        )
        model_com_tools = model.bind_tools(FERRAMENTAS)
    else:
        model = ChatXAI(
            model=nome_modelo,
            temperature=0,
            api_key=chave,
            timeout=60,
            max_retries=2,
        )
        model_com_tools = model.bind_tools(FERRAMENTAS, parallel_tool_calls=True)

    def call_model(state: MessagesState) -> dict[str, list]:
        """Nó AGENT: a LLM decide entre responder ou solicitar uma ferramenta."""
        try:
            resposta = model_com_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
            )
        except Exception as erro:
            raise _converter_erro_modelo(erro, provedor, nome_modelo) from erro
        return {"messages": [resposta]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(FERRAMENTAS, handle_tool_errors=True))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()
