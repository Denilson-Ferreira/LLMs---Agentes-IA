"""Ferramentas reais usadas pelo agente de utilidades do Brasil."""

from __future__ import annotations

import re

import requests
from langchain.tools import tool


VIACEP_URL = "https://viacep.com.br/ws/{cep}/json/"
AWESOMEAPI_URL = "https://economia.awesomeapi.com.br/json/last/{par}"
TIMEOUT_SEGUNDOS = 10


def _linhas_presentes(dados: dict, campos: tuple[tuple[str, str], ...]) -> list[str]:
    """Formata somente campos realmente presentes e não vazios na resposta."""
    linhas: list[str] = []
    for rotulo, chave in campos:
        valor = dados.get(chave)
        if valor not in (None, ""):
            linhas.append(f"{rotulo}: {valor}")
    return linhas


@tool
def consultar_cep(cep: str) -> str:
    """Consulta um CEP brasileiro real no ViaCEP.

    Use quando o usuário pedir endereço, cidade, estado, DDD ou dados de um CEP.
    O argumento pode conter hífen e espaços, por exemplo: ``01001-000``.
    """
    cep_normalizado = cep.replace("-", "").replace(" ", "")
    if not re.fullmatch(r"\d{8}", cep_normalizado):
        return (
            "CEP inválido. Informe exatamente 8 números; hífen e espaços são opcionais."
        )

    try:
        response = requests.get(
            VIACEP_URL.format(cep=cep_normalizado),
            timeout=TIMEOUT_SEGUNDOS,
        )
        response.raise_for_status()
        dados = response.json()
    except requests.exceptions.Timeout:
        return "A consulta ao ViaCEP excedeu o limite de 10 segundos. Tente novamente."
    except requests.exceptions.ConnectionError:
        return "Não foi possível conectar ao ViaCEP. Verifique a conexão e tente novamente."
    except requests.exceptions.HTTPError as erro:
        status = erro.response.status_code if erro.response is not None else "desconhecido"
        return f"O ViaCEP retornou um erro HTTP ({status})."
    except requests.exceptions.RequestException as erro:
        return f"Falha ao consultar o ViaCEP: {type(erro).__name__}."
    except ValueError:
        return "O ViaCEP retornou uma resposta que não é um JSON válido."

    if not isinstance(dados, dict):
        return "O ViaCEP retornou uma resposta em formato inesperado."
    erro_cep = dados.get("erro")
    if erro_cep is True or str(erro_cep).lower() == "true":
        return f"CEP {cep_normalizado} não encontrado no ViaCEP."

    linhas = _linhas_presentes(
        dados,
        (
            ("CEP", "cep"),
            ("Logradouro", "logradouro"),
            ("Bairro", "bairro"),
            ("Cidade", "localidade"),
            ("Estado", "estado"),
            ("UF", "uf"),
            ("DDD", "ddd"),
            ("IBGE", "ibge"),
        ),
    )
    if not linhas:
        return "O ViaCEP respondeu, mas não forneceu campos de endereço utilizáveis."
    return "\n".join(linhas)


def _normalizar_par_moeda(moeda: str) -> str | None:
    valor = moeda.strip().upper().replace(" ", "")
    if "-" not in valor:
        valor = f"{valor}-BRL"
    if not re.fullmatch(r"[A-Z0-9]{2,10}-[A-Z0-9]{2,10}", valor):
        return None
    return valor


@tool
def consultar_cotacao(moeda: str) -> str:
    """Consulta uma cotação real na AwesomeAPI, normalmente contra o real brasileiro.

    Aceita um código como ``USD``, ``EUR`` ou ``BTC`` e o transforma em ``*-BRL``.
    Também aceita um par explícito, como ``USD-BRL``.
    """
    par = _normalizar_par_moeda(moeda)
    if par is None:
        return (
            "Moeda inválida. Informe um código como USD, EUR, BTC ou um par como USD-BRL."
        )

    try:
        response = requests.get(
            AWESOMEAPI_URL.format(par=par),
            timeout=TIMEOUT_SEGUNDOS,
        )
        response.raise_for_status()
        resposta = response.json()
    except requests.exceptions.Timeout:
        return "A consulta à AwesomeAPI excedeu o limite de 10 segundos. Tente novamente."
    except requests.exceptions.ConnectionError:
        return (
            "Não foi possível conectar à AwesomeAPI. Verifique a conexão e tente novamente."
        )
    except requests.exceptions.HTTPError as erro:
        status = erro.response.status_code if erro.response is not None else None
        if status == 404:
            return f"Moeda ou par {par} não encontrado na AwesomeAPI."
        return f"A AwesomeAPI retornou um erro HTTP ({status or 'desconhecido'})."
    except requests.exceptions.RequestException as erro:
        return f"Falha ao consultar a AwesomeAPI: {type(erro).__name__}."
    except ValueError:
        return "A AwesomeAPI retornou uma resposta que não é um JSON válido."

    if not isinstance(resposta, dict) or not resposta:
        return f"A AwesomeAPI não retornou uma cotação para {par}."

    chave_esperada = par.replace("-", "")
    dados = resposta.get(chave_esperada)
    if not isinstance(dados, dict):
        return f"A AwesomeAPI retornou um formato inesperado para {par}."

    linhas = _linhas_presentes(
        dados,
        (
            ("Nome da moeda", "name"),
            ("Preço de compra", "bid"),
            ("Preço de venda", "ask"),
            ("Máxima", "high"),
            ("Mínima", "low"),
            ("Variação percentual", "pctChange"),
            ("Data/hora", "create_date"),
        ),
    )
    if not any(linha.startswith("Data/hora:") for linha in linhas):
        timestamp = dados.get("timestamp")
        if timestamp not in (None, ""):
            linhas.append(f"Data/hora (timestamp informado pela API): {timestamp}")

    if not linhas:
        return f"A AwesomeAPI respondeu, mas não forneceu campos de cotação para {par}."
    return "\n".join(linhas)
