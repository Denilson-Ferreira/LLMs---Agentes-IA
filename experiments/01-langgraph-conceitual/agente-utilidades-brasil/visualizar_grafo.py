"""Gera uma visualização HTML do StateGraph compilado."""

from __future__ import annotations

import argparse
import html
import webbrowser
from pathlib import Path

from agent import ErroConfiguracao, criar_grafo


HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grafo — Agente de Utilidades do Brasil</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 32px; color: #172033; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitulo {{ color: #526078; margin-top: 0; }}
    .painel {{ border: 1px solid #d9dfeb; border-radius: 14px; padding: 28px; background: #fafbfe; }}
    .legenda {{ margin-top: 24px; padding: 18px; border-left: 4px solid #6d5bd0; background: #f2f0ff; }}
    code {{ background: #edf0f6; border-radius: 4px; padding: 2px 5px; }}
  </style>
</head>
<body>
  <h1>Agente de Utilidades do Brasil</h1>
  <p class="subtitulo">StateGraph real compilado pelo LangGraph</p>
  <div class="painel">
    <pre class="mermaid">{mermaid}</pre>
  </div>
  <div class="legenda">
    <strong>Como ler:</strong>
    <code>START → agent</code>. Se a AIMessage possuir tool calls, o fluxo segue para
    <code>tools</code>; cada resultado vira ToolMessage e retorna para <code>agent</code>.
    Quando não há tool calls, o fluxo termina em <code>END</code>.
  </div>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "default", securityLevel: "loose" }});
  </script>
</body>
</html>
"""


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera o diagrama visual do LangGraph.")
    parser.add_argument(
        "--abrir",
        action="store_true",
        help="Abre o HTML gerado no navegador padrão.",
    )
    return parser


def main() -> int:
    args = criar_parser().parse_args()
    pasta = Path(__file__).resolve().parent

    try:
        grafo = criar_grafo()
    except ErroConfiguracao as erro:
        print(f"Erro: {erro}")
        return 2

    mermaid = grafo.get_graph().draw_mermaid()
    arquivo_mermaid = pasta / "grafo_langgraph.mmd"
    arquivo_html = pasta / "grafo_langgraph.html"

    arquivo_mermaid.write_text(mermaid, encoding="utf-8")
    arquivo_html.write_text(
        HTML_TEMPLATE.format(mermaid=html.escape(mermaid)),
        encoding="utf-8",
    )

    print(f"Mermaid: {arquivo_mermaid}")
    print(f"HTML: {arquivo_html}")

    if args.abrir:
        webbrowser.open(arquivo_html.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
