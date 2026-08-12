"""Gera visualizações locais do StateGraph compilado."""

from __future__ import annotations

import argparse
import html
import os
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
    .painel {{ border: 1px solid #d9dfeb; border-radius: 14px; padding: 28px; background: #fafbfe; overflow-x: auto; }}
    .painel svg {{ display: block; width: 100%; min-width: 760px; height: auto; }}
    .legenda {{ margin-top: 24px; padding: 18px; border-left: 4px solid #6d5bd0; background: #f2f0ff; }}
    details {{ margin-top: 24px; }}
    pre {{ overflow-x: auto; white-space: pre-wrap; }}
    code {{ background: #edf0f6; border-radius: 4px; padding: 2px 5px; }}
  </style>
</head>
<body>
  <h1>Agente de Utilidades do Brasil</h1>
  <p class="subtitulo">StateGraph real compilado pelo LangGraph — renderização local</p>
  <div class="painel">
    {svg}
  </div>
  <div class="legenda">
    <strong>Como ler:</strong>
    <code>START → agent</code>. Se a AIMessage possuir tool calls, o fluxo segue para
    <code>tools</code>; cada resultado vira ToolMessage e retorna para <code>agent</code>.
    Quando não há tool calls, o fluxo termina em <code>END</code>.
  </div>
  <details>
    <summary>Mostrar código Mermaid</summary>
    <pre>{mermaid}</pre>
  </details>
</body>
</html>
"""


SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" role="img" aria-labelledby="titulo descricao">
  <title id="titulo">Grafo do agente de utilidades do Brasil</title>
  <desc id="descricao">Fluxo START para AGENT, decisão entre TOOLS e END, e retorno de TOOLS para AGENT.</desc>
  <defs>
    <marker id="seta" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
    </marker>
    <filter id="sombra" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.14"/>
    </filter>
  </defs>
  <rect width="900" height="520" rx="24" fill="#f8fafc"/>

  <g fill="none" stroke="#334155" stroke-width="3" marker-end="url(#seta)">
    <path d="M450 100 L450 152"/>
    <path d="M390 230 C330 260 275 292 230 326" stroke-dasharray="8 7"/>
    <path d="M510 230 C570 260 625 292 670 326" stroke-dasharray="8 7"/>
    <path d="M230 406 C260 474 410 468 425 258"/>
  </g>

  <g font-family="Segoe UI, Arial, sans-serif" text-anchor="middle">
    <g filter="url(#sombra)">
      <rect x="375" y="40" width="150" height="60" rx="30" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
      <text x="450" y="77" font-size="21" font-weight="700" fill="#166534">START</text>
    </g>
    <g filter="url(#sombra)">
      <rect x="350" y="155" width="200" height="100" rx="18" fill="#ede9fe" stroke="#7c3aed" stroke-width="3"/>
      <text x="450" y="195" font-size="24" font-weight="700" fill="#5b21b6">AGENT</text>
      <text x="450" y="226" font-size="15" fill="#4c1d95">Groq + decisão</text>
    </g>
    <g filter="url(#sombra)">
      <rect x="100" y="330" width="260" height="105" rx="18" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
      <text x="230" y="370" font-size="24" font-weight="700" fill="#1e40af">TOOLS</text>
      <text x="230" y="400" font-size="14" fill="#1e3a8a">ViaCEP · AwesomeAPI</text>
    </g>
    <g filter="url(#sombra)">
      <rect x="595" y="330" width="150" height="75" rx="38" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
      <text x="670" y="376" font-size="21" font-weight="700" fill="#991b1b">END</text>
    </g>

    <g font-size="14" font-weight="600" fill="#475569">
      <rect x="255" y="265" width="105" height="28" rx="8" fill="#f8fafc"/>
      <text x="307" y="284">tool_calls</text>
      <rect x="552" y="265" width="112" height="28" rx="8" fill="#f8fafc"/>
      <text x="608" y="284">resposta final</text>
      <rect x="304" y="446" width="130" height="28" rx="8" fill="#f8fafc"/>
      <text x="369" y="465">ToolMessage</text>
    </g>
  </g>
</svg>"""


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera o diagrama visual do LangGraph.")
    parser.add_argument(
        "--abrir",
        action="store_true",
        help="Abre o HTML gerado no navegador padrão.",
    )
    return parser


def abrir_arquivo(arquivo: Path) -> bool:
    """Abre o arquivo no aplicativo padrão e informa se o pedido foi aceito."""
    try:
        if os.name == "nt":
            os.startfile(arquivo)  # type: ignore[attr-defined]
            return True
        return bool(webbrowser.open(arquivo.as_uri(), new=2))
    except OSError as erro:
        print(f"Não foi possível abrir automaticamente: {erro}")
        return False


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
    arquivo_svg = pasta / "grafo_langgraph.svg"
    arquivo_html = pasta / "grafo_langgraph.html"

    arquivo_mermaid.write_text(mermaid, encoding="utf-8")
    arquivo_svg.write_text(SVG_TEMPLATE + "\n", encoding="utf-8")
    arquivo_html.write_text(
        HTML_TEMPLATE.format(
            mermaid=html.escape(mermaid),
            svg=SVG_TEMPLATE,
        ),
        encoding="utf-8",
    )

    print(f"Mermaid: {arquivo_mermaid}")
    print(f"SVG local: {arquivo_svg}")
    print(f"HTML: {arquivo_html}")

    if args.abrir:
        if abrir_arquivo(arquivo_html):
            print("Diagrama aberto no navegador padrão.")
        else:
            print(f"Abra manualmente este arquivo: {arquivo_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
