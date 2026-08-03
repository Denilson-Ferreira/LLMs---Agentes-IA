"""Fluxo didático inspirado em grafos de estado.

Não usa LangGraph real. O objetivo é demonstrar nós, estado, arestas e decisões.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "sample_data" / "relatorio_ameaca_ficticio.txt"

@dataclass
class State:
    text: str
    cves: list[str] = field(default_factory=list)
    ips: list[str] = field(default_factory=list)
    risk: str = "desconhecido"
    history: list[str] = field(default_factory=list)


def receive_report(state: State) -> State:
    state.history.append("relatório recebido")
    return state


def extract_entities(state: State) -> State:
    state.cves = sorted(set(re.findall(r"CVE-\d{4}-\d{4,7}", state.text)))
    state.ips = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", state.text)))
    state.history.append("entidades extraídas")
    return state


def assess_risk(state: State) -> State:
    state.risk = "revisão necessária" if state.cves or state.ips else "baixo"
    state.history.append("risco avaliado")
    return state


def human_review(state: State) -> State:
    state.history.append("aguardando validação humana")
    return state


def run() -> None:
    state = State(text=REPORT.read_text(encoding="utf-8"))
    for node in (receive_report, extract_entities, assess_risk, human_review):
        state = node(state)

    print("\n[LangGraph conceitual]")
    print("CVEs:", state.cves)
    print("IPs:", state.ips)
    print("Risco:", state.risk)
    print("Histórico:", " -> ".join(state.history))


if __name__ == "__main__":
    run()
