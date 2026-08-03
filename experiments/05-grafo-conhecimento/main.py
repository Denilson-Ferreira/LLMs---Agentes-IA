"""Grafo de conhecimento simples para CTI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "cti_graph.dot"

@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str

class KnowledgeGraph:
    def __init__(self) -> None:
        self.edges: list[Edge] = []

    def add(self, source: str, relation: str, target: str) -> None:
        edge = Edge(source, relation, target)
        if edge not in self.edges:
            self.edges.append(edge)

    def neighbors(self, node: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node or edge.target == node]

    def export_dot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["digraph CTI {"]
        for edge in self.edges:
            s = edge.source.replace('"', '\\"')
            t = edge.target.replace('"', '\\"')
            r = edge.relation.replace('"', '\\"')
            lines.append(f'  "{s}" -> "{t}" [label="{r}"];')
        lines.append("}")
        path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    graph = KnowledgeGraph()
    graph.add("Operação Exemplo", "explora", "CVE-2026-12345")
    graph.add("Operação Exemplo", "usa", "PowerShell")
    graph.add("Operação Exemplo", "comunica-se com", "203.0.113.42")
    graph.add("203.0.113.42", "é", "endereço de documentação")
    graph.export_dot(OUTPUT)

    print("\n[Grafo de conhecimento]")
    for edge in graph.neighbors("Operação Exemplo"):
        print(f"- {edge.source} --{edge.relation}--> {edge.target}")
    print("Arquivo DOT:", OUTPUT)


if __name__ == "__main__":
    run()
