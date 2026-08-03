"""Memória persistente simples em JSON."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "memoria.json"

@dataclass
class MemoryItem:
    kind: str
    content: str
    source: str
    created_at: str

class AgentMemory:
    def __init__(self, path: Path):
        self.path = path
        self.items: list[MemoryItem] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.items = [MemoryItem(**item) for item in data]

    def add(self, kind: str, content: str, source: str) -> None:
        self.items.append(MemoryItem(
            kind=kind,
            content=content,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))

    def search(self, term: str) -> list[MemoryItem]:
        term = term.lower()
        return [item for item in self.items if term in item.content.lower()]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run() -> None:
    memory = AgentMemory(OUTPUT)
    if not memory.search("CVE-2026-12345"):
        memory.add(
            kind="episódica",
            content="O relatório fictício mencionou a CVE-2026-12345.",
            source="relatorio_ameaca_ficticio.txt",
        )
    if not memory.search("IOC"):
        memory.add(
            kind="semântica",
            content="Um IOC é um indício que precisa ser interpretado no contexto.",
            source="glossário do trilha de estudos",
        )
    memory.save()

    print("\n[Memória de longo prazo]")
    for item in memory.search("CVE"):
        print(f"- {item.kind}: {item.content} (fonte: {item.source})")


if __name__ == "__main__":
    run()
