"""Gerenciamento didático de contexto e arquivo."""
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class ContextManager:
    max_items: int = 3
    core_memory: list[str] = field(default_factory=list)
    archival_memory: list[str] = field(default_factory=list)

    def remember(self, item: str, essential: bool = False) -> None:
        if essential and item not in self.core_memory:
            self.core_memory.append(item)
        elif item not in self.archival_memory:
            self.archival_memory.append(item)
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        while len(self.core_memory) > self.max_items:
            self.archival_memory.append(self.core_memory.pop(0))

    def retrieve(self, term: str) -> list[str]:
        term = term.lower()
        return [x for x in self.archival_memory if term in x.lower()]

    def current_context(self) -> list[str]:
        return list(self.core_memory)


def run() -> None:
    manager = ContextManager(max_items=3)
    manager.remember("Objetivo: analisar o relatório fictício", essential=True)
    manager.remember("Regra: não tratar IOC isolado como prova", essential=True)
    manager.remember("CVE encontrada: CVE-2026-12345", essential=True)
    manager.remember("IP encontrado: 203.0.113.42")
    manager.remember("O endereço é reservado para documentação")

    print("\n[Gerenciamento de contexto]")
    print("Contexto atual:")
    for item in manager.current_context():
        print("-", item)
    print("Recuperação do arquivo por 'documentação':", manager.retrieve("documentação"))


if __name__ == "__main__":
    run()
