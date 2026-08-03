"""Equipe multiagente didática sem CrewAI real."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "sample_data" / "relatorio_ameaca_ficticio.txt"

@dataclass
class TaskResult:
    agent: str
    data: dict

class ExtractorAgent:
    name = "extrator"
    def run(self, text: str) -> TaskResult:
        return TaskResult(self.name, {
            "cves": sorted(set(re.findall(r"CVE-\d{4}-\d{4,7}", text))),
            "ips": sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))),
            "sha256": sorted(set(re.findall(r"\b[a-fA-F0-9]{64}\b", text))),
        })

class EnrichmentAgent:
    name = "enriquecedor"
    def run(self, extracted: dict) -> TaskResult:
        notes = []
        for ip in extracted.get("ips", []):
            if ip.startswith("203.0.113."):
                notes.append(f"{ip} é uma faixa reservada para documentação.")
        return TaskResult(self.name, {"notes": notes, "confidence": "didática"})

class ReporterAgent:
    name = "relator"
    def run(self, extracted: dict, enriched: dict) -> TaskResult:
        summary = (
            f"Foram encontradas {len(extracted.get('cves', []))} CVE(s), "
            f"{len(extracted.get('ips', []))} IP(s) e "
            f"{len(extracted.get('sha256', []))} hash(es). "
            "Os dados precisam de validação humana antes de qualquer ação."
        )
        return TaskResult(self.name, {"summary": summary, "notes": enriched.get("notes", [])})


def run() -> None:
    text = REPORT.read_text(encoding="utf-8")
    extracted = ExtractorAgent().run(text)
    enriched = EnrichmentAgent().run(extracted.data)
    report = ReporterAgent().run(extracted.data, enriched.data)

    print("\n[Equipe multiagente]")
    print(extracted)
    print(enriched)
    print(report)


if __name__ == "__main__":
    run()
