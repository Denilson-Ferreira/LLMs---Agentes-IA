"""Executa todos os experimentos do trilha de estudos."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "experiments" / "01-langgraph-conceitual" / "main.py",
    ROOT / "experiments" / "02-memoria-longo-prazo" / "main.py",
    ROOT / "experiments" / "03-rag-local" / "main.py",
    ROOT / "experiments" / "04-acp-protocolo" / "main.py",
    ROOT / "experiments" / "05-grafo-conhecimento" / "main.py",
    ROOT / "experiments" / "06-gerenciamento-contexto" / "main.py",
    ROOT / "experiments" / "07-multiagentes" / "main.py",
]


def main() -> int:
    failures = 0
    for script in SCRIPTS:
        print("\n" + "=" * 72)
        print("Executando:", script.relative_to(ROOT))
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode != 0:
            failures += 1
    print("\n" + "=" * 72)
    print(f"Concluído. Falhas: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
