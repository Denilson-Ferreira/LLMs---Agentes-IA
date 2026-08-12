import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPOSITORIO = Path(__file__).resolve().parents[1]
if str(REPOSITORIO) not in sys.path:
    sys.path.insert(0, str(REPOSITORIO))

os.environ.setdefault("GROQ_API_KEY", "chave-ficticia-para-testes")

MODULE_PATH = (
    REPOSITORIO
    / "experiments"
    / "01-langgraph"
    / "agente-cve-nvd"
    / "agente_cve_nvd.py"
)
SPEC = importlib.util.spec_from_file_location("agente_cve_nvd", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Não foi possível carregar {MODULE_PATH}")
modulo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = modulo
SPEC.loader.exec_module(modulo)


class ConsultarNvdTests(unittest.TestCase):
    def test_rejeita_cve_com_formato_invalido(self):
        resultado = modulo.consultar_nvd.invoke({"cve_id": "2021-44228"})
        self.assertIn("erro", resultado)

    @patch.object(modulo, "obter_json", return_value={"vulnerabilities": []})
    def test_informa_quando_cve_nao_e_encontrada(self, obter_json):
        resultado = modulo.consultar_nvd.invoke({"cve_id": "cve-2021-44228"})
        self.assertEqual(resultado, {"cve_id": "CVE-2021-44228", "encontrada": False})
        obter_json.assert_called_once_with(
            f"{modulo.NVD_URL}?cveId=CVE-2021-44228"
        )

    @patch.object(modulo, "obter_json")
    def test_normaliza_resposta_do_nvd(self, obter_json):
        obter_json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2021-44228",
                        "published": "2021-12-10T10:15:09.143",
                        "lastModified": "2025-10-27T17:41:54.913",
                        "descriptions": [
                            {"lang": "es", "value": "Descripción"},
                            {"lang": "en", "value": "Example description"},
                        ],
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "baseScore": 10.0,
                                        "baseSeverity": "CRITICAL",
                                        "vectorString": "CVSS:3.1/example",
                                    }
                                }
                            ]
                        },
                        "references": [
                            {"url": f"https://example.com/{numero}"}
                            for numero in range(7)
                        ],
                    }
                }
            ]
        }
        resultado = modulo.consultar_nvd.invoke({"cve_id": "CVE-2021-44228"})
        self.assertTrue(resultado["encontrada"])
        self.assertEqual(resultado["descricao"], "Example description")
        self.assertEqual(resultado["cvss"]["pontuacao"], 10.0)
        self.assertEqual(resultado["cvss"]["severidade"], "CRITICAL")
        self.assertEqual(len(resultado["referencias"]), 5)


if __name__ == "__main__":
    unittest.main()
