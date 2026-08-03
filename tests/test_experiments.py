import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ExperimentTests(unittest.TestCase):
    def test_all_experiments_run(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_all.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

if __name__ == "__main__":
    unittest.main()
