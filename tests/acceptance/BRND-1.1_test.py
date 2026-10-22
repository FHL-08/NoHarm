"""BRND-1.1@1 `import harmless` resolves after a fresh install of requirements.txt.

The interpreter runs from a directory outside the repository so that a bare
source tree on the working directory cannot satisfy the import: the package
must reach the environment via requirements.txt."""

import subprocess
import sys


def test_brnd_1_1_import_harmless_resolves_after_fresh_install(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import harmless"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "`import harmless` did not resolve in the installed environment: "
        + (result.stderr.strip() or f"exit {result.returncode}")
    )
