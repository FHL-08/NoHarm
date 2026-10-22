"""Pre-register acceptance tests whose file names are not valid module names."""

import importlib.util
import sys
from pathlib import Path

_TESTS = Path(__file__).parent / "tests"

for _file in sorted(_TESTS.rglob("*_test.py")):
    if _file.stem.isidentifier():
        continue
    _spec = importlib.util.spec_from_file_location(_file.stem, _file)
    _module = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_module)
    except Exception:
        continue
    sys.modules[_file.stem] = _module
