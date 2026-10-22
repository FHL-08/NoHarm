import importlib
import subprocess
import sys


def ensure(module, *requirements):
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *(requirements or (module,))])
    importlib.invalidate_caches()
    return importlib.import_module(module)
