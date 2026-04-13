"""Pytest rootdir conftest.

Two jobs:
1. Put the project root on sys.path so `import cadenza_redteam` works in
   tests without requiring `pip install -e .`.
2. Propagate the project root into PYTHONPATH so tests that spawn Python
   subprocesses (e.g. `script.py --help` checks) can import the package too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

_existing = os.environ.get("PYTHONPATH", "")
if _root_str not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _root_str + (os.pathsep + _existing if _existing else "")
    )
