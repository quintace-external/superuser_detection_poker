"""Helpers of the step-04 tests: the repository paths and the step scripts as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

STEP_DIR = Path(__file__).resolve().parents[1]
REPO = STEP_DIR.parent


def load_script(name: str, path: Path) -> Any:
    """A script of a step folder as a module (the folder names are not importable)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return sys.modules[name]
