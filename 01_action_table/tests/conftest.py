"""Shared fixtures of the step-01 tests: the repository on ``sys.path`` and a scratch warehouse folder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STEP_DIR = Path(__file__).resolve().parents[1]
REPO = STEP_DIR.parent
for path in (str(REPO), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A fresh ``SU_DATA_DIR`` for one test."""
    monkeypatch.setenv("SU_DATA_DIR", str(tmp_path / "work"))
    return tmp_path / "work"
