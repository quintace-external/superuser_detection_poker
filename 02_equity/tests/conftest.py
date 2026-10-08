"""Shared fixtures of the step-02 tests: a tiny warehouse built from simulated hands by the step-01 loader.

The warehouse holds 300 hands written in full-information mode (every hole
card known) and 120 hands written the way an observer sees them (cards known
only when shown), so both the exact paths and the NULL rules get exercised.
It is built once per session in a temporary folder; every test that writes
to it gets its own copy.
"""

from __future__ import annotations

import importlib.util
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

STEP_DIR = Path(__file__).resolve().parents[1]
REPO = STEP_DIR.parent
for _path in (str(REPO), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASET = "tiny"


def load_module(name: str, path: Path) -> Any:
    """A step script (its folder name is not importable) as a module."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def step01() -> Any:
    return load_module("step01_run_for_step02_tests", REPO / "01_action_table" / "run.py")


def step02() -> Any:
    return load_module("step02_run", STEP_DIR / "run.py")


def simulated_texts(seed: int, count: int, first_id: int, table: str, full_information: bool) -> list[str]:
    """Hands played by random legal agents at 2-, 3- and 6-seat tables, as hand-history text."""
    from sudetect.sim import RandomAgent, SeatState, TableConfig, play_hand, write_hand

    rng = random.Random(seed)
    deck = np.random.default_rng(seed)
    t0 = datetime(2001, 1, 1, 10, 0, 0)
    out = []
    for k in range(count):
        max_seats = rng.choice((2, 2, 3, 6, 6))
        n = max_seats if max_seats < 6 else rng.randint(3, 6)
        seat_nos = sorted(rng.sample(range(1, max_seats + 1), n))
        names = [f"P_{rng.getrandbits(24):06x}" for _ in range(n)]
        if len(set(names)) < n:
            names = [f"{x}{i}" for i, x in enumerate(names)]
        seats = [SeatState(s, p, rng.choice((40, 100, 100, 250)) * 100) for s, p in zip(seat_nos, names)]
        agents = {p: RandomAgent(rng.getrandbits(32), aggression=0.25, all_in_share=0.05, fold_share=0.2)
                  for p in names}
        rec = play_hand(TableConfig(max_seats=max_seats), seats, rng.choice(seat_nos), agents, first_id + k,
                        table, t0 + timedelta(seconds=61 * k), deck)
        out.append(write_hand(rec, full_information))
    return out


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory) -> Path:
    """Folder holding ``tiny.duckdb`` with the step-01 tables (built once, never written to by tests)."""
    base = tmp_path_factory.mktemp("step02_base")
    hands_dir = base / "hands"
    hands_dir.mkdir()
    (hands_dir / "full.txt").write_text("\n\n".join(simulated_texts(11, 300, 700001, "T01", True)) + "\n",
                                        encoding="utf-8")
    (hands_dir / "observed.txt").write_text("\n\n".join(simulated_texts(12, 120, 800001, "T02", False)) + "\n",
                                            encoding="utf-8")
    old = os.environ.get("SU_DATA_DIR")
    os.environ["SU_DATA_DIR"] = str(base)
    try:
        result = step01().load(DATASET, [str(hands_dir)], log=lambda _msg: None)
    finally:
        if old is None:
            os.environ.pop("SU_DATA_DIR", None)
        else:
            os.environ["SU_DATA_DIR"] = old
    assert result.hands_loaded == 420, result.reasons()
    assert result.violations == 0
    return base


@pytest.fixture
def tiny(tiny_base, tmp_path, monkeypatch) -> Path:
    """A private copy of the tiny warehouse; ``SU_DATA_DIR`` points at it for the test."""
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy2(tiny_base / f"{DATASET}.duckdb", work / f"{DATASET}.duckdb")
    monkeypatch.setenv("SU_DATA_DIR", str(work))
    return work
