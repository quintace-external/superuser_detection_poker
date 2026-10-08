"""Helpers for the step-01 tests: parse and replay text hands, read rows by column name."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sudetect.hhparse import GateReject, ParsedHand, ParseOptions, parse_hand
from sudetect.replay import Batch, ReplayContext, replay_hand

STEP_DIR = Path(__file__).resolve().parents[1]


def load_run_module() -> Any:
    """``01_action_table/run.py`` as a module (the folder name is not importable)."""
    name = "step01_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STEP_DIR / "run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def hand(text: str) -> str:
    """Dedent a hand written inside a test."""
    return textwrap.dedent(text).strip("\n") + "\n"


def gate(text: str, **options: Any) -> Optional[str]:
    """The gate reason of a hand, or None when it is kept."""
    parsed = parse_hand(hand(text), ParseOptions(**options))
    return parsed.reason if isinstance(parsed, GateReject) else None


@dataclass
class Replayed:
    """One replayed hand: the parse, and the three tables' rows as dicts (amounts in big blinds)."""

    parsed: ParsedHand
    hand: dict
    players: dict[str, dict]
    actions: list[dict]

    def decisions(self, player: Optional[str] = None, street: Optional[str] = None) -> list[dict]:
        """Betting rows (no posts), optionally of one player and/or one street."""
        return [a for a in self.actions if not a["is_post"]
                and (player is None or a["player_id"] == player)
                and (street is None or a["street"] == street)]

    def row(self, player: str, street: str, k: int = 0) -> dict:
        """The k-th betting row of a player on a street."""
        return self.decisions(player, street)[k]


def replay_text(text: str, source: str = "fixture", **options: Any) -> Replayed:
    """Parse and replay one hand; fails the test if the gate drops it."""
    parsed = parse_hand(hand(text), ParseOptions(**options))
    assert not isinstance(parsed, GateReject), parsed
    rows = replay_hand(parsed, ReplayContext(source=source))
    batch = Batch()
    batch.add(rows)
    tables = batch.to_arrow()
    players = {r["player_id"]: r for r in tables["hand_players"].to_pylist()}
    return Replayed(parsed, tables["hands"].to_pylist()[0], players, tables["actions"].to_pylist())


def simulated_hands(seed: int, count: int, first_id: int, table: str, full_information: bool = True,
                    start: Optional[Any] = None) -> list[str]:
    """Texts of ``count`` hands played by random legal agents (mixed table sizes, antes, stacks, rake)."""
    import random
    from datetime import datetime, timedelta

    import numpy as np

    from sudetect.sim import RandomAgent, SeatState, TableConfig, play_hand, write_hand

    rng = random.Random(seed)
    deck = np.random.default_rng(seed)
    t0 = start or datetime(2001, 1, 1, 10, 0, 0)
    out = []
    for k in range(count):
        max_seats = rng.choice((2, 2, 6))
        n = rng.randint(2, max_seats)
        seat_nos = sorted(rng.sample(range(1, max_seats + 1), n))
        cfg = TableConfig(max_seats=max_seats, ante=rng.choice((0, 10)))
        names = [f"P_{rng.getrandbits(24):06x}" for _ in range(n)]
        if len(set(names)) < n:
            names = [f"{x}{i}" for i, x in enumerate(names)]
        seats = [SeatState(s, p, int(round(40 * 100 ** rng.random())) * 10) for s, p in zip(seat_nos, names)]
        agents = {p: RandomAgent(rng.getrandbits(32), aggression=0.3, all_in_share=0.2) for p in names}
        rec = play_hand(cfg, seats, rng.choice(seat_nos), agents, first_id + k, table,
                        t0 + timedelta(seconds=53 * k), deck)
        out.append(write_hand(rec, full_information))
    return out
