"""Integration checks across the package.

* The simulator settles showdowns with :func:`sudetect.evaluator.eval_hands`
  unless it is given another evaluator.
* The hand-history writer prints exactly the text of
  :func:`sudetect.evaluator.describe_hand` for every shown hand.
* On a finished board the equity functions agree with the engine's
  settlement: the main pot's winners have equity 1/k, everybody else 0.
* The core modules need nothing but numpy, and every name a module lists in
  ``__all__`` exists.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import sudetect
import sudetect.sim.engine as engine_module
from sudetect.cards import parse_cards
from sudetect.equity import equity_vs_hand, equity_vs_hand_batch, multiway_equity
from sudetect.evaluator import describe_hand, eval_hands
from sudetect.sim import Decision, HandRecord, RandomAgent, SeatState, TableConfig, play_hand, write_hand

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2000, 1, 1, 12, 0, 0)


class CheckCall:
    """Checks or calls every decision, so every hand reaches a showdown."""

    def act(self, view):
        return Decision.check() if view.to_call == 0 else Decision.call()


def random_hands(n: int, seed: int, evaluate=None) -> list[HandRecord]:
    """``n`` hands at random tables: 2-9 players, 1-300 bb stacks, some antes.

    Half the tables seat only check/call agents (every hand goes to a
    showdown on a full board), the rest random agents (all-ins, folds, side
    pots).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out = []
    for hand_id in range(n):
        max_seats = rng.choice((2, 6, 9))
        k = rng.randint(2, max_seats)
        seat_nos = sorted(rng.sample(range(1, max_seats + 1), k))
        cfg = TableConfig(max_seats=max_seats, ante=rng.choice((0, 0, 10)))
        names = [f"P_{rng.getrandbits(24):06x}{i}" for i in range(k)]
        seats = [SeatState(s, p, rng.randint(100, 30_000)) for s, p in zip(seat_nos, names)]
        if rng.random() < 0.5:
            agents = {p: CheckCall() for p in names}
        else:
            agents = {p: RandomAgent(rng.getrandbits(32)) for p in names}
        out.append(play_hand(cfg, seats, rng.choice(seat_nos), agents, hand_id, "Alpha", T0, np_rng,
                             evaluate=evaluate))
    return out


# ------------------------------------------------------------------ wiring


def test_engine_settles_showdowns_with_the_package_evaluator(monkeypatch):
    calls = []

    def counting(cards, *, check=True):
        calls.append(np.asarray(cards).shape)
        return eval_hands(cards, check=check)

    monkeypatch.setattr(engine_module, "eval_hands", counting)
    records = random_hands(200, seed=1)
    showdowns = sum(r.showdown for r in records)
    assert showdowns > 50
    assert len(calls) == showdowns
    assert all(len(shape) == 2 and shape[1] == 7 for shape in calls)


def test_an_evaluator_passed_in_replaces_the_default(monkeypatch):
    def unused(cards, *, check=True):  # pragma: no cover - must not run
        raise AssertionError("the default evaluator ran although another one was passed")

    monkeypatch.setattr(engine_module, "eval_hands", unused)
    reversed_order = random_hands(200, seed=2, evaluate=lambda cards: -eval_hands(cards))
    monkeypatch.undo()
    normal = random_hands(200, seed=2)
    # same deals and decisions; with the order reversed some showdowns go the other way
    assert any(a.net != b.net for a, b in zip(reversed_order, normal) if a.showdown)


SHOWS = re.compile(r"^(\S+): shows \[(\S\S \S\S)\] \((.+)\)$")
SUMMARY_SHOWED = re.compile(
    r"^Seat \d+: (\S+)(?: \((?:button|small blind|big blind)\))* showed \[(\S\S \S\S)\] and"
    r" (?:won \(\d+\.\d\d\) with|lost with) (.+)$")


@pytest.mark.parametrize("full_information", [True, False])
def test_writer_prints_the_evaluator_description_of_every_shown_hand(full_information):
    checked = 0
    for rec in random_hands(400, seed=3):
        text = write_hand(rec, full_information)
        seat_of = {p: s for s, p in rec.players.items()}
        for line in text.splitlines():
            m = SHOWS.match(line) or SUMMARY_SHOWED.match(line)
            if m is None:
                continue
            player, cards, description = m.groups()
            hole = parse_cards(cards)
            assert tuple(hole) == rec.shown[seat_of[player]]
            assert description == describe_hand(hole + list(rec.board))
            checked += 1
    assert checked > 300


# ------------------------------------------------- equity agrees with the engine


def test_river_equity_matches_the_main_pot_winners():
    heads_up = []
    multiway = 0
    for rec in random_hands(600, seed=4):
        if not rec.showdown:
            continue
        assert len(rec.board) == 5
        live = rec.showdown_order  # every player still in the hand, all eligible for the main pot
        main = rec.pots[0]
        assert set(main.eligible) == set(live)
        hands = [list(rec.hole_cards[s]) for s in live]
        shares = multiway_equity(hands, list(rec.board))
        expected = [1 / len(main.winners) if s in main.winners else 0.0 for s in live]
        assert shares == pytest.approx(expected, abs=1e-12)
        if len(live) == 2:
            hero, villain = hands
            assert equity_vs_hand(hero, villain, list(rec.board)) == pytest.approx(expected[0], abs=1e-12)
            heads_up.append((hero, villain, list(rec.board), expected[0]))
        else:
            multiway += 1
    assert len(heads_up) > 30 and multiway > 30
    hero, villain, board, expected = (np.array(col) for col in zip(*heads_up))
    np.testing.assert_allclose(equity_vs_hand_batch(hero, villain, board), expected, atol=1e-12)


# ---------------------------------------------------------- package hygiene

CORE_MODULES = (
    "sudetect.cards",
    "sudetect.evaluator",
    "sudetect.equity",
    "sudetect.handclass",
    "sudetect.positions",
    "sudetect.situations",
    "sudetect.hhparse",
    "sudetect.sim",
    "sudetect.sim.engine",
    "sudetect.sim.hhtext",
    "sudetect.sim.policies",
    "sudetect.sim.population",
    "sudetect.sim.run",
    "sudetect.sim.calibrate",  # imports pandas only inside the functions that build tables
)


def test_core_modules_import_nothing_but_numpy():
    code = (
        "import importlib, sys\n"
        f"for name in {CORE_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        "top = {m.split('.')[0] for m in sys.modules}\n"
        "print(sorted(m for m in top - set(sys.stdlib_module_names) if not m.startswith('_')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert ast.literal_eval(out.stdout.strip()) == ["numpy", "sudetect"]


def _all_modules() -> list[str]:
    names = ["sudetect"]
    names += [m.name for m in pkgutil.walk_packages(sudetect.__path__, prefix="sudetect.")]
    return sorted(names)


@pytest.mark.parametrize("name", _all_modules())
def test_every_name_in_all_exists(name):
    module = importlib.import_module(name)
    missing = [n for n in getattr(module, "__all__", ()) if not hasattr(module, n)]
    assert missing == []
