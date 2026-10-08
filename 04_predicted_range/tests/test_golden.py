"""Golden decision: the hero faces the opponent's river raise after a flop bet and a turn bet (the narrowing example).

``golden/river_raise.json`` holds the slices of the fitted tables the model reads for this decision and the range
it produced when the file was made (``make_golden.py``, reviewed: the range ends on trips and full houses). The
test rebuilds the range from the file alone, so any change in the model's arithmetic shows up here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from sudetect import rangemodel as rm
from sudetect.cards import CLASS_LABELS, COMBOS, parse_cards
from sudetect.equity import equity_vs_range

# river_raise.json: the default settings; river_raise_cap240.json: the narrow-setting constants (240 combos).
GOLDENS = sorted((Path(__file__).resolve().parent / "golden").glob("*.json"))


@pytest.fixture(scope="module", params=GOLDENS, ids=lambda p: p.stem)
def golden(request):
    doc = json.loads(request.param.read_text(encoding="utf-8"))
    t = doc["tables"]
    tables = rm.FitTables.from_frames(pd.DataFrame(t["rng_lik_post"]), pd.DataFrame(t["rng_prior_pre"]),
                                      pd.DataFrame(t["rng_field"]))
    cfg_args = dict(doc["range_config"])
    cfg_args["bluff_exponent"] = tuple(cfg_args["bluff_exponent"])
    cfg = rm.RangeConfig(**cfg_args)
    nodes = tuple(rm.Node(street=n["street"], situation=n["situation"], pose=n["pose"], pot_type=n["pot_type"],
                          action=n["action"], board=tuple(parse_cards(n["board"]))) for n in doc["nodes"])
    line = rm.OpponentLine(nodes=nodes, **doc["line"])
    hero, board = parse_cards(doc["hero"]), parse_cards(doc["board"])
    return doc, tables, cfg, line, hero, board


def test_golden_range(golden):
    doc, tables, cfg, line, hero, board = golden
    want = doc["expected"]
    res = rm.predict_range(tables, cfg, line, hero + board)
    assert res.prior_level == want["prior_level"]
    assert res.ids.size == want["n_combos"]
    mass = rm.range_classes(res.ids, res.w)
    for label, share in want["top_classes"]:
        assert mass[CLASS_LABELS.index(label)] == pytest.approx(share, abs=2e-6), label
    shares = rm.group_shares(res.ids, res.w, board)
    for group, share in want["groups"].items():
        assert shares[group] == pytest.approx(share, abs=2e-6), group
    eq = equity_vs_range(hero, board, rm.range_dense(res.ids, res.w))
    assert eq == pytest.approx(want["eq_vs_range"], abs=2e-6)


def test_golden_river_raise_removes_air(golden):
    doc, tables, cfg, line, hero, board = golden
    before = rm.predict_range(tables, cfg, dataclasses.replace(line, nodes=line.nodes[:-1]), hero + board)
    after = rm.predict_range(tables, cfg, line, hero + board)
    air_before = rm.group_shares(before.ids, before.w, board)["air"]
    air_after = rm.group_shares(after.ids, after.w, board)["air"]
    assert air_before == pytest.approx(doc["expected"]["air_before_last_action"], abs=2e-6)
    assert air_after < air_before / 2        # the field also raises the river with air, so some stays
    strong = lambda r: sum(rm.group_shares(r.ids, r.w, board)[g] for g in ("monster", "strong"))
    assert strong(after) > max(strong(before), 0.75)      # two pair or better after the raise


def test_golden_range_is_deterministic_and_clean(golden):
    _, tables, cfg, line, hero, board = golden
    a = rm.predict_range(tables, cfg, line, hero + board)
    b = rm.predict_range(tables, cfg, line, hero + board)
    assert (a.ids == b.ids).all() and (a.w == b.w).all()
    assert a.w.sum() == pytest.approx(1.0, abs=1e-12) and a.ids.size <= cfg.cap
    dead = set(hero + board)
    assert not any(int(c) in dead for c in COMBOS[a.ids].ravel())


def test_both_goldens_are_present():
    assert {p.stem for p in GOLDENS} == {"river_raise", "river_raise_cap240"}

