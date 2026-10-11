"""The preflop and flop tests of step 05 (``sudetect/streets.py``) on synthetic decision rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

import step05_screen as step05
from sudetect import streets as ST


def preflop_rows(seed: int = 7, n_honest: int = 30, n_rows: int = 400) -> pd.DataFrame:
    """Responses to a raise and unopened spots. Honest players fold and raise on r alone; player 'SU' folds when
    behind the real hand (q < 0.40) and raises when ahead (q >= 0.55), whatever r says."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in [f"H{i:02d}" for i in range(n_honest)] + ["SU"]:
        for k in range(n_rows):
            sit = ["vs_open", "vs_3bet", "unopened"][k % 3]
            r = float(rng.uniform(0.25, 0.75))
            q = float(np.clip(r + rng.normal(0, 0.15), 0.02, 0.98))
            facing = sit != "unopened"
            if pid == "SU":
                fold = facing and q < 0.40
                raise_ = q >= 0.55 and not fold
            else:
                fold = facing and rng.random() < 1.2 - 2 * r
                raise_ = not fold and rng.random() < r - 0.2
            action = "fold" if fold else ("raise" if raise_ else ("call" if facing else "check"))
            rows.append((pid, f"{pid}-{k}", sit, facing, action, q, r, 0.3, bool(raise_ and rng.random() < 0.5)))
    d = pd.DataFrame(rows, columns=["player_id", "hand_id", "situation", "facing", "action", "q", "r", "req",
                                    "bluff_success"])
    d["is_fold"] = d["action"] == "fold"
    d["is_cont"] = d["action"].isin(["call", "raise"])
    d["is_aggr"] = d["action"] == "raise"
    return d


def test_preflop_tests_separate_the_informed_player():
    cfg = step05.load_config()
    out, meta = ST.preflop_players(preflop_rows(), ["player_id"], cfg["preflop"])
    out = out.set_index("player_id")
    honest = out.drop(index="SU")
    assert out.loc["SU", "pf_fold_cmh_z"] > 6 and out.loc["SU", "pf_aggr_z"] > 6
    assert out.loc["SU", "pf_fold_behind"] > 0.9 and out.loc["SU", "pf_fold_ahead"] == 0
    assert bool(out.loc["SU", "pf_screen"]) and not honest["pf_screen"].any()
    assert abs(honest["pf_zr"].median()) < 0.5                   # robust-standardised on the population
    assert meta["pf_players_eligible"] == 31


def test_robust_standardise_needs_a_population():
    z = pd.Series([1.0, 2.0, 3.0])
    assert ST.robust_standardise(z, pd.Series([True] * 3), 20).tolist() == [1.0, 2.0, 3.0]
    zz = pd.Series(np.arange(25, dtype=float))
    r = ST.robust_standardise(zz, pd.Series([True] * 25), 20)
    assert r[12] == 0 and r[24] > 0 > r[0]


def test_reraise_success_counts():
    d = pd.DataFrame({
        "player_id": ["A"] * 6,
        "situation": ["vs_open", "vs_open", "vs_3bet", "vs_3bet", "vs_4bet_plus", "unopened"],
        "is_aggr": [True, True, True, True, True, True],
        "bluff_success": [True, False, True, True, False, True],
        "r": [0.3, 0.6, 0.4, 0.5, 0.2, 0.3],
        "q": [0.7, 0.3, np.nan, 0.6, 0.4, 0.9],
    })
    g = ST.reraise_success(d, ["player_id"], 0.45).set_index("player_id").loc["A"]
    assert (g["pf_3bet_n"], g["pf_3bet_succ"]) == (2, 0.5)
    assert (g["pf_4bet_n"], g["pf_4bet_succ"]) == (2, 1.0)
    assert (g["pf_5bet_n"], g["pf_5bet_succ"]) == (1, 0.0)
    # light (r < 0.45): the 3-bet at 0.3, the 4-bet at 0.4, the 5-bet at 0.2; the unopened raise is no re-raise
    assert g["pf_rr_light_n"] == 3 and abs(g["pf_rr_light_succ"] - 2 / 3) < 1e-12
    assert g["pf_rr_light_ahead"] == 0.5                       # q known for two of them, one ahead


def test_flag_rules():
    cfg = step05.load_config()
    pre = pd.DataFrame([
        # PF1: fold, the range said continue (r - req = 0.2), the real hand far ahead (q 0.2 < req)
        ("P", "h1", 1, "preflop", "fold", True, 0.20, 0.50, 0.30),
        # PF2: call, the range said fold (r < req), the real hand behind ours (q 0.6)
        ("P", "h2", 1, "preflop", "call", True, 0.60, 0.25, 0.30),
        # PF3: re-raise light (r 0.40) with q 0.7
        ("P", "h3", 1, "preflop", "raise", True, 0.70, 0.40, 0.30),
        # PF4: flat a strong hand (r 0.65) that is beaten (q 0.2)
        ("P", "h4", 1, "preflop", "call", True, 0.20, 0.65, 0.30),
        # none: an ordinary fold (r - req = 0.05)
        ("P", "h5", 1, "preflop", "fold", True, 0.20, 0.35, 0.30),
    ], columns=["player_id", "hand_id", "action_index", "street", "action", "facing", "q", "r", "req"])
    post = pd.DataFrame([
        ("P", "h6", 5, "flop", "fold", True, 0.10, 0.55, 0.30),      # FL1
        ("P", "h7", 5, "flop", "call", True, 0.70, 0.10, 0.30),      # FL2
        ("P", "h8", 5, "turn", "fold", True, 0.00, 0.90, 0.30),      # not the flop
    ], columns=["player_id", "hand_id", "action_index", "street", "action", "facing", "q", "r", "req"])
    for d in (pre, post):
        d["is_fold"] = d["action"] == "fold"
        d["is_cont"] = d["action"].isin(["call", "raise"])
        d["is_aggr"] = d["action"] == "raise"
    f = ST.flag_streets(pre, post, cfg).set_index("hand_id")
    assert f["rule"].to_dict() == {"h1": "PF1", "h2": "PF2", "h3": "PF3", "h4": "PF4", "h6": "FL1", "h7": "FL2"}
    assert f.loc["h4", "gap"] == 0.45 and f.loc["h4", "tier"] == "BLATANT" and f.loc["h3", "tier"] == "STRONG"
