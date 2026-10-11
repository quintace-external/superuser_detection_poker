"""Pair outliers (sudetect/pairs.py): the pair hands of a tiny warehouse, the leave-pair-out residual, and a synthetic
field where one player sees one opponent's cards and another opponent loses to everybody."""
import math

import numpy as np
import pandas as pd
import pytest
from test_potsize import observer_warehouse

from sudetect import pairs as P
from sudetect import warehouse


def test_pair_rows_of_a_tiny_warehouse():
    con = observer_warehouse()
    rows, dec = P.load_inputs(con)
    con.close()
    assert dec.empty                                                    # no step 02 tables: money and showdowns only
    got = sorted(zip(rows["hand_id"], rows["A"], rows["B"]))
    assert got == [("H1", "P1", "P2"), ("H1", "P2", "P1"), ("H2", "P1", "P2"), ("H2", "P2", "P1"),
                   ("H3", "P1", "P3"), ("H3", "P3", "P1"), ("H4", "P2", "P3"), ("H4", "P3", "P2")]
    h3 = rows[rows["hand_id"] == "H3"].set_index("A")
    assert not h3["sd"].any() and h3.loc["P1", "net"] == -5.0          # P1 checked, then folded the flop: still a pair
    h4 = rows[rows["hand_id"] == "H4"].set_index("A")
    assert h4["won"].tolist() == [0.5, 0.5]
    h2 = rows[(rows["hand_id"] == "H2") & (rows["A"] == "P1")].iloc[0]
    assert bool(h2["rcall_big"]) and h2["river_in"] == 160.0 and not bool(h2["raggr"])


def test_split_resid_is_the_leave_pair_out_formula():
    df = pd.DataFrame({"A": ["a", "a", "a", "b", "b", "c"], "B": ["x", "x", "y", "x", "y", "y"],
                       "x": [1.0, 3.0, 0.0, 0.0, 1.0, 2.0], "v": 1.0})
    t = P.split_resid(df, prior=0).set_index(["A", "B"])
    m = df["x"].mean()
    # a vs x: mean 2; a vs the rest (y): 0; the rest vs x (b): 0
    assert t.loc[("a", "x"), "resid"] == pytest.approx(2 - 0 - 0 + m)
    # b vs y: mean 1; b vs the rest (x): 0; the rest vs y (a, c): (0 + 2) / 2
    assert t.loc[("b", "y"), "resid"] == pytest.approx(1 - 0 - 1 + m)
    assert t.loc[("a", "x"), "n"] == 2
    # c met nobody but y: without pseudo-cases his "rest" is empty; the default prior pulls it to the mean instead
    assert math.isnan(t.loc[("c", "y"), "z"]) and np.isfinite(t.drop(index=("c", "y"))["z"]).all()
    assert np.isfinite(P.split_resid(df)["z"]).all()


def synthetic_field(seed: int = 11, n_players: int = 24, hands: int = 40):
    """Every pair meets ``hands`` times (P00 and P01 half as often again). P00 sees P01's cards: against P01 he wins
    3bb a hand, wins 80% of the showdowns, calls only when ahead and folds only when behind. P23 is a weak player
    who loses 4bb a hand to everyone; nobody sees his cards."""
    rng = np.random.default_rng(seed)
    names = [f"P{i:02d}" for i in range(n_players)]
    rows, dec, hid = [], [], 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cheat = (a, b) == ("P00", "P01")
            for _ in range(hands * 3 // 2 if cheat else hands):
                hid += 1
                mu = 3.0 if cheat else 0.0
                mu += 4.0 if b == "P23" else (-4.0 if a == "P23" else 0.0)
                net = rng.normal(mu, 8.0)
                sd = rng.random() < 0.35
                p_win = 0.8 if cheat else (0.6 if b == "P23" else (0.4 if a == "P23" else 0.5))
                won = float(rng.random() < p_win) if sd else 0.0
                for x, y, n, w in ((a, b, net, won), (b, a, -net, (1.0 - won) if sd else 0.0)):
                    rows.append((f"H{hid}", x, y, n, sd, w))
                    q, r = rng.random(), rng.random()
                    sees = cheat and x == "P00"
                    action = ("call" if q >= 0.5 else "fold") if sees else ("call" if rng.random() < 0.5 else "fold")
                    dec.append((f"H{hid}", x, y, "river", None, action, False, 5.0, q, r, False))
    rows = pd.DataFrame(rows, columns=["hand_id", "A", "B", "net", "sd", "won"]).assign(
        pot=20.0, river_in=0.0, raggr=False, rcall_big=False, saw_river=False, stack=100.0)
    dec = pd.DataFrame(dec, columns=["hand_id", "A", "B", "street", "situation", "action", "is_allin", "to_call_bb",
                                     "q", "r", "faced_allin"])
    return rows, dec


def test_the_seen_opponent_is_the_top_pair_and_the_weak_one_is_nobody_victim():
    rows, dec = synthetic_field()
    t = P.pair_table(rows, dec)
    top = t.iloc[0]
    assert (top["A"], top["B"]) == ("P00", "P01") and top["field_rank"] == 1
    assert top["fold_z"] > 3 and top["call_z"] > 3                     # folds only behind, calls only ahead
    v = P.victims(t, "P00")
    assert v["B"].tolist() == ["P01"]
    # P23 loses to everybody: the raw win rate against him is high in every pair, the leave-pair-out residual is not
    vs_weak = t[t["B"] == "P23"]
    assert (vs_weak["bb100"] > 150).mean() > 0.6
    assert vs_weak["score"].max() < top["score"] / 2
    assert sum(len(P.victims(t, a)) > 0 for a in t["A"].unique() if a != "P00") <= 1
    assert P.pair_label(top) == f"pair #1 of {int(top['n_ranked']):,} (top {100 / top['n_ranked']:.1f}%)"


def test_parts_need_their_minimum_sample():
    rows, dec = synthetic_field(hands=8)                               # 8 hands a pair: below 30, wr never scores
    t = P.pair_table(rows, dec, P.PairConfig(min_cases=7))
    assert t["wr_z"].isna().all() and t["wr_n"].max() < 30
    assert (t.loc[t["call_n"] < 7, "call_z"].isna()).all()
    without = P.pair_table(rows, dec, P.PairConfig(score_parts=("wr", "sd")))
    assert (without["k"] <= 2).all()


def test_merge_aliases_makes_one_player():
    rows, dec = synthetic_field(n_players=6, hands=10)
    r2, d2 = P.merge_aliases(rows, dec, {"P00": "SU", "P02": "SU"})
    assert "P00" not in set(r2["A"]) and "SU" in set(r2["A"])
    assert not ((r2["A"] == "SU") & (r2["B"] == "SU")).any() and not (d2["A"] == d2["B"]).any()
    assert len(r2) == len(rows) - 2 * 10                               # the P00-P02 hands are gone, both sides


def test_pot_columns_on_the_tiny_warehouse():
    con = observer_warehouse()
    rows, dec = P.load_inputs(con)
    con.close()
    t = P.pot_columns(rows, P.PairConfig(min_cases=1)).set_index(["A", "B"])
    assert t.loc[("P1", "P2"), "wsd_big"] == 0.5 and t.loc[("P1", "P2"), "wsd_big_n"] == 2
    assert t.loc[("P1", "P2"), "rbet_win"] == 1.0 and t.loc[("P1", "P2"), "rcall_win"] == 0.0
    assert t.loc[("P1", "P2"), "stackoff"] == 1.0 and t.loc[("P2", "P1"), "stackoff"] == 0.0
    assert t.loc[("P1", "P2"), "net_big"] == -70.0
    assert math.isnan(t.loc[("P2", "P3"), "wsd_big"])                  # H4 is a 20bb pot


def test_config_from_toml():
    cfg = P.PairConfig.from_toml({"min_cases": 10, "score_parts": ["wr", "sd"], "victim_min_score": 2.5})
    assert cfg.min_cases == 10 and cfg.score_parts == ("wr", "sd") and cfg.victim_min_score == 2.5
    assert P.PairConfig.from_toml(None) == P.PairConfig()


def test_decisions_with_equity_tables():
    con = observer_warehouse()
    eq = pd.DataFrame([("H1", 4, 0.0, True), ("H2", 4, 0.0, True), ("H3", 5, 0.2, True)],
                      columns=["hand_id", "action_index", "eq_vs_real", "eq_real_known"])
    warehouse.insert_df(con, "action_equity", eq)
    warehouse.insert_df(con, "action_range_equity", eq[["hand_id", "action_index"]].assign(eq_vs_range=0.4))
    rows, dec = P.load_inputs(con)
    only_sd = P.load_inputs(con, showdown_only=True)[1]
    con.close()
    assert sorted(zip(dec["hand_id"], dec["A"], dec["B"])) == [("H1", "P2", "P1"), ("H2", "P1", "P2"),
                                                              ("H3", "P1", "P3")]
    assert set(only_sd["hand_id"]) == {"H1", "H2"} and dec["r"].eq(0.4).all()


def test_pair_label_without_the_field_size():
    row = {"field_rank": 4, "n_ranked": 10000}
    assert P.pair_label(row) == "pair #4 of 10,000 (top 0.04%)"
    assert P.pair_label(row, with_n=False) == "top 0.04% of the field's pairs"
    assert P.pair_label({"field_rank": 1, "n_ranked": 50000}, with_n=False) == "top <0.01% of the field's pairs"
    assert P.pair_label({"field_rank": float("nan"), "n_ranked": 10}) == ""


def correlated_field(seed: int = 5, n_players: int = 30, hands: int = 60):
    """An honest field whose win rate and showdowns come from the same hands (a showdown is won by the player who nets
    the chips) and whose calls on several streets of one hand share the same cards."""
    rng = np.random.default_rng(seed)
    names = [f"Q{i:02d}" for i in range(n_players)]
    rows, dec, hid = [], [], 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for _ in range(hands):
                hid += 1
                net = rng.normal(0.0, 8.0)
                sd = rng.random() < 0.4
                for x, y, n in ((a, b, net), (b, a, -net)):
                    rows.append((f"H{hid}", x, y, n, sd, float(n > 0) if sd else 0.0))
                    ahead = rng.random() < 0.5
                    for street in ("flop", "turn", "river"):      # the same cards on every street of the hand
                        q = rng.uniform(0.5, 1.0) if ahead else rng.uniform(0.0, 0.5)
                        dec.append((f"H{hid}", x, y, street, None, "call", False, 5.0, q, rng.random(), False))
    rows = pd.DataFrame(rows, columns=["hand_id", "A", "B", "net", "sd", "won"]).assign(
        pot=20.0, river_in=0.0, raggr=False, rcall_big=False, saw_river=False, stack=100.0)
    dec = pd.DataFrame(dec, columns=["hand_id", "A", "B", "street", "situation", "action", "is_allin", "to_call_bb",
                                     "q", "r", "faced_allin"])
    return rows, dec


def test_honest_scores_are_calibrated_despite_shared_hands():
    rows, dec = correlated_field()
    parts = pd.concat([P.hand_parts(rows), P.decision_parts(dec)], ignore_index=True)
    calls = parts[parts["part"] == "call"]
    assert P.design_effect(calls) > 2                       # three calls a hand on the same cards
    t = P.pair_table(rows, dec)
    s = t.loc[t["k"] >= 2, "score"]
    assert 0.8 < s.std() < 1.25                             # summing correlated parts as if independent gives ~1.4
    assert (s >= 3).mean() < 0.01 and len(P.victims(t, "Q00")) == 0


def test_second_pass_keeps_the_victims_other_pairs_quiet():
    rows, dec = synthetic_field(hands=40)
    t = P.pair_table(rows, dec)
    # P01 loses heavily to P00; without the second pass his "rest" is dragged down by those hands and his ordinary
    # pairs against third players look like wins, and so do the pairs of third players against P00
    assert P.victims(t, "P01").empty and t[t["A"] == "P01"]["score"].max() < 3
    assert t[(t["B"] == "P00") & (t["k"] >= 2)]["score"].max() < 3
    # elsewhere only chance: with an honest score of sd ~1, the largest of 551 honest pairs sits near 3
    assert sum(len(P.victims(t, a)) for a in t["A"].unique() if a != "P00") <= 1


def test_empty_input_gives_an_empty_table():
    rows, dec = synthetic_field(n_players=2, hands=0)
    t = P.pair_table(rows.iloc[0:0], dec.iloc[0:0])
    assert t.empty and {"A", "B", "score", "field_rank"} <= set(t.columns)
