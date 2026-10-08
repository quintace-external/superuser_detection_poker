"""Tests of the sizing-aware likelihood: size tiers, the back-off of rng_lik_size, parity of sizing = false with the
size-blind model, and a frozen river-overbet decision."""

from __future__ import annotations

import dataclasses
import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from step04_testlib import STEP_DIR
from sudetect import rangemodel as rm
from sudetect import warehouse
from sudetect.cards import COMBO_INDEX, parse_cards
from sudetect.equity import equity_vs_range
from sudetect.handclass import MADE_CLASSES, classes_for_all_combos
from test_rangemodel import CFG, CFG_POSTERIOR, line, make_tables, node

POSTERIOR = dataclasses.replace(CFG_POSTERIOR, sizing=True)
BLIND = dataclasses.replace(CFG_POSTERIOR, sizing=False)


def sized(node_: rm.Node, size_pot=None, faced_size_pot=None) -> rm.Node:
    return dataclasses.replace(node_, size_tier=rm.size_tier(size_pot), faced_tier=rm.faced_tier(faced_size_pot))


def counts13(**kw) -> np.ndarray:
    out = np.zeros(len(rm.SIZED_ACTIONS))
    for name, v in kw.items():
        out[rm.SIZED_INDEX[name]] = v
    return out


# ------------------------------------------------------------------ tiers


@pytest.mark.parametrize("frac, tier", [
    (0.01, "t1"), (0.3999, "t1"), (0.40, "t2"), (0.5999, "t2"), (0.60, "t3"), (0.8999, "t3"), (0.90, "t4"),
    (1.0999, "t4"), (1.10, "t5"), (1.25, "t5"), (23.6, "t5"),
    (None, ""), (float("nan"), ""), (0.0, ""), (-0.5, ""), ("x", ""),
])
def test_size_tier_edges(frac, tier):
    assert rm.size_tier(frac) == tier
    assert rm.faced_tier(frac) == (tier or rm.FACED_NONE)


def test_sized_outcomes():
    assert rm.SIZED_ACTIONS[:3] == ("fold", "check", "call") and len(rm.SIZED_ACTIONS) == 13
    assert rm.sized_outcome("bet", "t5") == rm.SIZED_INDEX["bet_t5"] == 7
    assert rm.sized_outcome("raise", "t1") == 8
    assert rm.sized_outcome("call", "") == 2
    assert rm.sized_outcome("bet", "") is None        # a bet of unknown size: read with the plain likelihood


def test_fit_sql_puts_each_action_in_its_tier():
    """fit_lik_size.sql assigns the same tiers as size_tier() and keys the faced tier; FitTables reads it back."""
    sizes = [0.2, 0.3999, 0.4, 0.5999, 0.6, 0.8999, 0.9, 1.0999, 1.1, 3.5]
    rows = [("river", "cbet_opp", True, "srp", "nothing", None, "bet", s, None) for s in sizes]
    rows += [("river", "cbet_opp", True, "srp", "nothing", None, "bet", None, None)]        # unknown size: left out
    rows += [("river", "cbet_opp", True, "srp", "nothing", None, "check", None, None)]
    rows += [("river", "vs_cbet", False, "srp", "middle_pair", "gutshot", "call", None, 1.25)] * 3
    rows += [("river", "vs_cbet", False, "srp", "middle_pair", "gutshot", "raise", 0.5, 0.3)]
    con = duckdb.connect()
    con.execute("""CREATE TABLE decisions (street VARCHAR, situation VARCHAR, is_ip BOOLEAN, pot_type_now VARCHAR,
                   made_hand VARCHAR, draw VARCHAR, action VARCHAR, size_pot DOUBLE, faced_size_pot DOUBLE)""")
    con.executemany("INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    edges = {f"e{i + 1}": e for i, e in enumerate(rm.SIZE_EDGES)}
    warehouse.run_sql_file(con, STEP_DIR / "sql" / "fit_lik_size.sql", **edges)
    df = con.execute("SELECT * FROM rng_lik_size").fetchdf()
    s6 = df[(df.level == "S6") & (df.situation == "cbet_opp")].iloc[0]
    assert s6.faced_tier == "none" and s6.n == len(sizes) + 1
    for t in rm.SIZE_TIERS:
        assert s6[f"c_bet_{t}"] == sum(rm.size_tier(s) == t for s in sizes), t
    vs = df[(df.level == "S6") & (df.situation == "vs_cbet")]
    assert set(vs.faced_tier) == {"t5", "t1"}
    assert vs[vs.faced_tier == "t5"].c_call.iat[0] == 3 and vs[vs.faced_tier == "t1"].c_raise_t2.iat[0] == 1
    f0 = df[df.level == "F0"].iloc[0]
    assert f0.n == len(rows) - 1 and pd.isna(f0.faced_tier)
    assert sorted(df.level.unique()) == sorted(rm.SIZED_LEVELS + rm.FIELD_SIZE_LEVELS)
    empty_lik = pd.DataFrame(columns=["level", "street", "situation", "pose", "pot_type", "made_hand", "draw",
                                      "c_fold", "c_check", "c_call", "c_bet", "c_raise", "n"])
    empty_pri = pd.DataFrame(columns=["level", "size_class", "preflop_situation", "position_class", "action",
                                      "hand_class", "n"])
    empty_fld = pd.DataFrame(columns=["size_class", "field_vpip", "field_pfr", "field_agg"])
    t = rm.FitTables.from_frames(empty_lik, empty_pri, empty_fld, df)
    got = t.lik_size["S3"][("river", "vs_cbet", "middle_pair", "t5")]
    assert got[rm.SIZED_INDEX["call"]] == 3 and got.sum() == 3
    assert t.lik_size["F1"][("river", "none")].sum() == len(sizes) + 1


# ------------------------------------------------------------------ back-off


def backoff_tables() -> rm.FitTables:
    """Plain K1 cells everywhere, a plain K3 cell for two pair on the river; sized cells: the two-pair cell facing
    an overbet is too thin (10 actions), the (river, vs_bet, t5) cell holds 30 raises but no bet."""
    t = make_tables({("river", "vs_bet", "two_pair"): [10, 0, 20, 0, 10]})
    t.lik_size = {lv: {} for lv in rm.SIZED_LEVELS + rm.FIELD_SIZE_LEVELS}
    t.lik_size["S3"][("river", "vs_bet", "two_pair", "t5")] = counts13(fold=3, call=6, raise_t5=1)
    t.lik_size["S1"][("river", "vs_bet", "t5")] = counts13(fold=300, call=200, raise_t3=10, raise_t4=10,
                                                          raise_t5=10)
    t.lik_size["F0"][("river",)] = counts13(check=50, bet_t1=10, bet_t2=20, bet_t3=30, bet_t4=40, bet_t5=50)
    return t


def test_backoff_sums_to_one_everywhere():
    t = backoff_tables()
    for street in ("flop", "turn", "river"):
        for sit in ("vs_bet", "cbet_opp"):
            for faced in (rm.FACED_NONE,) + rm.SIZE_TIERS:
                p, direct = t.sized_node_table(street, sit, "ip", "single_raised", faced, POSTERIOR)
                assert p.shape == (len(MADE_CLASSES), 5, 13)
                np.testing.assert_allclose(p.sum(axis=2), 1.0, rtol=0, atol=1e-12)
                assert (p >= 0).all()
                assert direct.any() == (street == "river" and sit == "vs_bet" and faced == "t5")


def test_backoff_is_plain_times_the_next_tier_mix():
    t = backoff_tables()
    cfg = POSTERIOR
    plain, level = t.node_table("river", "vs_bet", "ip", "single_raised", cfg)
    p, direct = t.sized_node_table("river", "vs_bet", "ip", "single_raised", "t5", cfg)
    m = MADE_CLASSES.index("two_pair")
    assert rm.LIK_LEVELS[level[m, 0]] == "K3" and not direct[m, 0]          # the sized K3 cell holds 10 actions
    np.testing.assert_allclose(p[m, 0, :3], plain[m, 0, :3], rtol=0, atol=0)
    raises = np.array([0, 0, 10, 10, 10.0])                                  # the S1 cell: first with >= 25 raises
    np.testing.assert_allclose(p[m, 0, 8:], plain[m, 0, 4] * (raises + cfg.eps) / (30 + 5 * cfg.eps), atol=1e-15)
    bets = np.array([10, 20, 30, 40, 50.0])                                  # no bet below the field row of the street
    np.testing.assert_allclose(p[m, 0, 3:8], plain[m, 0, 3] * (bets + cfg.eps) / (150 + 5 * cfg.eps), atol=1e-15)
    other = MADE_CLASSES.index("nothing")                                   # plain K1: its sized S1 cell is populated
    assert direct[other].all()
    s1 = t.lik_size["S1"][("river", "vs_bet", "t5")]
    np.testing.assert_allclose(p[other, 0], (s1 + cfg.eps) / (s1.sum() + 13 * cfg.eps), atol=1e-15)


# ------------------------------------------------------------------ parity


def a_line():
    flop, turn, river = parse_cards("Kh 9c 5d"), parse_cards("Kh 9c 5d 2s"), parse_cards("Kh 9c 5d 2s 3h")
    return line([
        sized(node("flop", "cbet_opp", "bet", flop), size_pot=0.33),
        sized(node("turn", "vs_raise", "call", turn), faced_size_pot=0.7),
        sized(node("river", "vs_bet", "raise", river), size_pot=1.4, faced_size_pot=1.25),
    ], raised_preflop=True, n=120.0, vpip=0.3, pfr=0.2, agg=0.45)


@pytest.mark.parametrize("base", [CFG, CFG_POSTERIOR], ids=["cap240", "posterior"])
def test_sizing_false_is_the_size_blind_model(base):
    """sizing = false gives exactly the ranges of the size-blind model, whatever sized data the tables and the
    nodes carry; so does sizing = true on tables without sized cells and nodes without sizes."""
    hero_board = tuple(parse_cards("Ac Qd Kh 9c 5d 2s 3h"))
    ln = a_line()
    blind_line = dataclasses.replace(ln, nodes=tuple(dataclasses.replace(n, size_tier="", faced_tier="none")
                                                     for n in ln.nodes))
    plain_tables = make_tables({("river", "vs_bet", "two_pair"): [10, 0, 20, 0, 10]})
    want = rm.predict_range(plain_tables, dataclasses.replace(base, sizing=False), blind_line, hero_board)
    got = rm.predict_range(backoff_tables(), dataclasses.replace(base, sizing=False), ln, hero_board)
    assert np.array_equal(got.ids, want.ids) and np.array_equal(got.w, want.w) and got.sized_mass == 0.0
    same = rm.predict_range(plain_tables, dataclasses.replace(base, sizing=True), blind_line, hero_board)
    assert np.array_equal(same.ids, want.ids) and np.array_equal(same.w, want.w)


# ------------------------------------------------------------------ golden overbet

# River, the opponent leads 125% of the pot. Plain counts: every made class checks 50 and bets 50, so the size-blind
# model learns nothing from the bet. Sized counts (same totals): strong hands overbet, pairs below top pair bet small,
# air overbets as a bluff now and then (the value-heavy mix of hands that reach a showdown).
OVERBET_T5 = {"air": 5, "weak_pair": 1, "middle_pair": 1, "top_pair": 10, "strong": 40, "monster": 40}
BOARD = "Kh 9c 5d 2s 3h"
HERO = "6c 5c"                     # third pair, no straight (no 4)


def overbet_tables() -> rm.FitTables:
    t = make_tables({("river", "cbet_opp", m): [0, 50, 0, 50, 0] for m in MADE_CLASSES})
    t.lik_size = {lv: {} for lv in rm.SIZED_LEVELS + rm.FIELD_SIZE_LEVELS}
    for m in MADE_CLASSES:
        big = OVERBET_T5[rm.MADE_GROUP[m]]
        t.lik_size["S3"][("river", "cbet_opp", m, "none")] = counts13(check=50, bet_t2=50 - big, bet_t5=big)
    return t


def overbet_line() -> rm.OpponentLine:
    return line([sized(node("river", "cbet_opp", "bet", parse_cards(BOARD)), size_pot=1.25)])


def test_golden_river_overbet():
    tables = overbet_tables()
    hero, board = parse_cards(HERO), parse_cards(BOARD)
    dead = tuple(hero) + tuple(board)
    blind = rm.predict_range(tables, BLIND, overbet_line(), dead)
    sized_res = rm.predict_range(tables, POSTERIOR, overbet_line(), dead)
    prior = rm.predict_range(tables, POSTERIOR, line([]), dead)
    eff = {k: 1.0 / float(np.sum(r.w ** 2)) for k, r in (("prior", prior), ("blind", blind), ("sized", sized_res))}
    eq = {k: equity_vs_range(hero, board, rm.range_dense(r.ids, r.w)) for k, r in (("blind", blind),
                                                                                    ("sized", sized_res))}
    # the size-blind model reads the bet as any bet: the range stays the prior
    assert blind.ids.size == prior.ids.size and math.isclose(eff["blind"], eff["prior"], rel_tol=1e-9)
    # the overbet is read as an overbet: the range narrows toward strong hands and the third pair's equity falls
    assert sized_res.ids.size == blind.ids.size and eff["sized"] < 0.8 * eff["blind"]
    assert eq["sized"] < eq["blind"] - 0.05
    assert math.isclose(sized_res.sized_mass, 1.0, rel_tol=1e-12)
    # hand check of one update: weight ratio of a set of nines to a hand of air = ratio of their overbet frequencies
    made, _ = classes_for_all_combos(board)
    cfg = POSTERIOR

    def p_t5(m: str) -> float:
        return (OVERBET_T5[rm.MADE_GROUP[m]] + cfg.eps) / (100 + 13 * cfg.eps)

    w = rm.range_dense(sized_res.ids, sized_res.w)
    nines, air = COMBO_INDEX[parse_cards("9h")[0], parse_cards("9d")[0]], COMBO_INDEX[parse_cards("Qs")[0],
                                                                                     parse_cards("Js")[0]]
    m9, ma = MADE_CLASSES[made[nines]], MADE_CLASSES[made[air]]
    assert (rm.MADE_GROUP[m9], rm.MADE_GROUP[ma]) == ("strong", "air")
    assert math.isclose(w[nines] / w[air], (p_t5(m9) + cfg.eps) / (p_t5(ma) + cfg.eps), rel_tol=1e-9)
    # frozen numbers of this decision (reviewed by hand: 990 live combos, the overbet keeps the strong hands)
    assert blind.ids.size == 990
    assert round(eff["blind"], 3) == pytest.approx(GOLDEN["eff_blind"], abs=1e-3)
    assert round(eff["sized"], 3) == pytest.approx(GOLDEN["eff_sized"], abs=1e-3)
    assert round(eq["blind"], 6) == pytest.approx(GOLDEN["eq_blind"], abs=1e-6)
    assert round(eq["sized"], 6) == pytest.approx(GOLDEN["eq_sized"], abs=1e-6)


# strong hands 12.6% of the range before the overbet, 60.4% after (40 / 100 overbets against 5 for air)
GOLDEN = {"eff_blind": 990.0, "eff_sized": 312.13, "eq_blind": 0.60101, "eq_sized": 0.261958}


# ------------------------------------------------------------------ worker


def test_score_chunk_reads_the_sizes(tmp_path):
    """The staged node columns size_pot and faced_size_pot reach the model through the worker."""
    import pickle

    tables = overbet_tables()
    with open(tmp_path / "fit.pkl", "wb") as fh:
        pickle.dump(tables, fh)
    dec = pd.DataFrame({"hand_id": ["h1"], "action_index": [9], "player_id": ["A"], "street": ["river"],
                        "hole_cards": ["6c5c"], "board": ["Kh9c5d2s3h"], "n_live_opp": [1], "action": ["call"],
                        "pot_odds": [0.36], "size_class": ["2"]})
    opp = pd.DataFrame({"hand_id": ["h1"], "action_index": [9], "opp_id": ["B"], "opp_cards": ["9h9d"],
                        "pf_sit": ["unopened"], "pf_act": ["raise"], "position_class": ["BTN"], "raised_pf": [False],
                        "t_n": [None], "t_vpip": [None], "t_pfr": [None], "t_agg": [None]})
    nodes = pd.DataFrame({"hand_id": ["h1"], "action_index": [8], "player_id": ["B"], "street": ["river"],
                          "action": ["bet"], "situation": ["cbet_opp"], "pose": ["ip"], "pot_type": ["single_raised"],
                          "board": ["Kh9c5d2s3h"], "size_pot": [1.25], "faced_size_pot": [None]})
    job = {}
    for name, df in (("dec", dec), ("opp", opp), ("nodes", nodes)):
        job[name] = str(tmp_path / f"{name}.parquet")
        df.to_parquet(job[name], index=False)
    job["out"] = str(tmp_path / "out.parquet")
    rm.init_worker(str(tmp_path / "fit.pkl"), {"range": dataclasses.asdict(POSTERIOR), "exact": True,
                                                "mc_samples": 1024, "seed": 1})
    stats = rm.score_chunk(job)
    out = pd.read_parquet(job["out"])
    hero, board = parse_cards(HERO), parse_cards(BOARD)
    ln = dataclasses.replace(overbet_line(), raised_preflop=False)
    res = rm.predict_range(tables, POSTERIOR, ln, tuple(hero) + tuple(board))
    assert math.isclose(out.eq_vs_range.iat[0], equity_vs_range(hero, board, rm.range_dense(res.ids, res.w)),
                        abs_tol=1e-12)
    assert math.isclose(out.n_eff_opp.iat[0], 1.0 / float(np.sum(res.w ** 2)), rel_tol=1e-12)
    assert math.isclose(stats["sized_mass"], 1.0, rel_tol=1e-12)
