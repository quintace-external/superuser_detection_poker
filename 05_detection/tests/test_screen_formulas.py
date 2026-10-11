"""Step 05 formulas against hand-computed values, and the decision query on a tiny synthetic warehouse."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import step05_screen as step05
from sudetect import metrics as M
from sudetect import screening as S
from sudetect import warehouse

SQL = step05.SQL_DIR / "scored_decisions.sql"


# --------------------------------------------------------------------------------------------- method (b)


def responses() -> pd.DataFrame:
    """Player A: 4 dead (p 0.5, folds 1 1 1 0), 5 winning (p 0.2, no fold), 1 middle row; player B: 2 dead
    (p 0.5, no fold), 2 winning (p 0.2, folds 1 0)."""
    rows = [("A", 0.0, y, 0.5) for y in (1, 1, 1, 0)] + [("A", 0.9, 0, 0.2)] * 5 + [("A", 0.3, 1, 0.4)]
    rows += [("B", 0.0, 0, 0.5)] * 2 + [("B", 0.9, 1, 0.2), ("B", 0.9, 0, 0.2)]
    return pd.DataFrame(rows, columns=["player_id", "q", "y", "p"])


def test_range_counters_scores_and_centring():
    c = M.range_counters(responses(), ["player_id"]).set_index("player_id")
    assert c.loc["A", ["nd", "fd", "ed", "vd", "nw", "fw", "ew", "vw"]].tolist() == pytest.approx(
        [4, 3, 2.0, 1.0, 5, 0, 1.0, 0.8])
    dd, dw, n = M.centring(c.reset_index(), min_dead=1, min_win=1)
    assert (dd, dw, n) == (pytest.approx(0.0), pytest.approx(0.4 / 7), 2)
    s = M.range_scores(c.reset_index(), 0.05, 0.10).set_index("player_id").loc["A"]
    assert s["x_dead"] == pytest.approx(0.25) and s["x_win"] == pytest.approx(0.2) and s["s"] == pytest.approx(0.45)
    assert s["z"] == pytest.approx(2 / math.sqrt(1.8))
    assert s["z_c"] == pytest.approx(1.3 / math.sqrt(1.8))
    assert s["z_b"] == pytest.approx(0.30 / math.sqrt(1.0 / 16 + 0.8 / 25))      # 0.975900
    assert s["var_s"] == pytest.approx(0.0945)


def test_excess_discrimination_and_cmh():
    rows = ([("A", "river", 3, 0.0, 1)] * 2 + [("A", "river", 3, 0.9, 0)] * 2
            + [("A", "turn", 5, 0.0, 1)] + [("A", "turn", 5, 0.9, 1)] + [("A", "turn", 5, 0.9, 0)] * 2
            + [("B", "river", 3, 0.0, 1), ("B", "river", 3, 0.0, 0), ("B", "river", 3, 0.9, 1),
               ("B", "river", 3, 0.9, 0)])
    df = pd.DataFrame(rows, columns=["player_id", "street", "rbin", "q", "y"])
    e = M.excess_discrimination(df, ["player_id"]).set_index("player_id")
    assert e.loc["A", "e_disc"] == pytest.approx(0.5 / 1.75)                     # 0.285714
    assert e.loc["A", "cmh_z"] == pytest.approx(1.5 / math.sqrt(1 / 3 + 0.25))   # 1.963961
    assert e.loc["B", "e_disc"] == pytest.approx(-0.5)
    assert e.loc["A", "e_strata"] == 2


def test_oracle_events():
    df = pd.DataFrame({"player_id": "A", "q": [0.0, 0.0, 0.9, 0.9, 0.3], "r": [0.7, 0.7, 0.1, 0.7, 0.6],
                       "req": [0.3, 0.3, 0.35, 0.3, 0.3], "y": [1, 0, 0, 1, 1], "p": [0.1, 0.1, 0.6, 0.1, 0.2]})
    g = M.oracle_events(df, ["player_id"]).iloc[0]
    assert (g["n_rc_dead"], g["k_or_fold"], g["n_rf_win"], g["k_or_call"]) == (2, 1, 1, 1)
    assert g["or_share"] == pytest.approx(2 / 3) and g["or_expected"] == pytest.approx((0.1 + 0.1 + 0.4) / 3)
    assert g["g_fold"] == pytest.approx(0.5 - 1.0)                 # dead folds 1/2 minus winning folds 1/1
    assert g["or_fold_range"] == pytest.approx(2 / 3)              # folds with r >= .5: 3, of them q < .5: 2


# --------------------------------------------------------------------------------------------- scorecard


def decisions(n_value=20, f_value=15, n_bluff=20, f_bluff=5) -> pd.DataFrame:
    rows = []
    for n, f, bq in ((n_value, f_value, 0.8), (n_bluff, f_bluff, 0.2)):
        for i in range(n):
            fold = i < f
            rows.append(dict(player_id="A", street="river", action="fold" if fold else "call", q=1 - bq,
                             facing=True, bettor_q=bq, bluff_success=False, won_at_showdown=not fold))
    rows += [dict(player_id="A", street="turn", action="bet", q=0.2, facing=False, bettor_q=np.nan,
                  bluff_success=s, won_at_showdown=False) for s in (True, True, False)]
    rows += [dict(player_id="A", street="turn", action="raise", q=0.3, facing=True, bettor_q=0.7,
                  bluff_success=True, won_at_showdown=False)]
    d = pd.DataFrame(rows)
    d["is_fold"], d["is_cont"] = d["action"] == "fold", d["action"].isin(["call", "raise"])
    d["is_aggr"] = d["action"].isin(["bet", "raise"])
    d["pair_own"], d["pair_full"] = False, True
    return d


def test_scorecard_gap_bluffs_and_switches():
    d = decisions()
    g = M.scorecard(d, ["player_id"], gap_min_n=20).iloc[0]
    assert g["fold_to_value"] == pytest.approx(15 / 21)            # 15 river folds; the turn raise faced 0.7
    assert g["fold_to_bluff"] == pytest.approx(0.25)
    assert g["disc_gap"] == pytest.approx(15 / 21 - 0.25)
    assert g["bluff_index"] == pytest.approx(3 / 4) and g["bluff_index__n"] == 4   # bets and the raise
    g2 = M.scorecard(d, ["player_id"], gap_min_n=22, bluff_includes_raises=False).iloc[0]
    assert math.isnan(g2["disc_gap"]) and g2["bluff_index"] == pytest.approx(2 / 3)
    assert g["ofr_river"] == pytest.approx(15 / 20)                 # the value-bet folds were behind (q 0.2)
    assert g["ofr_pair__n"] == 0 and M.scorecard(d, ["player_id"], pair_plus_own_cards=False).iloc[0][
        "ofr_pair__n"] == 20
    assert g["river_accuracy"] == pytest.approx((15 + 15) / 40)     # value folds right, bluff calls right
    assert g["river_call_win"] == pytest.approx(20 / 20)


# --------------------------------------------------------------------------------------------- pairs, BH, runs


def test_benjamini_hochberg_and_robust_z():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    q = M.benjamini_hochberg(p)
    assert q[:8] == pytest.approx([0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.084571, 0.205], abs=1e-6)
    assert math.isnan(q[8])
    assert M.robust_z(np.array([1, 2, 3, 4, 100.0]))[4] == pytest.approx(97 / 1.4826)
    assert M.upper_tail_p(np.array([1.959963985]))[0] == pytest.approx(0.025)


def test_two_way_fit_isolates_one_pair():
    # every pair is 0 except (i1 -> j1); i1 also meets j2..j5, j1 also meets i2..i5
    i = ["i1"] * 5 + ["i2", "i3", "i4", "i5"]
    j = ["j1", "j2", "j3", "j4", "j5", "j1", "j1", "j1", "j1"]
    s = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
    mu, a, b, eps = M.two_way_fit(s, np.ones(9), np.array(i), np.array(j), passes=3, prior_weight=1.0)
    assert eps[0] == eps.max() and eps[0] > 0.6 and np.all(np.abs(eps[1:]) < 0.2)


def runs_hands() -> pd.DataFrame:
    spec = {"S": [0.25] * 600, "A": [-0.5] * 500, "B": [0.25] * 500, "C": [1.0] * 500,
            "D": [0.5] * 720 + [1.5] * 720, "E": [5.0] * 499}
    rows = [(pid, pd.Timestamp("2001-01-01") + pd.Timedelta(minutes=k), f"{pid}{k:05d}", v)
            for pid, vals in spec.items() for k, v in enumerate(vals)]
    return pd.DataFrame(rows, columns=["player_id", "started_at", "hand_id", "net_bb"])


def test_win_rate_runs_percentile():
    rc = {"run_length": 720, "min_hands": 500, "n_draws": 50, "best_run_offsets": [0, 240, 480]}
    r = S.win_rate_runs(runs_hands(), rc, seed=1).set_index("player_id")
    assert "E" not in r.index                                        # 499 hands: out of the field
    assert r.loc["D", "wr_runs"] == 2 and r.loc["D", "wr_run_bb100"] == pytest.approx(50.0)
    s = r.loc["S"]                                                   # 25 bb/100 vs -50, 25 (tie), 100, {50, 150}
    assert s["wr_run_bb100"] == pytest.approx(25.0)
    assert (s["wr_pct_median"], s["wr_pct_lo"], s["wr_pct_hi"]) == (37.5, 37.5, 37.5)
    assert s["wr_pct_pooled"] == pytest.approx(37.5) and s["wr_pct_best_run"] == pytest.approx(37.5)
    assert s["wr_at_or_above"] == 3 and s["wr_ever_as_good"] == 3 and s["wr_field"] == 4


# --------------------------------------------------------------------------------------------- stages 1, 3, 4, 5


def test_stage1_top10_and_p95():
    cfg = step05.load_config()
    n = 10
    df = pd.DataFrame({"player_id": [f"P{k}" for k in range(n)], "time_window": "all", "hands": 600,
                       "vpip": np.linspace(0.2, 0.5, n), "pfr": np.linspace(0.1, 0.3, n),
                       "bb100": np.arange(n, 0, -1.0), "wsd": np.arange(n, 0, -1.0), "wtsd": np.arange(n, dtype=float),
                       "fold_river_bet": np.arange(n, 0, -1.0), "river_call_win": np.arange(n, 0, -1.0)})
    out = S.stage1(df, cfg).set_index("player_id")
    assert out.loc["P0", "s1_composite_pct"] == pytest.approx(0.95) and out.loc["P0", "s1_rank_group"] == "all|B2"
    assert out["s1_top10"].tolist() == [True] + [False] * 9 and out["s1_p95"].sum() == 1


def flag_row(**kw) -> pd.DataFrame:
    row = dict(player_id="A", opponent_id="B", hand_id="H1", action_index=9, hand_date=pd.Timestamp("2001-01-02"),
               started_at=pd.Timestamp("2001-01-02 10:00"), street="river", action="fold", is_fold=True,
               is_cont=False, is_aggr=False, facing=True, q=0.0, r=0.8, req=0.3, bluff_success=False, pot_bb=32.5,
               hole_cards="6s6h", opp_hole_cards="8c7d", board_at_action="6d9hTs2cKh", decision_s=1.5, p_all=0.1)
    row.update(kw)
    return pd.DataFrame([row])


def test_hand_risk_check_value():
    cfg = step05.load_config()
    f = S.flag_hands(flag_row(), cfg).set_index("rule_set")
    a = f.loc["article"]
    assert (a["category"], a["decision_label"], bool(a["snap"])) == ("A", "Perfect fold", True)
    assert a["risk_raw"] == pytest.approx(1.0 * 1.5 * math.log(1 + 32.5) * 1.2, abs=1e-12)   # 6.3197
    assert round(a["risk_raw"], 2) == 6.32 and a["risk"] == pytest.approx(100.0)
    assert a["risk_surprise_raw"] == pytest.approx(-math.log(0.1) * math.log(33.5) * 1.2)
    assert f.loc["range_conditioned", "label"] == "Perfect"
    slow = S.flag_hands(flag_row(decision_s=np.nan), cfg).set_index("rule_set").loc["article"]
    assert not slow["snap"] and slow["risk_raw"] == pytest.approx(1.5 * math.log(33.5))      # unknown time: 1
    # the range said fold (r - req < 0.24): the range-conditioned rule set does not flag it
    assert set(S.flag_hands(flag_row(r=0.4), cfg)["rule_set"]) == {"article"}


def test_escalation_rules():
    cfg = step05.load_config()
    cfg["hand_tests"]["rule_sets"] = ["article"]
    t0 = pd.Timestamp("2001-02-01")
    hands = pd.DataFrame([(p, f"{p}{k}", t0 + pd.Timedelta(hours=k)) for p in "XYZ" for k in range(100)],
                         columns=["player_id", "hand_id", "started_at"])
    hands = pd.concat([hands, pd.DataFrame([("X", "Xold", t0 - pd.Timedelta(days=30))],
                                           columns=hands.columns)])
    flagged = pd.DataFrame([("article", p, f"{p}{k}") for p, n in (("X", 6), ("Y", 5), ("Z", 10)) for k in range(n)]
                           + [("article", "X", "Xold")], columns=["rule_set", "player_id", "hand_id"])
    e = S.escalate(flagged, hands, cfg)
    fixed = e[e["rule"] == "article_fixed"].set_index("player_id")
    assert fixed["hands"].tolist() == [100, 100, 100] and fixed["flagged"].tolist() == [6, 5, 10]
    assert fixed["escalate"].tolist() == [True, False, True]         # > 5% and > 5 (Y has exactly 5)
    cal = e[e["rule"] == "population_calibrated"].set_index("player_id")
    assert cal["threshold_ratio"].iloc[0] == pytest.approx(0.06 + 2.326348 * 1.4826 * 0.01, abs=1e-5)


# --------------------------------------------------------------------------------------------- the decision query


def tiny_warehouse():
    """H1, H2: heads-up table. H3: three dealt, two see the flop. H4: three see the flop. H5: three dealt, a
    3-bet pot that ends preflop after the third player folded."""
    con = warehouse.connect(":memory:")
    acts = [  # hand, index, player, street, action, q, live opponents (n_live = 1 + their number)
        ("H1", 1, "P1", "preflop", "raise", None, ["P2"]), ("H1", 2, "P2", "preflop", "call", None, ["P1"]),
        ("H1", 10, "P1", "flop", "bet", 0.3, ["P2"]), ("H1", 11, "P2", "flop", "fold", 0.7, ["P1"]),
        ("H2", 20, "P1", "turn", "check", 0.6, ["P2"]), ("H2", 21, "P2", "turn", "bet", 0.4, ["P1"]),
        ("H2", 22, "P1", "turn", "raise", 0.6, ["P2"]), ("H2", 23, "P2", "turn", "call", 0.4, ["P1"]),
        ("H3", 30, "P3", "preflop", "raise", None, ["P1", "P2"]), ("H3", 31, "P1", "preflop", "call", None, ["P2", "P3"]),
        ("H3", 32, "P2", "preflop", "fold", None, ["P1", "P3"]),
        ("H3", 33, "P3", "flop", "bet", 0.8, ["P1"]), ("H3", 34, "P1", "flop", "call", 0.2, ["P3"]),
        ("H4", 40, "P1", "flop", "bet", 0.5, ["P2", "P3"]), ("H4", 41, "P2", "flop", "fold", 0.1, ["P1", "P3"]),
        ("H4", 42, "P3", "flop", "call", 0.4, ["P1"]), ("H4", 43, "P1", "turn", "bet", 0.6, ["P3"]),
        ("H5", 50, "P1", "preflop", "raise", 0.3, ["P2", "P3"]), ("H5", 51, "P2", "preflop", "fold", 0.2, ["P1", "P3"]),
        ("H5", 52, "P3", "preflop", "raise", 0.7, ["P1"]), ("H5", 53, "P1", "preflop", "fold", 0.3, ["P3"]),
    ]
    hu = {"H1", "H2"}
    warehouse.execute(con, """CREATE TABLE actions (hand_id VARCHAR, action_index SMALLINT, player_id VARCHAR,
        street VARCHAR, action VARCHAR, is_hu_table BOOLEAN, is_post BOOLEAN, n_live SMALLINT, hand_date DATE,
        is_allin BOOLEAN, decision_s DOUBLE, hole_cards VARCHAR, board_at_action VARCHAR, made_hand VARCHAR,
        pot_odds DOUBLE, pot_before_bb DOUBLE, to_call_bb DOUBLE, faced_size_pot DOUBLE,
        live_opponents STRUCT(player_id VARCHAR)[], situation VARCHAR, position VARCHAR, hand_class VARCHAR)""")
    for h, i, p, s, a, _, opp in acts:
        warehouse.execute(con, "INSERT INTO actions VALUES (:h, :i, :p, :s, :a, :hu, FALSE, :n, DATE '2001-01-01', "
                          "FALSE, NULL, 'AhKd', NULL, 'nothing', NULL, 1.0, 0.0, NULL, "
                          "list_transform(string_split(:opp, ','), x -> {'player_id': x}), NULL, NULL, 'AKo')",
                          h=h, i=i, p=p, s=s, a=a, hu=h in hu, n=len(opp) + 1, opp=",".join(opp))
    warehouse.execute(con, "CREATE VIEW decisions AS SELECT * FROM actions")
    warehouse.execute(con, "CREATE TABLE hands AS SELECT * FROM (VALUES "
                      "('H1', TIMESTAMP '2001-01-01 10:00', 3.0, 2), ('H2', TIMESTAMP '2001-01-01 10:01', 9.0, 2), "
                      "('H3', TIMESTAMP '2001-01-01 10:02', 9.0, 3), ('H4', TIMESTAMP '2001-01-01 10:03', 9.0, 3), "
                      "('H5', TIMESTAMP '2001-01-01 10:04', 9.0, 3)) t(hand_id, started_at, final_pot_bb, n_dealt)")
    players = [("H1", "P1", True), ("H1", "P2", True), ("H2", "P1", True), ("H2", "P2", True),
               ("H3", "P1", True), ("H3", "P2", False), ("H3", "P3", True),
               ("H4", "P1", True), ("H4", "P2", True), ("H4", "P3", True),
               ("H5", "P1", False), ("H5", "P2", False), ("H5", "P3", False)]
    warehouse.insert_df(con, "hand_players", pd.DataFrame(
        [(h, p, "AhKd" if p == "P1" else "7c2d", False, False, f) for h, p, f in players],
        columns=["hand_id", "player_id", "hole_cards", "went_to_showdown", "won_at_showdown", "saw_flop"]))
    eq = pd.DataFrame([(h, i, q, True, 0.5) for h, i, _, _, _, q, _ in acts if q is not None],
                      columns=["hand_id", "action_index", "eq_vs_real", "eq_real_known", "eq_vs_random"])
    warehouse.insert_df(con, "action_equity", eq)
    post = eq[~eq["hand_id"].eq("H5")]
    warehouse.insert_df(con, "action_range_equity", post[["hand_id", "action_index"]].assign(eq_vs_range=0.5))
    warehouse.insert_df(con, "action_range_equity_pre", pd.DataFrame(
        [("H5", 52, 0.35, "unopened", "raise", "P4"), ("H5", 53, 0.6, "vs_open", "raise", "P4")],
        columns=["hand_id", "action_index", "eq_vs_range", "opp_situation", "opp_action", "prior_level"]))
    return con


def test_scored_decisions_query_on_a_tiny_warehouse():
    con = tiny_warehouse()
    d = warehouse.run_sql_file(con, SQL, hu_tables_only=False).set_index("action_index")
    hu = warehouse.run_sql_file(con, SQL, hu_tables_only=True).set_index("action_index")
    con.close()
    # flop_hu: H3's flop (two players saw it) and H5's 3-bet pot (one live opponent) join; three-way spots do not
    assert sorted(d.index) == [1, 2, 10, 11, 20, 21, 22, 23, 33, 34, 52, 53]
    assert sorted(hu.index) == [1, 2, 10, 11, 20, 21, 22, 23]                   # hu_tables: the heads-up table only
    assert d.loc[33, "opponent_id"] == "P1" and d.loc[34, "opponent_id"] == "P3"
    assert d.loc[34, "table_class"] == "3-6" and d.loc[10, "table_class"] == "hu"
    assert bool(d.loc[34, "facing"]) and d.loc[34, "bettor_q"] == 0.8
    assert bool(d.loc[53, "facing"]) and d.loc[53, "bettor_id"] == "P3" and d.loc[53, "opponent_id"] == "P3"
    assert bool(d.loc[52, "facing"]) and bool(d.loc[52, "bluff_success"])     # P1 folded to the 3-bet
    assert d.loc[52, "r"] == 0.35 and d.loc[53, "opp_situation"] == "vs_open" and pd.isna(d.loc[1, "r"])
    assert bool(d.loc[10, "bluff_success"]) and not bool(d.loc[10, "facing"])
    assert bool(d.loc[11, "facing"]) and d.loc[11, "bettor_id"] == "P1" and d.loc[11, "bettor_q"] == 0.3
    assert pd.isna(d.loc[11, "bluff_success"]) and d.loc[11, "opponent_id"] == "P1"
    assert not bool(d.loc[21, "facing"]) and not bool(d.loc[21, "bluff_success"])   # P1 raised after the bet
    assert bool(d.loc[22, "facing"]) and d.loc[22, "bettor_q"] == 0.4 and not bool(d.loc[22, "bluff_success"])
    assert d.loc[23, "bettor_id"] == "P1" and d.loc[23, "bettor_q"] == 0.6 and d.loc[23, "pot_bb"] == 9.0
    assert not bool(d.loc[20, "facing"]) and pd.isna(d.loc[20, "bettor_q"])
    same = ["player_id", "opponent_id", "street", "action", "q", "r", "facing", "bettor_q", "pot_bb"]
    pd.testing.assert_frame_equal(d.loc[hu.index, same], hu[same], check_dtype=False)


def test_player_hands_query_scope():
    con = tiny_warehouse()
    warehouse.execute(con, "ALTER TABLE hands ADD COLUMN hand_date DATE DEFAULT DATE '2001-01-01'")
    warehouse.execute(con, "ALTER TABLE hands ADD COLUMN variant VARCHAR DEFAULT 'standard'")
    warehouse.execute(con, "ALTER TABLE hands ADD COLUMN error_flag BOOLEAN DEFAULT FALSE")
    for col, typ, val in (("net_bb", "DOUBLE", "0"), ("net_pre_rake_bb", "DOUBLE", "0"), ("vpip", "BOOLEAN", "TRUE"),
                          ("pfr", "BOOLEAN", "FALSE"), ("is_bb", "BOOLEAN", "FALSE"), ("n_decisions", "INTEGER", "1")):
        warehouse.execute(con, f"ALTER TABLE hand_players ADD COLUMN {col} {typ} DEFAULT {val}")
    sql = step05.SQL_DIR / "player_hands.sql"
    h = warehouse.run_sql_file(con, sql, hu_tables_only=False).set_index(["hand_id", "player_id"])
    hu = warehouse.run_sql_file(con, sql, hu_tables_only=True)
    con.close()
    assert len(h) == 13 and set(hu["hand_id"]) == {"H1", "H2"}
    opp = h["opponent_id"]
    assert opp[("H1", "P2")] == "P1" and opp[("H3", "P1")] == "P3" and opp[("H3", "P3")] == "P1"
    assert pd.isna(opp[("H3", "P2")]) and opp.loc["H4"].isna().all() and opp.loc["H5"].isna().all()
    assert h.loc[("H3", "P1"), "table_class"] == "3-6" and h.loc[("H1", "P1"), "table_class"] == "hu"
