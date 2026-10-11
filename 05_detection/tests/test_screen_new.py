"""Step 05 additions: the table ``pair_outliers`` and the ``obs_*`` columns the screen writes, the rule set
``street_tails``, the pair summary, the flop's excess discrimination ``fl_e_disc`` and the raw preflop gaps."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from test_screen_formulas import tiny_warehouse

import step05_screen as step05
from sudetect import pairs as PA
from sudetect import screening as S
from sudetect import streets as ST
from sudetect import warehouse

# --------------------------------------------------------------------------------------------- the whole screen


def screen_warehouse():
    """The tiny warehouse (H1, H2 heads-up table; H3 three dealt, P1 and P3 see the flop; H4 three see the flop; H5
    three dealt, a 3-bet pot that ends preflop) with every column the screen's queries read, the pair and observer
    queries included."""
    con = tiny_warehouse()
    for col, typ, val in (("hand_date", "DATE", "DATE '2001-01-01'"), ("variant", "VARCHAR", "'standard'"),
                          ("error_flag", "BOOLEAN", "FALSE"), ("table_id", "VARCHAR", "'T1'"),
                          ("n_showdown", "SMALLINT", "0"), ("pot_flop_bb", "DOUBLE", "3.0"),
                          ("pot_river_bb", "DOUBLE", "NULL")):
        warehouse.execute(con, f"ALTER TABLE hands ADD COLUMN {col} {typ} DEFAULT {val}")
    for col, typ, val in (("net_bb", "DOUBLE", "0"), ("net_pre_rake_bb", "DOUBLE", "0"), ("vpip", "BOOLEAN", "TRUE"),
                          ("pfr", "BOOLEAN", "FALSE"), ("is_bb", "BOOLEAN", "FALSE"), ("n_decisions", "INTEGER", "1"),
                          ("stack_start_bb", "DOUBLE", "100"), ("saw_river", "BOOLEAN", "FALSE")):
        warehouse.execute(con, f"ALTER TABLE hand_players ADD COLUMN {col} {typ} DEFAULT {val}")
    return con


TABLES = ("actions", "hands", "hand_players", "action_equity", "action_range_equity", "action_range_equity_pre")


def column_types(con, table: str) -> dict[str, str]:
    rows = warehouse.query_df(con, "SELECT column_name, data_type FROM information_schema.columns "
                                   "WHERE table_name = :t ORDER BY ordinal_position", t=table)
    return dict(zip(rows["column_name"], rows["data_type"]))


def test_screen_writes_pair_outliers_and_the_new_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("SU_DATA_DIR", str(tmp_path))
    con = screen_warehouse()
    warehouse.execute(con, f"ATTACH '{warehouse.db_path('tiny').as_posix()}' AS f")
    for t in TABLES:
        warehouse.execute(con, f"CREATE TABLE f.{t} AS SELECT * FROM {t}")
    con.close()
    con = warehouse.connect("tiny")
    warehouse.execute(con, "CREATE VIEW decisions AS SELECT * FROM actions")
    con.close()

    step05.main(["--dataset", "tiny"])

    con = warehouse.connect("tiny", read_only=True)
    try:
        po = column_types(con, "pair_outliers")
        ps = column_types(con, "player_scores")
        pairs = warehouse.query_df(con, "SELECT * FROM pair_outliers")
        players = warehouse.query_df(con, "SELECT * FROM player_scores")
        esc = warehouse.query_df(con, "SELECT * FROM escalations WHERE rule_set = 'street_tails'")
        meta = warehouse.query_df(con, "SELECT * FROM report_meta")
    finally:
        con.close()
    assert {"A": "VARCHAR", "B": "VARCHAR", "time_window": "VARCHAR", "scope": "VARCHAR", "hands": "BIGINT",
            "bb100": "DOUBLE", "k": "BIGINT", "score": "DOUBLE", "field_rank": "DOUBLE", "n_ranked": "BIGINT",
            "top_pct": "DOUBLE", "wr_z": "DOUBLE", "fold_z": "DOUBLE", "wsd_big_n": "BIGINT"}.items() <= po.items()
    # pair hands: P1-P2 at the heads-up table (H1, H2), P1-P3 in H3, H4 (P2 folded the flop) and H5 (P1 folded last)
    got = {(a, b): n for a, b, n in zip(pairs["A"], pairs["B"], pairs["hands"])}
    assert got == {("P1", "P2"): 2, ("P2", "P1"): 2, ("P1", "P3"): 3, ("P3", "P1"): 3}
    assert set(pairs["time_window"]) == {"all"} and set(pairs["scope"]) == {"flop_hu"}
    assert {"pf_fold_gap": "DOUBLE", "pf_aggr_gap": "DOUBLE", "fl_e_disc": "DOUBLE", "pf_hit": "BOOLEAN",
            "fl_hit": "BOOLEAN", "tr_hit": "BOOLEAN", "street_hits": "BIGINT", "esc_street_tails": "BOOLEAN",
            "review_street_tails": "BOOLEAN", "victims_n": "BIGINT", "best_pair_score": "DOUBLE",
            "best_pair_rank": "DOUBLE", "best_pair_opponent": "VARCHAR", "obs_hands": "BIGINT", "obs_bb100": "DOUBLE",
            "obs_wsd_n": "BIGINT", "obs_wsd_pct": "DOUBLE", "obs_wsd_rank": "DOUBLE", "obs_wsd_of": "BIGINT",
            "obs_stackoff": "DOUBLE"}.items() <= ps.items()
    allw = players[players["time_window"] == "all"].set_index("player_id")
    other = players[players["time_window"] != "all"]
    assert allw["obs_hands"].to_dict() == {"P1": 5, "P2": 5, "P3": 3}     # every hand of the dataset
    assert other["obs_hands"].isna().all() and other["esc_street_tails"].isna().all()
    assert (allw["victims_n"] == 0).all() and not allw["esc_street_tails"].any()
    assert (players["street_hits"] == 0).all()
    assert len(esc) == 3 and set(esc["rule"]) == {"several_streets"} and not esc["escalate"].any()
    assert (esc["threshold_count"] == 1).all() and (esc["flagged"] == 0).all()
    assert meta.loc[meta["item"] == "pairs_ranked", "value"].tolist() == [0.0]    # no pair has two parts yet


def test_pair_outliers_follow_the_screen_hands():
    """Under the scope ``hu_tables`` the screen's hands are the two-player table's only, and so are the pairs."""
    con = screen_warehouse()
    rows, dec = PA.load_inputs(con)
    con.close()
    cfg = step05.load_config()
    cfg["scope"] = "hu_tables"
    meta: list = []
    t = step05.pair_outliers({"pair_rows": rows, "pair_dec": dec}, pd.DataFrame({"hand_id": ["H1", "H2"]}), cfg, meta)
    assert set(zip(t["A"], t["B"])) == {("P1", "P2"), ("P2", "P1")} and set(t["scope"]) == {"hu_tables"}
    assert meta[0][0] == "pairs_ranked"
    empty = step05.pair_outliers({}, pd.DataFrame({"hand_id": ["H1"]}), cfg, [])
    assert empty.empty and list(empty.columns) == step05.PAIR_FIRST


def test_pair_summary_names_the_victims_and_the_best_pair():
    cfg = step05.load_config()
    pairs = pd.DataFrame([
        # A, B, parts, score, rank among the 40 ranked pairs
        ("S", "V1", 5, 6.0, 1.0), ("S", "V2", 4, 3.5, 3.0), ("S", "X", 3, 1.0, 20.0),
        ("H", "Y", 2, 2.9, 4.0),                         # best pair of H: ranked, below the victim score
        ("L", "Z", 4, 3.2, 12.0),                        # score high enough, rank outside max(10, 1% of 40)
        ("U", "S", 1, 9.0, np.nan),                      # one part only: not ranked
    ], columns=["A", "B", "k", "score", "field_rank"]).assign(n_ranked=40)
    out = step05.pair_summary(pairs, cfg).set_index("player_id")
    assert out.loc["S", "victims_n"] == len(PA.victims(pairs, "S", PA.PairConfig())) == 2
    assert (out.loc["S", "best_pair_score"], out.loc["S", "best_pair_rank"], out.loc["S", "best_pair_opponent"]) == (
        6.0, 1.0, "V1")
    assert out.loc["H", "victims_n"] == 0 and out.loc["H", "best_pair_opponent"] == "Y"
    assert out.loc["L", "victims_n"] == 0 and "U" not in out.index
    assert step05.pair_summary(pairs.iloc[0:0], cfg).empty


# --------------------------------------------------------------------------------------------- street_tails


def tails_frame() -> pd.DataFrame:
    """Window ``all``: P1 hits all three streets; P2 preflop and turn + river; P3 turn + river only (his flop effect
    is just short of the bar); P4 misses every sample floor (not preflop eligible, 14 flop decisions behind, 19 dead);
    P5 has no test at all; P6 sits exactly on every bar. Window ``last15``: P1 again, preflop and flop only."""
    base = dict(time_window="all", hands=600, pf_eligible=True, pf_zr=0.0, fl_nd=50.0, fl_nw=50.0, z_b_flop=0.0,
                fl_e_disc=0.0, nd=50.0, nw=50.0, z_b=0.0, e_disc=0.0)
    rows = [
        dict(player_id="P1", pf_zr=6.0, z_b_flop=5.0, fl_e_disc=0.5, z_b=5.0, e_disc=0.6),
        dict(player_id="P2", pf_zr=4.5, z_b=3.5, e_disc=0.35),
        dict(player_id="P3", z_b_flop=5.0, fl_e_disc=0.29, z_b=3.1, e_disc=0.4),
        dict(player_id="P4", pf_eligible=False, pf_zr=9.0, fl_nd=14.0, z_b_flop=9.0, fl_e_disc=0.9, nd=19.0,
             z_b=9.0, e_disc=0.9),
        dict(player_id="P5", pf_eligible=False, pf_zr=np.nan, fl_nd=np.nan, fl_nw=np.nan, z_b_flop=np.nan,
             fl_e_disc=np.nan, nd=np.nan, nw=np.nan, z_b=np.nan, e_disc=np.nan),
        dict(player_id="P6", pf_zr=4.0, fl_nd=15.0, fl_nw=15.0, z_b_flop=3.0, fl_e_disc=0.30, nd=20.0, nw=20.0,
             z_b=3.0, e_disc=0.30),
        dict(player_id="P1", time_window="last15", hands=200, pf_zr=6.0, z_b_flop=5.0, fl_e_disc=0.5),
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_street_tails_hits_and_verdicts():
    cfg = step05.load_config()
    out = S.street_tails(tails_frame(), cfg)
    a = out[out["time_window"] == "all"].set_index("player_id")
    hits = {p: (bool(r.pf_hit), bool(r.fl_hit), bool(r.tr_hit)) for p, r in a.iterrows()}
    assert hits == {"P1": (True, True, True), "P2": (True, False, True), "P3": (False, False, True),
                    "P4": (False, False, False), "P5": (False, False, False), "P6": (True, True, True)}
    assert a["street_hits"].to_dict() == {"P1": 3, "P2": 2, "P3": 1, "P4": 0, "P5": 0, "P6": 3}
    assert a["esc_street_tails"].astype(bool).to_dict() == {"P1": True, "P2": True, "P3": False, "P4": False,
                                                            "P5": False, "P6": True}
    assert a["review_street_tails"].astype(bool).sum() == 4                       # P1, P2, P3, P6
    last = out[out["time_window"] == "last15"].iloc[0]
    assert last["street_hits"] == 2 and pd.isna(last["esc_street_tails"]) and pd.isna(last["review_street_tails"])
    # the code's defaults are the shipped configuration
    assert S.street_tails(tails_frame(), {})["street_hits"].tolist() == out["street_hits"].tolist()
    cfg["escalation"]["street_tails"]["escalate_min_hits"] = 3
    strict = S.street_tails(tails_frame(), cfg)
    assert set(strict.loc[strict["esc_street_tails"].fillna(False).astype(bool), "player_id"]) == {"P1", "P6"}


def test_street_tails_escalation_rows():
    cfg = step05.load_config()
    players = S.street_tails(tails_frame(), cfg)
    t0 = pd.Timestamp("2001-01-01 10:00")
    hands = pd.DataFrame([(p, t0 + pd.Timedelta(days=k)) for p in ("P1", "P2", "P3", "P4", "P5", "P6") for k in (0, 9)],
                         columns=["player_id", "started_at"])
    e = S.street_tails_escalations(players, hands, cfg).set_index("player_id")
    assert list(e.reset_index().columns) == S.ESCALATION_COLUMNS and len(e) == 6        # the all window only
    assert set(e["rule_set"]) == {"street_tails"} and set(e["rule"]) == {"several_streets"}
    assert e["flagged"].to_dict() == {"P1": 3, "P2": 2, "P3": 1, "P4": 0, "P5": 0, "P6": 3}
    assert (e["escalate"] == (e["flagged"] > e["threshold_count"])).all() and (e["threshold_count"] == 1).all()
    assert e.loc["P2", "ratio"] == pytest.approx(2 / 3) and e.loc["P1", "band"] == "B2"
    assert e.loc["P1", "window_start"] == t0 and e.loc["P1", "window_end"] == t0 + pd.Timedelta(days=9)
    assert math.isnan(e.loc["P1", "threshold_ratio"])


# --------------------------------------------------------------------------------------------- street tests


def flop_rows() -> pd.DataFrame:
    """One stratum (r 0.55). A folds all 4 of his flop hands behind (q < 0.20: two of them would not be dead on the
    turn's 0.02 line) and none of his 4 winning ones; B folds half of each; one middle row each (0.20 <= q < 0.5)
    and a turn row stay out of the flop test."""
    rows = []
    for pid, dead_folds, win_folds in (("A", 4, 0), ("B", 2, 2)):
        rows += [(pid, "flop", q, int(k < dead_folds)) for k, q in enumerate((0.0, 0.01, 0.15, 0.19))]
        rows += [(pid, "flop", 0.7, int(k < win_folds)) for k in range(4)]
        rows.append((pid, "flop", 0.3, 1))
    rows.append(("A", "turn", 0.0, 0))
    return pd.DataFrame(rows, columns=["player_id", "street", "q", "y"]).assign(facing=True, r=0.55)


def test_flop_excess_discrimination():
    cfg = step05.load_config()
    out, meta = ST.flop_players(flop_rows(), ["player_id"], cfg["flop"], 10)
    out = out.set_index("player_id")
    # the field folds 6 of 8 hands behind and 2 of 8 winning ones in the stratum, h = 4 x 4 / 8 for both players:
    # A: (4/4 - 0/4) - (0.75 - 0.25) = 0.5; B: (2/4 - 2/4) - 0.5 = -0.5
    assert out.loc["A", "fl_e_disc"] == pytest.approx(0.5) and out.loc["B", "fl_e_disc"] == pytest.approx(-0.5)
    assert (out.loc["A", "fl_nd"], out.loc["A", "fl_nw"]) == (4, 4) and meta["fl_responses"] == 18
    assert ST.PLAYER_COLUMNS.index("fl_e_disc") == ST.PLAYER_COLUMNS.index("z_b_flop") + 1


def preflop_rows() -> pd.DataFrame:
    """Player A facing a raise: 10 hands behind (q 0.30): 8 folds, a call, a raise; 10 ahead (q 0.70): a fold, 4 calls,
    5 raises. Unopened: 4 hands ahead, 2 raised; 2 hands in the middle (q 0.50), in neither class."""
    spec = ([(True, 0.3, a) for a in ["fold"] * 8 + ["call", "raise"]]
            + [(True, 0.7, a) for a in ["fold"] + ["call"] * 4 + ["raise"] * 5]
            + [(False, 0.7, a) for a in ("raise", "raise", "check", "check")]
            + [(False, 0.5, "raise"), (True, 0.5, "fold")])
    d = pd.DataFrame(spec, columns=["facing", "q", "action"]).assign(player_id="A", r=0.5, bluff_success=False)
    d["situation"] = np.where(d["facing"], "vs_open", "unopened")
    d["is_fold"] = d["action"] == "fold"
    d["is_aggr"] = d["action"] == "raise"
    return d


def test_preflop_raw_gaps():
    cfg = step05.load_config()
    out = ST.preflop_players(preflop_rows(), ["player_id"], cfg["preflop"])[0].set_index("player_id").loc["A"]
    assert (out["pf_fold_behind"], out["pf_fold_ahead"]) == pytest.approx((0.8, 0.1))
    assert out["pf_fold_gap"] == pytest.approx(0.7)
    assert (out["pf_aggr_behind"], out["pf_aggr_ahead"]) == pytest.approx((0.1, 0.5))    # 1 of 10, 7 of 14
    assert out["pf_aggr_gap"] == pytest.approx(0.4)
    assert {"pf_fold_gap", "pf_aggr_gap", "fl_e_disc"} <= set(ST.PLAYER_COLUMNS)
