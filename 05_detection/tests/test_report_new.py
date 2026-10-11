"""The per-suspect report's field sections and page (05_detection/run.py, superuser/html_report.py): rare-decision
selection on a synthetic pool, the tabbed page and its pair labels, the metrics table against a field, the victims
table through an id map, the street-tails line, decision times, and where the real hand sits in a range."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sudetect import pairs  # noqa: E402
from sudetect.cards import COMBOS, class_combos  # noqa: E402
from superuser.html_report import tabbed_hands_html  # noqa: E402
from superuser.parser import Action, Hand, Seat  # noqa: E402
from superuser.stats import decision_times, player_stats  # noqa: E402


def _mod(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


run05 = _mod("step05_run", ROOT / "05_detection" / "run.py")
RC = run05.load_report_config()


# --------------------------------------------------------------------------------------------- rare decisions


def synthetic_pool():
    """Preflop counts per (situation, band of r, action) and flop counts per band of r facing a bet."""
    pre = [("vs_open", 10, "fold", 20), ("vs_open", 10, "call", 50), ("vs_open", 10, "raise", 30),
           ("vs_open", 6, "fold", 80), ("vs_open", 6, "call", 15), ("vs_open", 6, "raise", 5),
           ("unopened", 12, "raise", 4), ("unopened", 12, "call", 36), ("unopened", 12, "fold", 60),
           ("unopened", 14, "call", 50), ("unopened", 14, "fold", 50),           # nobody raises here
           ("vs_3bet", 10, "fold", 2), ("vs_3bet", 10, "call", 3)]               # too thin a band
    flop = [(5, 100, 10, 90), (1, 100, 88, 12), (2, 100, 70, 30)]
    return {"pre": pd.DataFrame(pre, columns=["situation", "rbin", "action", "n"]).astype({"rbin": float}),
            "flop": pd.DataFrame(flop, columns=["rbin", "n", "n_fold", "n_cont"]).astype({"rbin": float})}


def candidates():
    rows = [  # hand, index, street, situation, action, req, to call, q, r
        ("H1", 5, "preflop", "vs_open", "fold", 0.35, 2.0, 0.30, 0.52),   # behind, the pool folds 20%: pf_fold
        ("H2", 5, "preflop", "vs_open", "fold", 0.35, 2.0, 0.30, 0.32),   # the pool folds 80% there: ordinary
        ("H3", 4, "preflop", "unopened", "raise", None, 0.5, 0.60, 0.62),  # the pool raises 4%: pf_raise
        ("H3", 6, "preflop", "vs_open", "call", 0.30, 2.0, 0.70, 0.33),   # the pool calls 15%, less rare than H3/4
        ("H4", 5, "preflop", "vs_3bet", "fold", 0.35, 6.0, 0.30, 0.52),   # band of 5 decisions: no pool rate
        ("H5", 5, "preflop", "vs_open", "raise", 0.35, 2.0, 0.50, 0.52),  # q below 0.55
        ("H10", 5, "preflop", "vs_open", "fold", 0.30, 2.0, 0.38, 0.52),  # q above the pot odds
        ("H11", 4, "preflop", "unopened", "raise", None, 0.5, 0.70, 0.71),  # nobody in the pool raises there
        ("H6", 9, "flop", "vs_cbet", "fold", 0.25, 3.0, 0.10, 0.55),      # the pool folds 10%: fl_fold
        ("H7", 9, "flop", "vs_cbet", "call", 0.25, 3.0, 0.70, 0.15),      # the pool continues 12%: fl_call
        ("H8", 9, "flop", "vs_cbet", "fold", 0.25, 3.0, 0.30, 0.55),      # q above 0.20
        ("H9", 9, "flop", "vs_cbet", "call", 0.25, 3.0, 0.80, 0.25),      # the pool continues 30%: not under 30%
    ]
    df = pd.DataFrame(rows, columns=["hand_id", "action_index", "street", "situation", "action", "req",
                                     "to_call_bb", "q", "r"])
    return df.assign(hole_cards="AhKd", q_random=0.5, opp_id="V", opp_hole_cards="7c2d")


def test_rare_selection_on_a_synthetic_pool():
    sel = run05.select_rare(candidates(), synthetic_pool(), RC["rare"], top_n=2)
    assert list(zip(sel["hand_id"], sel["rule"])) == [("H11", "pf_raise"), ("H3", "pf_raise"), ("H6", "fl_fold"),
                                                      ("H7", "fl_call")]
    s = sel.set_index("hand_id")
    assert s.loc["H11", "p_pool"] == 0.0 and s.loc["H11", "score"] == pytest.approx(-math.log(0.005))
    assert s.loc["H3", "action_index"] == 4 and s.loc["H3", "score"] == pytest.approx(-math.log(0.04))
    assert s.loc["H3", "n_pool"] == 100 and s.loc["H3", "rbin"] == 12
    assert s["tier"].to_dict() == {"H11": "Perfect", "H3": "Perfect", "H6": "Good", "H7": "Good"}
    everything = run05.select_rare(candidates(), synthetic_pool(), RC["rare"], top_n=25)
    assert set(everything["hand_id"]) == {"H11", "H3", "H1", "H6", "H7"}
    assert everything.set_index("hand_id").loc["H1", "tier"] == "Average"      # the pool folds 20%: not rare enough
    thin = run05.select_rare(candidates(), synthetic_pool(), {**RC["rare"], "min_pool": 3}, top_n=25)
    assert "H4" in set(thin["hand_id"])                                         # a 5-decision band counts at 3
    none = run05.select_rare(candidates(), {"pre": None, "flop": None}, RC["rare"], top_n=25)
    assert none.empty and list(none.columns) == run05.RARE_COLUMNS


# --------------------------------------------------------------------------------------------- the page


def three_handed_hand(gid: str, street_acts: dict) -> Hand:
    seats = {1: Seat(1, 1, 10000, "BTN", [48, 44]), 2: Seat(2, 2, 10000, "SB", [20, 1]),
             3: Seat(3, 3, 10000, "BB", [8, 13])}
    acts = {st: street_acts.get(st, []) for st in ("preflop", "flop", "turn", "river")}
    return Hand(hand_uid=gid, start_ts=978307200, end_ts=978307200, table_no=1, stake_level=None, chip_type=None,
                blind=100, ante=0, game_type="No-Limit Hold'em", seats=seats, actions=acts, board=[0, 5, 10, 15, 22],
                street_end_pot={"preflop": 600}, settlement={1: 300.0, 2: -300.0, 3: 0.0})


def page_row(gid: str, street: str, opponent: str, category: str) -> dict:
    return {"hand_uid": gid, "category": category, "category_name": f"rule {category}",
            "decision_label": "Perfect fold", "street": street, "key_action": "folded", "decision_time_s": 2.0,
            "snap_decision": False, "equity_at_decision": 0.0, "opponent_uid": opponent, "player_hand": "Pair",
            "opponent_hand": "Two Pair", "pot_bb": 6.0, "net_win": 300.0}


def test_tabbed_page_postflop_default_and_labels_only_for_victims():
    """Three players: SU (the suspect), V (one of his victims) and X (not). The page prints the data's ids, opens on
    Postflop without the script, and names the pair next to V only."""
    pre = [Action("preflop", 1, 1, 1, 5, 300, 1.0, True), Action("preflop", 2, 2, 2, 3, 300, 2.0),
           Action("preflop", 3, 3, 3, 3, 200, 3.0)]
    hands = {"H1": three_handed_hand("H1", {"preflop": pre, "flop": [Action("flop", 4, 1, 1, 2, 0, 1.0)]}),
             "H2": three_handed_hand("H2", {"preflop": pre})}
    result = {"player": 1, "player_id": "SU", "opponent": None, "uid_of": {"SU": 1, "V": 2, "X": 3},
              "meta": {"hands_analysed": 2}, "escalation": {"hands": 0}, "hands": hands}
    post = pd.DataFrame([page_row("H1", "flop", "V", "A")])
    preflop = pd.DataFrame([page_row("H2", "preflop", "X", "S")])
    tabs = [{"key": "postflop", "label": "Postflop", "rows": post, "boxes": {}, "intro": "flop rules"},
            {"key": "preflop", "label": "Preflop", "rows": preflop, "boxes": {}, "intro": "preflop rules",
             "anchor": "pf", "title": "PREFLOP NO."}]
    label = "pair #1 of 50 (top 2.0%)"
    page = tabbed_hands_html(result, tabs, counted=post, labels={"V": label})
    assert '<body data-tab="postflop">' in page
    assert 'data-show="postflop" aria-selected="true">Postflop (1)' in page
    assert 'data-show="preflop" aria-selected="false">Preflop (1)' in page and 'data-show="all"' in page
    assert "All (2)" in page and 'body[data-tab="postflop"] [data-tab="preflop"]' in page   # hidden without script
    assert '<div class="hand" id="h1" data-tab="postflop">' in page
    assert '<div class="hand" id="pf1" data-tab="preflop"><h2>PREFLOP NO.1' in page
    assert "<script src" not in page and page.count("<script>") == 1
    assert f'player:V <span class="tag pair">{label}</span>' in page
    assert f'<b>Opponent:</b> V <span class="tag pair">{label}</span>' in page
    assert page.count('class="tag pair"') == 3                    # V's settlement lines and H1's decision box
    assert "<b>Opponent:</b> X</div>" in page and 'player:X <span class="tag pair">' not in page
    assert "player:SU" in page and "player:2<" not in page        # the data's ids, not the module's numbers


# --------------------------------------------------------------------------------------------- field sections


def field_scores() -> pd.DataFrame:
    """Six players: SU, four with the full preflop sample, T with a high pf_z on too few decisions."""
    ids = ["SU", "A", "B", "C", "D", "T"]
    return pd.DataFrame({
        "player_id": ids, "time_window": "all", "scope": "flop_hu",
        "pf_z": [5.0, 1.0, 2.0, 5.0, 3.0, 9.0], "pf_nd": [40, 40, 40, 40, 40, 5], "pf_nw": [40] * 5 + [5],
        "pf_aggr_z": [1.0] * 6, "pf_fold_cmh_z": [2.0] * 6,
        "pf_fold_behind": [0.9, 0.5, 0.5, 0.5, 0.5, 0.5], "pf_fold_ahead": [0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
        "pf_aggr_ahead": [0.4] * 6, "pf_aggr_behind": [0.1] * 6,
        "pf_rr_light_n": [12, 30, 2, 15, 10, 0], "pf_rr_light_succ": [0.9, 0.5, 1.0, 0.8, 0.95, np.nan],
        "z_b_flop": [3.0, 1.0, 0.5, 2.0, 0.0, 1.0], "fl_nd": [25] * 6, "fl_nw": [25] * 6,
        "e_disc": [0.5, 0.1, 0.2, 0.0, 0.1, 0.9], "nd": [30, 30, 30, 30, 30, 10], "nw": [30] * 6,
        "x_dead": [0.3] * 6, "x_win": [0.1] * 6, "z_b": [4.0, 1.0, 1.0, 1.0, 1.0, 9.0]})


def test_metrics_table_ranks_the_eligible_field():
    shown, raw, notes = run05.metrics_table(field_scores(), "SU", RC["metrics"], {})
    t = shown.set_index("Metric")
    r = raw.set_index("column")
    # pf_z: T is out (5 decisions); SU ties C at 5.0 among five players: (3 below + half of 2 equal) / 5
    assert t.loc["pf_z", "Players ranked"] == "5" and t.loc["pf_z", "Percentile"] == "80.0"
    assert t.loc["pf_z", "Rank"] == "#1" and t.loc["pf_z", "Best other player"] == "5.00"
    assert r.loc["pf_z", "best_other_id"] == "C"
    assert t.loc["Fold gap (raw)", "Value"] == "80.0 pp" and t.loc["Fold gap (raw)", "Rank"] == "#1"   # from the rates
    light = t.loc["Light re-raise success"]
    assert light["Players ranked"] == "4 (>= 10 light re-raises)" and light["Value"] == "90.0% (12)"
    assert light["Rank"] == "#2" and light["Best other player"] == "95.0% (10)"
    assert t.loc["Flop e_disc", "Value"] == "–" and t.loc["Flop e_disc", "Players ranked"] == "–"
    assert any("fl_e_disc" in n for n in notes)                    # an older screen: a note, no crash
    assert t.loc["e_disc", "Players ranked"] == "5" and t.loc["z_b", "Best other player"] == "1.00"
    assert list(shown.columns) == ["Street", "Metric", "What it measures", "Players ranked", "Value", "Percentile",
                                   "Rank", "Best other player"]
    small = field_scores().assign(pf_nd=lambda d: d["pf_nd"].where(d["player_id"] != "SU", 3))
    t2 = run05.metrics_table(small, "SU", RC["metrics"], {})[0].set_index("Metric")
    assert t2.loc["pf_z", "Rank"] == "–" and t2.loc["pf_z", "Value"].endswith("(sample below the minimum)")
    assert run05.metrics_table(None, "SU", RC["metrics"], {})[0].empty


def test_victims_through_the_id_map(tmp_path):
    rows = [  # A, B, score, field rank
        ("R1", "R2", 6.0, 1), ("R1", "R3", 4.0, 3), ("R1", "R4", 2.5, 5), ("R1", "R5", 5.0, 50), ("R9", "R1", 7.0, 2)]
    fp = pd.DataFrame(rows, columns=["A", "B", "score", "field_rank"]).assign(
        k=3, n_ranked=400, hands=500, bb100=50.0, sd_won=0.6, sd_n=40, wr_z=1.0, wsd_big=0.7, wsd_big_n=20,
        wsd_big_z=2.0, stackoff=0.0, stackoff_n=12, stackoff_z=1.5, net_big=120.0, net_big_n=20, net_big_z=2.2)
    ids_csv = tmp_path / "ids.csv"
    ids_csv.write_text("case_id,ref_id\nSUSPECT,R1\nVICTIM,R2\n", encoding="utf-8")
    ids = run05.read_id_map(ids_csv)
    back = {v: k for k, v in ids.items()}
    raw, parts, pot, best = run05.victims_tables(fp, ids["SUSPECT"], pairs.PairConfig(), back)
    # score >= 3 and a rank inside the top 10 of 400: R2 (mapped back to this dataset's id) and R3 (F's id)
    assert raw["opponent"].tolist() == ["VICTIM", "R3"]
    assert raw["label"].tolist() == [pairs.pair_label({"field_rank": k, "n_ranked": 400}) for k in (1, 3)]
    assert parts["Pair"].tolist() == raw["label"].tolist() and parts["Showdowns won"].iloc[0] == "60% of 40"
    assert pot["z W$SD"].iloc[0] == "2.00" and pot["River bets won"].iloc[0] == "–"     # no such column: a dash
    assert best["opponent"] == "VICTIM" and best["field_rank"] == 1
    assert run05.victims_tables(None, "R1", pairs.PairConfig(), back)[3] is None


def test_street_tails_line_and_old_tables():
    assert run05.street_tails(None, {}) == (None, {})
    assert run05.street_tails({"pf_z": 1.0}, {}) == (None, {})                      # a screen without the rule
    row = {"pf_hit": True, "fl_hit": False, "tr_hit": True, "street_hits": 2, "esc_street_tails": True,
           "review_street_tails": True, "pf_zr": 5.2, "z_b_flop": 1.0, "fl_e_disc": 0.1, "z_b": 4.4, "e_disc": 0.38}
    text, info = run05.street_tails(row, {})
    assert "2 of 3 streets: **ESCALATE for manual review**" in text and "preflop HIT (pf_zr 5.20)" in text
    assert info == {"street_hits": 2, "escalate": True, "review": True, "pf_hit": True, "fl_hit": False,
                    "tr_hit": True}


def test_scorecard_and_observer_degrade_without_data():
    s = field_scores().assign(ofr_turn=[0.9, 0.5, 0.6, 0.7, 0.8, 0.4], ofr_turn__n=[30, 30, 30, 30, 30, 5],
                              ofr_turn__k=[27, 15, 18, 21, 24, 2], hands=[700, 900, 800, 1000, 600, 100],
                              bb100=[100.0, 50.0, -20.0, 10.0, 5.0, 0.0])
    pair_rows = pd.DataFrame({"player_id": ["A"], "opponent_id": ["SU"], "ofr_turn__k": [10], "ofr_turn__n": [10]})
    card, notes = run05.scorecard_table(s, pair_rows, "SU", "A", {"min_n": 20, "top_winners": 2})
    row = card.set_index("Signal").loc["Oracle-fold ratio: turn"]
    assert row["Suspect"] == "90.0%" and row["Field percentile"] == "90th" and row["Field median"] == "70.0%"
    assert row["Top 2 winners (mean)"] == "60.0%"                 # A and C, the biggest net winners but SU
    assert row["Main opponent vs the rest"] == "25.0%"            # (15 - 10) / (30 - 10)
    assert card.set_index("Signal").loc["Bluff success", "Suspect"] == "–"
    assert run05.scorecard_table(None, None, "SU", None, {"min_n": 20, "top_winners": 10})[0].empty
    oc = run05.potsize.from_config(None)
    assert run05.observer_table(None, "SU", oc)[0].empty
    t, line, notes = run05.observer_table(pd.DataFrame({"obs_bb100": [12.0]}, index=["X"]), "SU", oc)
    assert t.empty and any("SU" in n for n in notes)


# --------------------------------------------------------------------------------------------- package pieces


def test_decision_times_by_action_type():
    seats = {1: Seat(1, 7, 10000, "BTN", [48, 44]), 2: Seat(2, 8, 10000, "BB", [20, 1])}
    acts = {"preflop": [Action("preflop", 1, 1, 7, 5, 300, 1.0, True), Action("preflop", 2, 2, 8, 3, 200, 9.0)],
            "flop": [Action("flop", 3, 2, 8, 1, 0, 2.0), Action("flop", 4, 1, 7, 4, 300, 3.0, True),
                     Action("flop", 5, 2, 8, 2, 0, 4.0)],
            "turn": [], "river": []}
    hand = Hand(hand_uid="H", start_ts=0, end_ts=0, table_no=1, stake_level=None, chip_type=None, blind=100,
                ante=0, game_type="No-Limit Hold'em", seats=seats, actions=acts, board=[0, 5, 10],
                street_end_pot={}, settlement={7: 300.0, 8: -300.0})
    nan_hand = Hand(**{**hand.__dict__, "hand_uid": "H2",
                       "actions": {"preflop": [Action("preflop", 1, 1, 7, 5, 300, float("nan"), True)], "flop": [],
                                   "turn": [], "river": []}})
    rows = {r["action"]: r for r in decision_times([hand, nan_hand], 7)}
    assert set(rows) == {"bet", "raise", "all"}
    assert rows["raise"]["n"] == 1 and rows["bet"]["max_s"] == 3.0                 # the NaN time is left out
    assert (rows["all"]["n"], rows["all"]["mean_s"], rows["all"]["median_s"], rows["all"]["min_s"]) == (2, 2.0, 2.0, 1.0)
    st = player_stats([hand, nan_hand], 7)
    assert st["median_think_time"] == 2.0 and st["mean_think_time"] == 2.0
    assert (st["min_think_time"], st["max_think_time"]) == (1.0, 3.0)
    assert decision_times([hand], 99) == []


def test_actual_hand_in_the_range():
    """A range of AA (3/4) and KK (1/4): where KK and a hand outside it sit."""
    aa, kk = class_combos("AA"), class_combos("KK")
    ids = np.concatenate([aa, kk])
    w = np.concatenate([np.full(len(aa), 0.75 / len(aa)), np.full(len(kk), 0.25 / len(kk))])
    view = run05._range_view(ids, w, tuple(int(c) for c in COMBOS[kk[0]]))
    assert view["actual_class"] == "KK" and view["actual_share"] == pytest.approx(0.25) and view["actual_rank"] == 2
    seven_two = tuple(int(c) for c in COMBOS[class_combos("72o")[0]])
    out = run05._range_view(ids, w, seven_two)
    assert out["actual_class"] == "72o" and out["actual_share"] == 0.0 and out["actual_rank"] == 3
    assert view["top_classes"][0] == ("AA", pytest.approx(0.75)) and "actual_class" not in run05._range_view(ids, w)


def test_field_size_can_be_kept_out_of_a_published_report():
    """show_field_size = false: no "Players ranked", no "of N" in pair labels, no field sizes in the CSV rows."""
    shown, raw, _ = run05.metrics_table(field_scores(), "SU", RC["metrics"], {}, with_n=False)
    assert "Players ranked" not in shown.columns and "players_ranked" not in raw.columns
    assert "Rank" not in shown.columns and shown.set_index("Metric").loc["pf_z", "Percentile"] == "80.0"
    fp = pd.DataFrame([("R1", "R2", 6.0, 1)], columns=["A", "B", "score", "field_rank"]).assign(
        k=3, n_ranked=400, hands=500, bb100=50.0, sd_won=0.6, sd_n=40)
    v, parts, _, best = run05.victims_tables(fp, "R1", pairs.PairConfig(), {}, with_n=False)
    assert v["label"].tolist() == ["top 0.2% of the field's pairs"] and "n_ranked" not in v.columns
    assert "400" not in parts.to_string() and "of 400" not in best["label"]
    assert "400" not in run05.victims_intro(pairs.PairConfig(), best, "the field", with_n=False)
    assert RC["show_field_size"] is True                            # the default keeps the size for an operator
