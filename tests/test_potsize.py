"""Observer metrics (sudetect/potsize.py) on a four-hand warehouse whose answers are worked out by hand."""
import math

import pandas as pd
import pytest

from sudetect import potsize, warehouse


def observer_warehouse():
    """H1: P1 bets the river, P2 calls, P1 wins a 60bb pot. H2: P2 bets the river, P1 calls and loses his whole stack
    in a 200bb pot. H3: P1 checks and folds the flop to P3, no showdown. H4: P2 and P3 split a 20bb pot. H5: a bomb
    pot P1 wins big; it must not count anywhere."""
    con = warehouse.connect(":memory:")
    warehouse.insert_df(con, "hands", pd.DataFrame([
        # hand, table, final pot, pot at the flop, pot at the river, players at the showdown
        ("H1", "T1", 60.0, 10.0, 40.0, 2), ("H2", "T1", 200.0, 20.0, 40.0, 2),
        ("H3", "T2", 10.0, 5.0, None, 0), ("H4", "T2", 20.0, 10.0, 20.0, 2), ("H5", "T3", 500.0, 50.0, 100.0, 2),
    ], columns=["hand_id", "table_id", "final_pot_bb", "pot_flop_bb", "pot_river_bb", "n_showdown"]).assign(
        hand_date=pd.Timestamp("2001-01-01").date(), went_to_showdown=lambda d: d["n_showdown"] > 0,
        variant=lambda d: d["hand_id"].map(lambda h: "bomb" if h == "H5" else "standard"), error_flag=False))
    warehouse.insert_df(con, "hand_players", pd.DataFrame([
        # hand, player, net, starting stack, saw the flop, saw the river, went to showdown, won at showdown
        ("H1", "P1", 30.0, 100.0, True, True, True, True), ("H1", "P2", -30.0, 100.0, True, True, True, False),
        ("H2", "P1", -100.0, 100.0, True, True, True, False), ("H2", "P2", 100.0, 150.0, True, True, True, True),
        ("H3", "P1", -5.0, 100.0, True, False, False, False), ("H3", "P3", 5.0, 100.0, True, False, False, False),
        ("H4", "P2", 0.0, 100.0, True, True, True, True), ("H4", "P3", 0.0, 100.0, True, True, True, True),
        ("H5", "P1", 250.0, 300.0, True, True, True, True), ("H5", "P2", -250.0, 300.0, True, True, True, False),
    ], columns=["hand_id", "player_id", "net_bb", "stack_start_bb", "saw_flop", "saw_river", "went_to_showdown",
                "won_at_showdown"]).assign(eff_stack_start_bb=lambda d: d["stack_start_bb"].clip(upper=100.0)
                                           .where(d["hand_id"] != "H5", 300.0)))
    acts = [  # hand, index, player, street, action, to call, pot before, opponent
        ("H1", 1, "P1", "preflop", "raise", 0.0, 1.5, "P2"), ("H1", 2, "P2", "preflop", "call", 2.0, 4.0, "P1"),
        ("H1", 3, "P1", "river", "bet", 0.0, 40.0, "P2"), ("H1", 4, "P2", "river", "call", 10.0, 50.0, "P1"),
        ("H2", 1, "P1", "preflop", "raise", 0.0, 1.5, "P2"), ("H2", 2, "P2", "preflop", "call", 2.0, 4.0, "P1"),
        ("H2", 3, "P2", "river", "bet", 0.0, 40.0, "P1"), ("H2", 4, "P1", "river", "call", 80.0, 120.0, "P2"),
        ("H3", 1, "P3", "preflop", "raise", 0.0, 1.5, "P1"), ("H3", 2, "P1", "preflop", "call", 2.0, 4.0, "P3"),
        ("H3", 3, "P1", "flop", "check", 0.0, 5.0, "P3"), ("H3", 4, "P3", "flop", "bet", 0.0, 5.0, "P1"),
        ("H3", 5, "P1", "flop", "fold", 3.0, 8.0, "P3"),
        ("H4", 1, "P2", "preflop", "raise", 0.0, 1.5, "P3"), ("H4", 2, "P3", "preflop", "call", 2.0, 4.0, "P2"),
        ("H4", 3, "P2", "river", "check", 0.0, 20.0, "P3"), ("H4", 4, "P3", "river", "check", 0.0, 20.0, "P2"),
        ("H5", 1, "P1", "river", "bet", 0.0, 100.0, "P2"), ("H5", 2, "P2", "river", "call", 200.0, 300.0, "P1"),
    ]
    frame = pd.DataFrame(acts, columns=["hand_id", "action_index", "player_id", "street", "action", "to_call_bb",
                                        "pot_before_bb", "opp"])
    frame = frame.assign(is_post=False, is_allin=False, n_live=2, situation=None)
    warehouse.insert_df(con, "actions_src", frame)
    warehouse.execute(con, """CREATE TABLE actions AS SELECT hand_id, action_index, player_id, street, action,
        is_post, is_allin, n_live, to_call_bb, pot_before_bb, situation,
        [{'player_id': opp}]::STRUCT(player_id VARCHAR)[] AS live_opponents FROM actions_src""")
    warehouse.execute(con, "CREATE VIEW decisions AS SELECT * FROM actions WHERE NOT is_post")
    return con


def test_player_metrics_by_hand():
    con = observer_warehouse()
    m = potsize.player_metrics(con)
    con.close()
    p1, p2, p3 = m.loc["P1"], m.loc["P2"], m.loc["P3"]
    assert p1["obs_hands"] == 3 and p1["obs_bb100"] == pytest.approx(-2500.0)     # the H5 bomb pot is left out
    assert [p1[f"obs_net_b{b}"] for b in range(1, 5)] == [-5.0, 30.0, 0.0, -100.0]
    assert p1["obs_big_net100"] == pytest.approx(-10000.0 / 3)
    assert p1["obs_wtsd"] == pytest.approx(2 / 3) and p1["obs_wtsd_n"] == 3
    assert p1["obs_wtsd_srp"] == pytest.approx(0.5) and p1["obs_wtsd_3bp"] == 1.0 and math.isnan(p1["obs_wtsd_4bp"])
    assert p1["obs_wsd"] == 0.5 and p1["obs_wsd_b2"] == 1.0 and p1["obs_wsd_b4"] == 0.0 and math.isnan(p1["obs_wsd_b1"])
    assert math.isnan(p1["obs_wsd_growth"]) and p1["obs_growth_n"] == 0
    assert p1["obs_rbet_win"] == 1.0 and p1["obs_rbet_win_n"] == 1                  # H1: his river bet, 60bb pot
    assert p1["obs_rcall_win"] == 0.0 and p1["obs_rcall_freq"] == 1.0               # H2: called 80 into 120, lost
    assert p1["obs_stackoff"] == 1.0 and p1["obs_stackoff_n"] == 1                  # H2: 160bb in on the river
    # P2 covers P1 in H2 (150 against 100): his effective stack is 100, so a 100bb loss would be a stack-off for him
    assert p2["obs_wsd"] == pytest.approx(0.5) and p2["obs_wsd_b1"] == 0.5         # the H4 split counts 1/2
    assert p2["obs_wsd_growth"] == pytest.approx(0.5) and p2["obs_growth_n"] == 1
    assert p2["obs_rbet_win"] == 1.0 and p2["obs_rcall_win"] == 0.0 and p2["obs_stackoff"] == 0.0
    assert p3["obs_bb100"] == pytest.approx(250.0) and p3["obs_wtsd"] == 0.5 and p3["obs_wsd_n"] == 1


def test_field_ranks_direction_and_eligibility():
    con = observer_warehouse()
    m = potsize.field_ranks(potsize.player_metrics(con), min_hands=1, min_cases=1)
    con.close()
    assert m.loc["P2", "obs_bb100_rank"] == 1 and m.loc["P1", "obs_bb100_rank"] == 3
    assert m.loc["P2", "obs_bb100_pct"] == pytest.approx(100 * 2.5 / 3)
    # stack-offs: fewer is more suspicious, so P2 (0 of 1) ranks above P1 (1 of 1); P3 has no such hand
    assert m.loc["P2", "obs_stackoff_rank"] == 1 and m.loc["P1", "obs_stackoff_rank"] == 2
    assert "obs_wsd_b1_rank" not in m.columns and "obs_rcall_freq_rank" in m.columns   # small pots: shown, not ranked
    assert math.isnan(m.loc["P3", "obs_stackoff_rank"]) and m.loc["P1", "obs_stackoff_of"] == 2
    base = m[[c for c in m if not c.endswith(("_pct", "_rank", "_of"))]]
    strict = potsize.field_ranks(base, min_hands=4)
    assert strict["obs_bb100_rank"].isna().all() and (strict["obs_bb100_of"] == 0).all()   # nobody has 4 hands


def test_from_config_reads_the_observer_table():
    kw = potsize.from_config({"big_pot_bb": 50, "pot_edges_bb": [50, 120, 200], "min_cases": 30})
    assert kw["big_pot"] == 50.0 and kw["pot_edges"] == (50, 120, 200) and kw["min_cases"] == 30
    assert potsize.from_config(None)["flop_edges"] == potsize.FLOP_EDGES
