"""Situation vocabulary: every branch of the pure functions, then whole hands street by street."""

from __future__ import annotations

import pytest
from step01_testlib import replay_text

from sudetect import situations as sit


@pytest.mark.parametrize("raises, limps, calls_after, expected", [
    (0, 0, 0, "unopened"),
    (0, 2, 0, "vs_limp"),
    (1, 0, 0, "vs_open"),
    (1, 3, 0, "vs_open"),        # limps before the raise do not make a squeeze spot
    (1, 0, 1, "vs_open_call"),
    (2, 0, 0, "vs_3bet"),
    (2, 0, 2, "vs_3bet"),
    (3, 0, 0, "vs_4bet_plus"),
    (5, 1, 1, "vs_4bet_plus"),
])
def test_preflop_situation(raises, limps, calls_after, expected):
    assert sit.preflop_situation(raises, limps, calls_after) == expected


@pytest.mark.parametrize("is_agg, active, checked, after, expected", [
    (True, True, False, False, "cbet_opp"),
    (False, False, False, False, "lead_opp"),   # no aggressor, or he folded / is all-in
    (False, True, True, False, "probe_opp"),
    (False, True, False, True, "donk_opp"),
    (False, True, False, False, "lead_opp"),    # unreachable in play; falls back to a lead
])
def test_postflop_spot(is_agg, active, checked, after, expected):
    assert sit.postflop_spot(is_agg, active, checked, after) == expected


@pytest.mark.parametrize("n_aggr, spot, expected", [
    (1, "cbet_opp", "vs_cbet"),
    (1, "donk_opp", "vs_donk"),
    (1, "probe_opp", "vs_probe"),
    (1, "lead_opp", "vs_bet"),
    (2, "cbet_opp", "vs_raise"),
    (3, "lead_opp", "vs_raise"),
    (1, None, "vs_bet"),
])
def test_facing_situation(n_aggr, spot, expected):
    assert sit.facing_situation(n_aggr, spot) == expected


def test_line_prefix_and_pot_type():
    had_bet = [True, True, False, False]
    assert sit.line_prefix(0, had_bet) is None and sit.line_prefix(1, had_bet) is None
    assert sit.line_prefix(2, had_bet) == "b" and sit.line_prefix(3, had_bet) == "bx"
    assert sit.line_prefix(3, [True, False, False, False]) == "xx"
    assert sit.pot_type(0, 0, preflop_row=True) == "unopened"
    assert sit.pot_type(0, 1, preflop_row=True) == "limped" and sit.pot_type(0) == "limped"
    assert [sit.pot_type(k) for k in (1, 2, 3, 4)] == ["single_raised", "3bet", "4bet_plus", "4bet_plus"]
    assert sit.is_in_position(6, [1, 6]) and not sit.is_in_position(1, [1, 6])
    assert not sit.is_in_position(1, [])


FOUR_BET_POT = """
    Hand #3001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #6 is the button
    Seat 1: P1 (200.00 in chips)
    Seat 2: P2 (200.00 in chips)
    Seat 3: P3 (200.00 in chips)
    Seat 4: P4 (200.00 in chips)
    Seat 5: P5 (200.00 in chips)
    Seat 6: P6 (200.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: calls 1.00
    P4: calls 1.00
    P5: raises 4.00 to 5.00
    P6: calls 5.00
    P1: raises 15.00 to 20.00
    P2: folds
    P3: folds
    P4: folds
    P5: raises 30.00 to 50.00
    P6: folds
    P1: calls 30.00
    *** FLOP *** [2c 7d 9h]
    P1: checks
    P5: bets 20.00
    P1: raises 40.00 to 60.00
    P5: calls 40.00
    *** TURN *** [2c 7d 9h] [Ts]
    P1: bets 30.00
    P5: calls 30.00
    *** RIVER *** [2c 7d 9h Ts] [3c]
    P1: checks
    P5: bets 50.00
    P1: folds
    Uncalled bet (50.00) returned to P5
    P5 collected 288.00 from pot
    *** SUMMARY ***
    Total pot 288.00 | Rake 0.00
    Board [2c 7d 9h Ts 3c]
"""


def test_situations_through_a_four_bet_pot():
    r = replay_text(FOUR_BET_POT)
    got = [(a["player_id"], a["street"], a["situation"]) for a in r.decisions()]
    assert got == [
        ("P3", "preflop", "unopened"),
        ("P4", "preflop", "vs_limp"),
        ("P5", "preflop", "vs_limp"),
        ("P6", "preflop", "vs_open"),
        ("P1", "preflop", "vs_open_call"),
        ("P2", "preflop", "vs_3bet"),
        ("P3", "preflop", "vs_3bet"),
        ("P4", "preflop", "vs_3bet"),
        ("P5", "preflop", "vs_3bet"),
        ("P6", "preflop", "vs_4bet_plus"),
        ("P1", "preflop", "vs_4bet_plus"),
        ("P1", "flop", "donk_opp"),
        ("P5", "flop", "cbet_opp"),
        ("P1", "flop", "vs_cbet"),
        ("P5", "flop", "vs_raise"),
        ("P1", "turn", "cbet_opp"),
        ("P5", "turn", "vs_cbet"),
        ("P1", "river", "cbet_opp"),
        ("P5", "river", "probe_opp"),
        ("P1", "river", "vs_probe"),
    ]
    flop = r.decisions(street="flop")
    assert [a["pot_type_now"] for a in flop] == ["4bet_plus"] * 4
    assert [a["is_ip"] for a in flop] == [False, True, False, True]
    assert all(a["is_ip"] is None for a in r.decisions(street="preflop"))
    assert [a["line"] for a in r.decisions(street="river")] == ["bb"] * 3
    assert r.decisions("P5", "flop")[0]["preflop_aggressor_id"] == "P5"
    assert r.decisions("P1", "turn")[0]["prev_street_aggressor_id"] == "P1"
    raise_row = r.decisions("P5", "flop")[1]
    assert raise_row["faced_bettor_id"] == "P1" and raise_row["last_aggressor_id"] == "P1"
    assert raise_row["bets_before"] == 2 and raise_row["faced_size_pot"] == pytest.approx(40 / 148)
    assert [a["n_limpers"] for a in r.decisions(street="preflop")[:4]] == [0, 1, 2, 2]
    assert r.players["P5"]["pfr"] and r.players["P5"]["preflop_raises_made"] == 2
    assert r.hand["pot_type"] == "4bet_plus" and r.hand["preflop_raises"] == 3


DONK_AND_LEAD = """
    Hand #3002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: raises 2.00 to 3.00
    P1: calls 2.50
    P2: calls 2.00
    *** FLOP *** [2c 7d 9h]
    P1: bets 5.00
    P2: calls 5.00
    P3: calls 5.00
    *** TURN *** [2c 7d 9h] [Ts]
    P1: checks
    P2: checks
    P3: checks
    *** RIVER *** [2c 7d 9h Ts] [3c]
    P1: bets 9.00
    P2: folds
    P3: folds
    Uncalled bet (9.00) returned to P1
    P1 collected 24.00 from pot
    *** SUMMARY ***
    Total pot 24.00 | Rake 0.00
    Board [2c 7d 9h Ts 3c]
"""


def test_donk_probe_and_lead_spots_multiway():
    r = replay_text(DONK_AND_LEAD)
    got = [(a["player_id"], a["street"], a["situation"]) for a in r.decisions() if a["street"] != "preflop"]
    assert got == [
        ("P1", "flop", "donk_opp"),
        ("P2", "flop", "vs_donk"),
        ("P3", "flop", "vs_donk"),
        ("P1", "turn", "cbet_opp"),
        ("P2", "turn", "probe_opp"),
        ("P3", "turn", "probe_opp"),
        ("P1", "river", "lead_opp"),
        ("P2", "river", "vs_bet"),
        ("P3", "river", "vs_bet"),
    ]
    assert [a["line"] for a in r.decisions(street="river")] == ["bx"] * 3
    assert [a["n_behind"] for a in r.decisions(street="flop")] == [2, 1, 0]
    assert [a["n_live_start_street"] for a in r.decisions(street="flop")] == [3, 3, 3]
    assert r.decisions("P3", "flop")[0]["is_ip"] and not r.decisions("P1", "flop")[0]["is_ip"]


AGGRESSOR_ALL_IN = """
    Hand #3003: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (10.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: raises 9.00 to 10.00 and is all-in
    P1: calls 9.50
    P2: calls 9.00
    *** FLOP *** [2c 7d 9h]
    P1: bets 10.00
    P2: folds
    Uncalled bet (10.00) returned to P1
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    P1: shows [Ah Kd] (high card Ace)
    P3: shows [Qs Qc] (a pair of Queens)
    P3 collected 30.00 from pot
    *** SUMMARY ***
    Total pot 30.00 | Rake 0.00
    Board [2c 7d 9h Ts 3c]
"""


def test_all_in_aggressor_leaves_a_lead_spot_and_still_counts_as_live():
    r = replay_text(AGGRESSOR_ALL_IN)
    lead = r.decisions("P1", "flop")[0]
    assert lead["situation"] == "lead_opp" and lead["is_ip"] is False
    assert lead["n_live"] == 3 and lead["n_live_start_street"] == 3
    opp = {o["player_id"]: o for o in lead["live_opponents"]}
    assert opp["P3"]["is_allin"] and opp["P3"]["behind_bb"] == 0
    assert opp["P2"]["behind_bb"] == pytest.approx(90.0) and opp["P2"]["street_put_bb"] == 0
    assert r.decisions("P2", "flop")[0]["situation"] == "vs_bet"
    assert r.decisions("P2", "flop")[0]["is_ip"] is True  # last of the players able to act on the flop
    assert r.hand["players_flop"] == 3 and r.hand["players_turn"] == 2 and r.hand["players_river"] == 2
    assert r.players["P3"]["went_to_showdown"] and r.players["P3"]["won_at_showdown"]
    assert r.players["P3"]["allin_street"] == "preflop" and r.players["P2"]["folded_street"] == "flop"
