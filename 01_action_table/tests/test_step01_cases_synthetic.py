"""Synthetic cases A-M of the action-table conventions, written as text hands (big blind = 1.00).

Each case pins the numbers a reader can check by hand: effective sizes, pot before the row,
amount to call, pot odds, uncalled excess and the hand-level results.
"""

from __future__ import annotations

import pytest
from step01_testlib import gate, replay_text

approx = pytest.approx

CASE_A = """
    Hand #1001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: H (100.00 in chips)
    Seat 2: V (40.00 in chips)
    H: posts small blind 0.50
    V: posts big blind 1.00
    *** HOLE CARDS ***
    Dealt to H [Ah Kd]
    Dealt to V [Qs Qc]
    H: raises 2.00 to 3.00
    V: calls 2.00
    *** FLOP *** [2c 7d 9h]
    V: checks
    H: bets 2.00
    V: calls 2.00
    *** TURN *** [2c 7d 9h] [Ts]
    V: checks
    H: bets 95.00 and is all-in
    V: calls 35.00 and is all-in
    Uncalled bet (60.00) returned to H
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    H: shows [Ah Kd] (high card Ace)
    V: shows [Qs Qc] (a pair of Queens)
    V collected 80.00 from pot
    *** SUMMARY ***
    Total pot 80.00 | Rake 0.00
    Board [2c 7d 9h Ts 3c]
    Seat 1: H (button) (small blind) showed [Ah Kd] and lost with high card Ace
    Seat 2: V (big blind) showed [Qs Qc] and won (80.00) with a pair of Queens
"""


def test_case_a_heads_up_shove_is_capped_at_the_shorter_stack():
    r = replay_text(CASE_A)
    shove = r.row("H", "turn")
    assert r.row("V", "turn")["pot_before_bb"] == approx(10.0)
    assert shove["amount_bb"] == approx(95.0)
    assert shove["amount_eff_bb"] == approx(35.0)
    assert shove["size_pot"] == approx(3.5)  # not 9.5
    assert shove["is_capped"] and shove["is_allin"] and shove["action"] == "bet"
    call = r.row("V", "turn", 1)
    assert call["to_call_bb"] == approx(35.0) and call["to_call_eff_bb"] == approx(35.0)
    assert call["pot_odds"] == approx(35 / 80)
    assert call["faced_size_pot"] == approx(3.5)
    assert r.hand["final_pot_bb"] == approx(80.0)
    assert r.hand["uncalled_bb"] == approx(60.0) and r.hand["uncalled_seat"] == 1
    assert r.players["H"]["net_bb"] == approx(-40.0) and r.players["V"]["net_bb"] == approx(40.0)
    assert r.hand["pot_flop_bb"] == approx(6.0) and r.hand["pot_turn_bb"] == approx(10.0)
    assert r.hand["pot_river_bb"] == approx(80.0)


CASE_B = """
    Hand #1002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:01:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: S (60.00 in chips)
    Seat 2: B (100.00 in chips)
    Seat 3: O (200.00 in chips)
    Seat 4: C (100.00 in chips)
    S: posts small blind 0.50
    B: posts big blind 1.00
    *** HOLE CARDS ***
    O: raises 2.00 to 3.00
    C: calls 3.00
    S: raises 17.00 to 20.00
    B: folds
    O: raises 180.00 to 200.00 and is all-in
    C: folds
    S: calls 40.00 and is all-in
    Uncalled bet (140.00) returned to O
    *** FLOP *** [2s 3s 4h]
    *** TURN *** [2s 3s 4h] [9c]
    *** RIVER *** [2s 3s 4h 9c] [Jd]
    *** SHOW DOWN ***
    S: shows [Kc Kd] (a pair of Kings)
    O: shows [Ac Ad] (a pair of Aces)
    O collected 124.00 from pot
    *** SUMMARY ***
    Total pot 124.00 | Rake 0.00
    Board [2s 3s 4h 9c Jd]
"""


def test_case_b_cap_set_by_a_deeper_player_who_then_folds():
    r = replay_text(CASE_B)
    shove = r.decisions("O")[1]
    assert shove["to_call_bb"] == approx(17.0) and shove["pot_before_bb"] == approx(27.0)
    assert shove["amount_eff_bb"] == approx(97.0) and shove["size_pot"] == approx(80 / 44)
    fold = r.decisions("C")[1]
    assert fold["to_call_bb"] == approx(97.0) and fold["pot_before_bb"] == approx(124.0)
    assert fold["pot_odds"] == approx(97 / 221)
    call = r.decisions("S")[1]
    assert call["to_call_bb"] == approx(80.0) and call["to_call_eff_bb"] == approx(40.0)
    assert call["pot_odds"] == approx(40 / 124)  # the naive 40/164 overstates the pot
    assert r.hand["final_pot_bb"] == approx(124.0) and r.hand["uncalled_bb"] == approx(140.0)
    assert r.hand["n_side_pots"] == 0


CASE_C = """
    Hand #1003: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:02:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: A (30.00 in chips)
    Seat 2: B (500.00 in chips)
    A: posts small blind 0.50
    B: posts big blind 1.00
    *** HOLE CARDS ***
    A: raises 29.00 to 30.00 and is all-in
    B: raises 470.00 to 500.00 and is all-in
    Uncalled bet (470.00) returned to B
    *** FLOP *** [2s 3s 4h]
    *** TURN *** [2s 3s 4h] [9c]
    *** RIVER *** [2s 3s 4h 9c] [Jd]
    *** SHOW DOWN ***
    A: shows [As Ah] (a pair of Aces)
    B: shows [Kc Kh] (a pair of Kings)
    A collected 60.00 from pot
    *** SUMMARY ***
    Total pot 60.00 | Rake 0.00
"""


def test_case_c_all_in_over_a_covered_all_in_is_a_call():
    r = replay_text(CASE_C)
    row = r.decisions("B")[0]
    assert row["action"] == "call" and row["action_raw"] == "raise"
    assert row["amount_eff_bb"] == approx(29.0) and row["size_pot"] is None
    assert row["is_allin"] and row["is_capped"]
    assert r.hand["final_pot_bb"] == approx(60.0) and r.hand["uncalled_bb"] == approx(470.0)
    assert r.players["B"]["pfr"] is False and r.players["B"]["vpip"] is True
    assert r.hand["preflop_raises"] == 1


CASE_D = """
    Hand #1004: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:03:00 UTC
    Table 'T01' 6-max Seat #6 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    Seat 4: P4 (100.00 in chips)
    Seat 5: P5 (100.00 in chips)
    Seat 6: P6 (100.00 in chips)
    P1: posts the ante 0.10
    P2: posts the ante 0.10
    P3: posts the ante 0.10
    P4: posts the ante 0.10
    P5: posts the ante 0.10
    P6: posts the ante 0.10
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: raises 1.50 to 2.50
    P4: folds
    P5: folds
    P6: folds
    P1: folds
    P2: folds
    Uncalled bet (1.50) returned to P3
    P3 collected 3.10 from pot
    *** SUMMARY ***
    Total pot 3.10 | Rake 0.00
"""


def test_case_d_antes_are_dead_money():
    r = replay_text(CASE_D)
    open_ = r.decisions("P3")[0]
    assert open_["pot_before_bb"] == approx(2.1) and open_["to_call_bb"] == approx(1.0)
    assert open_["size_pot"] == approx(1.5 / 3.1) and open_["size_x"] == approx(2.5)
    assert open_["situation"] == "unopened"
    assert r.hand["ante_bb"] == approx(0.1)
    assert r.players["P3"]["ante_bb"] == approx(0.1) and r.players["P3"]["net_bb"] == approx(0.5 + 1.0 + 0.5)
    posts = [a for a in r.actions if a["is_post"]]
    assert [a["post_kind"] for a in posts] == ["ante"] * 6 + ["sb", "bb"]


CASE_E = """
    Hand #1005: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:04:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: calls 0.50
    Y: checks
    *** FLOP *** [2c 7d 9h]
    Y: folds
    X collected 2.00 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
    Board [2c 7d 9h]
"""


def test_case_e_fold_with_nothing_to_call_is_a_free_fold():
    r = replay_text(CASE_E)
    fold = r.decisions("Y", "flop")[0]
    assert fold["free_fold"] and fold["to_call_bb"] == 0 and fold["is_decision"]
    assert r.decisions("Y", "preflop")[0]["free_fold"] is False


CASE_F = """
    Hand #1006: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:05:00 UTC
    Table 'T01' 6-max Seat #6 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    Seat 4: P4 (100.00 in chips)
    Seat 5: P5 (100.00 in chips)
    Seat 6: P6 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    P3: posts straddle 2.00
    *** HOLE CARDS ***
    P4: raises 4.00 to 6.00
    P5: folds
    P6: folds
    P1: folds
    P2: folds
    P3: folds
    Uncalled bet (4.00) returned to P4
    P4 collected 5.50 from pot
    *** SUMMARY ***
    Total pot 5.50 | Rake 0.00
"""


def test_case_f_raise_over_a_straddle_is_an_open():
    r = replay_text(CASE_F)
    open_ = r.decisions("P4")[0]
    assert open_["situation"] == "unopened" and open_["pot_type_now"] == "unopened"
    assert open_["to_call_bb"] == approx(2.0)
    assert open_["size_pot"] == approx(4 / 5.5) and open_["size_x"] == approx(3.0)
    assert open_["raise_to_eff_bb"] == approx(6.0)
    assert r.hand["straddle_bb"] == approx(2.0) and r.hand["n_straddles"] == 1
    assert r.players["P3"]["is_straddle"] and r.players["P4"]["preflop_order"] == 1
    assert r.players["P3"]["preflop_order"] == 6
    assert "STRADDLE" in r.hand["error_detail"] and not r.hand["error_flag"]


CASE_G = """
    Hand #1007: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:06:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: calls 1.00
    P1: calls 0.50
    P2: checks
    *** FLOP *** [2c 7d 9h]
    P1: checks
    P2: checks
    P3: checks
    *** TURN *** [2c 7d 9h] [Ts]
    P1: bets 1.00
    P2: folds
    P3: folds
    Uncalled bet (1.00) returned to P1
    P1 collected 3.00 from pot
    *** SUMMARY ***
    Total pot 3.00 | Rake 0.00
    Board [2c 7d 9h Ts]
"""


def test_case_g_big_blind_option_is_not_vpip():
    r = replay_text(CASE_G)
    option = r.decisions("P2", "preflop")[0]
    assert option["action"] == "check" and option["to_call_bb"] == 0
    assert option["situation"] == "vs_limp" and option["n_limpers"] == 2
    assert r.players["P2"]["vpip"] is False and r.players["P3"]["vpip"] and r.players["P1"]["vpip"]
    assert r.hand["pot_type"] == "limped"
    turn_bet = r.decisions("P1", "turn")[0]
    assert turn_bet["situation"] == "lead_opp" and turn_bet["line"] == "x"


CASE_H = """
    Hand #1008: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:07:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3 has timed out
    P3: folds
    P1: raises 2.00 to 3.00
    P2: folds
    Uncalled bet (2.00) returned to P1
    P1 collected 2.00 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
"""


def test_case_h_timeout_fold_is_not_a_decision():
    r = replay_text(CASE_H)
    fold = r.decisions("P3")[0]
    assert fold["auto_kind"] == "timeout" and fold["is_decision"] is False
    assert r.decisions("P1")[0]["auto_kind"] == "unknown" and r.decisions("P1")[0]["is_decision"]
    assert r.players["P3"]["n_decisions"] == 0


CASE_I = """
    Hand #1009: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:08:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: N (100.00 in chips)
    Seat 4: P4 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    N: posts small blind (dead) and big blind 1.50
    *** HOLE CARDS ***
    N: checks
    P4: raises 2.00 to 3.00
    P1: folds
    P2: folds
    N: calls 2.00
    *** FLOP *** [2c 7d 9h]
    N: checks
    P4: bets 3.00
    N: folds
    Uncalled bet (3.00) returned to P4
    P4 collected 8.00 from pot
    *** SUMMARY ***
    Total pot 8.00 | Rake 0.00
    Board [2c 7d 9h]
"""


def test_case_i_dead_blind_post_counts_only_its_live_part():
    r = replay_text(CASE_I)
    call = r.decisions("N")[1]
    assert call["street_put_before_bb"] == approx(1.0) and call["to_call_bb"] == approx(2.0)
    assert call["pot_before_bb"] == approx(6.0)  # 0.5 + 1 + 1.5 (0.5 of it dead) + 3
    n_posts = [a for a in r.actions if a["is_post"] and a["player_id"] == "N"]
    assert [(a["post_kind"], a["amount_bb"]) for a in n_posts] == [("dead", 0.5), ("post", 1.0)]
    assert r.players["N"]["dead_bb"] == approx(0.5) and r.players["N"]["blind_bb"] == approx(1.0)
    assert r.players["P2"]["is_bb"] and not r.players["N"]["is_bb"]
    assert r.players["N"]["position"] == "CO"


CASE_J = """
    Hand #1010: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:09:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P1: posts the ante 0.10
    P2: posts the ante 0.10
    P3: posts the ante 0.10
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: folds
    P1: folds
    Uncalled bet (0.50) returned to P2
    P2 collected 1.30 from pot
    *** SUMMARY ***
    Total pot 1.30 | Rake 0.00
"""


def test_case_j_walk_big_blind_has_no_decision_and_wins_the_blind():
    r = replay_text(CASE_J)
    assert r.decisions("P2") == []
    assert r.players["P2"]["n_decisions"] == 0
    assert r.players["P2"]["net_bb"] == approx(0.7)  # the small blind and two antes
    assert r.hand["pot_type"] == "walk" and r.hand["n_dealt"] == 3
    assert r.hand["final_pot_bb"] == approx(1.3)


CASE_K = """
    Hand #1011: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:10:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P1: posts the ante 2.00
    P2: posts the ante 2.00
    P3: posts the ante 2.00
    *** HOLE CARDS ***
    *** FLOP *** [2c 7d 9h]
    P1: checks
    P2: bets 3.00
    P3: folds
    P1: folds
    Uncalled bet (3.00) returned to P2
    P2 collected 6.00 from pot
    *** SUMMARY ***
    Total pot 6.00 | Rake 0.00
    Board [2c 7d 9h]
"""


def test_case_k_bomb_pot_is_dropped_by_default_and_kept_on_request():
    assert gate(CASE_K) == "bomb_pot_no_blinds"
    r = replay_text(CASE_K, keep_bomb_pots=True)
    assert r.hand["variant"] == "bomb_pot"
    assert [a["situation"] for a in r.decisions(street="preflop")] == []
    assert r.decisions("P1", "flop")[0]["situation"] == "lead_opp"
    assert r.players["P2"]["net_bb"] == approx(4.0)


CASE_L = """
    Hand #1012: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:11:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 2.00 to 3.00
    Y: calls 2.00
    *** FLOP *** [2c 7d 9h]
    Y: bets 97.00 and is all-in
    X: calls 97.00 and is all-in
    *** FIRST TURN *** [2c 7d 9h] [Ts]
    *** FIRST RIVER *** [2c 7d 9h Ts] [3c]
    *** SECOND TURN *** [2c 7d 9h] [As]
    *** SECOND RIVER *** [2c 7d 9h As] [4d]
    *** FIRST SHOW DOWN ***
    Y: shows [9c 9s] (three of a kind, Nines)
    X: shows [Ah Ad] (a pair of Aces)
    Y collected 100.00 from pot
    *** SECOND SHOW DOWN ***
    Y: shows [9c 9s] (three of a kind, Nines)
    X: shows [Ah Ad] (three of a kind, Aces)
    X collected 100.00 from pot
    *** SUMMARY ***
    Total pot 200.00 | Rake 0.00
    Hand was run twice
    FIRST Board [2c 7d 9h Ts 3c]
    SECOND Board [2c 7d 9h As 4d]
"""


def test_case_l_run_it_twice_keeps_both_boards_and_sums_the_runs():
    r = replay_text(CASE_L)
    assert r.hand["n_runs"] == 2 and r.hand["board"] == "2c7d9hTs3c"
    assert r.hand["board_run2"] == "2c7d9hAs4d" and r.hand["runout_street"] == "turn"
    assert all(a["board_at_action"] == "2c7d9h" for a in r.decisions(street="flop"))
    assert r.players["X"]["won_bb"] == approx(100.0) and r.players["Y"]["won_bb"] == approx(100.0)
    assert r.players["X"]["net_bb"] == approx(0.0) and r.players["Y"]["net_bb"] == approx(0.0)
    assert "RIT" in r.hand["error_detail"] and not r.hand["error_flag"]


CASE_M = """
    Hand #1013: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:12:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 9.00 to 10.00
    Y: calls 9.00
    *** FLOP *** [2c 7d 9h]
    Y: checks
    X: bets 10.00
    Y: folds
    Uncalled bet (10.00) returned to X
    X collected 20.00 from pot
    *** SUMMARY ***
    Total pot 20.00 | Rake 0.00
    Board [2c 7d 9h]
"""


def test_case_m_uncalled_bet_when_everyone_folds():
    r = replay_text(CASE_M)
    assert r.hand["uncalled_bb"] == approx(10.0) and r.hand["final_pot_bb"] == approx(20.0)
    assert r.players["X"]["net_bb"] == approx(10.0) and r.players["X"]["returned_bb"] == approx(10.0)
    assert r.players["Y"]["net_bb"] == approx(-10.0)
    assert r.hand["went_to_showdown"] is False and r.players["X"]["went_to_showdown"] is False
    bet = r.decisions("X", "flop")[0]
    assert bet["situation"] == "cbet_opp" and bet["size_pot"] == approx(0.5)
    assert r.decisions("Y", "flop")[1]["situation"] == "vs_cbet"


ALL_CASES = [CASE_A, CASE_B, CASE_C, CASE_D, CASE_E, CASE_F, CASE_G, CASE_H, CASE_I, CASE_J, CASE_L, CASE_M]


@pytest.mark.parametrize("text", ALL_CASES)
def test_every_case_passes_the_gate_and_reconciles(text):
    r = replay_text(text)
    assert not r.hand["error_flag"], r.hand["error_detail"]
    total = sum(p["net_bb"] for p in r.players.values())
    assert total + r.hand["rake_bb"] + r.hand["fees_bb"] == approx(0.0, abs=1e-9)
    pot = 0.0
    for a in r.actions:  # the pot before each row is the sum of the effective amounts before it
        assert a["pot_before_bb"] == approx(pot)
        pot += a["amount_eff_bb"]
