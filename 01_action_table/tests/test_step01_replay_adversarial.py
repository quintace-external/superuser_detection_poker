"""Adversarial replay cases: numbers worked out by hand for the hands most likely to break the replay.

Side pots in three layers, blinds all-in for less than the blind, uncalled bets after an
all-in call, dead money (dead blinds and a dead button blind), straddles, run it twice,
the older all-in raise form, names that contain colons, heads-up tables whose button
seat is not the small blind, effective sizes (cap the raise-to, never the increment),
side-pot-aware pot odds, free folds and the big blind's option. Big blind = 1.00.
"""

from __future__ import annotations

import pytest
from step01_testlib import replay_text

from sudetect.positions import assign_positions

approx = pytest.approx


def money_and_pot_checks(r) -> None:
    """Checks every hand here must pass: no hard error, nets + rake = 0, pot accumulates, the
    final pot equals the printed total pot, and the uncalled excess equals the printed return."""
    assert not r.hand["error_flag"], r.hand["error_detail"]
    assert "UNCALLED_TEXT_DIFFERS" not in (r.hand["error_detail"] or "")
    assert sum(p["net_bb"] for p in r.players.values()) + r.hand["rake_bb"] + r.hand["fees_bb"] == approx(0.0)
    pot = 0.0
    for a in r.actions:
        assert a["pot_before_bb"] == approx(pot)
        pot += a["amount_eff_bb"]
    bb = r.parsed.bb
    assert r.hand["final_pot_bb"] == approx(r.parsed.total_pot / bb)
    assert r.hand["uncalled_bb"] == approx(sum(r.parsed.returned) / bb)
    assert sum(p["returned_bb"] for p in r.players.values()) == approx(r.hand["uncalled_bb"])


# ------------------------------------------------------------------ side pots, short blinds, uncalled

SIDE_POTS_3 = """
    Hand #2001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: A (10.00 in chips)
    Seat 2: B (30.00 in chips)
    Seat 3: C (60.00 in chips)
    Seat 4: D (200.00 in chips)
    A: posts small blind 0.50
    B: posts big blind 1.00
    *** HOLE CARDS ***
    Dealt to A [As Ah]
    Dealt to B [Ks Kh]
    Dealt to C [Qs Qh]
    Dealt to D [Js Jh]
    C: raises 59.00 to 60.00 and is all-in
    D: raises 140.00 to 200.00 and is all-in
    A: calls 9.50 and is all-in
    B: calls 29.00 and is all-in
    Uncalled bet (140.00) returned to D
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    A: shows [As Ah] (a pair of Aces)
    B: shows [Ks Kh] (a pair of Kings)
    C: shows [Qs Qh] (a pair of Queens)
    D: shows [Js Jh] (a pair of Jacks)
    C collected 60.00 from side pot-2
    B collected 60.00 from side pot-1
    A collected 37.00 from main pot
    *** SUMMARY ***
    Total pot 160.00 Main pot 37.00. Side pot-1 60.00. Side pot-2 60.00. | Rake 3.00
    Board [2c 7d 9h Ts 3c]
"""


def test_three_layer_side_pots_pot_odds_and_payouts():
    r = replay_text(SIDE_POTS_3)
    money_and_pot_checks(r)
    shove = r.decisions("C")[0]
    assert shove["action"] == "raise" and shove["raise_to_eff_bb"] == approx(60.0)
    assert shove["size_pot"] == approx(59 / 2.5)
    # D's all-in over a covered all-in raises nothing: a call of 60, 140 comes back
    over = r.decisions("D")[0]
    assert over["action"] == "call" and over["action_raw"] == "raise" and over["is_capped"]
    assert over["amount_bb"] == approx(200.0) and over["amount_eff_bb"] == approx(60.0)
    assert over["size_pot"] is None and over["pot_odds"] == approx(60 / 121.5)
    # the short callers can only win their layer: A 9.5 of 31, B 29 of 100
    a_call, b_call = r.decisions("A")[0], r.decisions("B")[0]
    assert a_call["to_call_bb"] == approx(59.5) and a_call["to_call_eff_bb"] == approx(9.5)
    assert a_call["pot_odds"] == approx(9.5 / 31) and a_call["situation"] == "vs_open_call"
    assert b_call["to_call_bb"] == approx(59.0) and b_call["to_call_eff_bb"] == approx(29.0)
    assert b_call["pot_odds"] == approx(29 / 100) and b_call["pot_before_bb"] == approx(131.0)
    h = r.hand
    assert h["n_side_pots"] == 2 and h["final_pot_bb"] == approx(160.0) and h["players_flop"] == 4
    assert h["uncalled_bb"] == approx(140.0) and h["uncalled_seat"] == 4
    assert h["pot_flop_bb"] == approx(160.0) and h["pot_river_bb"] == approx(160.0)
    assert h["n_showdown"] == 4 and h["rake_bb"] == approx(3.0)
    nets = {k: v["net_bb"] for k, v in r.players.items()}
    assert nets == {"A": approx(27.0), "B": approx(30.0), "C": approx(0.0), "D": approx(-60.0)}
    assert r.players["D"]["returned_bb"] == approx(140.0) and r.players["D"]["won_at_showdown"] is False
    assert all(p["went_to_showdown"] for p in r.players.values())


SB_SHORT = """
    Hand #2002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: S (0.30 in chips)
    Seat 2: B (100.00 in chips)
    Seat 3: N (100.00 in chips)
    S: posts small blind 0.30 and is all-in
    B: posts big blind 1.00
    *** HOLE CARDS ***
    N: raises 2.00 to 3.00
    B: folds
    Uncalled bet (2.00) returned to N
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    S: shows [As Ah] (a pair of Aces)
    N: shows [Ks Kh] (a pair of Kings)
    N collected 1.40 from side pot
    S collected 0.90 from main pot
    *** SUMMARY ***
    Total pot 2.30 | Rake 0.00
"""


def test_small_blind_all_in_for_less_than_the_blind_wins_only_the_main_pot():
    r = replay_text(SB_SHORT)
    money_and_pot_checks(r)
    post = [a for a in r.actions if a["is_post"] and a["player_id"] == "S"][0]
    assert post["is_allin"] and post["amount_bb"] == approx(0.3) and post["post_kind"] == "sb"
    open_ = r.decisions("N")[0]
    assert open_["to_call_bb"] == approx(1.0) and open_["pot_before_bb"] == approx(1.3)
    assert open_["size_pot"] == approx(2 / 2.3) and not open_["is_capped"]
    fold = r.decisions("B")[0]
    assert fold["to_call_bb"] == approx(2.0) and fold["pot_odds"] == approx(2 / 6.3)
    assert r.hand["n_side_pots"] == 1 and r.hand["final_pot_bb"] == approx(2.3)
    assert r.hand["uncalled_bb"] == approx(2.0) and r.hand["uncalled_seat"] == 3
    assert r.players["S"]["position"] == "SB" and r.players["S"]["is_sb"]
    assert r.players["S"]["net_bb"] == approx(0.6) and r.players["N"]["net_bb"] == approx(0.4)
    assert r.players["S"]["allin_street"] == "preflop" and r.players["S"]["n_decisions"] == 0


BB_BELOW_SB = """
    Hand #2003: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (0.30 in chips)
    X: posts small blind 0.50
    Y: posts big blind 0.30 and is all-in
    *** HOLE CARDS ***
    Uncalled bet (0.20) returned to X
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    X: shows [As Ah] (a pair of Aces)
    Y: shows [Ks Kh] (a pair of Kings)
    X collected 0.60 from pot
    *** SUMMARY ***
    Total pot 0.60 | Rake 0.00
"""


def test_big_blind_all_in_below_the_small_blind_returns_the_small_blind_excess():
    r = replay_text(BB_BELOW_SB)
    money_and_pot_checks(r)
    assert r.decisions() == [] and r.hand["went_to_showdown"]
    assert r.hand["uncalled_bb"] == approx(0.2) and r.hand["uncalled_seat"] == 1
    assert r.hand["final_pot_bb"] == approx(0.6) and r.hand["pot_flop_bb"] == approx(0.6)
    assert r.players["X"]["net_bb"] == approx(0.3) and r.players["Y"]["net_bb"] == approx(-0.3)
    assert r.players["X"]["position"] == "BTN" and r.players["X"]["is_sb"]


UNCALLED_AFTER_ALLIN_CALL = """
    Hand #2004: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #1 is the button
    Seat 1: A (100.00 in chips)
    Seat 2: S (100.00 in chips)
    Seat 3: C (100.00 in chips)
    Seat 4: B (23.00 in chips)
    S: posts small blind 0.50
    C: posts big blind 1.00
    *** HOLE CARDS ***
    B: calls 1.00
    A: raises 2.00 to 3.00
    S: folds
    C: calls 2.00
    B: calls 2.00
    *** FLOP *** [2c 7d 9h]
    C: checks
    B: checks
    A: bets 50.00
    C: folds
    B: calls 20.00 and is all-in
    Uncalled bet (30.00) returned to A
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    B: shows [Ks Kh] (a pair of Kings)
    A: shows [As Ah] (a pair of Aces)
    A collected 49.50 from pot
    *** SUMMARY ***
    Total pot 49.50 | Rake 0.00
"""


def test_uncalled_bet_after_an_all_in_call_with_a_folded_third_player():
    r = replay_text(UNCALLED_AFTER_ALLIN_CALL)
    money_and_pot_checks(r)
    bet = r.decisions("A", "flop")[0]
    assert bet["action"] == "bet" and not bet["is_capped"] and bet["amount_eff_bb"] == approx(50.0)
    assert bet["situation"] == "cbet_opp" and bet["pot_before_bb"] == approx(9.5)
    fold = r.decisions("C", "flop")[1]
    # C would put 50 and could win A's 53, B's 3 so far and the folded small blind
    assert fold["to_call_bb"] == approx(50.0) and fold["pot_odds"] == approx(50 / 109.5)
    call = r.decisions("B", "flop")[1]
    assert call["to_call_bb"] == approx(50.0) and call["to_call_eff_bb"] == approx(20.0)
    assert call["pot_odds"] == approx(20 / 49.5) and call["is_allin"]
    h = r.hand
    assert h["uncalled_bb"] == approx(30.0) and h["uncalled_seat"] == 1
    assert h["final_pot_bb"] == approx(49.5) and h["pot_turn_bb"] == approx(49.5)
    assert h["n_side_pots"] == 0 and h["players_turn"] == 2 and h["n_showdown"] == 2
    assert r.players["A"]["net_bb"] == approx(26.5) and r.players["C"]["net_bb"] == approx(-3.0)


# ------------------------------------------------------------------ dead money

DEAD_SB_RAISE = """
    Hand #2101: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: N (100.00 in chips)
    Seat 4: P3 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    N: posts small blind (dead) 0.50
    *** HOLE CARDS ***
    N: raises 2.00 to 3.00
    P3: folds
    P1: folds
    P2: folds
    Uncalled bet (2.00) returned to N
    N collected 3.00 from pot
    *** SUMMARY ***
    Total pot 3.00 | Rake 0.00
"""


def test_a_dead_blind_is_never_returned_as_uncalled():
    r = replay_text(DEAD_SB_RAISE)
    money_and_pot_checks(r)
    assert r.hand["uncalled_bb"] == approx(2.0) and r.hand["uncalled_seat"] == 3
    assert r.hand["final_pot_bb"] == approx(3.0)
    assert r.players["N"]["net_bb"] == approx(1.5) and r.players["N"]["dead_bb"] == approx(0.5)
    assert r.players["N"]["position"] == "CO" and r.players["N"]["preflop_order"] == 1
    raise_ = r.decisions("N")[0]
    assert raise_["street_put_before_bb"] == 0 and raise_["to_call_bb"] == approx(1.0)
    assert raise_["pot_before_bb"] == approx(2.0) and raise_["size_pot"] == approx(2 / 3)


def test_pot_odds_count_dead_money_whole():
    r = replay_text(DEAD_SB_RAISE)
    # a call of 3 by P3 makes a pot of 8: the 0.5 dead is won whole, not capped at the caller's 3
    assert r.decisions("P3")[0]["pot_odds"] == approx(3 / 8)
    assert r.decisions("P1")[0]["pot_odds"] == approx(2.5 / 7.5)
    assert r.decisions("P2")[0]["pot_odds"] == approx(2 / 7)


def test_case_i_dead_blind_hand_level_numbers():
    from test_step01_cases_synthetic import CASE_I

    r = replay_text(CASE_I)
    money_and_pot_checks(r)
    assert r.hand["uncalled_bb"] == approx(3.0) and r.hand["uncalled_seat"] == 4
    assert r.hand["final_pot_bb"] == approx(8.0) and r.hand["pot_flop_bb"] == approx(8.0)
    assert r.hand["error_detail"] is None
    assert r.players["P4"]["net_bb"] == approx(5.0) and r.players["N"]["net_bb"] == approx(-3.5)


DEAD_MONEY_SHOWDOWN = """
    Hand #2102: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: N (100.00 in chips)
    Seat 4: P4 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    N: posts small & big blinds 1.50
    *** HOLE CARDS ***
    N: checks
    P4: calls 1.00
    P1: calls 0.50
    P2: checks
    *** FLOP *** [2c 7d 9h]
    P1: checks
    P2: checks
    N: checks
    P4: checks
    *** TURN *** [2c 7d 9h] [Ts]
    P1: checks
    P2: checks
    N: checks
    P4: checks
    *** RIVER *** [2c 7d 9h Ts] [3c]
    P1: checks
    P2: checks
    N: checks
    P4: checks
    *** SHOW DOWN ***
    P1: shows [As Ah] (a pair of Aces)
    P2: mucks hand
    N: mucks hand
    P4: mucks hand
    P1 collected 4.50 from pot
    *** SUMMARY ***
    Total pot 4.50 | Rake 0.00
"""


def test_dead_money_makes_no_side_pot_and_no_uncalled_excess():
    r = replay_text(DEAD_MONEY_SHOWDOWN)
    money_and_pot_checks(r)
    h = r.hand
    assert h["n_side_pots"] == 0 and h["uncalled_bb"] == 0 and h["uncalled_seat"] is None
    assert h["final_pot_bb"] == approx(4.5) and h["pot_flop_bb"] == approx(4.5)
    assert r.players["N"]["dead_bb"] == approx(0.5) and r.players["N"]["blind_bb"] == approx(1.0)
    check = r.decisions("N")[0]
    assert check["action"] == "check" and check["to_call_bb"] == 0 and check["pot_before_bb"] == approx(3.0)


BUTTON_BLIND_WALK = """
    Hand #2103: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    P3: posts button blind 1.00
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: folds
    P1: folds
    Uncalled bet (0.50) returned to P2
    P2 collected 2.00 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
"""


def test_a_dead_button_blind_does_not_hide_the_big_blinds_return():
    r = replay_text(BUTTON_BLIND_WALK)
    money_and_pot_checks(r)
    assert r.hand["uncalled_bb"] == approx(0.5) and r.hand["uncalled_seat"] == 2
    assert r.hand["final_pot_bb"] == approx(2.0) and r.hand["button_blind_bb"] == approx(1.0)
    assert r.hand["pot_type"] == "walk" and r.players["P2"]["net_bb"] == approx(1.5)
    assert r.decisions("P3")[0]["to_call_bb"] == approx(1.0)  # the button blind is dead money


# ------------------------------------------------------------------ heads-up button

HU_BUTTON_ON_BIG_BLIND = """
    Hand #2201: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 5: Y (100.00 in chips)
    Y: posts small blind 0.50
    X: posts big blind 1.00
    *** HOLE CARDS ***
    Y: raises 2.00 to 3.00
    X: calls 2.00
    *** FLOP *** [2c 7d 9h]
    X: checks
    Y: bets 2.00
    X: folds
    Uncalled bet (2.00) returned to Y
    Y collected 6.00 from pot
    *** SUMMARY ***
    Total pot 6.00 | Rake 0.00
"""


@pytest.mark.parametrize("button_line, flag", [
    ("Seat #1 is the button", "HU_BUTTON_NOT_SB"),   # the text puts the button on the big blind
    ("Seat #3 is the button", "DEAD_BUTTON"),        # the button sits on an empty seat
])
def test_heads_up_positions_follow_the_small_blind(button_line, flag):
    r = replay_text(HU_BUTTON_ON_BIG_BLIND.replace("Seat #1 is the button", button_line))
    money_and_pot_checks(r)
    x, y = r.players["X"], r.players["Y"]
    assert (y["position"], y["is_sb"], y["preflop_order"], y["postflop_order"]) == ("BTN", True, 1, 2)
    assert (x["position"], x["is_bb"], x["preflop_order"], x["postflop_order"]) == ("BB", True, 2, 1)
    assert r.hand["sb_seat"] == 5 and r.hand["bb_seat"] == 1
    assert flag in r.hand["error_detail"] and "HU_BUTTON_NOT_SB" in r.hand["error_detail"]
    check = r.decisions("X", "flop")[0]
    assert check["situation"] == "donk_opp" and check["is_ip"] is False and check["n_behind"] == 1
    bet = r.decisions("Y", "flop")[0]
    assert bet["situation"] == "cbet_opp" and bet["is_ip"] is True and bet["n_behind"] == 0
    assert r.decisions("Y", "preflop")[0]["n_behind"] == 1


def test_heads_up_assign_positions_takes_the_button_from_the_small_blind():
    p = assign_positions([1, 5], button_seat=1, max_seats=6, sb_player=1, bb_player=0)
    assert p.labels == ("BB", "BTN") and p.button == 1 and p.sb == 1 and p.bb == 0
    assert p.preflop_order == (2, 1) and p.postflop_order == (1, 2)
    assert p.flags == frozenset({"hu_button_not_sb"})
    only_bb = assign_positions([1, 5], button_seat=3, max_seats=6, sb_player=None, bb_player=0)
    assert only_bb.labels == ("BB", "BTN") and "dead_button" in only_bb.flags
    neither = assign_positions([1, 5], button_seat=1, max_seats=6)
    assert neither.labels == ("BTN", "BB") and not neither.flags


# ------------------------------------------------------------------ straddles, option, free folds

STRADDLE_LIMPED = """
    Hand #2301: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #1 is the button
    Seat 1: BT (100.00 in chips)
    Seat 2: SB (100.00 in chips)
    Seat 3: BB (100.00 in chips)
    Seat 4: ST (100.00 in chips)
    Seat 5: U1 (100.00 in chips)
    SB: posts small blind 0.50
    BB: posts big blind 1.00
    ST: posts straddle 2.00
    *** HOLE CARDS ***
    U1: folds
    BT: calls 2.00
    SB: calls 1.50
    BB: calls 1.00
    ST: checks
    *** FLOP *** [2c 7d 9h]
    SB: checks
    BB: checks
    ST: bets 4.00
    BT: folds
    SB: folds
    BB: folds
    Uncalled bet (4.00) returned to ST
    ST collected 8.00 from pot
    *** SUMMARY ***
    Total pot 8.00 | Rake 0.00
"""


def test_straddler_option_and_limps_of_the_straddle():
    r = replay_text(STRADDLE_LIMPED)
    money_and_pot_checks(r)
    order = {k: v["preflop_order"] for k, v in r.players.items()}
    assert order == {"U1": 1, "BT": 2, "SB": 3, "BB": 4, "ST": 5}
    assert r.decisions("BT")[0]["situation"] == "unopened" and r.decisions("BT")[0]["to_call_bb"] == approx(2.0)
    assert r.decisions("SB")[0]["to_call_bb"] == approx(1.5) and r.decisions("SB")[0]["situation"] == "vs_limp"
    option = r.decisions("ST")[0]
    assert option["action"] == "check" and option["to_call_bb"] == 0 and option["level_before_bb"] == approx(2.0)
    assert option["situation"] == "vs_limp" and option["n_limpers"] == 3
    assert r.players["ST"]["vpip"] is False and r.players["ST"]["is_straddle"]
    assert r.hand["preflop_raises"] == 0 and r.hand["pot_type"] == "limped"
    lead = r.decisions("ST", "flop")[0]
    assert lead["situation"] == "lead_opp" and lead["pot_type_now"] == "limped"
    assert lead["size_pot"] == approx(0.5)
    assert r.decisions("BT", "flop")[0]["is_ip"] is True and r.decisions("BT", "flop")[0]["situation"] == "vs_bet"


DOUBLE_STRADDLE = """
    Hand #2302: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
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
    P4: posts straddle 4.00
    *** HOLE CARDS ***
    P5: folds
    P6: calls 4.00
    P1: folds
    P2: folds
    P3: folds
    P4: checks
    *** FLOP *** [2c 7d 9h]
    P4: bets 5.00
    P6: folds
    Uncalled bet (5.00) returned to P4
    P4 collected 11.50 from pot
    *** SUMMARY ***
    Total pot 11.50 | Rake 0.00
"""


def test_two_straddles_both_post_and_the_second_has_the_option():
    r = replay_text(DOUBLE_STRADDLE)
    money_and_pot_checks(r)
    order = {k: v["preflop_order"] for k, v in r.players.items()}
    assert order == {"P5": 1, "P6": 2, "P1": 3, "P2": 4, "P3": 5, "P4": 6}
    assert r.hand["straddle_bb"] == approx(4.0) and r.hand["n_straddles"] == 2
    to_call = {a["player_id"]: a["to_call_bb"] for a in r.decisions(street="preflop")}
    assert to_call == {"P5": approx(4.0), "P6": approx(4.0), "P1": approx(3.5), "P2": approx(3.0),
                       "P3": approx(2.0), "P4": approx(0.0)}
    assert r.decisions("P4")[0]["situation"] == "vs_limp"
    assert r.hand["final_pot_bb"] == approx(11.5) and r.hand["uncalled_bb"] == approx(5.0)


BB_OPTION_RAISE = """
    Hand #2303: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: calls 0.50
    Y: raises 3.00 to 4.00
    X: calls 3.00
    *** FLOP *** [2c 7d 9h]
    Y: bets 4.00
    X: folds
    Uncalled bet (4.00) returned to Y
    Y collected 8.00 from pot
    *** SUMMARY ***
    Total pot 8.00 | Rake 0.00
"""


def test_big_blind_raising_his_option_is_a_raise_with_nothing_to_call():
    r = replay_text(BB_OPTION_RAISE)
    money_and_pot_checks(r)
    limp = r.decisions("X")[0]
    assert limp["situation"] == "unopened" and limp["pot_type_now"] == "unopened"
    raise_ = r.decisions("Y")[0]
    assert raise_["action"] == "raise" and raise_["to_call_bb"] == 0 and raise_["level_before_bb"] == approx(1.0)
    assert raise_["size_pot"] == approx(3 / 2) and raise_["size_x"] == approx(4.0)
    assert raise_["situation"] == "vs_limp" and raise_["pot_odds"] is None
    call = r.decisions("X")[1]
    assert call["situation"] == "vs_open" and call["pot_odds"] == approx(3 / 8)
    assert call["faced_size_pot"] == approx(1.5) and call["faced_bettor_id"] == "Y"
    assert r.players["Y"]["pfr"] and r.players["X"]["vpip"] and r.hand["pot_type"] == "single_raised"
    cbet = r.decisions("Y", "flop")[0]
    assert cbet["situation"] == "cbet_opp" and cbet["preflop_aggressor_id"] == "Y"


FREE_FOLDS = """
    Hand #2304: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #4 is the button
    Seat 1: P1 (100.00 in chips)
    Seat 2: P2 (100.00 in chips)
    Seat 3: P3 (100.00 in chips)
    Seat 4: P4 (100.00 in chips)
    P1: posts small blind 0.50
    P2: posts big blind 1.00
    *** HOLE CARDS ***
    P3: calls 1.00
    P4: calls 1.00
    P1: calls 0.50
    P2: folds
    *** FLOP *** [2c 7d 9h]
    P1: folds
    P3: checks
    P4: bets 2.00
    P3: folds
    Uncalled bet (2.00) returned to P4
    P4 collected 4.00 from pot
    *** SUMMARY ***
    Total pot 4.00 | Rake 0.00
"""


def test_folds_with_nothing_to_call_are_free_folds_preflop_and_postflop():
    r = replay_text(FREE_FOLDS)
    money_and_pot_checks(r)
    bb_fold = r.decisions("P2")[0]
    assert bb_fold["free_fold"] and bb_fold["to_call_bb"] == 0 and bb_fold["is_decision"]
    assert bb_fold["pot_odds"] is None and bb_fold["situation"] == "vs_limp"
    flop_fold = r.decisions("P1", "flop")[0]
    assert flop_fold["free_fold"] and flop_fold["situation"] == "lead_opp"
    assert not r.decisions("P1", "preflop")[0]["free_fold"]
    assert r.players["P2"]["folded_street"] == "preflop" and r.players["P2"]["vpip"] is False


# ------------------------------------------------------------------ effective sizes

CAP_THE_RAISE_TO = """
    Hand #2401: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #2 is the button
    Seat 1: A (18.00 in chips)
    Seat 2: B (200.00 in chips)
    B: posts small blind 0.50
    A: posts big blind 1.00
    *** HOLE CARDS ***
    B: raises 2.00 to 3.00
    A: calls 2.00
    *** FLOP *** [2c 7d 9h]
    A: bets 10.00
    B: raises 187.00 to 197.00 and is all-in
    A: calls 5.00 and is all-in
    Uncalled bet (182.00) returned to B
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    A: shows [9c 9s] (three of a kind, Nines)
    B: shows [Ah Ad] (a pair of Aces)
    A collected 36.00 from pot
    *** SUMMARY ***
    Total pot 36.00 | Rake 0.00
"""


def test_a_shove_over_a_short_bet_is_capped_at_the_bettors_capacity_not_his_chips_behind():
    r = replay_text(CAP_THE_RAISE_TO)
    money_and_pot_checks(r)
    shove = r.decisions("B", "flop")[0]
    # A can match 10 (in) + 5 (behind) = 15: the shove is a raise to 15, still above the bet of 10
    assert shove["action"] == "raise" and shove["is_capped"] and shove["is_allin"]
    assert shove["amount_bb"] == approx(197.0) and shove["amount_eff_bb"] == approx(15.0)
    assert shove["raise_to_eff_bb"] == approx(15.0) and shove["eff_stack_bb"] == approx(15.0)
    assert shove["size_pot"] == approx(5 / 26) and shove["size_x"] == approx(1.5)
    assert shove["situation"] == "vs_donk"
    call = r.decisions("A", "flop")[1]
    assert call["to_call_bb"] == approx(5.0) and call["to_call_eff_bb"] == approx(5.0)
    assert call["pot_odds"] == approx(5 / 36) and call["situation"] == "vs_raise"
    assert call["faced_size_pot"] == approx(5 / 26)
    assert r.hand["uncalled_bb"] == approx(182.0) and r.hand["uncalled_seat"] == 2
    assert r.players["B"]["net_bb"] == approx(-18.0)


RUN_IT_TWICE_PREFLOP = """
    Hand #2501: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (50.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 99.00 to 100.00 and is all-in
    Y: calls 49.00 and is all-in
    Uncalled bet (50.00) returned to X
    *** FIRST FLOP *** [2c 7d 9h]
    *** FIRST TURN *** [2c 7d 9h] [Ts]
    *** FIRST RIVER *** [2c 7d 9h Ts] [3c]
    *** SECOND FLOP *** [Kd 4s 5s]
    *** SECOND TURN *** [Kd 4s 5s] [6d]
    *** SECOND RIVER *** [Kd 4s 5s 6d] [8c]
    *** FIRST SHOW DOWN ***
    X: shows [As Ah] (a pair of Aces)
    Y: shows [Qs Qh] (a pair of Queens)
    X collected 50.00 from pot
    *** SECOND SHOW DOWN ***
    X: shows [As Ah] (a pair of Aces)
    Y: shows [Qs Qh] (a straight)
    Y collected 50.00 from pot
    *** SUMMARY ***
    Total pot 100.00 | Rake 0.00
    Hand was run twice
    FIRST Board [2c 7d 9h Ts 3c]
    SECOND Board [Kd 4s 5s 6d 8c]
"""


def test_run_it_twice_from_preflop_keeps_both_runs_and_caps_the_shove():
    r = replay_text(RUN_IT_TWICE_PREFLOP)
    money_and_pot_checks(r)
    h = r.hand
    assert h["n_runs"] == 2 and h["board"] == "2c7d9hTs3c" and h["board_run2"] == "Kd4s5s6d8c"
    assert h["runout_street"] == "flop" and h["last_street"] == "river" and "RIT" in h["error_detail"]
    shove = r.decisions("X")[0]
    assert shove["is_capped"] and shove["amount_eff_bb"] == approx(49.5) and shove["action"] == "raise"
    assert r.decisions("Y")[0]["to_call_bb"] == approx(49.0)
    assert r.players["X"]["won_bb"] == approx(50.0) and r.players["Y"]["won_bb"] == approx(50.0)
    assert r.players["X"]["net_bb"] == approx(0.0) and r.players["Y"]["saw_river"]
    assert r.decisions(street="flop") == []


LEGACY_ALLIN_RAISE = """
    Hand #2601: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (20.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 2.00 to 3.00
    Y: raises 19.00 and is all-in
    X: calls 17.00
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    X: shows [As Ah] (a pair of Aces)
    Y: shows [Qs Qh] (a pair of Queens)
    X collected 40.00 from pot
    *** SUMMARY ***
    Total pot 40.00 | Rake 0.00
"""


def test_the_older_all_in_raise_form_moves_the_chips_written():
    r = replay_text(LEGACY_ALLIN_RAISE)
    money_and_pot_checks(r)
    shove = r.decisions("Y")[0]
    assert shove["amount_bb"] == approx(19.0) and shove["raise_to_eff_bb"] == approx(20.0)
    assert shove["action"] == "raise" and shove["is_allin"] and shove["situation"] == "vs_open"
    assert shove["size_pot"] == approx(17 / 6)
    call = r.decisions("X")[1]
    assert call["to_call_bb"] == approx(17.0) and call["pot_odds"] == approx(17 / 40)
    assert call["situation"] == "vs_3bet" and r.hand["pot_type"] == "3bet"


PREFIX_NAMES = """
    Hand #2701: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'room: 1' 6-max Seat #3 is the button
    Seat 1: a (100.00 in chips)
    Seat 2: a: b (100.00 in chips)
    Seat 3: a b (100.00 in chips)
    a: posts small blind 0.50
    a: b: posts big blind 1.00
    *** HOLE CARDS ***
    Dealt to a [Ah Kd]
    Dealt to a: b [Qs Qc]
    Dealt to a b [7h 7c]
    a b: raises 2.00 to 3.00
    a b said, "a: b: folds"
    a: folds
    a: b: calls 2.00
    *** FLOP *** [2c 7d 9h]
    a: b: checks
    a b: bets 2.00
    a: b: folds
    Uncalled bet (2.00) returned to a b
    a b collected 6.50 from pot
    *** SUMMARY ***
    Total pot 6.50 | Rake 0.00
"""


def test_names_that_are_prefixes_of_each_other_and_contain_colons():
    r = replay_text(PREFIX_NAMES)
    money_and_pot_checks(r)
    assert r.parsed.players == ["a", "a: b", "a b"] and r.hand["table_id"] == "room: 1"
    assert [(a["player_id"], a["action"]) for a in r.decisions()] == [
        ("a b", "raise"), ("a", "fold"), ("a: b", "call"), ("a: b", "check"), ("a b", "bet"), ("a: b", "fold")]
    assert r.players["a: b"]["hole_cards"] == "QsQc" and r.players["a"]["hole_cards"] == "AhKd"
    assert r.players["a: b"]["position"] == "BB" and r.players["a b"]["position"] == "BTN"
    assert r.players["a b"]["net_bb"] == approx(3.5)


ANTE_ALL_IN = """
    Hand #2801: Hold'em No Limit (0.50/1.00 - Ante 0.20) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #1 is the button
    Seat 1: BT (100.00 in chips)
    Seat 2: SB (100.00 in chips)
    Seat 3: BB (100.00 in chips)
    Seat 4: SH (0.10 in chips)
    BT: posts the ante 0.20
    SB: posts the ante 0.20
    BB: posts the ante 0.20
    SH: posts the ante 0.10 and is all-in
    SB: posts small blind 0.50
    BB: posts big blind 1.00
    *** HOLE CARDS ***
    BT: raises 2.00 to 3.00
    SB: folds
    BB: folds
    Uncalled bet (2.00) returned to BT
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    BT: shows [As Ah] (a pair of Aces)
    SH: shows [Qs Qh] (a pair of Queens)
    BT collected 2.80 from side pot
    BT collected 0.40 from main pot
    *** SUMMARY ***
    Total pot 3.20 | Rake 0.00
"""


def test_a_player_all_in_from_a_short_ante_plays_for_the_antes_only():
    r = replay_text(ANTE_ALL_IN)
    money_and_pot_checks(r)
    assert r.hand["n_side_pots"] == 1 and r.hand["ante_bb"] == approx(0.2)
    assert r.players["SH"]["ante_bb"] == approx(0.1) and r.players["SH"]["went_to_showdown"]
    assert r.players["SH"]["position"] == "CO" and r.players["SH"]["n_decisions"] == 0
    assert r.decisions("BT")[0]["pot_odds"] == approx(1 / 3.2)
    assert r.decisions("SB")[0]["pot_odds"] == approx(2.5 / 7.7)
    assert r.players["BT"]["net_bb"] == approx(2.0) and r.players["SH"]["net_bb"] == approx(-0.1)
