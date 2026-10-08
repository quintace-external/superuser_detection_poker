"""Adversarial parser cases: text that must be dropped with the right reason, or kept intact.

Chat lines that quote grammar (a player can type anything, including a hand header or the
cancel marker, and must not be able to drop his own hands that way), table lines without a
usable button, conflicting seats and cards, impossible betting, time zones, antes on and off,
run it three times.
"""

from __future__ import annotations

import pytest
from step01_testlib import gate, hand, replay_text

from sudetect.hhparse import GATE_REASONS, ParsedHand, parse_hand, split_hands

approx = pytest.approx

BASE = """
    Hand #6000: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    Dealt to X [Ah Kd]
    Dealt to Y [Qs Qc]
    X: raises 2.00 to 3.00
    Y: calls 2.00
    *** FLOP *** [2c 7d 9h]
    Y: checks
    X: bets 2.00
    Y: folds
    Uncalled bet (2.00) returned to X
    X collected 6.00 from pot
    *** SUMMARY ***
    Total pot 6.00 | Rake 0.00
    Board [2c 7d 9h]
"""

ALL_IN = """
    Hand #6001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:01:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (10.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 9.00 to 10.00
    Y: calls 9.00 and is all-in
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    X: shows [Ah Kd] (high card Ace)
    Y: shows [Qs Qc] (a pair of Queens)
    Y collected 20.00 from pot
    *** SUMMARY ***
    Total pot 20.00 | Rake 0.00
"""


def variant(old: str, new: str, text: str = BASE) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def test_base_hands_are_kept():
    assert gate(BASE) is None and gate(ALL_IN) is None


# ------------------------------------------------------------------ chat cannot drop a hand

HEADER_IN_CHAT = 'Y said, "Hand #6000: Hold\'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC"'


def test_a_header_quoted_in_the_chat_does_not_split_the_hand():
    first = hand(variant("    Y: checks\n", "    Y: checks\n    " + HEADER_IN_CHAT + "\n"))
    text = first + "\n" + hand(ALL_IN)
    hands = split_hands(text)
    assert len(hands) == 2 and hands[0].startswith("Hand #6000") and hands[1].startswith("Hand #6001")
    parsed = parse_hand(hands[0])
    assert isinstance(parsed, ParsedHand), parsed
    assert parsed.collected == [600, 0]


def test_the_cancel_marker_typed_in_the_chat_does_not_cancel_the_hand():
    text = variant("    Y: checks\n", '    Y: checks\n    X said, "*** HAND CANCELLED *** gg"\n')
    assert gate(text) is None
    assert gate(variant("    Y: checks\n", "    *** HAND CANCELLED ***\n")) == "cancelled"


# ------------------------------------------------------------------ table line and button

def test_a_table_line_without_the_button_is_dropped():
    assert gate(variant("Table 'T01' 6-max Seat #1 is the button", "Table 'T01' 6-max")) == "bad_table_line"


@pytest.mark.parametrize("button", ["Seat #9 is the button", "Seat #0 is the button"])
def test_a_button_seat_outside_the_dealt_seats_is_a_dead_button(button):
    r = replay_text(variant("Seat #1 is the button", button))
    assert "DEAD_BUTTON" in r.hand["error_detail"] and not r.hand["error_flag"]
    assert r.players["X"]["position"] == "BTN" and r.players["X"]["is_sb"]
    assert r.players["Y"]["position"] == "BB"


# ------------------------------------------------------------------ the gate, one bad line at a time

@pytest.mark.parametrize("text, reason", [
    # a post after the hole cards
    (variant("    Dealt to X [Ah Kd]\n", "    Dealt to X [Ah Kd]\n    X: posts the ante 0.10\n"), "unknown_grammar"),
    # the same name in two seats, two names in one seat
    (variant("Seat 2: Y (100.00 in chips)", "Seat 2: Y (100.00 in chips)\n    Seat 3: Y (50.00 in chips)"),
     "unknown_grammar"),
    (variant("Seat 2: Y (100.00 in chips)", "Seat 2: Y (100.00 in chips)\n    Seat 2: Z (50.00 in chips)"),
     "unknown_grammar"),
    # cards dealt to or chips returned to a name with no seat
    (variant("    Dealt to Y [Qs Qc]\n", "    Dealt to Y [Qs Qc]\n    Dealt to Z [Js Jc]\n"), "unknown_grammar"),
    (variant("returned to X", "returned to Z"), "unknown_grammar"),
    # streets out of order
    (variant("*** FLOP *** [2c 7d 9h]", "*** TURN *** [2c 7d 9h] [Ts]"), "unknown_grammar"),
    # four cards on the flop, two different hands for one player, a shown hand that was not dealt
    (variant("*** FLOP *** [2c 7d 9h]", "*** FLOP *** [2c 7d 9h Ts]"), "bad_card"),
    (variant("    Dealt to Y [Qs Qc]\n", "    Dealt to Y [Qs Qc]\n    Dealt to Y [Js Jc]\n"), "bad_card"),
    (variant("    Y: folds\n", "    Y: folds\n    Y: shows [Js Jc]\n"), "bad_card"),
    # more than the stack, an all-in marker with chips behind
    (variant("    X: posts small blind 0.50\n", "    X: posts the ante 150.00\n    X: posts small blind 0.50\n"),
     "stack_inconsistent"),
    (variant("Y: posts big blind 1.00", "Y: posts big blind 1.00 and is all-in"), "stack_inconsistent"),
    (variant("X: raises 2.00 to 3.00", "X: raises 2.50 and is all-in"), "stack_inconsistent"),
    # acting while all-in
    (variant("*** FLOP *** [2c 7d 9h]\n", "*** FLOP *** [2c 7d 9h]\n    Y: checks\n    X: checks\n", ALL_IN),
     "action_order_invalid"),
    # a raise that does not raise, a bet facing a bet, a call above the amount to call
    (variant("X: raises 2.00 to 3.00\n    Y: calls 2.00", "X: raises 0.50 to 1.00\n    Y: checks"),
     "amount_inconsistent"),
    (variant("    Y: folds\n    Uncalled bet (2.00) returned to X",
             "    Y: bets 4.00\n    X: folds\n    Uncalled bet (2.00) returned to Y"), "amount_inconsistent"),
    (variant("    Y: folds\n", "    Y: calls 3.00\n"), "amount_inconsistent"),
])
def test_gate_drops_inconsistent_text(text, reason):
    assert reason in GATE_REASONS
    assert gate(text) == reason


# ------------------------------------------------------------------ time zones and antes

def test_time_zone_flags():
    assert "time_zone_not_utc" in parse_hand(hand(variant("10:00:00 UTC", "10:00:00 ET"))).flags
    for tz in ("10:00:00 GMT", "10:00:00"):
        assert "time_zone_not_utc" not in parse_hand(hand(variant("10:00:00 UTC", tz))).flags


ANTES = """
    Hand #6002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:02:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts ante 0.10
    Y: posts the ante 0.10
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 2.00 to 3.00
    Y: folds
    Uncalled bet (2.00) returned to X
    X collected 2.20 from pot
    *** SUMMARY ***
    Total pot 2.20 | Rake 0.00
"""


@pytest.mark.parametrize("stakes", ["(0.50/1.00)", "(0.50/1.00 - Ante 0.10)", "(0.50/1.00 ante 0.10)"])
def test_antes_posted_with_or_without_the_header_saying_so(stakes):
    r = replay_text(ANTES.replace("(0.50/1.00)", stakes))
    assert r.hand["ante_bb"] == approx(0.1) and r.hand["final_pot_bb"] == approx(2.2)
    assert [a["post_kind"] for a in r.actions if a["is_post"]] == ["ante", "ante", "sb", "bb"]
    raise_ = r.decisions("X")[0]
    assert raise_["pot_before_bb"] == approx(1.7) and raise_["to_call_bb"] == approx(0.5)
    assert raise_["size_pot"] == approx(2 / 2.2)
    assert r.players["X"]["net_bb"] == approx(1.1) and r.players["Y"]["ante_bb"] == approx(0.1)


def test_no_antes_at_all():
    r = replay_text(BASE)
    assert r.hand["ante_bb"] == 0 and all(p["ante_bb"] == 0 for p in r.players.values())


# ------------------------------------------------------------------ run it three times

RUN_THREE_TIMES = """
    Hand #6003: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:03:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (30.00 in chips)
    Seat 2: Y (30.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 2.00 to 3.00
    Y: calls 2.00
    *** FLOP *** [2c 7d 9h]
    Y: bets 27.00 and is all-in
    X: calls 27.00 and is all-in
    *** FIRST TURN *** [2c 7d 9h] [Ts]
    *** FIRST RIVER *** [2c 7d 9h Ts] [3c]
    *** SECOND TURN *** [2c 7d 9h] [As]
    *** SECOND RIVER *** [2c 7d 9h As] [4d]
    *** THIRD TURN *** [2c 7d 9h] [Kh]
    *** THIRD RIVER *** [2c 7d 9h Kh] [5s]
    *** FIRST SHOW DOWN ***
    X collected 20.00 from pot
    *** SECOND SHOW DOWN ***
    Y collected 20.00 from pot
    *** THIRD SHOW DOWN ***
    Y collected 20.00 from pot
    *** SUMMARY ***
    Total pot 60.00 | Rake 0.00
    Hand was run three times
    FIRST Board [2c 7d 9h Ts 3c]
"""


def test_run_it_three_times_sums_the_runs_and_keeps_run_two():
    r = replay_text(RUN_THREE_TIMES)
    h = r.hand
    assert h["n_runs"] == 3 and h["board"] == "2c7d9hTs3c" and h["board_run2"] == "2c7d9hAs4d"
    assert h["runout_street"] == "turn" and "RIT" in h["error_detail"] and not h["error_flag"]
    assert r.players["X"]["won_bb"] == approx(20.0) and r.players["Y"]["won_bb"] == approx(40.0)
    assert r.players["X"]["net_bb"] == approx(-10.0) and r.players["Y"]["net_bb"] == approx(10.0)
    assert all(a["board_at_action"] == "2c7d9h" for a in r.decisions(street="flop"))
