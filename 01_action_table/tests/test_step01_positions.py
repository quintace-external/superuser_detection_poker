"""Positions from the button and the dealt seats: labels, classes, acting orders, edge cases."""

from __future__ import annotations

import pytest
from step01_testlib import replay_text

from sudetect.positions import assign_positions, position_class, postflop_sequence

FULL_RING = ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def labels_in_preflop_order(n: int) -> list[str]:
    seats = list(range(1, n + 1))
    p = assign_positions(seats, button_seat=n, max_seats=9, sb_player=0, bb_player=1)
    order = sorted(range(n), key=lambda i: p.preflop_order[i])
    return [p.labels[i] for i in order]


@pytest.mark.parametrize("n", range(3, 10))
def test_ring_labels_drop_from_the_front(n):
    expected = FULL_RING[9 - n:]
    assert labels_in_preflop_order(n) == expected


def test_heads_up_button_is_btn_and_acts_first_before_the_flop():
    p = assign_positions([3, 7], button_seat=7, max_seats=9, sb_player=1, bb_player=0)
    assert p.labels == ("BB", "BTN")
    assert p.preflop_order == (2, 1) and p.postflop_order == (1, 2)
    assert p.sb == 1 and p.bb == 0 and not p.flags


def test_position_classes():
    assert [position_class(x) for x in FULL_RING] == ["EP", "EP", "EP", "MP", "MP", "CO", "BTN", "SB", "BB"]
    assert position_class("UTG+3") == "EP" and position_class(None) is None


def test_ten_handed_table_adds_utg_plus_three():
    seats = list(range(1, 11))
    p = assign_positions(seats, button_seat=10, max_seats=10, sb_player=0, bb_player=1)
    order = sorted(range(10), key=lambda i: p.preflop_order[i])
    assert [p.labels[i] for i in order] == ["UTG", "UTG+1", "UTG+2", "UTG+3", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def test_postflop_sequence_wraps_around_the_button():
    assert postflop_sequence([2, 5, 9], button_seat=5, max_seats=9) == [2, 0, 1]
    # dead button: the button seat is empty, the seat before it still acts last
    assert postflop_sequence([3, 4, 5, 6, 7], button_seat=1, max_seats=7) == [0, 1, 2, 3, 4]


def test_dead_button_keeps_a_btn_label_and_is_flagged():
    p = assign_positions([3, 4, 5, 6, 7], button_seat=1, max_seats=7, sb_player=0, bb_player=1)
    assert p.labels == ("SB", "BB", "HJ", "CO", "BTN") and "dead_button" in p.flags


def test_dead_small_blind_leaves_no_sb_label():
    # the first seat after the button posted the big blind; nobody posted a small blind
    p = assign_positions([1, 2, 3, 4], button_seat=4, max_seats=6, sb_player=None, bb_player=0)
    assert p.labels == ("BB", "HJ", "CO", "BTN") and p.sb is None
    assert p.preflop_order == (4, 1, 2, 3)


def test_small_blind_all_in_from_the_ante_is_taken_from_seat_order():
    p = assign_positions([1, 2, 3, 4], button_seat=4, max_seats=6, sb_player=None, bb_player=1)
    assert p.labels == ("SB", "BB", "CO", "BTN") and "sb_from_seat_order" in p.flags


def test_big_blind_all_in_from_the_ante_is_taken_from_seat_order():
    p = assign_positions([1, 2, 3, 4], button_seat=4, max_seats=6, sb_player=0, bb_player=None)
    assert p.labels == ("SB", "BB", "CO", "BTN") and "bb_from_seat_order" in p.flags


def test_straddle_moves_the_first_preflop_actor():
    p = assign_positions([1, 2, 3, 4, 5, 6], button_seat=6, max_seats=6, sb_player=0, bb_player=1,
                         straddlers=[2])
    first = min(range(6), key=lambda i: p.preflop_order[i])
    assert first == 3 and p.preflop_order[2] == 6


def test_heads_up_rows_label_the_button_btn_with_is_sb():
    r = replay_text("""
        Hand #2001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 2-max Seat #2 is the button
        Seat 1: A (100.00 in chips)
        Seat 2: B (100.00 in chips)
        B: posts small blind 0.50
        A: posts big blind 1.00
        *** HOLE CARDS ***
        B: folds
        Uncalled bet (0.50) returned to A
        A collected 1.00 from pot
        *** SUMMARY ***
        Total pot 1.00 | Rake 0.00
    """)
    assert r.players["B"]["position"] == "BTN" and r.players["B"]["is_sb"] is True
    assert r.players["A"]["position"] == "BB" and r.players["A"]["is_bb"] is True
    assert r.hand["sb_seat"] == 2 and r.hand["bb_seat"] == 1 and r.hand["button_seat"] == 2
    assert r.decisions("B")[0]["is_hu_table"] and r.decisions("B")[0]["position_class"] == "BTN"


def test_dead_button_hand_end_to_end():
    r = replay_text("""
        Hand #2002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 7-max Seat #1 is the button
        Seat 3: P3 (100.00 in chips)
        Seat 4: P4 (100.00 in chips)
        Seat 5: P5 (100.00 in chips)
        Seat 7: P7 (100.00 in chips)
        P3: posts small blind 0.50
        P4: posts big blind 1.00
        *** HOLE CARDS ***
        P5: folds
        P7: folds
        P3: folds
        Uncalled bet (0.50) returned to P4
        P4 collected 1.00 from pot
        *** SUMMARY ***
        Total pot 1.00 | Rake 0.00
    """)
    assert [r.players[x]["position"] for x in ("P3", "P4", "P5", "P7")] == ["SB", "BB", "CO", "BTN"]
    assert "DEAD_BUTTON" in r.hand["error_detail"] and not r.hand["error_flag"]
