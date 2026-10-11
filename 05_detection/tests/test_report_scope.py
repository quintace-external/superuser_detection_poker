"""The per-suspect report at tables of any size: which hands ``adapter.hands_in_scope`` reads under each scope, and the
opponent the package names for a flagged decision when players have folded before it."""

from __future__ import annotations

import pytest
from test_screen_formulas import tiny_warehouse

import adapter                       # 05_detection/adapter.py
from sudetect import warehouse
from superuser.parser import Action, Hand, Seat
from superuser.suspicious import _opponent_uid


def scope_warehouse():
    """The screen's tiny warehouse (H1, H2 heads-up table; H3 three dealt, P1 and P3 see the flop; H4 three see the
    flop; H5 three dealt, no flop) with the columns the hand query reads."""
    con = tiny_warehouse()
    warehouse.execute(con, "ALTER TABLE hands ADD COLUMN variant VARCHAR DEFAULT 'standard'")
    warehouse.execute(con, "ALTER TABLE hands ADD COLUMN error_flag BOOLEAN DEFAULT FALSE")
    return con


@pytest.mark.parametrize("scope, expected", [
    ("flop_hu", {"H1", "H2", "H3"}),          # heads-up table, and the three-handed hand two players took to the flop
    ("hu_tables", {"H1", "H2"}),              # the article's scope
    ("all", {"H1", "H2", "H3", "H4", "H5"}),  # every hand, multiway pots included
    (True, {"H1", "H2"}),                     # the old heads_up_only flag
    (False, {"H1", "H2", "H3", "H4", "H5"}),
])
def test_hands_in_scope(scope, expected):
    con = scope_warehouse()
    try:
        assert set(adapter.hands_in_scope(con, "P1", scope=scope)) == expected
    finally:
        con.close()


def test_flop_hu_needs_the_player_on_the_flop():
    con = scope_warehouse()
    try:
        assert "H3" not in adapter.hands_in_scope(con, "P2", scope="flop_hu")   # P2 folded H3 preflop
        assert set(adapter.hands_in_scope(con, "P3", scope="flop_hu")) == {"H3"}
    finally:
        con.close()


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        adapter.scope_name("six_max")


def test_bluff_opponent_is_a_live_player():
    """Three-handed: seat 3 folds aces preflop, seat 1 bluffs the river and seat 2 folds. The opponent of the bluff
    is seat 2, not the player with the strongest cards on the board."""
    cards = adapter.module_cards
    seats = {1: Seat(1, 1, 10000, "BTN", cards("7c6c")), 2: Seat(2, 2, 10000, "SB", cards("KdQd")),
             3: Seat(3, 3, 10000, "BB", cards("AhAs"))}
    acts = {"preflop": [Action("preflop", 1, 3, 3, 2, 0, 1.0), Action("preflop", 2, 1, 1, 5, 300, 1.0, True),
                        Action("preflop", 3, 2, 2, 3, 300, 1.0)],
            "flop": [], "turn": [],
            "river": [Action("river", 10, 1, 1, 4, 600, 1.0, True), Action("river", 11, 2, 2, 2, 0, 1.0)]}
    hand = Hand(hand_uid="X", start_ts=0, end_ts=0, table_no=1, stake_level=None, chip_type=None, blind=100, ante=0,
                game_type="No-Limit Hold'em", seats=seats, actions=acts, board=cards("2h9sJc3d4h"),
                street_end_pot={}, settlement={})
    row = {"street": "river", "seq": 10, "bettor_uid": None}
    assert _opponent_uid(hand, row, "C", 1) == 2
