"""Hand-history grammar, the gate (one test per reason) and the quirks that are kept with a flag."""

from __future__ import annotations

import pytest
from step01_testlib import gate, hand, replay_text

from sudetect.hhparse import GATE_REASONS, GateReject, ParsedHand, parse_amount, parse_hand, split_hands

approx = pytest.approx

BASE = """
    Hand #5000: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
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


def variant(old: str, new: str, text: str = BASE) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def test_base_hand_is_kept():
    assert gate(BASE) is None


@pytest.mark.parametrize("text, reason", [
    (variant("*** FLOP *** [2c 7d 9h]", "*** HAND CANCELLED ***"), "cancelled"),
    (variant("Hand #5000: Hold'em", "Hand 5000 Hold'em"), "bad_header"),
    (variant("Hold'em No Limit", "Omaha Pot Limit"), "unsupported_game"),
    (variant("Table 'T01' 6-max Seat #1 is the button", "Table T01 six seats"), "bad_table_line"),
    (variant("(0.50/1.00)", "(0.00/1.00)"), "unsupported_structure_no_sb"),
    (variant("    Y: posts big blind 1.00\n", ""), "no_big_blind_post"),
    (variant("""    X: raises 2.00 to 3.00
    Y: calls 2.00
    *** FLOP *** [2c 7d 9h]
    Y: checks
    X: bets 2.00
    Y: folds
    Uncalled bet (2.00) returned to X
    X collected 6.00 from pot""", "    X collected 1.50 from pot"), "no_actions"),
    (variant("    Y: checks\n", "    Y: dances on the table\n"), "unknown_grammar"),
    (variant("[2c 7d 9h]\n    Y: checks", "[2c 7d 9z]\n    Y: checks"), "bad_card"),
    (variant("Dealt to Y [Qs Qc]", "Dealt to Y [Qs Ah]"), "bad_card"),
    (variant("[2c 7d 9h]\n    Y: checks", "[2c 7d Ah]\n    Y: checks"), "bad_card"),
    (variant("Seat 2: Y (100.00 in chips)", "Seat 2: Y (100.00 in chips)\n    Seat 3: Z (50.00 in chips)",
             variant("X collected 6.00 from pot", "Z collected 6.00 from pot")), "payee_not_dealt"),
    (variant("X: bets 2.00", "X: bets 200.00"), "stack_inconsistent"),
    (variant("Y: calls 2.00", "Y: calls 2.00 and is all-in"), "stack_inconsistent"),
    (variant("    Y: folds\n", "    Y: folds\n    Y: checks\n"), "action_order_invalid"),
    (variant("    Y: folds\n", "    Y: checks\n    Y: folds\n"), "amount_inconsistent"),
    (variant("    Y: folds\n", "    Y: calls 1.00\n    Y: folds\n"), "amount_inconsistent"),
    (variant("X collected 6.00 from pot", "X collected 5.99 from pot"), "money_unreconciled"),
    (variant("Total pot 6.00 | Rake 0.00", "Board line only"), "money_unreconciled"),
])
def test_gate_reasons(text, reason):
    assert reason in GATE_REASONS
    assert gate(text) == reason


HIDDEN_POST = """
    Hand #5001: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: calls 0.20
    Y: checks
    *** FLOP *** [2c 7d 9h]
    Y: bets 2.00
    X: folds
    Uncalled bet (2.00) returned to Y
    Y collected 2.00 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
    Board [2c 7d 9h]
"""


def test_hidden_post_is_dropped_by_default_and_flagged_when_kept():
    assert gate(HIDDEN_POST) == "hidden_post_inferred"
    assert gate(HIDDEN_POST, infer_hidden_posts=False) == "money_unreconciled"
    r = replay_text(HIDDEN_POST, drop_hidden_post_hands=False)
    assert r.hand["error_flag"] and "HIDDEN_POST_INFERRED" in r.hand["error_detail"]
    hidden = [a for a in r.actions if a["post_kind"] == "hidden"]
    assert len(hidden) == 1 and hidden[0]["amount_bb"] == approx(0.3) and hidden[0]["player_id"] == "X"
    assert r.players["X"]["committed_bb"] == approx(1.0)


def test_uncalled_amount_printed_at_half_is_kept_with_correct_nets():
    text = """
        Hand #5002: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 2-max Seat #2 is the button
        Seat 1: Y (100.00 in chips)
        Seat 2: X (100.00 in chips)
        X: posts small blind 0.50
        Y: posts big blind 1.00
        *** HOLE CARDS ***
        X: raises 3.00 to 4.00
        Y: folds
        Uncalled bet (1.50) returned to X
        X collected 3.50 from pot
        *** SUMMARY ***
        Total pot 2.00 | Rake 0.00
    """
    r = replay_text(text)
    assert r.players["X"]["net_bb"] == approx(1.0) and r.players["Y"]["net_bb"] == approx(-1.0)
    assert r.hand["uncalled_bb"] == approx(3.0) and r.hand["final_pot_bb"] == approx(2.0)
    assert "UNCALLED_TEXT_DIFFERS" in r.hand["error_detail"] and not r.hand["error_flag"]


def test_negative_rake_and_other_drops_keep_the_money_identity():
    bonus = variant("X collected 6.00 from pot", "X collected 6.00 from pot\n    Y collected 0.50 from pot",
                    variant("Rake 0.00", "Rake -0.50"))
    r = replay_text(bonus)
    assert r.hand["rake_bb"] == approx(-0.5) and "NEGATIVE_RAKE" in r.hand["error_detail"]
    assert sum(p["net_bb"] for p in r.players.values()) + r.hand["rake_bb"] == approx(0.0)
    drops = variant("X collected 6.00 from pot", "X collected 5.70 from pot",
                    variant("Rake 0.00", "Rake 0.20 | Jackpot 0.10"))
    r = replay_text(drops)
    assert r.hand["rake_bb"] == approx(0.2) and r.hand["fees_bb"] == approx(0.1)
    assert "OTHER_DROPS" in r.hand["error_detail"]
    assert sum(p["net_bb"] for p in r.players.values()) + 0.3 == approx(0.0)
    assert r.players["X"]["rake_share_bb"] == approx(0.2)
    assert r.players["X"]["net_pre_rake_bb"] == approx(r.players["X"]["net_bb"] + 0.2)
    printed = replay_text(variant("Total pot 6.00", "Total pot 7.00"))
    assert "TOTAL_POT_DIFFERS" in printed.hand["error_detail"] and not printed.hand["error_flag"]


GRAMMAR = """
    Sample Hand #A-77: Hold'em No Limit ($5/$10 - Ante $1 USD) - 2001/02/03 4:05:06 ET
    Table 'Big Table' 9-max (Real Money) Seat #2 is the button
    Seat 1: Mr: X ($1,500 in chips)
    Seat 2: big shot ($300.50 in chips)
    Seat 3: Z ($2,000 in chips) is sitting out
    Seat 4: Q ($1,200 in chips)
    Mr: X: posts the ante $1
    big shot: posts the ante $1
    Q: posts ante $1
    Q: posts small blind $5
    Mr: X: posts big blind $10
    *** HOLE CARDS ***
    Dealt to big shot [Ah Kh]
    big shot: raises $20 to $30
    big shot said, "gl: hf"
    Z joins the table at seat #3
    Q: folds [7c 2d]
    Q is disconnected
    Mr: X: raises $70 to $100
    big shot: raises $269.50 and is all-in
    Mr: X has timed out
    Mr: X: calls $199.50
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    big shot: shows [Ah Kh] (high card Ace)
    Mr: X: shows [Qd Qs] (a pair of Queens)
    Mr: X collected $603 from pot
    *** SUMMARY ***
    Total pot $607 | Rake $3 | Jackpot $1
    Board [2c 7d 9h Ts 3c]
    Seat 1: Mr: X (big blind) showed [Qd Qs] and won ($603) with a pair of Queens
    Seat 2: big shot (button) showed [Ah Kh] and lost with high card Ace
    Seat 3: Z is sitting out
    Seat 4: Q (small blind) folded before Flop
"""


def test_general_grammar_currency_names_legacy_raise_and_info_lines():
    parsed = parse_hand(hand(GRAMMAR))
    assert isinstance(parsed, ParsedHand), parsed
    assert parsed.hand_id == "A-77" and parsed.players == ["Mr: X", "big shot", "Q"]
    assert parsed.bb == 1000 and parsed.sb == 500 and parsed.ante == 100
    assert parsed.stacks == [150000, 30050, 120000]
    assert "time_zone_not_utc" in parsed.flags and parsed.time_zone == "ET"
    r = replay_text(GRAMMAR)
    shove = r.decisions("big shot")[1]
    assert shove["amount_bb"] == approx(26.95) and shove["raise_to_eff_bb"] == approx(29.95)
    assert shove["is_allin"] and shove["action"] == "raise"
    call = r.decisions("Mr: X")[1]
    assert call["auto_kind"] == "unknown"  # a timeout followed by a call is the player's own action
    assert call["amount_bb"] == approx(19.95) and call["pot_odds"] == approx(19.95 / 60.7)
    assert r.hand["n_dealt"] == 3 and r.hand["ante_bb"] == approx(0.1) and r.hand["sb_bb"] == approx(0.5)
    assert r.hand["cards_visibility"] == "showdown"
    assert r.players["big shot"]["cards_source"] == "dealt" and r.players["Mr: X"]["cards_source"] == "showdown"
    assert r.players["Q"]["hole_cards"] is None and r.players["Q"]["cards_known"] is False
    assert r.players["Mr: X"]["position"] == "BB" and r.players["Q"]["position"] == "SB"
    assert r.hand["rake_bb"] == approx(0.3) and r.hand["fees_bb"] == approx(0.1)
    assert r.players["Mr: X"]["net_bb"] == approx(30.25)
    assert "TIME_ZONE_NOT_UTC" in r.hand["error_detail"]


SIDE_POTS = """
    Hand #5003: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
    Table 'T01' 6-max Seat #3 is the button
    Seat 1: S (20.00 in chips)
    Seat 2: M (50.00 in chips)
    Seat 3: L (100.00 in chips)
    S: posts small blind 0.50
    M: posts big blind 1.00
    *** HOLE CARDS ***
    L: raises 99.00 to 100.00 and is all-in
    S: calls 19.50 and is all-in
    M: calls 49.00 and is all-in
    Uncalled bet (50.00) returned to L
    *** FLOP *** [2c 7d 9h]
    *** TURN *** [2c 7d 9h] [Ts]
    *** RIVER *** [2c 7d 9h Ts] [3c]
    *** SHOW DOWN ***
    S: shows [Ac Ad] (a pair of Aces)
    M: shows [Kc Kd] (a pair of Kings)
    L: shows [Qc Qd] (a pair of Queens)
    M collected 60.00 from side-pot 1
    S collected 60.00 from main pot
    *** SUMMARY ***
    Total pot 120.00 | Rake 0.00
    Board [2c 7d 9h Ts 3c]
"""


def test_side_pots_and_collect_variants():
    r = replay_text(SIDE_POTS)
    assert r.hand["n_side_pots"] == 1 and r.hand["final_pot_bb"] == approx(120.0)
    assert r.hand["uncalled_bb"] == approx(50.0) and r.hand["uncalled_seat"] == 3
    assert r.players["S"]["net_bb"] == approx(40.0) and r.players["M"]["net_bb"] == approx(10.0)
    assert r.players["L"]["net_bb"] == approx(-50.0)
    s_call = r.decisions("S")[0]
    # S is short: of the chips in so far he can win 20 from himself and L, and the 1 of M
    assert s_call["to_call_eff_bb"] == approx(19.5) and s_call["pot_odds"] == approx(19.5 / 41.0)
    m_call = r.decisions("M")[0]
    assert m_call["amount_eff_bb"] == approx(49.0) and m_call["pot_odds"] == approx(49 / 120)
    shove = r.decisions("L")[0]
    assert shove["is_capped"] and shove["raise_to_eff_bb"] == approx(50.0)
    assert all(p["went_to_showdown"] for p in r.players.values()) and r.hand["n_showdown"] == 3


def test_summary_cards_are_a_fallback_and_shown_without_showdown():
    text = variant("Y: folds\n    Uncalled bet (2.00) returned to X\n    X collected 6.00 from pot",
                   "Y: folds\n    Uncalled bet (2.00) returned to X\n    X collected 6.00 from pot\n"
                   "    X: shows [Ah Kd] (high card Ace)")
    text = text.replace("    Dealt to X [Ah Kd]\n    Dealt to Y [Qs Qc]\n", "")
    r = replay_text(text)
    assert r.players["X"]["cards_source"] == "shown" and r.players["X"]["cards_shown"]
    assert r.players["X"]["went_to_showdown"] is False and r.hand["n_shown"] == 1
    assert r.players["Y"]["hole_cards"] is None


def test_a_hand_where_nobody_had_to_act_is_kept():
    text = """
        Hand #5004: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 2-max Seat #2 is the button
        Seat 1: A (0.57 in chips)
        Seat 2: B (0.36 in chips)
        A: posts the ante 0.25
        B: posts the ante 0.25
        B: posts small blind 0.11 and is all-in
        A: posts big blind 0.32 and is all-in
        *** HOLE CARDS ***
        Uncalled bet (0.21) returned to A
        *** FLOP *** [Ts 6c 7s]
        *** TURN *** [Ts 6c 7s] [7h]
        *** RIVER *** [Ts 6c 7s 7h] [9h]
        *** SHOW DOWN ***
        A: shows [8s 9s] (a straight, Six to Ten)
        B: mucks hand
        A collected 0.72 from pot
        *** SUMMARY ***
        Total pot 0.72 | Rake 0.00
        Board [Ts 6c 7s 7h 9h]
    """
    r = replay_text(text)
    assert r.decisions() == [] and r.hand["went_to_showdown"]
    assert r.players["A"]["net_bb"] == approx(0.36) and r.players["B"]["cards_shown"] is False
    assert r.players["B"]["went_to_showdown"] is True


def test_short_big_blind_is_called_at_the_full_blind_while_others_can_act():
    text = """
        Hand #5005: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 6-max Seat #3 is the button
        Seat 3: BT (11.84 in chips)
        Seat 5: SB (0.96 in chips)
        Seat 6: BB (0.84 in chips)
        SB: posts the ante 0.10
        BB: posts the ante 0.10
        BT: posts the ante 0.10
        SB: posts small blind 0.50
        BB: posts big blind 0.74 and is all-in
        *** HOLE CARDS ***
        BT: calls 1.00
        SB: calls 0.36 and is all-in
        Uncalled bet (0.14) returned to BT
        *** FLOP *** [6s 4s 4h]
        *** TURN *** [6s 4s 4h] [8h]
        *** RIVER *** [6s 4s 4h 8h] [Qh]
        *** SHOW DOWN ***
        SB: shows [7d Kc] (a pair of Fours)
        BB: shows [Ad 5h] (a pair of Fours)
        BT: mucks hand
        SB collected 0.24 from side pot-1
        BB collected 2.52 from main pot
        *** SUMMARY ***
        Total pot 2.76 | Rake 0.00
        Board [6s 4s 4h 8h Qh]
    """
    r = replay_text(text)
    call = r.decisions("BT")[0]
    assert call["to_call_bb"] == approx(1.0) and call["to_call_eff_bb"] == approx(1.0)
    sb = r.decisions("SB")[0]
    assert sb["to_call_eff_bb"] == approx(0.36) and sb["is_allin"]
    assert r.hand["uncalled_bb"] == approx(0.14) and r.hand["n_side_pots"] == 1
    assert not r.hand["error_flag"]


def test_parse_amount():
    assert parse_amount("1.50") == 150 and parse_amount("$1,234.5") == 123450
    assert parse_amount("-0.20") == -20 and parse_amount("€7") == 700 and parse_amount("0.010") == 1
    assert parse_amount("12", decimals=0) == 12
    with pytest.raises(ValueError):
        parse_amount("0.016")
    with pytest.raises(ValueError):
        parse_amount("abc")


def test_split_hands_ignores_preambles_and_chat_lines():
    one = hand(BASE)
    text = "export of the day\n\n" + one + "\n\n" + one.replace("#5000", "#5001").replace(
        "    Y: checks\n", "") + "\n" + one.replace("#5000", "#5002")
    chat = 'Y said, "Hand #5000: what a hand - 2001/01/01"'
    text = text.replace("Y: checks", "Y: checks\n" + chat, 1)
    hands = split_hands("﻿" + text.replace("\n", "\r\n"))
    assert [h.split(":")[0] for h in hands] == ["Hand #5000", "Hand #5001", "Hand #5002"]
    assert isinstance(parse_hand(hands[0]), ParsedHand)


def test_unknown_grammar_never_raises():
    for bad in ("", "\n\n", "Hand #1: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC\n",
                hand(BASE).replace("Seat 1: X (100.00 in chips)", "Seat 1: X (lots in chips)")):
        assert isinstance(parse_hand(bad), GateReject)


SECOND_BIG_BLIND = """
    Hand #5006: Hold'em No Limit (1.00/2.00) - 2001/01/01 10:00:00 UTC
    Table 'T02' 6-max Seat #6 is the button
    Seat 1: SB (200.00 in chips)
    Seat 2: BB (200.00 in chips)
    Seat 4: W (200.00 in chips)
    Seat 5: R (200.00 in chips)
    Seat 6: BT (200.00 in chips)
    Seat 3: AWAY (200.00 in chips) is sitting out
    SB: posts the ante 0.20
    BB: posts the ante 0.20
    W: posts the ante 0.20
    R: posts the ante 0.20
    BT: posts the ante 0.20
    AWAY: didn't post the ante
    SB: posts small blind 1.00
    BB: posts big blind 2.00
    R: posts big blind 2.00
    *** HOLE CARDS ***
    W: calls 2.00
    R: checks
    BT: folds
    SB: folds
    BB: raises 6.00 to 8.00
    W: folds
    R: folds
    Uncalled bet (6.00) returned to BB
    BB collected 8.00 from pot
    BB: doesn't show hand
    *** SUMMARY ***
    Total pot 8.00 | Rake 0.00
    Game ended: 2001/01/01 10:01:00 UTC
    Seat 5: R folded before Flop
"""


def test_a_second_big_blind_post_is_live_and_only_the_first_names_the_big_blind():
    r = replay_text(SECOND_BIG_BLIND)
    assert r.hand["n_dealt"] == 5 and "AWAY" not in r.players
    assert r.players["BB"]["is_bb"] and r.players["BB"]["position"] == "BB"
    assert r.players["R"]["post_kind"] == "post" and r.players["R"]["position"] == "CO"
    assert r.players["W"]["position"] == "HJ"
    check = r.decisions("R")[0]
    assert check["action"] == "check" and check["to_call_bb"] == 0 and check["street_put_before_bb"] == 1.0
    assert r.players["BB"]["net_bb"] == approx(4.0 - 4.1 + 3.0)


def test_an_actor_without_a_seat_line_is_dropped():
    text = """
        Hand #5007: Hold'em No Limit (0.50/1.00) - 2001/01/01 10:00:00 UTC
        Table 'T01' 6-max Seat #1 is the button
        Seat 1: X (100.00 in chips)
        Seat 2: Y (100.00 in chips)
        Y: posts small blind 0.50
        Z: posts big blind 1.00
        *** HOLE CARDS ***
        X: raises 2.00 to 3.00
        Y: folds
        Z: folds
        Uncalled bet (2.00) returned to X
        X collected 2.50 from pot
        *** SUMMARY ***
        Total pot 2.50 | Rake 0.00
    """
    assert gate(text) == "no_big_blind_post"
    text = variant("    Y: checks\n", "    Ghost: checks\n")
    assert gate(text) == "unknown_grammar"
