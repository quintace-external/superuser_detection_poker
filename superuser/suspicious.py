"""Suspicious-hand detection and decision labels."""
from __future__ import annotations

import pandas as pd

from .cards import CATEGORY_NAMES, evaluate, to_pretty
from .parser import BOARD_LEN, Hand

CATEGORIES = {
    "A": "River fold when beaten",
    "B": "River bluff-catch",
    "C": "Successful river bluff",
    "D": "Early fold, near dead",
}
LABEL_ORDER = {"Perfect": 0, "Good": 1, "Average": 2}
SNAP_SECONDS, SNAP_POT_BB = 2, 30        # snap decision: Perfect label, <= 2 s, pot >= 30 bb


def _tier(cat: int | None) -> int:
    if cat is None:
        return 0
    return 0 if cat == 0 else 1 if cat == 1 else 2 if cat == 2 else 3


def _category(hand: Hand, row: dict) -> str | None:
    eq, st = row["hero_eq"], row["street"]
    if eq is None:
        return None
    if st == "river" and row["facing_bet"] and row["is_fold"] and eq <= 1e-9:
        return "A"
    if st == "river" and row["facing_bet"] and row["is_continue"] and eq >= 0.999:
        return "B"
    if st == "river" and row["is_aggr"] and eq < 0.5 and row["bluff_success"]:
        return "C"
    if st in ("flop", "turn") and row["is_fold"] and eq < 0.10:
        return "D"
    return None


def _opponent_uid(hand: Hand, row: dict, cat: str, uid: int) -> int | None:
    if cat in ("A", "B", "D") and row.get("bettor_uid") is not None:
        return int(row["bettor_uid"])
    board = hand.board[:BOARD_LEN[row["street"]]]
    others = [s for s in hand.seats.values() if s.uid != uid and s.hole]
    if not others:
        return None
    return max(others, key=lambda s: evaluate(board + s.hole)).uid


def _label(cat: str, eq: float, tier: int, opp_tier: int) -> str:
    if cat in ("A", "D"):
        if eq <= 0.05 and tier >= 1:
            return "Perfect fold"
        if eq <= 0.05 or (eq <= 0.20 and tier >= 1):
            return "Good fold"
        return "Average fold"
    if cat == "B":
        return "Perfect call" if tier <= 1 else "Good call" if tier == 2 else "Average call"
    if tier == 0 and opp_tier >= 1:
        return "Perfect bluff"
    return "Good bluff" if opp_tier >= 1 else "Average bluff"


def find_suspicious(hands: dict[str, Hand], dec: pd.DataFrame, uid: int) -> pd.DataFrame:
    """One row per suspicious hand (first qualifying decision, priority A > B > C > D).
    Sorted by decision label (Perfect, Good, Average), then pot size."""
    mine = dec[dec["uid"] == uid]
    out = []
    for gid, grp in mine.groupby("hand_uid", sort=False):
        hand = hands[gid]
        rows = grp.to_dict("records")
        found = [(c, r) for r in rows if (c := _category(hand, r))]
        if not found:
            continue
        cat, row = min(found, key=lambda x: "ABCD".index(x[0]))
        seat = hand.seat_of(uid)
        board = hand.board[:BOARD_LEN[row["street"]]]
        hero_cat = evaluate(board + seat.hole)[0] if len(board) >= 3 else None
        opp = _opponent_uid(hand, row, cat, uid)
        opp_seat = hand.seat_of(opp) if opp is not None else None
        opp_cat = evaluate(board + opp_seat.hole)[0] if opp_seat and opp_seat.hole and len(board) >= 3 else None
        tier, opp_tier = _tier(hero_cat), _tier(opp_cat)
        eq = row["hero_eq"]
        label = _label(cat, eq, tier, opp_tier)
        pot_bb = hand.contested_pot() / hand.blind if hand.blind else 0.0
        snap = label.startswith("Perfect") and row["think_time"] <= SNAP_SECONDS and pot_bb >= SNAP_POT_BB

        bet = row.get("bet_faced")
        if cat in ("A", "D", "B") and bet:
            pot_pre = max(row["pot_before"] - bet, 1)
            verb = "folded" if cat != "B" else f"{row['action']} {row['amount']:,.0f}"
            key = f"faced {bet:,.0f} into ~{pot_pre:,.0f} ({bet / pot_pre:.0%} pot), {verb}"
            bet_to_pot = bet / pot_pre
        else:
            pot_pre = max(row["pot_before"], 1)
            key = f"{row['action']} {row['amount']:,.0f} into ~{pot_pre:,.0f} ({row['amount'] / pot_pre:.0%} pot), opponent folded"
            bet_to_pot = row["amount"] / pot_pre

        out.append({
            "hand_uid": gid,
            "start_utc": pd.to_datetime(hand.start_ts, unit="s"),
            "category": cat,
            "category_name": CATEGORIES[cat],
            "decision_label": label,
            "street": row["street"],
            "key_action": key,
            "bet_to_pot": round(bet_to_pot, 2),
            "decision_time_s": row["think_time"],
            "snap_decision": snap,
            "equity_at_decision": round(eq, 3),
            "player_cards": to_pretty(seat.hole),
            "opponent_uid": opp,
            "opponent_cards": to_pretty(opp_seat.hole) if opp_seat else "",
            "board": to_pretty(hand.board),
            "player_hand": CATEGORY_NAMES[hero_cat] if hero_cat is not None else "",
            "opponent_hand": CATEGORY_NAMES[opp_cat] if opp_cat is not None else "",
            "pot_bb": round(pot_bb, 1),
            "net_win": hand.settlement.get(uid, 0.0),
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["_order"] = df["decision_label"].str.split().str[0].map(LABEL_ORDER)
    return (df.sort_values(["_order", "pot_bb"], ascending=[True, False])
              .drop(columns="_order").reset_index(drop=True))
