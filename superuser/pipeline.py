"""End-to-end analysis for one player from hand records."""
from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .decisions import decision_rows
from .parser import Hand, parse_hand
from .signals import player_signals
from .stats import player_stats
from .suspicious import find_suspicious


def load_hands(rows: Iterable[Mapping], uid: int, opponent: int | None = None, min_blind: float | None = None,
               heads_up_only: bool = False, chip_type=None) -> tuple[dict[str, Hand], dict]:
    hands, meta = {}, {"rows": 0, "unparseable": 0, "not_in_hand": 0, "filtered": 0}
    for row in rows:
        meta["rows"] += 1
        h = parse_hand(row)
        if h is None:
            meta["unparseable"] += 1
            continue
        if uid not in h.uids or (opponent is not None and opponent not in h.uids):
            meta["not_in_hand"] += 1
            continue
        if (min_blind and h.blind < min_blind) or (heads_up_only and h.n_players != 2) \
                or (chip_type is not None and str(h.chip_type) != str(chip_type)):
            meta["filtered"] += 1
            continue
        hands[h.hand_uid] = h
    meta["hands_analysed"] = len(hands)
    return hands, meta


def analyse_player(rows: Iterable[Mapping] | pd.DataFrame, uid: int, opponent: int | None = None,
                   min_blind: float | None = None, heads_up_only: bool = False, chip_type=None,
                   flop_samples: int = 200) -> dict:
    """rows: hand records (dicts or a DataFrame with hand_uid, hand_replay, hand_result, start_ts, ...).
    opponent: if given, only hands where both players sat are analysed."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    hands, meta = load_hands(rows, uid, opponent, min_blind, heads_up_only, chip_type)
    dec = pd.DataFrame([r for h in hands.values() for r in decision_rows(h, flop_samples)])
    if dec.empty:
        dec = pd.DataFrame(columns=["hand_uid", "uid", "street", "hero_eq", "is_fold", "is_aggr", "is_continue",
                                    "hero_pair_plus", "facing_bet", "bettor_eq", "bluff_success", "think_time"])
    hand_list = list(hands.values())
    suspicious = find_suspicious(hands, dec, uid)
    return {
        "player": uid,
        "opponent": opponent,
        "meta": meta,
        "stats": player_stats(hand_list, uid),
        "signals": player_signals(dec, uid),
        "suspicious_hands": suspicious,
        "escalation": escalation(hand_list, suspicious),
        "decisions": dec,
        "hands": hands,
    }


def escalation(hands: list[Hand], suspicious: pd.DataFrame, window_days: int = 15,
               min_ratio: float = 0.05, min_count: int = 5) -> dict:
    """Stage 5 rule over the last `window_days` of submitted hands:
    suspicious / total > min_ratio and suspicious > min_count."""
    if not hands:
        return {"window_days": window_days, "hands": 0, "suspicious": 0, "ratio": None, "escalate": False}
    end = max(h.start_ts for h in hands)
    start = end - window_days * 86400
    in_window = {h.hand_uid for h in hands if h.start_ts >= start}
    k = int(suspicious["hand_uid"].isin(in_window).sum()) if not suspicious.empty else 0
    ratio = k / len(in_window)
    return {"window_days": window_days, "window_start_utc": str(pd.to_datetime(start, unit="s")),
            "window_end_utc": str(pd.to_datetime(end, unit="s")), "hands": len(in_window), "suspicious": k,
            "ratio": ratio, "escalate": ratio > min_ratio and k > min_count}
