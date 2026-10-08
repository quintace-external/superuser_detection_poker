"""One row per action, with equity against the real hole cards of every player still in the hand."""
from __future__ import annotations

import random

from .cards import evaluate
from .parser import BOARD_LEN, STREETS, Hand

DECK = list(range(52))


def _equity(hand: Hand, seat: int, opps: list[int], board: list[int], rng: random.Random,
            flop_samples: int, cache: dict) -> float:
    """Share of the pot the player wins against `opps`: exact on turn/river, sampled on the flop."""
    key = (seat, len(board), tuple(sorted(opps)))
    if key in cache:
        return cache[key]
    h = hand.seats[seat].hole
    oh = [hand.seats[o].hole for o in opps]
    used = set(h + board + [c for o in oh for c in o])
    rest = [c for c in DECK if c not in used]
    need = 5 - len(board)
    if need == 0:
        runs = [[]]
    elif need == 1:
        runs = [[c] for c in rest]
    else:
        runs = [rng.sample(rest, need) for _ in range(flop_samples)]
    total = 0.0
    for r in runs:
        b = board + r
        hs = evaluate(b + h)
        os_ = [evaluate(b + o) for o in oh]
        best = max([hs] + os_)
        if hs == best:
            total += 1.0 / (1 + sum(1 for x in os_ if x == best))
    cache[key] = total / len(runs)
    return cache[key]


def decision_rows(hand: Hand, flop_samples: int = 200) -> list[dict]:
    """Rows for every action of every player. Postflop rows carry equity fields."""
    rng = random.Random(hand.hand_uid)
    cache: dict = {}
    folded: set[int] = set()
    rows: list[dict] = []
    for st in STREETS:
        acts = hand.actions[st]
        if not acts:
            continue
        if len(hand.board) < BOARD_LEN[st]:
            break
        board = hand.board[:BOARD_LEN[st]]
        pot = hand.street_end_pot.get(STREETS[STREETS.index(st) - 1], 0.0) if st != "preflop" else 0.0
        last_aggr = None
        street_rows = []
        for a in acts:
            opps = [s for s in hand.seats if s != a.seat_no and s not in folded]
            facing = last_aggr is not None and last_aggr["seat"] != a.seat_no
            eq = pair_plus = None
            if st != "preflop" and hand.seats.get(a.seat_no) and hand.seats[a.seat_no].hole and opps \
                    and all(hand.seats[o].hole for o in opps):
                eq = _equity(hand, a.seat_no, opps, board, rng, flop_samples, cache)
                pair_plus = evaluate(board + hand.seats[a.seat_no].hole)[0] >= 1
            row = dict(hand_uid=hand.hand_uid, uid=a.uid, seat_no=a.seat_no, street=st, seq=a.seq,
                       action=a.name, amount=a.amount, think_time=a.time,
                       is_fold=a.type == 2, is_aggr=a.aggressive, is_continue=a.type in (3, 5, 6),
                       hero_eq=eq, hero_pair_plus=pair_plus, facing_bet=facing,
                       bettor_uid=last_aggr["uid"] if facing else None,
                       bettor_eq=last_aggr["eq"] if facing else None,
                       bet_faced=last_aggr["amount"] if facing else None,
                       pot_before=pot, bluff_success=None, n_active=len(opps) + 1)
            if a.aggressive:
                last_aggr = {"seat": a.seat_no, "uid": a.uid, "eq": eq, "amount": a.amount}
            pot += a.amount
            rows.append(row)
            street_rows.append((a.seat_no, a.type, row))
            if a.type == 2:
                folded.add(a.seat_no)
        for i, (sn, _, row) in enumerate(street_rows):    # bluff success: every later action by others is a fold
            if row["is_aggr"]:
                later = [t for (s2, t, _) in street_rows[i + 1:] if s2 != sn]
                row["bluff_success"] = bool(later) and all(t == 2 for t in later)
    return rows
