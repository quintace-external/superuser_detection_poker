"""Standard player statistics computed directly from the replay actions."""
from __future__ import annotations

from statistics import median

from .parser import Hand


def _first(lst, pred):
    return next((x for x in lst if pred(x)), None)


def hand_flags(hand: Hand, uid: int) -> dict | None:
    """Per-hand opportunities (*_opp) and actions for one player; None if the player is not in the hand."""
    seat = hand.seat_of(uid)
    if seat is None:
        return None
    sn = seat.seat_no
    pre = hand.actions["preflop"]
    mine_pre = [a for a in pre if a.seat_no == sn]
    f = dict(win=hand.settlement.get(uid, 0.0), blind=hand.blind,
             vpip=any(a.type in (3, 4, 5, 6) for a in mine_pre),
             pfr=any(a.aggressive for a in mine_pre))

    # 3-bet: first decision facing exactly one raise (the open)
    f["threebet_opp"] = f["threebet"] = False
    raises_before = 0
    for a in pre:
        if a.seat_no == sn and raises_before == 1:
            f["threebet_opp"], f["threebet"] = True, a.aggressive
            break
        if a.aggressive and a.seat_no != sn:
            raises_before += 1
        elif a.aggressive:
            break

    # fold to 3-bet: player made the open raise and then faced a re-raise
    f["fold_to_3bet_opp"] = f["fold_to_3bet"] = False
    aggr = [a for a in pre if a.aggressive]
    if aggr and aggr[0].seat_no == sn and len(aggr) >= 2 and aggr[1].seat_no != sn:
        nxt = _first(pre[pre.index(aggr[1]) + 1:], lambda a: a.seat_no == sn)
        if nxt is not None:
            f["fold_to_3bet_opp"], f["fold_to_3bet"] = True, nxt.type == 2

    # flop c-bet and fold to c-bet
    pfa = aggr[-1].seat_no if aggr else None
    flop = hand.actions["flop"]
    f["cbet_opp"] = f["cbet"] = f["fold_to_cbet_opp"] = f["fold_to_cbet"] = False
    if pfa is not None and flop:
        first_pfa = _first(flop, lambda a: a.seat_no == pfa)
        first_aggr = _first(flop, lambda a: a.aggressive)
        if pfa == sn and first_pfa is not None and (first_aggr is None or flop.index(first_aggr) >= flop.index(first_pfa)):
            f["cbet_opp"], f["cbet"] = True, first_pfa.aggressive
        if pfa != sn and first_aggr is not None and first_aggr.seat_no == pfa and first_aggr is first_pfa:
            resp = _first(flop[flop.index(first_aggr) + 1:], lambda a: a.seat_no == sn)
            if resp is not None:
                f["fold_to_cbet_opp"], f["fold_to_cbet"] = True, resp.type == 2

    # postflop aggression
    post = [a for st in ("flop", "turn", "river") for a in hand.actions[st] if a.seat_no == sn]
    f["post_aggr"] = sum(a.aggressive for a in post)
    f["post_calls"] = sum(a.type == 3 or (a.type == 6 and not a.aggressive) for a in post)
    f["post_folds"] = sum(a.type == 2 for a in post)

    # showdown
    folded_seats = {a.seat_no for st in hand.actions for a in hand.actions[st] if a.type == 2}
    f["saw_flop"] = len(hand.board) >= 3 and not any(a.type == 2 for a in mine_pre)
    alive = [s for s in hand.seats if s not in folded_seats]
    f["wtsd"] = f["saw_flop"] and sn not in folded_seats and len(hand.board) == 5 and len(alive) >= 2
    f["wsd"] = f["wtsd"] and f["win"] > 0

    # river responses to a bet
    river = hand.actions["river"]
    f["river_faced"] = f["river_fold"] = f["river_call"] = f["river_call_won"] = 0
    last = None
    for a in river:
        if a.seat_no == sn and last is not None and last != sn:
            f["river_faced"] += 1
            if a.type == 2:
                f["river_fold"] += 1
            elif a.type == 3 or (a.type == 6 and not a.aggressive):
                f["river_call"] += 1
                f["river_call_won"] += int(f["wtsd"] and f["win"] > 0)
        if a.aggressive:
            last = a.seat_no
    f["times"] = [a.time for st in hand.actions for a in hand.actions[st] if a.seat_no == sn]
    return f


def _ratio(n, d):
    return n / d if d else None


def player_stats(hands: list[Hand], uid: int) -> dict:
    fl = [x for x in (hand_flags(h, uid) for h in hands) if x is not None]
    n = len(fl)
    s = lambda k: sum(1 for x in fl if x[k])
    times = [t for x in fl for t in x["times"]]
    aggr, calls, folds = (sum(x[k] for x in fl) for k in ("post_aggr", "post_calls", "post_folds"))
    return {
        "hands": n,
        "net_win": sum(x["win"] for x in fl),
        "bb_100": 100 * sum(x["win"] / x["blind"] for x in fl if x["blind"]) / n if n else None,
        "vpip": _ratio(s("vpip"), n),
        "pfr": _ratio(s("pfr"), n),
        "three_bet": _ratio(s("threebet"), s("threebet_opp")),
        "fold_to_3bet": _ratio(s("fold_to_3bet"), s("fold_to_3bet_opp")),
        "flop_cbet": _ratio(s("cbet"), s("cbet_opp")),
        "fold_to_cbet": _ratio(s("fold_to_cbet"), s("fold_to_cbet_opp")),
        "aggression_factor": _ratio(aggr, calls),
        "aggression_frequency": _ratio(aggr, aggr + calls + folds),
        "wtsd": _ratio(s("wtsd"), s("saw_flop")),
        "w_sd": _ratio(s("wsd"), s("wtsd")),
        "fold_to_river_bet": _ratio(sum(x["river_fold"] for x in fl), sum(x["river_faced"] for x in fl)),
        "river_call_win_rate": _ratio(sum(x["river_call_won"] for x in fl), sum(x["river_call"] for x in fl)),
        "median_think_time": median(times) if times else None,
    }
