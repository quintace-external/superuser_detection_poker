"""Run the per-suspect forensic package (``superuser/`` at the repository root) on the warehouse tables.

The module was written for hand-replay records. This adapter builds the two inputs it needs straight from the
step 01 and step 02 tables, so the module runs unchanged on any dataset of this repository:

* ``dec`` - one row per action with the module's column names (``hero_eq`` = equity vs the real cards of every
  player still in the hand, ``facing_bet``, ``bettor_eq``, ``bluff_success``, ...);
* ``hands`` - ``superuser.parser.Hand`` objects (seats, hole cards, board, actions by street, net results).

Amounts are converted to chips with the big blind = 100. The module expects integer player ids: numeric ids are
kept as they are, any other ids are numbered 1, 2, ... and mapped back in the suspicious-hands table. The text
format carries no decision times; ``times`` (from ``01_action_table/from_replay_json.py``) puts them back.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from superuser.cards import evaluate  # noqa: E402
from superuser.parser import Action, Hand, Seat  # noqa: E402
from superuser.pipeline import escalation  # noqa: E402
from superuser.signals import player_signals  # noqa: E402
from superuser.stats import player_stats  # noqa: E402
from superuser.suspicious import find_suspicious  # noqa: E402

CHIPS_PER_BB = 100.0
RANKS = "23456789TJQKA"
MODULE_SUITS = "dchs"                      # the module's suit order
TYPE = {"check": 1, "fold": 2, "call": 3, "bet": 4, "raise": 5}
STREETS = ("preflop", "flop", "turn", "river")


def module_card(token: str) -> int:
    """'Ah' -> the module's card id (rank * 4 + suit, suits in d, c, h, s order)."""
    return RANKS.index(token[0]) * 4 + MODULE_SUITS.index(token[1])


def module_cards(text: str | None) -> list[int]:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return []
    return [module_card(text[i:i + 2]) for i in range(0, len(text), 2)]


def _chips(bb) -> float:
    return 0.0 if bb is None or (isinstance(bb, float) and math.isnan(bb)) else float(bb) * CHIPS_PER_BB


SCOPES = ("flop_hu", "hu_tables", "all")


def scope_name(scope: str | bool) -> str:
    """A scope name; the old ``heads_up_only`` flag maps True -> 'hu_tables', False -> 'all'."""
    if isinstance(scope, bool):
        return "hu_tables" if scope else "all"
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}: {scope!r}")
    return scope


def hands_in_scope(con, player: str, opponent: str | None = None, scope: str | bool = "flop_hu") -> list[str]:
    """The player's hands the analysis reads (standard variant, no error):

    * ``flop_hu`` (default, the scope of ``screen.py``): every hand dealt to two players, and at larger tables every
      hand he took to the flop with exactly one other player. A pot three or more players see the flop of is left
      out: the categories read equity against every live opponent, which is not what they were built for;
    * ``hu_tables``: hands dealt to exactly two players (the article's scope);
    * ``all``: every hand, multiway pots included.
    """
    scope = scope_name(scope)
    sql = """
        SELECT DISTINCT p.hand_id
        FROM hand_players p JOIN hands h USING (hand_id)
        WHERE p.player_id = $player AND h.variant = 'standard' AND NOT h.error_flag
    """
    params = {"player": player}
    if scope == "hu_tables":
        sql += " AND h.n_dealt = 2"
    elif scope == "flop_hu":
        sql += """ AND (h.n_dealt = 2 OR (p.saw_flop AND (SELECT count(*) FROM hand_players f
                                                           WHERE f.hand_id = p.hand_id AND f.saw_flop) = 2))"""
    if opponent is not None:
        sql += " AND EXISTS (SELECT 1 FROM hand_players q WHERE q.hand_id = p.hand_id AND q.player_id = $opponent)"
        params["opponent"] = opponent
    return [r[0] for r in con.execute(sql, params).fetchall()]


def load_tables(con, hand_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con.execute("CREATE OR REPLACE TEMP TABLE _scope AS SELECT unnest($ids) AS hand_id", {"ids": hand_ids})
    hands = con.execute("SELECT h.* FROM hands h JOIN _scope USING (hand_id)").df()
    players = con.execute("SELECT p.* FROM hand_players p JOIN _scope USING (hand_id)").df()
    actions = con.execute("""
        SELECT a.hand_id, a.action_index, a.player_id, a.seat, a.street, a.action, a.is_allin, a.decision_s,
               a.amount_bb, a.raise_to_eff_bb, a.pot_before_bb, a.n_live,
               e.eq_vs_real, e.eq_real_known
        FROM actions a JOIN _scope USING (hand_id)
        LEFT JOIN action_equity e USING (hand_id, action_index)
        WHERE NOT a.is_post
        ORDER BY a.hand_id, a.action_index
    """).df()
    return hands, players, actions


def read_times(path) -> dict[tuple[str, str, int], float]:
    """``decision_times.csv`` -> {(hand_id, street, order on the street): seconds}; {} when the file is missing."""
    p = Path(path) if path else None
    if p is None or not p.exists():
        return {}
    t = pd.read_csv(p, dtype={"hand_id": str, "street": str})
    return {(h, s, int(o)): float(x) for h, s, o, x in t[["hand_id", "street", "order", "seconds"]].itertuples(index=False)}


def player_uids(player_ids) -> dict[str, int]:
    ids = sorted(set(player_ids))
    if all(str(p).isdigit() for p in ids):
        return {p: int(p) for p in ids}
    return {p: i + 1 for i, p in enumerate(ids)}


def build(con, hand_ids: list[str], times: dict | None = None) -> tuple[pd.DataFrame, dict[str, Hand], dict[str, int]]:
    """Returns (dec, hands, uid_of) for the module, uid_of maps player_id -> integer id.

    ``times``: {(hand_id, street, order on the street from 1): seconds}; without it ``decision_s`` is used (NaN
    when the source has no times)."""
    times = times or {}
    hands_df, players_df, actions_df = load_tables(con, hand_ids)
    uid_of = player_uids(players_df["player_id"])
    seats_by_hand = {hid: g for hid, g in players_df.groupby("hand_id", sort=False)}
    acts_by_hand = {hid: g for hid, g in actions_df.groupby("hand_id", sort=False)}

    hands: dict[str, Hand] = {}
    rows: list[dict] = []
    for h in hands_df.itertuples(index=False):
        seats_df = seats_by_hand[h.hand_id]
        seats = {int(s.seat): Seat(int(s.seat), uid_of[s.player_id], int(round(_chips(s.stack_start_bb))),
                                   str(s.position), module_cards(s.hole_cards))
                 for s in seats_df.itertuples(index=False)}
        board = module_cards(h.board)
        acts: dict[str, list[Action]] = {st: [] for st in STREETS}
        hand_acts = acts_by_hand.get(h.hand_id)
        if hand_acts is not None:
            for st, grp in hand_acts.groupby("street", sort=False):
                last_aggr = None                      # module semantics: the last bet or raise on this street
                street_rows = []
                for k, a in enumerate(grp.itertuples(index=False), start=1):
                    uid = uid_of[a.player_id]
                    aggressive = a.action in ("bet", "raise")
                    amount = _chips(a.raise_to_eff_bb if aggressive else a.amount_bb)
                    t = times.get((h.hand_id, st, k))
                    if t is None:
                        t = float(a.decision_s) if a.decision_s is not None and not pd.isna(a.decision_s) else float("nan")
                    acts[st].append(Action(st, int(a.action_index), int(a.seat), uid, TYPE[a.action],
                                           int(round(amount)), t, aggressive))
                    eq = None
                    known = a.eq_real_known is not None and not pd.isna(a.eq_real_known) and bool(a.eq_real_known)
                    if st != "preflop" and known and a.eq_vs_real is not None and not pd.isna(a.eq_vs_real):
                        eq = float(a.eq_vs_real)
                    facing = last_aggr is not None and last_aggr["seat"] != int(a.seat)
                    hole = seats[int(a.seat)].hole
                    n_board = {"flop": 3, "turn": 4, "river": 5}.get(st, 0)
                    pair_plus = evaluate(board[:n_board] + hole)[0] >= 1 if eq is not None and hole else None
                    row = dict(hand_uid=h.hand_id, uid=uid, seat_no=int(a.seat), street=st, seq=int(a.action_index),
                               action=a.action, amount=amount, think_time=t,
                               is_fold=a.action == "fold", is_aggr=aggressive, is_continue=a.action in ("call", "raise"),
                               hero_eq=eq, hero_pair_plus=pair_plus, facing_bet=facing,
                               bettor_uid=last_aggr["uid"] if facing else None,
                               bettor_eq=last_aggr["eq"] if facing else None,
                               bet_faced=last_aggr["amount"] if facing else None,
                               pot_before=_chips(a.pot_before_bb), bluff_success=None, n_active=int(a.n_live))
                    if aggressive:
                        last_aggr = {"seat": int(a.seat), "uid": uid, "eq": eq, "amount": amount}
                    rows.append(row)
                    street_rows.append(row)
                for i, row in enumerate(street_rows):  # module rule: every later action by the others is a fold
                    if row["is_aggr"]:
                        later = [r["is_fold"] for r in street_rows[i + 1:] if r["seat_no"] != row["seat_no"]]
                        row["bluff_success"] = bool(later) and all(later)
        start = int(pd.Timestamp(h.started_at).timestamp())
        net = {uid_of[s.player_id]: _chips(s.net_bb) for s in seats_df.itertuples(index=False)}
        committed = {uid_of[s.player_id]: _chips(s.committed_bb) for s in seats_df.itertuples(index=False)}
        hands[h.hand_id] = Hand(
            hand_uid=h.hand_id, start_ts=start, end_ts=start, table_no=h.table_id, stake_level=None, chip_type=None,
            blind=CHIPS_PER_BB, ante=_chips(h.ante_bb), game_type="No-Limit Hold'em", seats=seats, actions=acts,
            board=board,
            street_end_pot={"preflop": _chips(h.pot_flop_bb), "flop": _chips(h.pot_turn_bb),
                            "turn": _chips(h.pot_river_bb), "river": _chips(h.final_pot_bb)},
            settlement=net, committed=committed)
    return pd.DataFrame(rows), hands, uid_of


def analyse_player(con, player: str, opponent: str | None = None, scope: str | bool = "flop_hu",
                   times: dict | None = None) -> dict:
    """The module's analyse_player, fed from the warehouse, on the hands of ``scope`` (see :func:`hands_in_scope`).
    ``player`` in the result is the module's integer id, ``player_id`` the warehouse id."""
    ids = hands_in_scope(con, player, opponent, scope)
    dec, hands, uid_of = build(con, ids, times)
    name_of = {v: k for k, v in uid_of.items()}
    uid = uid_of.get(player)
    hand_list = list(hands.values())
    if uid is None or dec.empty:
        return {"player": player, "player_id": player, "opponent": opponent, "meta": {"hands_analysed": 0}}
    suspicious = find_suspicious(hands, dec, uid)
    if not suspicious.empty:
        suspicious["opponent_uid"] = suspicious["opponent_uid"].map(lambda u: name_of.get(u, u))
    return {
        "player": uid,
        "player_id": player,
        "opponent": opponent,
        "meta": {"rows": len(ids), "unparseable": 0, "not_in_hand": 0, "filtered": 0, "hands_analysed": len(hands)},
        "stats": player_stats(hand_list, uid),
        "signals": player_signals(dec, uid),
        "suspicious_hands": suspicious,
        "escalation": escalation(hand_list, suspicious),
        "decisions": dec,
        "hands": hands,
        "uid_of": uid_of,                 # warehouse player_id -> the module's integer id in these hands
    }
