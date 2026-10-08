"""Parse one hand record (hand_replay + hand_result JSON) into a Hand object."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .cards import InvalidCard, decode

STREETS = ["preflop", "flop", "turn", "river"]
BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
ACTION_NAMES = {1: "check", 2: "fold", 3: "call", 4: "bet", 5: "raise", 6: "allin", 7: "walk"}
ACT_CODES = {name: code for code, name in ACTION_NAMES.items()}   # the records write action names


def _act_code(v) -> int:
    """check / fold / call / bet / raise / allin / walk, or the integer codes 1-7; anything else is 0 (unknown)."""
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
        return int(v)
    return ACT_CODES.get(str(v or "").strip().lower().replace("-", "").replace(" ", ""), 0)


@dataclass
class Seat:
    seat_no: int
    uid: int
    stake: int
    label: str
    hole: list[int]


@dataclass
class Action:
    street: str
    seq: int
    seat_no: int
    uid: int
    type: int
    amount: int
    time: float
    aggressive: bool = False       # bet / raise / all-in that raises the price

    @property
    def name(self) -> str:
        return ACTION_NAMES.get(self.type, str(self.type))


@dataclass
class Hand:
    hand_uid: str
    start_ts: int
    end_ts: int
    table_no: Any
    stake_level: Any
    chip_type: Any
    blind: float
    ante: float
    game_type: str
    seats: dict[int, Seat]
    actions: dict[str, list[Action]]
    board: list[int]
    street_end_pot: dict[str, float]
    settlement: dict[int, float]                 # uid -> net win
    committed: dict[int, float] = field(default_factory=dict)   # uid -> chips put in (incl. uncalled)

    @property
    def n_players(self) -> int:
        return len(self.seats)

    @property
    def uids(self) -> list[int]:
        return [s.uid for s in self.seats.values()]

    def seat_of(self, uid: int) -> Seat | None:
        return next((s for s in self.seats.values() if s.uid == uid), None)

    def contested_pot(self) -> float:
        """Total pot excluding the uncalled part of the last bet."""
        if self.committed:
            bets = sorted(self.committed.values(), reverse=True)
            cap = bets[1] if len(bets) > 1 else bets[0]
            return float(sum(min(b, cap) for b in bets))
        pots = [v for v in self.street_end_pot.values() if v]
        return float(pots[-1]) if pots else 0.0


def _load(value):
    if value is None or (isinstance(value, float) and value != value):
        return {}
    return json.loads(value) if isinstance(value, str) else value


def _num(v) -> float:
    """A number, possibly given as text."""
    return float(v or 0)


def _pot(round_info) -> float:
    return float(sum(p.get("amount", 0) for p in ((round_info or {}).get("pots") or [])))


def _mark_aggression(actions: list[Action]) -> None:
    """Flag aggressive actions. Bet/raise amounts are street totals, call amounts are chips added;
    an all-in is aggressive only when it exceeds the current highest street total."""
    totals: dict[int, float] = {}
    for a in actions:
        cur_max = max([v for k, v in totals.items() if k != a.seat_no], default=0)
        if a.type in (4, 5):
            a.aggressive = True
            totals[a.seat_no] = a.amount
        elif a.type == 6:
            a.aggressive = a.amount > cur_max
            totals[a.seat_no] = max(totals.get(a.seat_no, 0), a.amount)
        elif a.type == 3:
            totals[a.seat_no] = totals.get(a.seat_no, 0) + a.amount


def parse_hand(row: Mapping) -> Hand | None:
    """Returns None when the hand cannot be used (missing replay, invalid cards)."""
    rp = _load(row.get("hand_replay"))
    rec = _load(row.get("hand_result"))
    if not rp:
        return None
    ri = rp.get("Streets") or {}
    room = rp.get("TableSetup") or {}
    try:
        seats = {}
        for s in (rp.get("Seats") or {}).get("seat_list") or []:
            hole = [decode(c) for c in s.get("hole_cards") or []]
            seats[s.get("seat", 0)] = Seat(s.get("seat", 0), int(s["player_id"]), s.get("stack", 0),
                                              s.get("position", ""), hole if len(hole) == 2 else [])
        board = [decode(c) for c in (rec.get("board") or [])]
    except (InvalidCard, KeyError):
        return None

    actions: dict[str, list[Action]] = {}
    for st in STREETS:
        lst = [Action(st, a.get("order", 0), a.get("seat", 0), seats[a.get("seat", 0)].uid if a.get("seat", 0) in seats else -1,
                      _act_code(a.get("act")), a.get("amount", 0) or 0, float(a.get("think_s", 0) or 0))
               for a in (ri.get(st) or [])]
        _mark_aggression(lst)
        actions[st] = lst

    settlement = {}
    for x in ri.get("settlement") or []:
        sn = x.get("seat", 0)
        if sn in seats:
            settlement[seats[sn].uid] = float(x.get("net", 0) or 0)
    committed = {}
    for r in rec.get("contributions") or []:
        try:
            committed[int(_num(r.get("player_id")))] = _num(r.get("total_bet"))
        except (TypeError, ValueError):
            pass

    return Hand(
        hand_uid=str(row.get("hand_uid")),
        start_ts=int(row.get("start_ts") or 0),
        end_ts=int(row.get("end_ts") or 0),
        table_no=row.get("table_no"),
        stake_level=row.get("stake_level"),
        chip_type=row.get("chip_type"),
        blind=float(room.get("big_blind") or 0),
        ante=float(room.get("ante") or 0),
        game_type=room.get("game", ""),
        seats=seats,
        actions=actions,
        board=board,
        street_end_pot={st: _pot(ri.get(f"{st}_end")) for st in STREETS},
        settlement=settlement,
        committed=committed,
    )
