"""Write hand records as generic PokerStars-style hand-history text.

Two modes produce the same text except for one block:

* full information (``full_information=True``): a ``Dealt to <player> [Xx Yy]``
  line for every seat after ``*** HOLE CARDS ***``. This is the operator's
  view, where every hole card is known.
* mined style (``full_information=False``): no ``Dealt to`` lines, so hole
  cards appear only when a player shows at the showdown. This is what hand
  histories collected at the table look like.

Amounts are chip units divided by 100 with exactly two decimals; with a big
blind of 100 units they read as big blinds. A heads-up hand in mined style::

    Hand #17: Hold'em No Limit (0.50/1.00) - 2000/01/01 12:00:00 UTC
    Table 'Alpha' 6-max Seat #3 is the button
    Seat 1: P_0a1b2c (100.00 in chips)
    Seat 3: P_3d4e5f (100.00 in chips)
    P_3d4e5f: posts small blind 0.50
    P_0a1b2c: posts big blind 1.00
    *** HOLE CARDS ***
    P_3d4e5f: raises 2.00 to 3.00
    P_0a1b2c: folds
    Uncalled bet (2.00) returned to P_3d4e5f
    P_3d4e5f collected 2.00 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
    Seat 1: P_0a1b2c (big blind) folded before Flop
    Seat 3: P_3d4e5f (button) (small blind) collected (2.00)

In full-information mode the same hand has two more lines after
``*** HOLE CARDS ***``: ``Dealt to P_0a1b2c [7c 2d]`` and
``Dealt to P_3d4e5f [As Kh]``.

Showdown rules are the same in both modes: the river's last aggressor shows
first (or, when nobody bet the river, the first player still in the hand left
of the button); every later player shows only if their hand wins or ties a
pot they are eligible for, and mucks otherwise. With side pots the winners
collect "from side pot-2", "from side pot-1", ..., "from main pot", side pots
first. Shown hands carry the text of :func:`sudetect.evaluator.describe_hand`,
e.g. "a flush, King high".
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime, timezone

from ..cards import format_cards
from .engine import STREETS, ActionRecord, HandRecord

__all__ = ["format_amount", "write_hand", "write_hands", "write_file"]

_STREET_HEADERS = {"flop": "Flop", "turn": "Turn", "river": "River"}


def format_amount(units: int) -> str:
    """Chip units as a decimal with two places: 150 -> '1.50'."""
    if units < 0:
        raise ValueError("amounts in hand histories are never negative")
    return f"{units // 100}.{units % 100:02d}"


def _timestamp(t: datetime) -> str:
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc)
    return f"{t.year:04d}/{t.month:02d}/{t.day:02d} {t.hour:02d}:{t.minute:02d}:{t.second:02d}"


def _action_line(a: ActionRecord) -> str:
    kind = a.kind
    if kind == "fold":
        text = "folds"
    elif kind == "check":
        text = "checks"
    elif kind == "call":
        text = f"calls {format_amount(a.amount)}"
    elif kind == "bet":
        text = f"bets {format_amount(a.amount)}"
    elif kind == "raise":
        text = f"raises {format_amount(a.raise_by)} to {format_amount(a.to)}"
    elif kind == "post_ante":
        text = f"posts the ante {format_amount(a.amount)}"
    elif kind == "post_sb":
        text = f"posts small blind {format_amount(a.amount)}"
    elif kind == "post_bb":
        text = f"posts big blind {format_amount(a.amount)}"
    else:
        raise ValueError(f"unknown action kind {kind!r}")
    if a.is_all_in:
        text += " and is all-in"
    return f"{a.player}: {text}"


def write_hand(record: HandRecord, full_information: bool) -> str:
    """One hand as text, ending with a newline.

    Args:
        record: The hand to write.
        full_information: Add a ``Dealt to`` line for every seat.
    """
    amt = format_amount
    players = record.players
    board = record.board
    lines = [
        f"Hand #{record.hand_id}: Hold'em No Limit ({amt(record.sb)}/{amt(record.bb)})"
        f" - {_timestamp(record.start_time)} UTC",
        f"Table '{record.table_name}' {record.max_seats}-max Seat #{record.button_seat} is the button",
    ]
    lines += [f"Seat {s.seat}: {s.player} ({amt(s.stack_start)} in chips)" for s in record.seats]

    actions = record.actions
    k = 0
    while k < len(actions) and actions[k].kind.startswith("post_"):
        lines.append(_action_line(actions[k]))
        k += 1
    lines.append("*** HOLE CARDS ***")
    if full_information:
        lines += [f"Dealt to {s.player} [{format_cards(record.hole_cards[s.seat])}]" for s in record.seats]

    decisions = actions[k:]
    # betting closed for good on the street of the last decision; the uncalled bet comes back there
    closed_on = decisions[-1].street if decisions else "preflop"
    n_streets = {0: 1, 3: 2, 4: 3, 5: 4}[len(board)]
    for street in STREETS[:n_streets]:
        if street == "flop":
            lines.append(f"*** FLOP *** [{format_cards(board[:3])}]")
        elif street == "turn":
            lines.append(f"*** TURN *** [{format_cards(board[:3])}] [{format_cards(board[3:4])}]")
        elif street == "river":
            lines.append(f"*** RIVER *** [{format_cards(board[:4])}] [{format_cards(board[4:5])}]")
        lines += [_action_line(a) for a in decisions if a.street == street]
        if street == closed_on:
            lines += [f"Uncalled bet ({amt(x)}) returned to {players[seat]}" for seat, x in record.uncalled.items()]

    if record.showdown:
        lines.append("*** SHOW DOWN ***")
        for seat in record.showdown_order:
            if seat in record.shown:
                lines.append(f"{players[seat]}: shows [{format_cards(record.shown[seat])}]"
                             f" ({record.descriptions[seat]})")
            else:
                lines.append(f"{players[seat]}: mucks hand")

    pots = record.pots
    names = ["pot"] if len(pots) == 1 else ["main pot"] + [f"side pot-{i}" for i in range(1, len(pots))]
    for i in reversed(range(len(pots))):  # side pots are settled before the main pot
        for seat in pots[i].winners:
            share = pots[i].shares[seat]
            if share > 0:
                lines.append(f"{players[seat]} collected {amt(share)} from {names[i]}")

    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {amt(record.total_pot)} | Rake {amt(record.rake)}")
    if board:
        lines.append(f"Board [{format_cards(board)}]")
    won = {seat for pot in pots for seat in pot.winners}
    for s in record.seats:
        lines.append(_summary_line(record, s.seat, s.player, won))
    return "\n".join(lines) + "\n"


def _summary_line(record: HandRecord, seat: int, player: str, won: set[int]) -> str:
    text = f"Seat {seat}: {player}"
    if seat == record.button_seat:
        text += " (button)"
    if seat == record.sb_seat:
        text += " (small blind)"
    if seat == record.bb_seat:
        text += " (big blind)"
    street = record.folded_on.get(seat)
    if street is not None:
        if street == "preflop":
            return f"{text} folded before Flop"
        return f"{text} folded on the {_STREET_HEADERS[street]}"
    if not record.showdown:
        return f"{text} collected ({format_amount(record.collected[seat])})"
    if seat not in record.shown:
        return f"{text} mucked"
    cards = format_cards(record.shown[seat])
    if seat in won:
        return (f"{text} showed [{cards}] and won ({format_amount(record.collected[seat])})"
                f" with {record.descriptions[seat]}")
    return f"{text} showed [{cards}] and lost with {record.descriptions[seat]}"


def write_hands(records: Iterable[HandRecord], full_information: bool) -> str:
    """Several hands, separated by a blank line."""
    return "\n".join(write_hand(r, full_information) for r in records)


def write_file(path: str | os.PathLike[str], records: Iterable[HandRecord], full_information: bool) -> int:
    """Write hands to a UTF-8 text file; returns the number of hands written."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            if count:
                f.write("\n")
            f.write(write_hand(r, full_information))
            count += 1
    return count
