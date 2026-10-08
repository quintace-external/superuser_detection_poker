"""Table positions computed from the button and the seats dealt into a hand.

Positions are derived, never copied from a hand history: the button seat and
the list of seats dealt in are enough, with the blind posts used to settle the
two blind seats.

Acting order
------------
After the flop the first seat left of the button acts first and the button
acts last. The order is the dealt seats sorted by
``(seat - button_seat - 1) mod max_seats``; when the button sits on an empty
seat (a "dead button") the last seat in this order still acts last and is
labelled ``BTN``. Before the flop the first player to act sits left of the
big blind (left of the straddles that follow the big blind, if any) and the
big blind acts last; heads-up the button acts first.

Labels
------
Heads-up, the button posts the small blind; it is labelled ``BTN`` (with
``is_sb`` set by the loader) and the other player ``BB``. When the text puts
the button on an empty seat or on the big blind, the small-blind poster is
still the button, since he is the one who acts first before the flop and last
after it (flag ``hu_button_not_sb``).

With three or more players ``BTN`` is the last to act after the flop, ``SB``
and ``BB`` are the first small-blind and big-blind posters, and every other
seat is named by its distance from the button counting backwards: ``CO`` one
seat before the button, then ``HJ``, ``LJ``, ``UTG+2``, ``UTG+1``, ``UTG``. A
full nine-handed table reads, in acting order before the flop::

    UTG  UTG+1  UTG+2  LJ  HJ  CO  BTN  SB  BB

and smaller tables drop labels from the front (six-handed:
``LJ HJ CO BTN SB BB``; four-handed: ``CO BTN SB BB``). A label therefore
always means the same distance from the button whatever the table size. More
than six seats between the blinds and the button (ten-handed tables) add
``UTG+3`` and up.

A blind seat whose player was all-in from the ante posts nothing; the blind
seats are then taken from seat order (flags ``sb_from_seat_order`` /
``bb_from_seat_order``). A hand with a big blind but no small blind (a "dead
small blind") has no ``SB`` label.

Position classes group the labels for frequency tables:
``EP`` (UTG, UTG+1, UTG+2, ...), ``MP`` (LJ, HJ), ``CO``, ``BTN``, ``SB``, ``BB``.

Example::

    >>> p = assign_positions(seats=[1, 2, 3, 5, 6, 8], button_seat=5, max_seats=9,
    ...                      sb_player=4, bb_player=5)
    >>> p.labels
    ('LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB')
    >>> p.preflop_order
    (1, 2, 3, 4, 5, 6)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "MIDDLE_LABELS",
    "POSITION_CLASSES",
    "LABEL_CLASS",
    "Positions",
    "postflop_sequence",
    "assign_positions",
    "position_class",
]

MIDDLE_LABELS: tuple[str, ...] = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO")
"""Labels of the seats between the big blind and the button, in acting order (nine-handed)."""

POSITION_CLASSES: tuple[str, ...] = ("EP", "MP", "CO", "BTN", "SB", "BB")
"""Coarse position classes, earliest first."""

LABEL_CLASS: dict[str, str] = {
    "UTG": "EP", "UTG+1": "EP", "UTG+2": "EP", "LJ": "MP", "HJ": "MP",
    "CO": "CO", "BTN": "BTN", "SB": "SB", "BB": "BB",
}
"""Position label -> position class."""


def position_class(label: Optional[str]) -> Optional[str]:
    """Class of a position label (``"HJ" -> "MP"``); ``UTG+k`` for any k is ``EP``."""
    if label is None:
        return None
    cls = LABEL_CLASS.get(label)
    if cls is None and label.startswith("UTG"):
        return "EP"
    return cls


@dataclass(frozen=True)
class Positions:
    """Positions of the players of one hand; every tuple follows the input seat order.

    Attributes:
        labels: Position label per player (``BTN``, ``SB``, ``BB``, ``CO``, ...).
        classes: Position class per player (``EP``, ``MP``, ``CO``, ``BTN``, ``SB``, ``BB``).
        preflop_order: 1 = first to act before the flop.
        postflop_order: 1 = first to act after the flop; the button has the largest value.
        button: Index of the player labelled ``BTN``.
        sb: Index of the small blind (heads-up: the button), or None for a dead small blind.
        bb: Index of the big blind, or None when there is no big blind seat.
        flags: ``dead_button``, ``sb_from_seat_order``, ``bb_from_seat_order``,
            ``hu_button_not_sb`` (heads-up and the button seat of the text is not the small
            blind's; the small-blind poster is labelled ``BTN``).
    """

    labels: tuple[str, ...]
    classes: tuple[str, ...]
    preflop_order: tuple[int, ...]
    postflop_order: tuple[int, ...]
    button: int
    sb: Optional[int]
    bb: Optional[int]
    flags: frozenset[str]


def postflop_sequence(seats: Sequence[int], button_seat: int, max_seats: int) -> list[int]:
    """Player indices in acting order after the flop (the button, or the dead button's seat, last).

    Args:
        seats: Seat number of every player dealt in.
        button_seat: Seat the button is on (it may be empty).
        max_seats: Seats at the table; a larger seat number in ``seats`` widens it.
    """
    if not seats:
        return []
    ring = max(max_seats, max(seats), button_seat)
    return sorted(range(len(seats)), key=lambda i: (seats[i] - button_seat - 1) % ring)


def _middle_labels(m: int) -> tuple[str, ...]:
    if m <= len(MIDDLE_LABELS):
        return MIDDLE_LABELS[len(MIDDLE_LABELS) - m:]
    return ("UTG",) + tuple(f"UTG+{k}" for k in range(1, m - 3)) + ("LJ", "HJ", "CO")


def assign_positions(
    seats: Sequence[int],
    button_seat: int,
    max_seats: int,
    sb_player: Optional[int] = None,
    bb_player: Optional[int] = None,
    straddlers: Sequence[int] = (),
) -> Positions:
    """Labels, classes and acting orders for one hand.

    Args:
        seats: Seat number of every player dealt in (any order; results follow it).
        button_seat: The button's seat as written in the hand history.
        max_seats: Seats at the table.
        sb_player: Index of the first player who posted a live small blind, if any.
        bb_player: Index of the first player who posted the big blind, if any.
        straddlers: Indices of players who posted a straddle; preflop action
            starts after the straddles that follow the big blind.

    Raises:
        ValueError: Fewer than two players.
    """
    n = len(seats)
    if n < 2:
        raise ValueError("a hand needs at least two players")
    order = postflop_sequence(seats, button_seat, max_seats)
    flags: set[str] = set()
    if button_seat not in seats:
        flags.add("dead_button")
    btn = order[-1]
    labels = [""] * n

    if n == 2:
        # heads-up the small blind is the button whatever seat the text names: he acts first
        # before the flop and last after it
        if sb_player is not None:
            hu_btn = sb_player
        elif bb_player is not None:
            hu_btn = 1 - bb_player
        else:
            hu_btn = btn
        if hu_btn != btn:
            flags.add("hu_button_not_sb")
            btn = hu_btn
            order = [1 - btn, btn]
        sb: Optional[int] = btn
        bb: Optional[int] = order[0]
        labels[btn], labels[order[0]] = "BTN", "BB"
        pre = [btn, order[0]]
    else:
        bb = bb_player if bb_player is not None and bb_player != btn else None
        sb = sb_player if sb_player is not None and sb_player != btn and sb_player != bb else None
        if bb is None:
            # nobody posted the big blind (its player was all-in from the ante)
            if sb is None:
                sb = order[0]
            k = order.index(sb) + 1
            bb = order[k] if k < n - 1 else None
            flags.add("bb_from_seat_order")
        elif sb is None and order[0] != bb:
            sb = order[0]  # the small-blind seat could not post (all-in from the ante)
            flags.add("sb_from_seat_order")
        labels[btn] = "BTN"
        if sb is not None:
            labels[sb] = "SB"
        if bb is not None:
            labels[bb] = "BB"
        middle = [i for i in order if not labels[i]]
        for i, label in zip(middle, _middle_labels(len(middle))):
            labels[i] = label
        if bb is not None:
            k = order.index(bb)
            skip = set(straddlers)
            steps = 0
            while steps < n - 1 and order[(k + 1) % n] in skip:
                k += 1
                steps += 1
            pre = [order[(k + 1 + j) % n] for j in range(n)]
        else:
            pre = list(order)

    preflop_rank = [0] * n
    for r, i in enumerate(pre, start=1):
        preflop_rank[i] = r
    postflop_rank = [0] * n
    for r, i in enumerate(order, start=1):
        postflop_rank[i] = r
    classes = tuple(position_class(x) or "" for x in labels)
    return Positions(tuple(labels), classes, tuple(preflop_rank), tuple(postflop_rank),
                     btn, sb, bb, frozenset(flags))
