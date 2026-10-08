"""Betting situations: what a player faced when he acted.

Every function here is pure; the replay (:mod:`sudetect.replay`) keeps the
per-street counters and calls them once per action row. All counts are taken
*before* the row, after all-ins are resolved (an all-in "raise" that nobody
can call more is a call) and with forced bets (blinds, antes, straddles)
treated as posts, never as raises.

Preflop situations
------------------
=================  ===========================================================
``unopened``       no voluntary action yet other than folds (posts and
                   straddles only): the open-raise spot
``vs_limp``        at least one limp (a call of the blind level), no raise;
                   includes the big blind's option after limps
``vs_open``        exactly one raise and no call after it
``vs_open_call``   one raise and at least one call after it (squeeze spot)
``vs_3bet``        two raises
``vs_4bet_plus``   three or more raises
=================  ===========================================================

Postflop situations
-------------------
The *aggressor* is the player who made the last bet or raise of the previous
street (on the flop: the last preflop raiser).

Not facing a bet:

=============  ==============================================================
``cbet_opp``   the actor is the aggressor (continuation-bet spot)
``donk_opp``   the aggressor acts after the actor and has not acted yet
``probe_opp``  the aggressor already checked on this street
``lead_opp``   there is no aggressor (the previous street was checked
               through, or the aggressor folded or is all-in)
=============  ==============================================================

Facing a bet: ``vs_raise`` when two or more bets/raises were made on the
street (check-raises included); otherwise the single bet is typed by the spot
its bettor was in: ``vs_cbet`` (made from ``cbet_opp``), ``vs_donk``,
``vs_probe``, ``vs_bet`` (from ``lead_opp``).

Line and pot type
-----------------
``line`` (turn and river rows) spells the earlier postflop streets with one
letter each, ``b`` when somebody bet on that street and ``x`` when it was
checked through: a river row after a bet flop and a checked turn has
``line = "bx"``.

``pot_type`` counts the raises before the flop: ``limped`` (none),
``single_raised``, ``3bet``, ``4bet_plus``; preflop rows use the count so far
and read ``unopened`` before the first limp or raise.

Position
--------
``is_ip`` (after the flop) is true for the player who acts last on the street
among the players who could still act when the street began (not folded, not
all-in). It is a property of the street, not of the row.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

__all__ = [
    "PREFLOP_SITUATIONS",
    "POSTFLOP_SITUATIONS",
    "POT_TYPES",
    "FACING_OF_SPOT",
    "preflop_situation",
    "postflop_spot",
    "facing_situation",
    "line_prefix",
    "pot_type",
    "is_in_position",
]

PREFLOP_SITUATIONS: tuple[str, ...] = (
    "unopened", "vs_limp", "vs_open", "vs_open_call", "vs_3bet", "vs_4bet_plus",
)
POSTFLOP_SITUATIONS: tuple[str, ...] = (
    "cbet_opp", "donk_opp", "probe_opp", "lead_opp",
    "vs_cbet", "vs_donk", "vs_probe", "vs_bet", "vs_raise",
)
POT_TYPES: tuple[str, ...] = ("unopened", "limped", "single_raised", "3bet", "4bet_plus")
"""``unopened`` occurs on preflop rows only."""

FACING_OF_SPOT: dict[str, str] = {
    "cbet_opp": "vs_cbet",
    "donk_opp": "vs_donk",
    "probe_opp": "vs_probe",
    "lead_opp": "vs_bet",
}
"""Spot in which a bet was made -> situation of the players facing that bet."""


def preflop_situation(n_raises: int, n_limps: int, n_calls_after_raise: int) -> str:
    """Preflop situation from the counts before the row.

    Args:
        n_raises: Voluntary raises so far (posts and straddles excluded).
        n_limps: Calls of the blind level made before the first raise.
        n_calls_after_raise: Calls made after the last raise.
    """
    if n_raises <= 0:
        return "vs_limp" if n_limps > 0 else "unopened"
    if n_raises == 1:
        return "vs_open_call" if n_calls_after_raise > 0 else "vs_open"
    if n_raises == 2:
        return "vs_3bet"
    return "vs_4bet_plus"


def postflop_spot(
    actor_is_aggressor: bool,
    aggressor_active: bool,
    aggressor_checked: bool,
    aggressor_acts_after: bool,
) -> str:
    """Postflop situation of a player who does not face a bet.

    Args:
        actor_is_aggressor: The actor made the last bet or raise of the previous street.
        aggressor_active: There is such an aggressor and he is still in the hand
            with chips behind.
        aggressor_checked: The aggressor already checked on this street.
        aggressor_acts_after: The aggressor acts after the actor on this street
            and has not acted yet.
    """
    if actor_is_aggressor and aggressor_active:
        return "cbet_opp"
    if not aggressor_active:
        return "lead_opp"
    if aggressor_checked:
        return "probe_opp"
    if aggressor_acts_after:
        return "donk_opp"
    return "lead_opp"


def facing_situation(n_aggressive: int, first_bet_spot: Optional[str]) -> str:
    """Postflop situation of a player facing a bet.

    Args:
        n_aggressive: Bets and raises made on this street before the row (>= 1).
        first_bet_spot: The spot (``cbet_opp``, ...) in which the street's bet was made.
    """
    if n_aggressive >= 2:
        return "vs_raise"
    return FACING_OF_SPOT.get(first_bet_spot or "lead_opp", "vs_bet")


def line_prefix(street: int, street_had_bet: Sequence[bool]) -> Optional[str]:
    """``x``/``b`` pattern of the postflop streets before ``street`` (2 = turn, 3 = river).

    ``street_had_bet[k]`` tells whether street ``k`` (0 = preflop, 1 = flop, ...)
    saw a bet or raise. Preflop and flop rows have no line (``None``).
    """
    if street < 2:
        return None
    return "".join("b" if street_had_bet[k] else "x" for k in range(1, street))


def pot_type(n_raises: int, n_limps: int = 0, preflop_row: bool = False) -> str:
    """Pot type from the number of preflop raises (so far on preflop rows)."""
    if n_raises <= 0:
        return "unopened" if preflop_row and n_limps <= 0 else "limped"
    if n_raises == 1:
        return "single_raised"
    if n_raises == 2:
        return "3bet"
    return "4bet_plus"


def is_in_position(actor_order: int, active_orders: Sequence[int]) -> bool:
    """True if ``actor_order`` is the last postflop order among the players active at the street start."""
    return bool(active_orders) and actor_order == max(active_orders)
