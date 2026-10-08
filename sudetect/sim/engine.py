"""No-limit hold'em rules engine: one table, one hand at a time.

Chips are integers ("units"). With the default blinds of 50/100 one big blind
is 100 units, which lets the text writer print every amount as big blinds with
two decimals.

:func:`play_hand` deals one hand, asks each player's agent for decisions
through ``agent.act(view)`` and returns a :class:`HandRecord` holding
everything that happened: forced bets, actions, the board, pots, winners, rake
and every seat's net result.

Rules
-----
* Antes are posted first, starting left of the button. A stack smaller than
  the ante posts all of it and is all-in.
* Heads-up, the button posts the small blind, acts first before the flop and
  last after it. With three or more players the small blind is the first seat
  left of the button and the big blind the next one; the first player to act
  before the flop sits left of the big blind, and after the flop the first
  player still able to act left of the button starts.
* A stack smaller than its blind posts what it has and is all-in. The full big
  blind is still the amount to call before the flop.
* Legal actions: fold (only when facing a bet), check, call (the amount to
  call, or the whole stack if smaller), bet or raise to a street total between
  the minimum raise and all-in. The minimum raise increment is
  ``max(last full raise increment, big blind)``; a stack that cannot reach it
  may still go all-in.
* An all-in that raises by less than a full raise does not reopen the betting.
  A player who has already acted on this street (a check counts) may raise
  again only if the bet has grown by at least a full raise since their last
  action; several short all-ins can add up to one. Otherwise they may only
  call or fold. An all-in bet smaller than the big blind is a short bet in the
  same sense.
* Raising is not offered when no other player could respond to it, and a
  player who is the only one left able to act is asked to act only while their
  street contribution is below somebody else's.
* When betting is over and at most one player still has chips behind, the
  rest of the board is dealt without further actions.
* The uncalled part of the largest contribution (its excess over the second
  largest) is returned before the pots are formed. Pots follow the
  contribution levels of the players still in the hand; folded players' chips
  stay in the pots they reached.
* Pots are awarded by a hand evaluator, :func:`sudetect.evaluator.eval_hands`
  unless another one is passed in. Ties share a pot equally and the odd chips
  go to the first tied winner left of the button.
* Rake is ``floor(rake_pct * total pot)``, capped at ``rake_cap``, taken only
  when a flop was dealt ("no flop, no drop", unless the table switches it
  off) and removed from the main pot first, then from the side pots in order.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Any, NamedTuple, Optional, Protocol, Union

import numpy as np

from ..evaluator import describe_hand, eval_hands

__all__ = [
    "STREETS",
    "ACTION_KINDS",
    "DECISION_KINDS",
    "Evaluator",
    "InfoGrants",
    "TableConfig",
    "SeatState",
    "Decision",
    "DecisionView",
    "Agent",
    "RandomAgent",
    "ActionRecord",
    "SeatRecord",
    "PotRecord",
    "HandRecord",
    "play_hand",
]

STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
"""Street names in order. ``ActionRecord.street`` takes these values."""

ACTION_KINDS: tuple[str, ...] = (
    "post_ante", "post_sb", "post_bb", "fold", "check", "call", "bet", "raise",
)
"""Every ``ActionRecord.kind``; the first three are forced bets."""

DECISION_KINDS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise")
"""Every ``Decision.kind`` an agent may return."""

_BOARD_LEN = (0, 3, 4, 5)  # cards on the board on each street

Evaluator = Callable[[np.ndarray], Any]
"""``evaluate(cards)`` takes an (n, 7) int array and returns n strengths (higher wins)."""

InfoGrants = Mapping[str, Union[str, Collection[str]]]
"""Player name -> names of the players whose hole cards that player sees, or ``"all"``."""


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class TableConfig:
    """Stakes and house rules of one table. Amounts are integer chip units.

    Attributes:
        max_seats: Number of seats at the table (2-9); seats are numbered from 1.
        sb: Small blind.
        bb: Big blind (also the minimum bet).
        ante: Ante posted by every player before the blinds (0 for none).
        rake_pct: Share of the total pot taken as rake, e.g. 0.05.
        rake_cap: Largest rake per hand, or ``None`` for no cap.
        no_flop_no_drop: Take rake only when a flop was dealt.
    """

    max_seats: int
    sb: int = 50
    bb: int = 100
    ante: int = 0
    rake_pct: float = 0.05
    rake_cap: Optional[int] = 300
    no_flop_no_drop: bool = True
    _rake_ratio: tuple[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("max_seats", "sb", "bb", "ante"):
            if not _is_int(getattr(self, name)):
                raise TypeError(f"{name} must be an integer number of chips")
        if self.rake_cap is not None and not _is_int(self.rake_cap):
            raise TypeError("rake_cap must be an integer number of chips or None")
        if not 2 <= self.max_seats <= 9:
            raise ValueError("max_seats must be between 2 and 9")
        if self.bb <= 0:
            raise ValueError("the big blind must be positive")
        if not 0 <= self.sb <= self.bb:
            raise ValueError("the small blind must be between 0 and the big blind")
        if self.ante < 0:
            raise ValueError("the ante cannot be negative")
        if self.rake_cap is not None and self.rake_cap < 0:
            raise ValueError("rake_cap cannot be negative")
        ratio = Fraction(str(self.rake_pct))  # exact decimal: 0.05 -> 1/20
        if not 0 <= ratio < 1:
            raise ValueError("rake_pct must be in [0, 1)")
        object.__setattr__(self, "_rake_ratio", (ratio.numerator, ratio.denominator))

    def rake_for(self, total_pot: int, flop_dealt: bool) -> int:
        """Rake taken from a pot of ``total_pot`` chips."""
        if self.no_flop_no_drop and not flop_dealt:
            return 0
        num, den = self._rake_ratio
        rake = total_pot * num // den
        if self.rake_cap is not None and rake > self.rake_cap:
            rake = self.rake_cap
        return rake


@dataclass(frozen=True)
class SeatState:
    """A player at a seat with ``stack`` chips at the start of a hand."""

    seat_no: int
    player: str
    stack: int


@dataclass(frozen=True)
class Decision:
    """What an agent wants to do.

    ``kind`` is one of ``fold``, ``check``, ``call``, ``bet`` or ``raise``.
    For ``bet`` and ``raise``, ``to`` is the player's total for the street
    after the action (not the increment). It is ignored for other kinds.
    """

    kind: str
    to: Optional[int] = None

    @staticmethod
    def fold() -> "Decision":
        return _FOLD

    @staticmethod
    def check() -> "Decision":
        return _CHECK

    @staticmethod
    def call() -> "Decision":
        return _CALL

    @staticmethod
    def bet(to: int) -> "Decision":
        return Decision("bet", to)

    @staticmethod
    def raise_to(to: int) -> "Decision":
        return Decision("raise", to)


_FOLD, _CHECK, _CALL = Decision("fold"), Decision("check"), Decision("call")


# -------------------------------------------------------------------------- records


class ActionRecord(NamedTuple):
    """One forced bet or decision, in the order it happened (immutable).

    Attributes:
        order: 0-based position in the hand's action list.
        street: ``preflop``, ``flop``, ``turn`` or ``river``.
        seat, player: Who acted.
        kind: One of :data:`ACTION_KINDS`.
        amount: Chips this action put in (0 for fold and check).
        to: Street total after a bet or raise; ``None`` for other kinds.
        raise_by: ``to`` minus the highest street bet before the action
            (for a bet: the bet itself); ``None`` for other kinds.
        is_all_in: The action put the player's last chip in.
        pot_before: All chips committed in the hand before this action.
        to_call_before: Chips the player needed to call before acting
            (not capped by the stack; 0 for forced bets).
        stack_before: The player's chips behind before acting.
    """

    order: int
    street: str
    seat: int
    player: str
    kind: str
    amount: int
    to: Optional[int]
    raise_by: Optional[int]
    is_all_in: bool
    pot_before: int
    to_call_before: int
    stack_before: int


@dataclass(slots=True)
class DecisionView:
    """Everything an agent may know when it is asked to act.

    Seats are table seat numbers. ``seat_order`` lists the seats dealt into
    the hand clockwise from the first seat left of the button; the button is
    last. ``players_in_hand`` and ``players_all_in`` follow the same order.

    Attributes:
        hand_id, street, seat, player: Where and who.
        hole_cards: The acting player's two cards (ints, rank*4 + suit).
        board: Community cards dealt so far.
        pot: All chips committed in the hand so far (antes, blinds, bets).
        to_call: Chips needed to call (may exceed the stack: calling then
            puts the whole stack in).
        current_bet: The street total to match (``to_call`` + own street
            contribution).
        min_raise_to, max_raise_to: Allowed street totals for a bet or raise
            (``max_raise_to`` is all-in). Both 0 when ``can_raise`` is False.
        can_raise: A bet or raise is allowed. Use ``bet`` when
            ``current_bet`` is 0 and ``raise`` otherwise (see
            :attr:`raise_kind`).
        stacks: Chips behind for every seat.
        street_contrib: Chips each seat committed on this street.
        players_in_hand: Seats that have not folded.
        players_all_in: Seats with no chips behind.
        button_seat: The button.
        seat_order: Seats clockwise from left of the button.
        action_history: Every action of the hand so far, forced bets included.
        raises_this_street: Bets and raises made on this street so far.
        last_aggressor: Seat of the last bettor or raiser on this street.
        revealed_cards: Hole cards of other players this agent is allowed to
            see (empty for an honest player).
        bb: The big blind, for sizing.
    """

    hand_id: int
    street: str
    seat: int
    player: str
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    pot: int
    to_call: int
    current_bet: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool
    stacks: dict[int, int]
    street_contrib: dict[int, int]
    players_in_hand: tuple[int, ...]
    players_all_in: tuple[int, ...]
    button_seat: int
    seat_order: tuple[int, ...]
    action_history: list[ActionRecord]
    raises_this_street: int
    last_aggressor: Optional[int]
    revealed_cards: dict[int, tuple[int, int]]
    bb: int

    @property
    def raise_kind(self) -> str:
        """``"bet"`` if nobody has bet on this street yet, else ``"raise"``."""
        return "bet" if self.current_bet == 0 else "raise"


class Agent(Protocol):
    """Anything with an ``act`` method can play a seat."""

    def act(self, view: DecisionView) -> Decision:  # pragma: no cover - protocol
        ...


class RandomAgent:
    """Baseline agent that picks random legal actions; useful for tests and benchmarks.

    Args:
        seed: Seed of the agent's own random generator.
        aggression: Probability of betting or raising whenever that is allowed.
        all_in_share: Share of bets and raises that go all-in. Half of the
            rest are minimum raises and half a uniform size up to all-in.
        fold_share: Probability of folding when facing a bet and not raising.
    """

    def __init__(self, seed: Optional[int] = None, aggression: float = 0.25,
                 all_in_share: float = 0.2, fold_share: float = 0.3) -> None:
        self.rng = random.Random(seed)
        self.aggression = aggression
        self.all_in_share = all_in_share
        self.fold_share = fold_share

    def act(self, view: DecisionView) -> Decision:
        rng = self.rng
        if view.can_raise and rng.random() < self.aggression:
            u = rng.random()
            if u < self.all_in_share:
                to = view.max_raise_to
            elif u < (1.0 + self.all_in_share) / 2:
                to = view.min_raise_to
            else:
                to = rng.randint(view.min_raise_to, view.max_raise_to)
            return Decision(view.raise_kind, to)
        if view.to_call == 0:
            return _CHECK
        return _FOLD if rng.random() < self.fold_share else _CALL


@dataclass(slots=True)
class SeatRecord:
    """A seat dealt into the hand and its stack before the antes and blinds."""

    seat: int
    player: str
    stack_start: int


@dataclass(slots=True)
class PotRecord:
    """The main pot (index 0) or a side pot, after the uncalled bet came back.

    Attributes:
        amount: Chips in the pot before rake.
        rake: Part of the hand's rake taken from this pot.
        eligible: Seats still in the hand that can win it.
        winners: Seats with the best hand among ``eligible`` (several on a tie).
        shares: Chips each winner receives (``amount - rake`` in total).
    """

    amount: int
    rake: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: dict[int, int]


@dataclass(slots=True)
class HandRecord:
    """Complete record of one played hand.

    All per-seat dicts are keyed by table seat number. ``seats`` follows
    seat-number order; ``seat_order`` runs clockwise from left of the button.

    Attributes:
        hand_id, table_name, max_seats, button_seat, sb, bb, ante, start_time:
            Header information.
        seats: Seat, player and starting stack of every seat dealt in.
        seat_order: Seats clockwise from the first seat left of the button.
        sb_seat, bb_seat: Seats in the blind positions (heads-up the button
            is the small blind).
        hole_cards: Both hole cards of every seat.
        board: Community cards, as far as they were dealt.
        actions: Forced bets and decisions in order.
        uncalled: Seat -> chips returned as an uncalled bet (at most one entry).
        pots: Main pot first, then side pots. There is always a main pot; it
            holds 0 chips only when everyone folds to the big blind at a
            table without a small blind or antes.
        rake: Total rake.
        contributed: Chips each seat put in, after the uncalled return.
        collected: Chips each seat won from the pots.
        net: ``collected - contributed`` per seat; ``sum(net) + rake == 0``.
        final_stacks: Stacks after the hand.
        folded_on: Seat -> street on which that seat folded.
        showdown: True when two or more players reached the showdown.
        showdown_order: Order in which players showed or mucked.
        shown: Seat -> cards shown at the showdown.
        mucked: Seats that mucked at the showdown.

    The best-hand text of every shown hand is available as
    :attr:`descriptions`; it is computed on first use, so playing hands that
    are never written out costs nothing extra.
    """

    hand_id: int
    table_name: str
    max_seats: int
    button_seat: int
    sb: int
    bb: int
    ante: int
    start_time: datetime
    seats: tuple[SeatRecord, ...]
    seat_order: tuple[int, ...]
    sb_seat: int
    bb_seat: int
    hole_cards: dict[int, tuple[int, int]]
    board: tuple[int, ...]
    actions: list[ActionRecord]
    uncalled: dict[int, int]
    pots: list[PotRecord]
    rake: int
    contributed: dict[int, int]
    collected: dict[int, int]
    net: dict[int, int]
    final_stacks: dict[int, int]
    folded_on: dict[int, str]
    showdown: bool
    showdown_order: tuple[int, ...]
    shown: dict[int, tuple[int, int]]
    mucked: tuple[int, ...]
    _descriptions: Optional[dict[int, str]] = field(default=None, repr=False, compare=False)

    @property
    def descriptions(self) -> dict[int, str]:
        """Seat -> best-hand text of every shown hand, e.g. ``"a flush, King high"``."""
        if self._descriptions is None:
            self._descriptions = {seat: describe_hand(cards + self.board) for seat, cards in self.shown.items()}
        return self._descriptions

    @property
    def players(self) -> dict[int, str]:
        """Seat -> player name."""
        return {s.seat: s.player for s in self.seats}

    @property
    def total_pot(self) -> int:
        """All chips in the pots, rake included."""
        return sum(p.amount for p in self.pots)


# --------------------------------------------------------------------------- engine


def play_hand(
    config: TableConfig,
    seats: Sequence[SeatState],
    button_seat: int,
    agents: Mapping[str, Agent],
    hand_id: int,
    table_name: str,
    start_time: datetime,
    rng: Optional[np.random.Generator],
    info_grants: Optional[InfoGrants] = None,
    evaluate: Optional[Evaluator] = None,
    *,
    deck: Optional[Sequence[int]] = None,
) -> HandRecord:
    """Play one hand and return its record.

    Args:
        config: Stakes and house rules.
        seats: 2 to ``config.max_seats`` occupied seats with positive stacks.
            Player names must be unique single words without spaces that
            neither start with ``*`` (a section line) nor end with ``:`` (an
            action line), so that the hand-history text stays unambiguous.
        button_seat: The dealer button; must be one of the occupied seats.
        agents: Player name -> agent deciding for that player.
        hand_id, table_name, start_time: Copied into the record.
        rng: numpy Generator that shuffles the deck (one permutation per hand).
        info_grants: Player name -> names of the players whose hole cards that
            player sees, or ``"all"`` for every other player at the table.
            Names not seated in this hand are ignored. This is how a simulated
            superuser gets its information; honest players are left out and
            see nothing.
        evaluate: Hand evaluator for the showdown, ``evaluate(cards)`` with an
            (n, 7) int array returning n strengths (higher wins). Defaults to
            :func:`sudetect.evaluator.eval_hands`.
        deck: Optional fixed card order (at least ``2 * players + 5`` distinct
            cards) used instead of ``rng``, for replays and tests. Cards are
            dealt one at a time starting left of the button, then the board.

    Raises:
        ValueError: Invalid seating, or an agent returned an illegal decision.
    """
    hand = _Hand(config, seats, button_seat, agents, hand_id, rng, info_grants, deck)
    return hand.play(evaluate, table_name, start_time)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _default_evaluate(cards: np.ndarray) -> np.ndarray:
    # the engine deals distinct cards from one deck, so the input checks are skipped
    return eval_hands(cards, check=False)


class _Hand:
    """State of one hand in play. Index 0 is the first seat left of the button."""

    def __init__(
        self,
        config: TableConfig,
        seats: Sequence[SeatState],
        button_seat: int,
        agents: Mapping[str, Agent],
        hand_id: int,
        rng: Optional[np.random.Generator],
        info_grants: Optional[InfoGrants],
        deck: Optional[Sequence[int]],
    ) -> None:
        if not isinstance(config, TableConfig):
            raise TypeError("config must be a TableConfig")
        n = len(seats)
        if not 2 <= n <= config.max_seats:
            raise ValueError(f"need 2 to {config.max_seats} players, got {n}")
        by_seat = sorted(seats, key=lambda s: s.seat_no)
        seat_nos = [s.seat_no for s in by_seat]
        if len(set(seat_nos)) != n:
            raise ValueError("two players share a seat")
        names = [s.player for s in by_seat]
        if len(set(names)) != n:
            raise ValueError("player names must be unique")
        for s in by_seat:
            if not _is_int(s.seat_no) or not 1 <= s.seat_no <= config.max_seats:
                raise ValueError(f"seat {s.seat_no!r} is outside 1..{config.max_seats}")
            if not _is_int(s.stack) or s.stack <= 0:
                raise ValueError(f"{s.player} needs a positive integer stack, got {s.stack!r}")
            if (not isinstance(s.player, str) or s.player.split() != [s.player]
                    or s.player.startswith("*") or s.player.endswith(":")):
                raise ValueError(f"invalid player name {s.player!r}: use one word without spaces"
                                 " that does not start with '*' or end with ':'")
            if s.player not in agents:
                raise ValueError(f"no agent for player {s.player}")
        if button_seat not in seat_nos:
            raise ValueError(f"the button must be on an occupied seat, got {button_seat}")

        b = seat_nos.index(button_seat)
        order = by_seat[b + 1:] + by_seat[:b + 1]  # clockwise from left of the button
        self.cfg = config
        self.n = n
        self.hand_id = hand_id
        self.button_seat = int(button_seat)
        self.seat_nos: tuple[int, ...] = tuple(int(s.seat_no) for s in order)
        self.names: tuple[str, ...] = tuple(s.player for s in order)
        self.agents = [agents[s.player] for s in order]
        self.stack_start = [int(s.stack) for s in order]
        self.stack = list(self.stack_start)

        if deck is None:
            if rng is None:
                raise ValueError("pass a numpy Generator as rng, or a fixed deck")
            cards = rng.permutation(52).tolist()
        else:
            cards = [int(c) for c in deck]
            if (len(cards) < 2 * n + 5 or len(set(cards)) != len(cards)
                    or not all(0 <= c < 52 for c in cards)):
                raise ValueError(f"deck must hold at least {2 * n + 5} distinct cards in 0..51")
        self.hole = [(cards[i], cards[n + i]) for i in range(n)]
        self.board_all = tuple(cards[2 * n:2 * n + 5])
        self.board: tuple[int, ...] = ()

        # heads-up the button (last index) is the small blind and acts first preflop
        if n == 2:
            self.sb_idx, self.bb_idx, self.preflop_first = 1, 0, 1
        else:
            self.sb_idx, self.bb_idx, self.preflop_first = 0, 1, 2

        self.revealed = self._grants(info_grants or {})

        self.total = [0] * n  # chips committed in the hand
        self.folded = [False] * n
        self.all_in = [False] * n
        self.folded_on: dict[int, str] = {}
        self.n_live = n  # not folded
        self.n_can_act = n  # not folded and not all-in
        self.pot = 0
        self.actions: list[ActionRecord] = []
        self.uncalled: dict[int, int] = {}
        self._in_hand_cache: Optional[tuple[int, ...]] = None
        self._all_in_cache: Optional[tuple[int, ...]] = None
        self._new_street(0)

    # ------------------------------------------------------------------ setup

    def _grants(self, info_grants: InfoGrants) -> dict[int, dict[int, tuple[int, int]]]:
        """Index -> {seat: hole cards} of the players that index may see."""
        revealed: dict[int, dict[int, tuple[int, int]]] = {}
        index_of = {name: i for i, name in enumerate(self.names)}
        for player, grant in info_grants.items():
            i = index_of.get(player)
            if i is None or not grant:
                continue
            if isinstance(grant, str):
                if grant != "all":
                    raise ValueError(f"info grant for {player} must be a set of players or 'all'")
                targets = [k for k in range(self.n) if k != i]
            else:
                targets = [index_of[p] for p in grant if p in index_of and index_of[p] != i]
            if targets:
                revealed[i] = {self.seat_nos[k]: self.hole[k] for k in sorted(targets)}
        return revealed

    def _new_street(self, street: int) -> None:
        n = self.n
        self.street = street
        self.board = self.board_all[:_BOARD_LEN[street]]
        self.street_bet = [0] * n
        self.acted = [False] * n
        self.acted_level = [0] * n  # bet level each player last acted at
        self.current_bet = 0
        self.full_raise = 0  # last full bet/raise increment on this street
        self.raises = 0
        self.last_aggressor = -1

    # --------------------------------------------------------------- the hand

    def play(self, evaluate: Optional[Evaluator], table_name: str, start_time: datetime) -> HandRecord:
        cfg = self.cfg
        if cfg.ante > 0:
            for i in range(self.n):
                self._post(i, "post_ante", cfg.ante)
        if cfg.sb > 0 and not self.all_in[self.sb_idx]:
            self._post(self.sb_idx, "post_sb", cfg.sb)
        if not self.all_in[self.bb_idx]:
            self._post(self.bb_idx, "post_bb", cfg.bb)
        # the full big blind sets the price before the flop, even if it was posted short
        self.current_bet = cfg.bb
        self.full_raise = cfg.bb
        self._betting_round(self.preflop_first)

        for street in (1, 2, 3):
            if self.n_live < 2:
                break
            self._new_street(street)
            if self.n_can_act >= 2:
                self._betting_round(0)

        self._return_uncalled()
        return self._settle(evaluate, table_name, start_time)

    def _post(self, i: int, kind: str, amount: int) -> None:
        stack_before = self.stack[i]
        amount = min(amount, stack_before)
        pot_before = self.pot
        self.stack[i] -= amount
        self.total[i] += amount
        self.pot += amount
        if kind != "post_ante":  # antes are dead money, not part of the street bet
            self.street_bet[i] += amount
        is_all_in = self.stack[i] == 0
        if is_all_in:
            self._mark_all_in(i)
        self._record(i, kind, amount, None, None, is_all_in, pot_before, 0, stack_before)

    def _betting_round(self, first: int) -> None:
        start = first
        while self.n_live > 1:
            j = self._next_actor(start)
            if j < 0:
                return
            self._act(j)
            start = (j + 1) % self.n

    def _next_actor(self, start: int) -> int:
        """First index from ``start`` clockwise that still has to act, or -1."""
        if self.n_can_act == 0:
            return -1
        sole = self.n_can_act == 1
        n = self.n
        for k in range(n):
            j = (start + k) % n
            if self.folded[j] or self.all_in[j]:
                continue
            if sole:
                # nobody can answer a raise: act only to match a larger contribution
                return j if self.street_bet[j] < self._max_other_bet(j) else -1
            if not self.acted[j] or self.street_bet[j] < self.current_bet:
                return j
        return -1

    def _max_other_bet(self, j: int) -> int:
        return max(b for k, b in enumerate(self.street_bet) if k != j)

    def _act(self, j: int) -> None:
        bb = self.cfg.bb
        others_can_act = self.n_can_act > 1
        own = self.street_bet[j]
        level = self.current_bet if others_can_act else self._max_other_bet(j)
        to_call = level - own
        stack = self.stack[j]
        all_in_to = own + stack
        full = self.full_raise if self.full_raise > bb else bb
        can_raise = (
            others_can_act
            and all_in_to > level
            and (not self.acted[j] or level - self.acted_level[j] >= full)
        )
        if can_raise:
            min_to = level + full
            if min_to > all_in_to:
                min_to = all_in_to
            max_to = all_in_to
        else:
            min_to = max_to = 0

        view = DecisionView(
            hand_id=self.hand_id,
            street=STREETS[self.street],
            seat=self.seat_nos[j],
            player=self.names[j],
            hole_cards=self.hole[j],
            board=self.board,
            pot=self.pot,
            to_call=to_call,
            current_bet=level,
            min_raise_to=min_to,
            max_raise_to=max_to,
            can_raise=can_raise,
            stacks=dict(zip(self.seat_nos, self.stack)),
            street_contrib=dict(zip(self.seat_nos, self.street_bet)),
            players_in_hand=self._players_in_hand(),
            players_all_in=self._players_all_in(),
            button_seat=self.button_seat,
            seat_order=self.seat_nos,
            action_history=list(self.actions),
            raises_this_street=self.raises,
            last_aggressor=self.seat_nos[self.last_aggressor] if self.last_aggressor >= 0 else None,
            revealed_cards=dict(self.revealed[j]) if j in self.revealed else {},
            bb=bb,
        )
        decision = self.agents[j].act(view)
        kind = getattr(decision, "kind", None)
        who = self.names[j]
        pot_before = self.pot
        amount = 0
        to: Optional[int] = None
        raise_by: Optional[int] = None

        if kind == "fold":
            if to_call <= 0:
                raise ValueError(f"{who} cannot fold with nothing to call; check instead")
            self.folded[j] = True
            self.folded_on[self.seat_nos[j]] = STREETS[self.street]
            self.n_live -= 1
            self.n_can_act -= 1
            self._in_hand_cache = None
        elif kind == "check":
            if to_call != 0:
                raise ValueError(f"{who} cannot check facing {to_call} to call")
            self.acted[j] = True
            self.acted_level[j] = level
        elif kind == "call":
            if to_call <= 0:
                raise ValueError(f"{who} has nothing to call; check instead")
            amount = to_call if to_call < stack else stack
            self.acted[j] = True
            self.acted_level[j] = level
        elif kind == "bet" or kind == "raise":
            if not can_raise:
                raise ValueError(f"{who} may not bet or raise here")
            if kind == "bet" and level != 0:
                raise ValueError(f"{who} faces a bet of {level}; use 'raise'")
            if kind == "raise" and level == 0:
                raise ValueError(f"nobody has bet yet; {who} should use 'bet'")
            target = getattr(decision, "to", None)
            if not _is_int(target):
                raise ValueError(f"{kind} needs an integer street total in 'to', got {target!r}")
            to = int(target)
            if not min_to <= to <= max_to:
                raise ValueError(f"{who} must {kind} to between {min_to} and {max_to}, got {to}")
            amount = to - own
            raise_by = to - level
            if raise_by >= full:  # a full raise sets the next minimum increment
                self.full_raise = raise_by
            self.current_bet = to
            self.raises += 1
            self.last_aggressor = j
            self.acted[j] = True
            self.acted_level[j] = to
        else:
            raise ValueError(f"unknown decision kind {kind!r}; expected one of {DECISION_KINDS}")

        is_all_in = False
        if amount:
            self.stack[j] -= amount
            self.street_bet[j] += amount
            self.total[j] += amount
            self.pot += amount
            if self.stack[j] == 0:
                is_all_in = True
                self._mark_all_in(j)
        self._record(j, kind, amount, to, raise_by, is_all_in, pot_before, to_call, stack)

    def _mark_all_in(self, i: int) -> None:
        self.all_in[i] = True
        self.n_can_act -= 1
        self._all_in_cache = None

    def _players_in_hand(self) -> tuple[int, ...]:
        if self._in_hand_cache is None:
            self._in_hand_cache = tuple(s for s, f in zip(self.seat_nos, self.folded) if not f)
        return self._in_hand_cache

    def _players_all_in(self) -> tuple[int, ...]:
        if self._all_in_cache is None:
            self._all_in_cache = tuple(s for s, a in zip(self.seat_nos, self.all_in) if a)
        return self._all_in_cache

    def _record(self, i: int, kind: str, amount: int, to: Optional[int], raise_by: Optional[int],
                is_all_in: bool, pot_before: int, to_call: int, stack_before: int) -> None:
        self.actions.append(ActionRecord(
            len(self.actions), STREETS[self.street], self.seat_nos[i], self.names[i], kind,
            amount, to, raise_by, is_all_in, pot_before, to_call, stack_before,
        ))

    def _return_uncalled(self) -> None:
        """Give back the part of the largest contribution nobody matched."""
        totals = self.total
        top = max(totals)
        leaders = [i for i, t in enumerate(totals) if t == top]
        if len(leaders) != 1:
            return
        i = leaders[0]
        excess = top - max(t for k, t in enumerate(totals) if k != i)
        if excess > 0:
            self.stack[i] += excess
            totals[i] -= excess
            self.pot -= excess
            self.uncalled[self.seat_nos[i]] = excess

    # ----------------------------------------------------------- settlement

    def _settle(self, evaluate: Optional[Evaluator], table_name: str, start_time: datetime) -> HandRecord:
        n, seat_nos, totals = self.n, self.seat_nos, self.total
        live = [i for i in range(n) if not self.folded[i]]
        pot_total = sum(totals)
        rake = self.cfg.rake_for(pot_total, len(self.board) >= 3)

        # pots by the contribution levels of the players still in the hand
        pots: list[tuple[int, list[int]]] = []
        previous = 0
        for level in sorted({totals[i] for i in live}):
            amount = sum(min(t, level) - previous for t in totals if t > previous)
            pots.append((amount, [i for i in live if totals[i] >= level]))
            previous = level
        if sum(a for a, _ in pots) != pot_total:  # pragma: no cover - invariant
            raise AssertionError("pot formation lost chips")

        showdown = len(live) > 1
        strength: dict[int, Any] = {}
        if showdown:
            rows = np.array([self.hole[i] + self.board for i in live], dtype=np.int64)
            fn = evaluate if evaluate is not None else _default_evaluate
            values = np.asarray(fn(rows)).tolist()
            if len(values) != len(live):
                raise ValueError("evaluate() must return one strength per row")
            strength = dict(zip(live, values))

        collected = [0] * n
        pot_records: list[PotRecord] = []
        winners_any: set[int] = set()
        rake_left = rake
        for amount, eligible in pots:
            pot_rake = min(rake_left, amount)
            rake_left -= pot_rake
            if len(eligible) == 1:
                winners = eligible
            else:
                best = max(strength[i] for i in eligible)
                winners = [i for i in eligible if strength[i] == best]
            share, odd = divmod(amount - pot_rake, len(winners))
            shares = {seat_nos[w]: share for w in winners}
            shares[seat_nos[winners[0]]] += odd  # first tied winner left of the button
            for w in winners:
                collected[w] += shares[seat_nos[w]]
            winners_any.update(winners)
            pot_records.append(PotRecord(
                amount, pot_rake, tuple(seat_nos[i] for i in eligible),
                tuple(seat_nos[w] for w in winners), shares,
            ))

        showdown_order: tuple[int, ...] = ()
        shown: dict[int, tuple[int, int]] = {}
        mucked: tuple[int, ...] = ()
        if showdown:
            first = live[0]
            if self.street == 3 and self.last_aggressor >= 0 and not self.folded[self.last_aggressor]:
                first = self.last_aggressor  # the river's last aggressor shows first
            k = live.index(first)
            order = live[k:] + live[:k]
            muck = []
            for pos, i in enumerate(order):
                # the first player always shows; later ones only with a hand that wins or ties a pot
                if pos == 0 or i in winners_any:
                    shown[seat_nos[i]] = self.hole[i]
                else:
                    muck.append(seat_nos[i])
            showdown_order = tuple(seat_nos[i] for i in order)
            mucked = tuple(muck)

        by_seat = sorted(range(n), key=lambda i: seat_nos[i])
        return HandRecord(
            hand_id=self.hand_id,
            table_name=table_name,
            max_seats=self.cfg.max_seats,
            button_seat=self.button_seat,
            sb=self.cfg.sb,
            bb=self.cfg.bb,
            ante=self.cfg.ante,
            start_time=start_time,
            seats=tuple(SeatRecord(seat_nos[i], self.names[i], self.stack_start[i]) for i in by_seat),
            seat_order=seat_nos,
            sb_seat=seat_nos[self.sb_idx],
            bb_seat=seat_nos[self.bb_idx],
            hole_cards={seat_nos[i]: self.hole[i] for i in by_seat},
            board=self.board,
            actions=self.actions,
            uncalled=self.uncalled,
            pots=pot_records,
            rake=rake,
            contributed={seat_nos[i]: totals[i] for i in by_seat},
            collected={seat_nos[i]: collected[i] for i in by_seat},
            net={seat_nos[i]: collected[i] - totals[i] for i in by_seat},
            final_stacks={seat_nos[i]: self.stack[i] + collected[i] for i in by_seat},
            folded_on=self.folded_on,
            showdown=showdown,
            showdown_order=showdown_order,
            shown=shown,
            mucked=mucked,
        )
