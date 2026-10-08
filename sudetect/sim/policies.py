"""Betting policies for the simulator: an honest field and planted superusers.

Every simulated player is an agent with an ``act(view) -> Decision`` method
(see :mod:`sudetect.sim.engine`). Two kinds exist.

Honest players (:class:`HonestPolicy`) decide from what a real player can
know: their own two cards, the board, the betting so far, stack sizes and,
for players whose style says so, the style class of each opponent (the read a
HUD gives after a few hundred hands). They never look at
``DecisionView.revealed_cards``.

* Preflop, a hand is placed on a 169-class ranking by its equity against one
  random hand (:data:`sudetect.equity.PREFLOP_VS_RANDOM`) and expressed as a
  percentile of all 1,326 combos (0 = aces, 1 = the worst hand). Opening,
  3-bet, calling and 4-bet decisions compare that percentile, plus gaussian
  noise, with widths that depend on the position and the player's style.
* After the flop the hand's strength ``h`` is its equity against one random
  hand on the current board: Monte Carlo with a seed fixed per decision on the
  flop, exact on the turn and river. Draws add a small implied-odds bonus
  when the player calls. Facing a bet, the player turns ``h`` into a
  perceived equity against the bettor's range: a logistic curve centred on
  the strength the bettor is believed to show, which grows with the street,
  the bet size, raises and the number of opponents. The player continues when
  that perceived equity, plus noise, clears the pot odds plus a style margin,
  and raises with the top of it (value) and with some draws (semi-bluffs).
  At heads-up tables, where ranges are wide, bets are believed weaker when
  calling (not when raising) and value bets are thinner. Not facing a bet,
  the player bets hands above a value threshold and bluffs at style rates
  that are higher for the previous street's aggressor (continuation bets),
  lower multiway, and on the turn and river apply mostly to air: medium hands
  check for showdown value.

Planted superusers (:class:`SuperuserPolicy`) play the honest policy of their
own style (a strong regular), and on a share ``u`` of their postflop
decisions in which they can see every live opponent's cards they use the
informed rules instead: continue facing a bet only when the equity against
the real cards clears the pot odds by a margin, raise for value with very
high equity, bet for value with high equity (bigger when the opponent holds
something he will call with), and bluff with low equity when every opponent
is weak enough to fold.

All randomness is drawn from a numpy Generator seeded per hand and per
player (``(seed, hand_id, player_index, stream tag)``), so a run is
reproducible no matter how hands are split across processes, and no
player's stream is the deck's (``(seed, hand_id)`` in the runner).

The numbers that shape the policies (style parameters, base opening ranges,
the bettor-strength model, the informed rules) live in the ``[sim]`` section
of ``config/sim.toml`` and reach this module through
:mod:`sudetect.sim.population`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from .. import equity as _equity
from ..cards import CLASS_SIZE, COMBO_INDEX, COMBO_TO_CLASS, COMBOS, N_CLASSES
from .engine import Decision, DecisionView

__all__ = [
    "ARCHETYPES",
    "POSITIONS",
    "Style",
    "PreflopModel",
    "PostflopModel",
    "SuperuserRules",
    "PolicyModel",
    "HonestPolicy",
    "SuperuserPolicy",
    "position_class",
    "preflop_percentile",
    "preflop_percentile_table",
    "draw_kind",
    "hand_strength",
]

ARCHETYPES: tuple[str, ...] = ("tag", "lag", "nit", "loose_passive", "maniac", "strong")
"""Style archetypes of the honest field ("strong" = a winning regular with no information)."""

POSITIONS: tuple[str, ...] = ("EP", "HJ", "CO", "BTN", "SB", "BB")
"""Position classes used by the preflop model (heads-up: BTN and BB only)."""

_STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_AGGRESSIVE = ("bet", "raise")
_INFORMED, _DECISIONS = 1, 2  # last word of the seed keys of a player's two random streams (never 0)


# --------------------------------------------------------------------------- parameters


def _from_mapping(cls: type, data: Mapping[str, Any], what: str) -> Any:
    """Build a frozen dataclass from a mapping; every field must be present."""
    names = [f.name for f in fields(cls)]
    missing = [n for n in names if n not in data]
    unknown = [k for k in data if k not in names]
    if missing or unknown:
        raise ValueError(f"{what}: missing {missing}, unknown {unknown}")
    values = {}
    for f in fields(cls):
        v = data[f.name]
        if isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        elif isinstance(v, dict):
            v = dict(v)
        values[f.name] = v
    return cls(**values)


@dataclass(frozen=True)
class Style:
    """Style parameters of one player (an archetype with jitter applied).

    Widths are fractions of all 1,326 combos, taken from the top of the
    preflop ranking. Probabilities are per decision. ``h`` thresholds are
    equities against one random hand.

    Preflop (ring tables):
        rfi: multiplier on the base open-raise widths per position.
        limp: extra width limped when first in (passive players).
        overlimp: width that limps behind other limpers.
        iso: iso-raise width over limpers, as a share of the open width.
        threebet: 3-bet width against a single open (scaled per position).
        cold_call: calling width against a single open, beyond the 3-bets.
        bb_defend: big-blind calling width against a standard open.
        fold_3bet: share of the player's own opening range that folds to a 3-bet.
        fourbet: 4-bet width (fraction of all hands).
        call_4bet: calling width against a 4-bet.
        open_size: open-raise size in big blinds; an iso-raise over limpers
            adds one big blind plus one per limper.
    Preflop (heads-up):
        hu_open, hu_limp: button raise width and extra limp width.
        hu_defend: big-blind calling width against a standard open.
        hu_3bet: big-blind 3-bet width.
        hu_iso: big-blind raise width against a limp (to ``hu_open_size`` + 1 bb).
        hu_open_size: button open size in big blinds.
    Postflop:
        value_thr: ``h`` needed to bet for value on the flop (later streets
            add the model's per-street shift).
        cbet: probability of a continuation bet without a value hand.
        barrel: probability of betting again on the turn or river as the
            previous street's aggressor without a value hand.
        stab: probability of betting when checked to in position.
        bluff: probability of leading without a value hand otherwise.
        semibluff: probability of betting or raising with a draw.
        call_margin: margin over the pot odds needed to continue.
        belief: shift of the bettor's believed strength (negative = sceptical).
        raise_thr: perceived equity needed to raise for value.
        raise_prob: probability of raising when ``raise_thr`` is met.
        slowplay: probability of checking a very strong hand when not the aggressor.
        sizes: weights of the bet sizes in the model's size grid (pot fractions).
        polar: probability of moving one size up with very strong hands
            (``h`` at least the model's ``strong_h``) and with bluffs.
        raise_mult: postflop raise size as a multiple of the bet faced.
        noise: standard deviation of the gaussian noise added to every
            postflop threshold comparison.
        pf_noise: standard deviation of the noise added to the preflop percentile.
        reads: uses the opponents' style classes (HUD reads) to adjust bluffs,
            value thresholds and beliefs.
    """

    rfi: float
    limp: float
    overlimp: float
    iso: float
    threebet: float
    cold_call: float
    bb_defend: float
    fold_3bet: float
    fourbet: float
    call_4bet: float
    open_size: float
    hu_open: float
    hu_limp: float
    hu_defend: float
    hu_3bet: float
    hu_iso: float
    hu_open_size: float
    value_thr: float
    cbet: float
    barrel: float
    stab: float
    bluff: float
    semibluff: float
    call_margin: float
    belief: float
    raise_thr: float
    raise_prob: float
    slowplay: float
    sizes: tuple[float, ...]
    polar: float
    raise_mult: float
    noise: float
    pf_noise: float
    reads: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Style":
        return _from_mapping(cls, data, "style")

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


@dataclass(frozen=True)
class PreflopModel:
    """Position tables shared by all players.

    Attributes:
        rfi: base open-raise width per position class (a regular's ranges).
        threebet_pos: 3-bet width multiplier per position class of the 3-bettor.
        cold_call_pos: calling-width multiplier per position class of the caller.
        standard_open: open size in big blinds that the calling widths assume;
            larger raises shrink them (``(standard / size) ** size_power``).
        size_power: exponent of that shrink.
        threebet_ip, threebet_oop: 3-bet size as a multiple of the open, in
            position and out of position.
        fourbet_mult: 4-bet size as a multiple of the 3-bet.
        shove_frac: a raise that would put in more than this share of the
            stack goes all-in instead.
        value_share: share of a 3-bet or 4-bet range taken from the top of
            the ranking; the rest are bluffs drawn from just below the
            calling range.
        squeeze_tighten: calling-width multiplier for every caller already in.
        size_round_bb: preflop raise sizes are rounded to this many big blinds.
        size_factor_clip: lower and upper bound of the calling-width multiplier
            for the raise size.
        sb_limp_mult, sb_overlimp_mult: the small blind completes this many
            times wider than its style's ``limp`` / ``overlimp`` widths.
        bb_iso_multi: the big blind's iso-raise width is multiplied by this
            when two or more players limped.
        limp_call_keep: a limper facing a raise calls with this share of his
            limping range.
        cold_3bet_call: a caller or limper facing a 3-bet calls with
            ``call_4bet`` times this.
        fivebet_share: share of the 4-bet width that moves all-in against a 4-bet.
        bluff_band_min: the band below the calling range that 3-bet bluffs
            come from is at least this wide.
    """

    rfi: dict[str, float]
    threebet_pos: dict[str, float]
    cold_call_pos: dict[str, float]
    standard_open: float
    size_power: float
    threebet_ip: float
    threebet_oop: float
    fourbet_mult: float
    shove_frac: float
    value_share: float
    squeeze_tighten: float
    size_round_bb: float
    size_factor_clip: tuple[float, float]
    sb_limp_mult: float
    sb_overlimp_mult: float
    bb_iso_multi: float
    limp_call_keep: float
    cold_3bet_call: float
    fivebet_share: float
    bluff_band_min: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PreflopModel":
        return _from_mapping(cls, data, "preflop model")


@dataclass(frozen=True)
class PostflopModel:
    """The bettor-strength model and the rest of the shared postflop constants.

    Attributes:
        m_street: believed strength (equity vs a random hand) of a bettor's
            range by street, for a bet of ``size_ref`` of the pot.
        m_size: added per pot of bet size above ``size_ref``.
        size_ref: reference bet size (fraction of the pot).
        m_raise: added per raise on the street.
        m_barrel: added per earlier street the same player bet or raised on.
        m_multi: added per live opponent beyond the first.
        width: scale of the logistic curve from ``h`` to perceived equity.
        value_shift: added to ``value_thr`` by street (flop, turn, river).
        value_multi: added to ``value_thr`` per opponent beyond the first.
        bluff_multi: bluff and continuation-bet probabilities are multiplied
            by this per opponent beyond the first.
        weak_h: below this ``h`` a hand counts as air.
        medium_barrel: on the turn and river, bluff probabilities (barrels,
            stabs, leads) of hands between air and value are multiplied by
            this: medium hands check for showdown value.
        implied: implied-odds bonus to ``h`` when calling with a draw, by draw
            kind (gutshot, open-ended or flush draw, combo draw); halved on the turn.
        flop_mc_samples: Monte Carlo deals for ``h`` on the flop.
        size_grid: bet sizes (fractions of the pot) that ``Style.sizes`` weights.
        allin_snap: a bet or raise within this share of the stack goes all-in.
        bluff_raise: probability of a pure bluff raise with air facing a bet,
            as a multiple of the player's ``bluff``.
        m_hu: added to the believed strength at heads-up tables by street
            (flop, turn, river), where ranges are wide and bets mean less;
            used for calling decisions only.
        value_hu: added to the value threshold at heads-up tables.
        strong_h: ``h`` from which a hand counts as very strong (slow-plays and
            bigger value sizes).
        semibluff_raise: facing a bet, a draw raises with probability
            ``semibluff`` times this.
        size_cap: bet sizes above this many pots count as this in the believed strength.
    """

    m_street: tuple[float, float, float]
    m_size: float
    size_ref: float
    m_raise: float
    m_barrel: float
    m_multi: float
    width: float
    value_shift: tuple[float, float, float]
    value_multi: float
    bluff_multi: float
    weak_h: float
    medium_barrel: float
    implied: tuple[float, float, float]
    flop_mc_samples: int
    size_grid: tuple[float, ...]
    allin_snap: float
    bluff_raise: float
    m_hu: tuple[float, float, float]
    value_hu: float
    strong_h: float
    semibluff_raise: float
    size_cap: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PostflopModel":
        return _from_mapping(cls, data, "postflop model")


@dataclass(frozen=True)
class SuperuserRules:
    """The informed rules a planted superuser uses on postflop decisions.

    Attributes:
        call_margin: continue facing a bet iff real equity >= pot odds + this
            (pot odds as the action table measures them: the call capped by
            the stack, side pots counted).
        raise_eq, raise_prob: raise for value with real equity at least
            ``raise_eq``, with this probability.
        value_eq: bet for value with real equity at least this.
        bluff_eq: bluff only with real equity below this ...
        bluff_opp_max: ... and when every live opponent's own ``h`` is below this ...
        bluff_prob: ... with this probability.
        opp_strong: an opponent whose own ``h`` is at least this will call a
            big bet, so the value bet uses ``value_size_strong``.
        value_size_strong, value_size_weak, bluff_size: bet sizes (pot fractions).
        preflop_info: use the information preflop too (off by default).
    """

    call_margin: float
    raise_eq: float
    raise_prob: float
    value_eq: float
    bluff_eq: float
    bluff_opp_max: float
    bluff_prob: float
    opp_strong: float
    value_size_strong: float
    value_size_weak: float
    bluff_size: float
    preflop_info: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SuperuserRules":
        return _from_mapping(cls, data, "superuser rules")


@dataclass(frozen=True)
class PolicyModel:
    """Everything the policies share: the preflop and postflop models, the
    informed rules, and how a reads-using player adjusts to each style class.

    ``exploit`` maps an opponent archetype to ``(bluff_mult, value_shift,
    belief_shift)``: bluffs against him are multiplied by ``bluff_mult``, the
    value threshold moves by ``value_shift`` and his bets are believed
    ``belief_shift`` stronger.
    """

    preflop: PreflopModel
    postflop: PostflopModel
    superuser: SuperuserRules
    exploit: dict[str, tuple[float, float, float]]


# --------------------------------------------------------------------------- hand helpers


@lru_cache(maxsize=1)
def preflop_percentile_table() -> np.ndarray:
    """Percentile (0 = best) of each of the 169 classes in the preflop ranking.

    Classes are ranked by equity against one random hand; a class's
    percentile is the share of the 1,326 combos ranked above it plus half of
    its own combos. Uses :data:`sudetect.equity.PREFLOP_VS_RANDOM`; if that
    table is empty, the equities are estimated once with a seeded Monte
    Carlo of 20,000 deals per class (same ranking up to near-ties).
    """
    eq = np.asarray(_equity.PREFLOP_VS_RANDOM, dtype=float)
    if eq.shape != (N_CLASSES,):
        first = np.array([COMBOS[np.flatnonzero(COMBO_TO_CLASS == c)[0]] for c in range(N_CLASSES)])
        board = np.full((N_CLASSES, 5), -1)
        eq = _equity.equity_vs_random_batch(first, board, exact=False, n_samples=20_000, seed=169)
    order = np.argsort(-eq, kind="stable")
    sizes = np.asarray(CLASS_SIZE, dtype=float)[order]
    before = np.concatenate([[0.0], np.cumsum(sizes)[:-1]])
    pct = np.empty(N_CLASSES)
    pct[order] = (before + sizes / 2) / sizes.sum()
    return pct


@lru_cache(maxsize=1)
def _percentile_by_cards() -> tuple[tuple[float, ...], ...]:
    """52 x 52 nested tuple: percentile of the hand made of two card ids."""
    pct = preflop_percentile_table()
    table = np.zeros((52, 52))
    for a in range(52):
        for b in range(52):
            if a != b:
                table[a, b] = pct[COMBO_TO_CLASS[COMBO_INDEX[a, b]]]
    return tuple(tuple(row) for row in table.tolist())


def preflop_percentile(hole: Sequence[int]) -> float:
    """Percentile of two hole cards in the preflop ranking (0 = aces, 1 = worst)."""
    return _percentile_by_cards()[hole[0]][hole[1]]


def position_class(n_dealt: int, index: int) -> str:
    """Position class of the player at ``index`` of ``seat_order`` (0 = left of the button).

    Heads-up the button (index 1) is ``BTN`` and the other player ``BB``. With
    three or more players index 0 is ``SB``, index 1 ``BB``, the button
    ``BTN``, then ``CO`` and ``HJ``; earlier seats are ``EP``.
    """
    if n_dealt == 2:
        return "BTN" if index == 1 else "BB"
    if index == 0:
        return "SB"
    if index == 1:
        return "BB"
    d = n_dealt - 1 - index
    return ("BTN", "CO", "HJ")[d] if d < 3 else "EP"


def _window_ranks(lo: int) -> tuple[int, ...]:
    return tuple(12 if r == -1 else r for r in range(lo - 1, lo + 4))


_WINDOWS = tuple(_window_ranks(lo) for lo in range(0, 10))  # A2345 .. TJQKA


def draw_kind(hole: Sequence[int], board: Sequence[int]) -> int:
    """Draw class of a hand on the flop or turn, with the hero's cards contributing.

    Returns 0 (none or a made straight or flush), 1 (gutshot), 2 (open-ended
    straight draw, double gutshot or flush draw) or 3 (flush draw plus a
    straight draw). Always 0 on the river or preflop.
    """
    if len(board) not in (3, 4):
        return 0
    cards = list(hole) + list(board)
    suits = [0, 0, 0, 0]
    for c in cards:
        suits[c & 3] += 1
    hole_suits = (hole[0] & 3, hole[1] & 3)
    flush_draw = False
    for s in range(4):
        if suits[s] >= 5:
            return 0  # a made flush
        if suits[s] == 4 and s in hole_suits:
            flush_draw = True
    ranks = {c >> 2 for c in cards}
    hole_ranks = {hole[0] >> 2, hole[1] >> 2}
    completing = set()
    for window in _WINDOWS:
        have = [r for r in window if r in ranks]
        if len(have) == 5:
            return 0  # a made straight
        if len(have) == 4 and hole_ranks & set(window):
            completing.add(next(r for r in window if r not in ranks))
    straight = 2 if len(completing) >= 2 else (1 if completing else 0)
    if flush_draw:
        return 3 if straight else 2
    return straight


def hand_strength(hole: Sequence[int], board: Sequence[int], *, mc_samples: int, seed: int) -> float:
    """Equity against one random hand on the current board.

    Exact on the turn and river; Monte Carlo with ``mc_samples`` deals and
    ``seed`` on the flop; the preflop table before the flop.
    """
    if len(board) == 3:
        return _equity.equity_vs_random(list(hole), list(board), exact=False, n_samples=mc_samples, seed=seed)
    if not board:
        return float(1.0 - preflop_percentile(hole))
    return _equity.equity_vs_random(list(hole), list(board))


def _mix(*values: int) -> int:
    """Deterministic 63-bit hash of a few integers (seeds for per-decision Monte Carlo)."""
    x = 0x9E3779B97F4A7C15
    for v in values:
        x ^= (int(v) + 0x632BE59BD9B4E019 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)) & 0xFFFFFFFFFFFFFFFF
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return x & 0x7FFFFFFFFFFFFFFF


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# --------------------------------------------------------------------------- hand state


class _PreflopState:
    """What a player can read from the preflop betting so far."""

    __slots__ = ("raises", "limpers", "callers_after", "last_raise_to", "first_raiser", "last_raiser",
                 "my_first_kind", "my_raises", "i_opened")

    def __init__(self, history: Sequence[Any], seat: int, bb: int) -> None:
        self.raises = 0
        self.limpers = 0
        self.callers_after = 0
        self.last_raise_to = bb
        self.first_raiser: Optional[int] = None
        self.last_raiser: Optional[int] = None
        self.my_first_kind: Optional[str] = None
        self.my_raises = 0
        self.i_opened = False
        for a in history:
            if a.street != "preflop":
                break
            kind = a.kind
            if kind == "raise":
                if self.raises == 0:
                    self.first_raiser = a.seat
                    if a.seat == seat:
                        self.i_opened = True
                self.raises += 1
                self.last_raise_to = a.to
                self.last_raiser = a.seat
                self.callers_after = 0
            elif kind == "call":
                if self.raises == 0:
                    self.limpers += 1
                else:
                    self.callers_after += 1
            if a.seat == seat and kind in ("fold", "check", "call", "raise"):
                if self.my_first_kind is None:
                    self.my_first_kind = kind
                if kind == "raise":
                    self.my_raises += 1


def _street_aggressors(history: Sequence[Any]) -> dict[str, Optional[int]]:
    """Seat of the last bettor or raiser on each street (None if nobody bet)."""
    last: dict[str, Optional[int]] = {"preflop": None, "flop": None, "turn": None, "river": None}
    for a in history:
        if a.kind in _AGGRESSIVE:
            last[a.street] = a.seat
    return last


# --------------------------------------------------------------------------- policies


class HonestPolicy:
    """A player who decides only from public information and their own cards.

    Args:
        name: Player name (as seated).
        index: Player index in the population (part of the random seed).
        archetype: Style class (one of :data:`ARCHETYPES`).
        style: The player's own parameters.
        model: Shared model constants.
        seed: Base seed of the run.
        reads: Player name -> style class of every player, used only when
            ``style.reads`` is true (HUD reads).
    """

    def __init__(self, name: str, index: int, archetype: str, style: Style, model: PolicyModel,
                 seed: int, reads: Optional[Mapping[str, str]] = None) -> None:
        self.name = name
        self.index = index
        self.archetype = archetype
        self.style = style
        self.model = model
        self.seed = int(seed)
        self.reads = dict(reads) if (reads and style.reads) else {}
        self._hand_id: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None
        self._h_cache: dict[tuple[int, int], float] = {}
        cum, total = [], 0.0
        for w in style.sizes:
            total += w
            cum.append(total)
        self._size_cum = tuple(c / total for c in cum)

    # ---------------------------------------------------------------- per hand

    def _begin(self, view: DecisionView) -> np.random.Generator:
        if view.hand_id != self._hand_id or self._rng is None:
            self._hand_id = view.hand_id
            # numpy pads a seed key with zeros, so [seed, hand, 0] would replay the deck's
            # stream [seed, hand]; the trailing tag keeps every player's stream apart from it
            self._rng = np.random.default_rng([self.seed, int(view.hand_id) & 0x7FFFFFFF, self.index, _DECISIONS])
            self._h_cache = {}
        return self._rng

    def strength(self, view: DecisionView, seat: int, hole: Sequence[int]) -> float:
        """``h`` of the given seat's cards on the current board, cached per street."""
        key = (seat, len(view.board))
        h = self._h_cache.get(key)
        if h is None:
            h = hand_strength(hole, view.board, mc_samples=self.model.postflop.flop_mc_samples,
                              seed=_mix(self.seed, view.hand_id, seat, len(view.board)))
            self._h_cache[key] = h
        return h

    def act(self, view: DecisionView) -> Decision:
        rng = self._begin(view)
        if view.street == "preflop":
            return self._preflop(view, rng)
        return self._postflop(view, rng)

    # --------------------------------------------------------------- sizing

    def _raise_to(self, view: DecisionView, to: float, fallback: Decision) -> Decision:
        """Bet or raise to a street total, clipped to the legal range."""
        if not view.can_raise:
            return fallback
        if view.street == "preflop":
            unit = max(1, int(round(self.model.preflop.size_round_bb * view.bb)))
            to_i = int(round(to / unit)) * unit
        else:
            to_i = int(round(to))
        if to_i < view.min_raise_to:
            to_i = view.min_raise_to
        if to_i > view.max_raise_to:
            to_i = view.max_raise_to
        own = view.street_contrib.get(view.seat, 0)
        behind = view.max_raise_to - own
        if behind > 0 and (to_i - own) >= (1.0 - self.model.postflop.allin_snap) * behind:
            to_i = view.max_raise_to
        return Decision(view.raise_kind, to_i)

    def _pick_size(self, rng: np.random.Generator, polar_up: bool) -> float:
        u = rng.random()
        k = 0
        cum = self._size_cum
        while k < len(cum) - 1 and u > cum[k]:
            k += 1
        grid = self.model.postflop.size_grid
        if polar_up and k < len(grid) - 1 and rng.random() < self.style.polar:
            k += 1
        return grid[k]

    # --------------------------------------------------------------- preflop

    def _preflop(self, v: DecisionView, rng: np.random.Generator) -> Decision:
        s, pm = self.style, self.model.preflop
        bb = v.bb
        n = len(v.seat_order)
        idx = v.seat_order.index(v.seat)
        pos = position_class(n, idx)
        st = _PreflopState(v.action_history, v.seat, bb)
        pct = preflop_percentile(v.hole_cards) + rng.normal(0.0, s.pf_noise)
        fold_or_check = Decision.check() if v.to_call == 0 else Decision.fold()
        stack_total = v.stacks.get(v.seat, 0) + v.street_contrib.get(v.seat, 0)

        def shove_guard(to: float) -> float:
            return v.max_raise_to if to >= pm.shove_frac * stack_total else to

        if st.raises == 0:
            if n == 2:
                if v.to_call > 0:  # the button, first in
                    if pct <= s.hu_open:
                        return self._raise_to(v, shove_guard(s.hu_open_size * bb), Decision.call())
                    if pct <= s.hu_open + s.hu_limp:
                        return Decision.call()
                    return Decision.fold()
                # big blind after a limp
                if pct <= s.hu_iso:
                    return self._raise_to(v, shove_guard((s.hu_open_size + 1.0) * bb), Decision.check())
                return Decision.check()
            iso_to = (s.open_size + 1.0 + st.limpers) * bb
            if v.to_call == 0:  # big blind, limped pot
                width = s.iso * pm.rfi["BTN"] * s.rfi * (pm.bb_iso_multi if st.limpers > 1 else 1.0)
                if pct <= width:
                    return self._raise_to(v, shove_guard(iso_to), Decision.check())
                return Decision.check()
            open_w = pm.rfi[pos] * s.rfi
            if st.limpers == 0:
                if pct <= open_w:
                    return self._raise_to(v, shove_guard(s.open_size * bb), Decision.call())
                limp_w = s.limp * (pm.sb_limp_mult if pos == "SB" else 1.0)
                if pct <= open_w + limp_w:
                    return Decision.call()
                return Decision.fold()
            iso_w = open_w * s.iso
            if pct <= iso_w:
                return self._raise_to(v, shove_guard(iso_to), Decision.call())
            over_w = s.overlimp * (pm.sb_overlimp_mult if pos == "SB" else 1.0)
            if pct <= iso_w + over_w:
                return Decision.call()
            return fold_or_check

        size_bb = st.last_raise_to / bb
        if st.raises == 1:
            if st.my_first_kind == "call" and not st.i_opened:  # limped, now raised
                if pct <= s.fourbet:
                    return self._raise_to(v, shove_guard(pm.threebet_oop * st.last_raise_to), Decision.call())
                keep = (s.hu_limp + s.hu_open if n == 2 else pm.rfi[pos] * s.rfi + s.limp) * pm.limp_call_keep
                return Decision.call() if pct <= keep * self._size_factor(size_bb, n) else Decision.fold()
            if n == 2:
                tb, cc = s.hu_3bet, s.hu_defend
            else:
                tb = s.threebet * pm.threebet_pos[pos]
                cc = s.bb_defend if pos == "BB" else s.cold_call * pm.cold_call_pos[pos]
                cc *= pm.squeeze_tighten ** st.callers_after
            cc *= self._size_factor(size_bb, n)
            value = tb * pm.value_share
            in_position = pos == "BTN" or (
                n > 2 and pos not in ("SB", "BB") and idx > v.seat_order.index(st.last_raiser))
            mult = pm.threebet_ip if in_position else pm.threebet_oop
            to3 = mult * st.last_raise_to + st.callers_after * st.last_raise_to
            if pct <= value:
                return self._raise_to(v, shove_guard(to3), Decision.call())
            if pct <= value + cc:
                return Decision.call()
            band = max(cc, pm.bluff_band_min)
            if pct <= value + cc + band and rng.random() < (tb - value) / band:
                return self._raise_to(v, shove_guard(to3), Decision.call())
            return Decision.fold()

        if st.raises == 2:
            if st.i_opened and st.my_raises == 1:
                open_w = (s.hu_open if n == 2 else pm.rfi[pos] * s.rfi)
                keep = open_w * (1.0 - s.fold_3bet) * self._size_factor(size_bb / 3.0, n)
                value4 = s.fourbet * pm.value_share
                to4 = pm.fourbet_mult * st.last_raise_to
                if pct <= value4:
                    return self._raise_to(v, shove_guard(to4), Decision.call())
                if pct <= max(keep, value4):
                    if rng.random() < (s.fourbet - value4) / max(keep, 1e-9):
                        return self._raise_to(v, shove_guard(to4), Decision.call())
                    return Decision.call()
                return Decision.fold()
            # cold, or a caller / limper facing a 3-bet
            if pct <= s.fourbet * pm.value_share:
                return self._raise_to(v, shove_guard(pm.fourbet_mult * st.last_raise_to), Decision.call())
            if pct <= s.call_4bet * pm.cold_3bet_call:
                return Decision.call()
            return Decision.fold()

        # facing a 4-bet or more
        if pct <= s.fourbet * pm.fivebet_share:
            return self._raise_to(v, v.max_raise_to, Decision.call())
        if pct <= s.call_4bet:
            return Decision.call()
        return Decision.fold()

    def _size_factor(self, size_bb: float, n: int) -> float:
        """Calling widths shrink against raises above the standard size."""
        pm = self.model.preflop
        standard = pm.standard_open if n > 2 else self.style.hu_open_size
        ratio = standard / max(size_bb, 1.0)
        lo, hi = pm.size_factor_clip
        return min(hi, max(lo, ratio ** pm.size_power))

    # -------------------------------------------------------------- postflop

    def _opponent_adjust(self, opponents: Sequence[str]) -> tuple[float, float, float]:
        """(bluff_mult, value_shift, belief_shift) against these opponents (HUD reads).

        Bluff multipliers multiply over the opponents, value shifts and
        belief shifts are averaged.
        """
        if not self.reads or not opponents:
            return 1.0, 0.0, 0.0
        neutral = (1.0, 0.0, 0.0)
        adj = [self.model.exploit.get(self.reads.get(name, ""), neutral) for name in opponents]
        bluff = 1.0
        for a in adj:
            bluff *= a[0]
        return bluff, sum(a[1] for a in adj) / len(adj), sum(a[2] for a in adj) / len(adj)

    def _postflop(self, v: DecisionView, rng: np.random.Generator) -> Decision:
        s, pf = self.style, self.model.postflop
        street = _STREET_INDEX[v.street]
        si = street - 1
        h = self.strength(v, v.seat, v.hole_cards)
        draw = draw_kind(v.hole_cards, v.board)
        live = v.players_in_hand
        n_opp = len(live) - 1
        names = _seat_names(v.action_history)
        opponents = [names[x] for x in live if x != v.seat and x in names]

        if v.to_call > 0:
            return self._facing_bet(v, rng, h, draw, n_opp, opponents)

        aggressors = _street_aggressors(v.action_history)
        prev = ("preflop", "flop", "turn")[si]
        was_aggressor = aggressors[prev] == v.seat
        bluff_mult, value_shift, _ = self._opponent_adjust(opponents)
        thr = s.value_thr + pf.value_shift[si] + pf.value_multi * (n_opp - 1) + value_shift
        if len(v.seat_order) == 2:
            thr += pf.value_hu
        if h + rng.normal(0.0, s.noise) >= thr:
            if h >= pf.strong_h and not was_aggressor and rng.random() < s.slowplay:
                return Decision.check()
            frac = self._pick_size(rng, polar_up=h >= pf.strong_h)
            return self._raise_to(v, frac * v.pot, Decision.check())
        if was_aggressor:
            p = s.cbet if street == 1 else s.barrel
        else:
            p = s.bluff
            order = [x for x in v.seat_order if x in live]
            if order and order[-1] == v.seat:  # in position: everyone checked to us
                p = max(p, s.stab)
        if street >= 2 and h >= pf.weak_h:  # medium hands mostly check the turn and river
            p *= pf.medium_barrel
        if draw >= 2 and street < 3:
            p = max(p, s.semibluff)
        p *= (pf.bluff_multi ** (n_opp - 1)) * bluff_mult
        if rng.random() < p:
            frac = self._pick_size(rng, polar_up=h < pf.weak_h)
            return self._raise_to(v, frac * v.pot, Decision.check())
        return Decision.check()

    def _facing_bet(self, v: DecisionView, rng: np.random.Generator, h: float, draw: int, n_opp: int,
                    opponents: Sequence[str]) -> Decision:
        s, pf = self.style, self.model.postflop
        street = _STREET_INDEX[v.street]
        si = street - 1
        to_call = v.to_call
        pot_odds = to_call / (v.pot + to_call)
        size = to_call / max(v.pot - to_call, v.bb)
        bettor = v.last_aggressor
        barrels = 0
        if bettor is not None:
            done = set()
            for a in v.action_history:
                if a.seat == bettor and a.kind in _AGGRESSIVE and a.street not in ("preflop", v.street):
                    done.add(a.street)
            barrels = len(done)
        names = _seat_names(v.action_history)
        bettor_name = [names[bettor]] if bettor is not None and bettor in names else list(opponents[:1])
        bluff_mult, _, belief_shift = self._opponent_adjust(bettor_name)
        m = (pf.m_street[si] + pf.m_size * (min(size, pf.size_cap) - pf.size_ref)
             + pf.m_raise * max(0, v.raises_this_street - 1) + pf.m_barrel * barrels
             + pf.m_multi * (n_opp - 1) + s.belief + belief_shift)
        h_call = h + (pf.implied[draw - 1] * (0.5 if street == 2 else 1.0) if draw and street < 3 else 0.0)
        noise = rng.normal(0.0, s.noise)
        e_raise = _sigmoid((h - m) / pf.width) + noise
        if len(v.seat_order) == 2:  # heads-up ranges are wide: call lighter, raise as usual
            m += pf.m_hu[si]
        e = _sigmoid((h_call - m) / pf.width) + noise
        if v.can_raise:
            if e_raise >= s.raise_thr and rng.random() < s.raise_prob:
                return self._raise_to(v, s.raise_mult * v.current_bet, Decision.call())
            multi = pf.bluff_multi ** (n_opp - 1)
            if draw >= 2 and street < 3 and rng.random() < s.semibluff * pf.semibluff_raise * multi:
                return self._raise_to(v, s.raise_mult * v.current_bet, Decision.call())
            if h < pf.weak_h and rng.random() < s.bluff * pf.bluff_raise * bluff_mult * multi:
                return self._raise_to(v, s.raise_mult * v.current_bet, Decision.call())
        if e >= pot_odds + s.call_margin:
            return Decision.call()
        return Decision.fold()


def _seat_names(history: Sequence[Any]) -> dict[int, str]:
    """Seat -> player name for every seat that has acted or posted so far."""
    return {a.seat: a.player for a in history}


def _call_pot_odds(v: DecisionView) -> float:
    """Pot odds of calling, as the action table measures them.

    The call is capped by the caller's stack and the pot counts only what
    the caller can win: each other seat's chips up to the caller's total
    after the call (side pots). Chips of the pot that no action in the
    history accounts for count in full. With nobody covering the caller this
    is ``to_call / (pot + to_call)``.
    """
    put: dict[int, int] = {}
    for a in v.action_history:
        put[a.seat] = put.get(a.seat, 0) + a.amount
    call = min(v.to_call, v.stacks.get(v.seat, v.to_call))
    if call <= 0:
        return 0.0
    total = put.get(v.seat, 0) + call
    unattributed = max(0, v.pot - sum(put.values()))
    return call / (total + unattributed + sum(min(x, total) for s, x in put.items() if s != v.seat))


class SuperuserPolicy(HonestPolicy):
    """A planted player who sees some opponents' hole cards.

    Plays :class:`HonestPolicy` with its own style, except on postflop
    decisions where the engine revealed the cards of every live opponent:
    there, with probability ``u`` per decision, the informed rules of
    :class:`SuperuserRules` decide. Which opponents are visible, and from
    when, is the runner's business (the engine's ``info_grants``); the
    policy only uses ``view.revealed_cards``.

    Args:
        u: Share of the eligible postflop decisions played with the information.
        All other arguments as for :class:`HonestPolicy`.
    """

    def __init__(self, name: str, index: int, archetype: str, style: Style, model: PolicyModel,
                 seed: int, reads: Optional[Mapping[str, str]] = None, *, u: float = 1.0) -> None:
        super().__init__(name, index, archetype, style, model, seed, reads)
        self.u = float(u)
        self._u_rng: Optional[np.random.Generator] = None
        self.informed_decisions = 0
        self.eligible_decisions = 0

    def _begin(self, view: DecisionView) -> np.random.Generator:
        if view.hand_id != self._hand_id or self._rng is None:
            # a separate stream decides which decisions use the information, so the
            # honest decisions of a plant are the same for every u
            self._u_rng = np.random.default_rng([self.seed, int(view.hand_id) & 0x7FFFFFFF, self.index, _INFORMED])
        return super()._begin(view)

    def act(self, view: DecisionView) -> Decision:
        rng = self._begin(view)
        rules = self.model.superuser
        known = view.revealed_cards
        if known and (view.street != "preflop" or rules.preflop_info):
            opp_seats = [x for x in view.players_in_hand if x != view.seat]
            if opp_seats and all(x in known for x in opp_seats):
                self.eligible_decisions += 1
                if self._u_rng.random() < self.u:
                    self.informed_decisions += 1
                    return self._informed(view, rng, [known[x] for x in opp_seats], opp_seats)
        if view.street == "preflop":
            return self._preflop(view, rng)
        return self._postflop(view, rng)

    def real_equity(self, view: DecisionView, opp_cards: Sequence[tuple[int, int]]) -> float:
        """Equity against the real cards of every live opponent (exact)."""
        hero = list(view.hole_cards)
        board = list(view.board)
        if len(opp_cards) == 1:
            return _equity.equity_vs_hand(hero, list(opp_cards[0]), board)
        return _equity.multiway_equity([hero] + [list(c) for c in opp_cards], board)[0]

    def _informed(self, v: DecisionView, rng: np.random.Generator,
                  opp_cards: Sequence[tuple[int, int]], opp_seats: Sequence[int]) -> Decision:
        r = self.model.superuser
        eq = self.real_equity(v, opp_cards)
        if v.to_call > 0:
            pot_odds = _call_pot_odds(v)
            if v.can_raise and eq >= r.raise_eq and rng.random() < r.raise_prob:
                return self._raise_to(v, self.style.raise_mult * v.current_bet, Decision.call())
            if eq >= pot_odds + r.call_margin:
                return Decision.call()
            return Decision.fold()
        if v.street == "preflop":  # only with preflop_info: complete like a regular
            return self._preflop(v, rng)
        opp_h = [self.strength(v, seat, cards) for seat, cards in zip(opp_seats, opp_cards)]
        if eq >= r.value_eq:
            frac = r.value_size_strong if max(opp_h) >= r.opp_strong else r.value_size_weak
            return self._raise_to(v, frac * v.pot, Decision.check())
        if eq < r.bluff_eq and max(opp_h) < r.bluff_opp_max and rng.random() < r.bluff_prob:
            return self._raise_to(v, r.bluff_size * v.pot, Decision.check())
        return Decision.check()
