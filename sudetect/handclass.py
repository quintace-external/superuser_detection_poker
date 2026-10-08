"""Hand and board classes: what a holding makes on a board, what it draws to,
what the board looks like, and ten preflop buckets.

These classes are the coarse vocabulary the rest of the package groups
decisions by. Step 03 keys the field's action frequencies on them, and the
range model of step 04 weights every opponent holding by how often players
act the way he did with a hand of that class. Each class has a scalar
function for one holding on one board. :func:`classes_for_all_combos`
classifies all 1,326 holdings on one board at once and :func:`classify_batch`
classifies many (holding, board) rows; the test suite checks that the
vectorised and the scalar forms agree on every holding of random boards.

Cards are ids from :mod:`sudetect.cards`; every function also accepts what
:func:`sudetect.cards.parse_cards` accepts, such as ``"Ah Kd"``. A board is
given in dealing order, so its first three cards are the flop. Below, the
*board ranks* are the distinct ranks on the board and ``t1 > t2 > ...`` are
those ranks from the highest down: on ``Kc Kd 7h 2s`` t1 is a king, t2 a
seven.

Made hand (:func:`made_hand`, :data:`MADE_CLASSES`)
---------------------------------------------------
Flop, turn or river; the first rule that matches decides.

When the hero's best five-card hand (hole cards plus board) is a straight or
better, it counts only if the hole cards make it better than the board does
on its own:

``full_house_plus``
    The best hand is a full house, four of a kind or a straight flush and it
    beats the best hand of the five board cards alone (always the case with
    fewer than five board cards). Four of a kind needs a hole card of its
    rank: four of a kind on the board with a hole card as the kicker does not
    count.
``flush``
    The best hand is a flush with at least one hole card among its five
    cards, i.e. it beats any flush on the board.
``straight``
    The best hand is a straight higher than any straight on the board.
``nothing``
    The best hand is a straight or better that the board makes alone and the
    hero plays the board. This happens on the river and when four of a kind
    lies on the board.

Otherwise the best hand is three of a kind or less, and the class comes
from the pairs that include a hole card. Pairs and trips that lie on the
board alone are ignored.

``trips_set``
    A pocket pair with a third card of its rank on the board (a set), or one
    hole card with two of its rank on the board (trips).
``two_pair``
    Each hole card pairs a different board rank.
``overpair``
    A pocket pair above every board card.
``top_pair_good``
    One hole card pairs t1 and the other one, the kicker, is one of the two
    best kickers possible: at most one rank above the kicker is missing from
    the board. On A-7-2 the good kickers are K and Q, on K-Q-5 they are A and
    J, on 9-6-2 they are A and K.
``top_pair_weak``
    One hole card pairs t1 and the kicker is not one of the two best.
``middle_pair``
    One hole card pairs t2, or a pocket pair lies between t2 and t1.
``weak_pair``
    One hole card pairs a board rank below t2, or a pocket pair lies below t2
    and above the lowest board card.
``underpair``
    A pocket pair below every board card.
``nothing``
    No pair that includes a hole card.

Pairs are read from the hole cards. A pocket pair on a paired board counts
as one pair (``Qs Qh`` on ``Kc Kd 7h`` is a ``middle_pair``), and a pair that
two higher board pairs counterfeit on the river still counts (``Ad 2s`` on
``Kc Kd Qh Qs 2c`` is a ``weak_pair``).

Draw (:func:`draw_class`, :data:`DRAW_CLASSES`)
-----------------------------------------------
Flop and turn; always ``none`` on the river. A draw has to use a hole card,
and it is reported only to a hand it would improve.

* Flush draw: four cards of one suit among the hole cards and the board, at
  least one of them a hole card, with a made hand below ``flush``.
* Straight outs: a rank x is an *out rank* when no card of rank x is among
  the hole cards and the board yet, and a card of rank x would give the hero
  a straight higher than any straight the board and that card make alone.
  The ace plays high and low, so wheel draws count. Out ranks count only
  with a made hand below ``straight``.

The class is the first that matches: ``combo_draw`` (a flush draw and at
least one out rank), ``flush_draw``, ``oesd`` (two or more out ranks: an
open-ended straight draw or a double gutshot), ``gutshot`` (one out rank),
``none``. :func:`backdoor_flush_draw` reports three to a flush on the flop
separately; it does not change the class.

Board texture (:func:`board_texture`, :func:`texture_class`)
------------------------------------------------------------
Both read the flop, the first three board cards. :func:`board_texture` joins
four parts into one key such as ``twotone_unpaired_KQ_c12``:

* suits: ``rainbow`` (three suits), ``twotone`` (two), ``monotone`` (one);
* pairing: ``unpaired``, ``paired``, ``trips``;
* high card: ``A``, ``KQ`` (king or queen high), ``JT``, ``96`` (nine to six
  high), ``52`` (five to two high);
* connectivity, the number of two-card straights the flop allows, i.e. of
  five-rank windows (A-5, 2-6, ..., T-A) that hold all three flop ranks:
  ``c0``, ``c12`` (one or two) or ``c3`` (three, the most a flop allows:
  9-8-7 is completed by 6-5, T-6 and J-T). Paired flops are ``c0``.

:func:`texture_class` is a coarse version with eight values, the first that
matches: ``paired`` (paired or trips), ``monotone``, ``A_high``, ``KQ_high``;
then ``connected_mid`` / ``dry_mid`` for jack- or ten-high flops and
``connected_low`` / ``dry_low`` for nine-high or lower, where ``connected``
means that at least one two-card straight is possible.

Board change (:func:`board_change`, :data:`BOARD_CHANGES`)
----------------------------------------------------------
What the last card of a turn or river board did to the board before it; the
first that matches:

``flush_completes``
    The new card is the third or a later card of its suit on the board, so a
    flush is now possible with two suited hole cards (third card), with one
    (fourth) or on the board alone (fifth).
``straight_completes``
    The new card adds a new rank to a five-rank window (wheel included) that
    then holds three or more board ranks, so a straight with at most two hole
    cards becomes possible or needs one hole card fewer.
``board_pairs_top``
    The new card pairs the highest card of the board before it.
``board_pairs_other``
    The new card pairs (or trips) another card of the board before it.
``overcard``
    The new card is higher than every card of the board before it.
``blank``
    Anything else.

Preflop
-------
:func:`hand_class169` gives the class label of :mod:`sudetect.cards`
(``"AKs"``, ``"T9o"``, ``"77"``). :func:`hand_bucket10` sorts the 169 classes
into ten buckets, the first that matches:

====================  =========================================================
``premium``           AA, KK, QQ, AK
``strong``            JJ, TT, AQ, AJs, KQs, KJs
``mid_pair``          99, 88, 77
``low_pair``          66 down to 22
``ax_suited``         the other suited aces, ATs down to A2s
``ax_offsuit``        the other offsuit aces, AJo down to A2o
``broadway``          two cards ten or higher: KQo, KJo, KT, QJ, QT, JT
``suited_connector``  suited, five to ten high, at most one gap: 54s to T9s
                      and 53s to T8s
``semi_trash``        the other king-x hands and the other suited hands
``trash``             everything else
====================  =========================================================

:data:`PREFLOP_RANKING` lists the 169 classes from the strongest to the
weakest by exact equity against one random hand, the table
:data:`sudetect.equity.PREFLOP_VS_RANDOM` (AA 0.852 first, 32o 0.323 last).
It is built once at import by :func:`rank_classes`. :data:`CHEN_PERCENTILE`
turns that order into a strength percentile per class, (number of weaker
classes) / 168, so AA is 1 and 32o is 0. The range model tilts a loose or
tight opponent's preflop range with it. The name comes from the Chen formula,
a quick hand score often used for this purpose; the ranking here uses exact
equity in its place. :data:`PREFLOP_TOP_SHARE` is the share of all 1,326
holdings that belong to classes at least as strong, so "the top 20% of
hands" are the classes with a value of at most 0.20.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .cards import (
    CLASS_LABELS,
    CLASS_SIZE,
    COMBO_TO_CLASS,
    COMBOS,
    N_CLASSES,
    N_COMBOS,
    RANKS,
    CardsLike,
    class_index,
    hand_class,
    parse_cards,
)
from .equity import PREFLOP_VS_RANDOM
from .evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    card_keys,
    value_from_keys,
)

__all__ = [
    "MADE_CLASSES",
    "DRAW_CLASSES",
    "MADE_INDEX",
    "DRAW_INDEX",
    "BUCKET10_CLASSES",
    "TEXTURE_CLASSES",
    "BOARD_CHANGES",
    "made_hand",
    "draw_class",
    "classify",
    "backdoor_flush_draw",
    "classes_for_all_combos",
    "classify_batch",
    "board_texture",
    "texture_class",
    "board_change",
    "hand_class169",
    "hand_bucket10",
    "BUCKET10_BY_CLASS",
    "COMBO_BUCKET10",
    "PREFLOP_RANKING",
    "rank_classes",
    "strength_percentile",
    "CHEN_PERCENTILE",
    "PREFLOP_TOP_SHARE",
    "chen_percentile",
]

#: Made-hand classes, roughly from the weakest to the strongest. The vectorised
#: functions return indices into this tuple.
MADE_CLASSES: tuple[str, ...] = (
    "nothing",
    "underpair",
    "weak_pair",
    "middle_pair",
    "top_pair_weak",
    "top_pair_good",
    "overpair",
    "two_pair",
    "trips_set",
    "straight",
    "flush",
    "full_house_plus",
)
#: Draw classes; the vectorised functions return indices into this tuple.
DRAW_CLASSES: tuple[str, ...] = ("none", "gutshot", "oesd", "flush_draw", "combo_draw")
MADE_INDEX: dict[str, int] = {name: i for i, name in enumerate(MADE_CLASSES)}
DRAW_INDEX: dict[str, int] = {name: i for i, name in enumerate(DRAW_CLASSES)}

#: The ten preflop buckets of :func:`hand_bucket10`.
BUCKET10_CLASSES: tuple[str, ...] = (
    "premium",
    "strong",
    "mid_pair",
    "low_pair",
    "ax_suited",
    "ax_offsuit",
    "broadway",
    "suited_connector",
    "semi_trash",
    "trash",
)
#: The eight coarse flop classes of :func:`texture_class`.
TEXTURE_CLASSES: tuple[str, ...] = (
    "paired",
    "monotone",
    "A_high",
    "KQ_high",
    "connected_mid",
    "dry_mid",
    "connected_low",
    "dry_low",
)
#: The turn and river change classes of :func:`board_change`, in matching order.
BOARD_CHANGES: tuple[str, ...] = (
    "flush_completes",
    "straight_completes",
    "board_pairs_top",
    "board_pairs_other",
    "overcard",
    "blank",
)

(_NOTHING, _UNDERPAIR, _WEAK_PAIR, _MIDDLE_PAIR, _TOP_PAIR_WEAK, _TOP_PAIR_GOOD, _OVERPAIR,
 _TWO_PAIR, _TRIPS_SET, _STRAIGHT, _FLUSH, _FULL_HOUSE_PLUS) = range(len(MADE_CLASSES))
_NO_DRAW, _GUTSHOT, _OESD, _FLUSH_DRAW, _COMBO_DRAW = range(len(DRAW_CLASSES))

_ACE, _KING, _QUEEN, _JACK, _TEN = 12, 11, 10, 9, 8


# ----------------------------------------------------------------------------
# Straights over rank sets
# ----------------------------------------------------------------------------

#: The ten five-rank windows by the rank of their top card: 3 is the wheel
#: (A-2-3-4-5, the ace playing low), 12 is ten to ace.
_WINDOWS: tuple[tuple[int, frozenset[int]], ...] = tuple(
    (top, frozenset({12, 0, 1, 2, 3}) if top == 3 else frozenset(range(top - 4, top + 1)))
    for top in range(3, 13)
)


def _straight_top(ranks: set[int] | frozenset[int]) -> int:
    """Top rank of the highest straight inside a set of ranks (3 for the wheel), -1 if none."""
    for top, window in reversed(_WINDOWS):
        if window <= ranks:
            return top
    return -1


def _build_straight_top() -> np.ndarray:
    """:func:`_straight_top` of every 13-bit rank mask (bit r = rank r)."""
    masks = np.arange(1 << 13)
    out = np.full(1 << 13, -1, dtype=np.int8)
    for top, window in _WINDOWS:                      # ascending, so the highest straight wins
        bits = sum(1 << r for r in window)
        out[(masks & bits) == bits] = top
    out.setflags(write=False)
    return out


_STRAIGHT_TOP = _build_straight_top()
_POPCOUNT = np.array([bin(m).count("1") for m in range(1 << 13)], dtype=np.int8)
_POPCOUNT.setflags(write=False)


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------

def _hole_and_board(hole: CardsLike, board: CardsLike) -> tuple[list[int], list[int]]:
    h = parse_cards(hole)
    b = _board_cards(board, (3, 4, 5))
    if len(h) != 2:
        raise ValueError(f"a holding has two cards, got {len(h)}")
    if set(h) & set(b):
        raise ValueError("the holding and the board share a card")
    return h, b


def _board_cards(board: CardsLike, sizes: tuple[int, ...]) -> list[int]:
    b = parse_cards(board)
    if len(b) not in sizes:
        raise ValueError(f"expected a board of {' or '.join(map(str, sizes))} cards, got {len(b)}")
    return b


# ----------------------------------------------------------------------------
# Made hand and draw: scalar reference
# ----------------------------------------------------------------------------

def _pack(cat: int, *ranks: int) -> int:
    value = cat << 20
    for i, r in enumerate(ranks):
        value |= r << (16 - 4 * i)
    return value


def _strength(cards: Sequence[int]) -> int:
    """Best five-card strength of 5 to 7 cards, in the layout of :mod:`sudetect.evaluator`.

    A plain-Python twin of the numpy evaluator, so that the scalar functions
    here do not share code with the vectorised ones they are tested against.
    """
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    flush: list[int] | None = None
    for s in range(4):
        in_suit = [r for r, t in zip(ranks, suits) if t == s]
        if len(in_suit) >= 5:                        # only one suit can have five of seven cards
            top = _straight_top(set(in_suit))
            if top >= 0:
                return _pack(STRAIGHT_FLUSH, top)
            flush = sorted(in_suit, reverse=True)[:5]
    counts = Counter(ranks)
    distinct = sorted(counts, reverse=True)
    quads = [r for r in distinct if counts[r] == 4]
    trips = [r for r in distinct if counts[r] == 3]
    pairs = [r for r in distinct if counts[r] == 2]
    if quads:
        return _pack(QUADS, quads[0], max(r for r in distinct if r != quads[0]))
    if trips and (len(trips) > 1 or pairs):
        return _pack(FULL_HOUSE, trips[0], max(trips[1:] + pairs))
    if flush is not None:
        return _pack(FLUSH, *flush)
    top = _straight_top(set(distinct))
    if top >= 0:
        return _pack(STRAIGHT, top)
    if trips:
        return _pack(TRIPS, trips[0], *[r for r in distinct if r != trips[0]][:2])
    if len(pairs) >= 2:
        return _pack(TWO_PAIR, pairs[0], pairs[1], max(r for r in distinct if r not in pairs[:2]))
    if pairs:
        return _pack(PAIR, pairs[0], *[r for r in distinct if r != pairs[0]][:3])
    return _pack(HIGH_CARD, *distinct[:5])


def _made_index(hole: list[int], board: list[int]) -> int:
    """Index into MADE_CLASSES; the rules are those of the module docstring, in order."""
    best = _strength(hole + board)
    cat = best >> 20
    a, b = hole[0] >> 2, hole[1] >> 2
    if cat >= STRAIGHT:
        if cat == QUADS:
            improves = ((best >> 16) & 15) in (a, b)      # the quads rank is a hole card
        else:
            improves = len(board) < 5 or best > _strength(board)
        if not improves:
            return _NOTHING
        if cat >= FULL_HOUSE:
            return _FULL_HOUSE_PLUS
        return _FLUSH if cat == FLUSH else _STRAIGHT

    on_board = Counter(c >> 2 for c in board)
    ca, cb = on_board[a], on_board[b]
    if a == b:
        if ca:
            return _TRIPS_SET
    else:
        if ca == 2 or cb == 2:
            return _TRIPS_SET
        if ca and cb:
            return _TWO_PAIR
    board_ranks = sorted(on_board, reverse=True)
    t1, lowest = board_ranks[0], board_ranks[-1]
    t2 = board_ranks[1] if len(board_ranks) > 1 else -1
    if a == b:
        if a > t1:
            return _OVERPAIR
        if a < lowest:
            return _UNDERPAIR
        return _MIDDLE_PAIR if a > t2 else _WEAK_PAIR
    if ca or cb:
        pair, kicker = (a, b) if ca else (b, a)
        if pair == t1:
            missing_above = sum(1 for r in range(kicker + 1, 13) if r not in on_board)
            return _TOP_PAIR_GOOD if missing_above <= 1 else _TOP_PAIR_WEAK
        return _MIDDLE_PAIR if pair == t2 else _WEAK_PAIR
    return _NOTHING


def _draw_index(hole: list[int], board: list[int], made: int) -> int:
    """Index into DRAW_CLASSES given the made-hand index of the same holding."""
    if len(board) >= 5:
        return _NO_DRAW
    cards = hole + board
    suit_count = Counter(c & 3 for c in cards)
    flush_draw = made < _FLUSH and any(suit_count[c & 3] == 4 for c in hole)
    outs = 0
    if made < _STRAIGHT:
        have = {c >> 2 for c in cards}
        board_ranks = {c >> 2 for c in board}
        for x in range(13):
            if x not in have and _straight_top(have | {x}) > _straight_top(board_ranks | {x}):
                outs += 1
    if flush_draw:
        return _COMBO_DRAW if outs else _FLUSH_DRAW
    if outs >= 2:
        return _OESD
    return _GUTSHOT if outs == 1 else _NO_DRAW


def made_hand(hole: CardsLike, board: CardsLike) -> str:
    """Made-hand class of a holding on a flop, turn or river (see the module docstring).

    Example: ``made_hand("Ah Qs", "Ad 7c 2s")`` is ``"top_pair_good"``.

    Raises:
        ValueError: not two hole cards, not 3-5 board cards, or a repeated card.
    """
    h, b = _hole_and_board(hole, board)
    return MADE_CLASSES[_made_index(h, b)]


def draw_class(hole: CardsLike, board: CardsLike) -> str:
    """Draw class of a holding on a flop or turn; ``"none"`` on the river.

    Example: ``draw_class("As 2s", "3d 5c Qh")`` is ``"gutshot"`` (a four
    makes the wheel).
    """
    h, b = _hole_and_board(hole, board)
    return DRAW_CLASSES[_draw_index(h, b, _made_index(h, b))]


def classify(hole: CardsLike, board: CardsLike) -> tuple[str, str]:
    """``(made_hand, draw_class)`` of a holding in one pass."""
    h, b = _hole_and_board(hole, board)
    made = _made_index(h, b)
    return MADE_CLASSES[made], DRAW_CLASSES[_draw_index(h, b, made)]


def backdoor_flush_draw(hole: CardsLike, board: CardsLike) -> bool:
    """True on a flop where some suit has exactly three of the five cards, at least one a hole card.

    Kept apart from :func:`draw_class`: a backdoor draw never changes the class.
    Always False on the turn and river.
    """
    h, b = _hole_and_board(hole, board)
    if len(b) != 3:
        return False
    suit_count = Counter(c & 3 for c in h + b)
    return any(suit_count[c & 3] == 3 for c in h)


# ----------------------------------------------------------------------------
# Made hand and draw: vectorised
# ----------------------------------------------------------------------------

def _classify_rows(hole: np.ndarray, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Made-hand and draw indices of n rows: ``hole`` (n, 2), ``board`` (n, k) with
    k in 3..5, valid and distinct card ids. Mirrors :func:`_made_index` and
    :func:`_draw_index` with the numpy evaluator."""
    n, k = board.shape
    rows = np.arange(n)
    r0, r1 = hole[:, 0] >> 2, hole[:, 1] >> 2
    board_rank = board >> 2

    on_board = np.zeros((n, 13), dtype=np.int8)       # board count of each rank
    board_mask = np.zeros(n, dtype=np.int64)          # bit r set when rank r is on the board
    for j in range(k):
        on_board[rows, board_rank[:, j]] += 1
        board_mask |= np.left_shift(1, board_rank[:, j])

    key_b, flush_b = card_keys(board)
    key_h, flush_h = card_keys(hole)
    best = value_from_keys(key_b + key_h, flush_b + flush_h, k + 2).astype(np.int64)
    cat = best >> 20
    strong = cat >= STRAIGHT
    if k == 5:
        beats_board = best > value_from_keys(key_b, flush_b, 5)
    else:
        beats_board = np.ones(n, dtype=bool)
    quads_rank = (best >> 16) & 15
    improves = np.where(cat == QUADS, (r0 == quads_rank) | (r1 == quads_rank), beats_board)

    ca, cb = on_board[rows, r0], on_board[rows, r1]
    pocket = r0 == r1
    present = on_board > 0
    t1 = 12 - np.argmax(present[:, ::-1], axis=1)
    lowest = np.argmax(present, axis=1)
    below_t1 = present & (np.arange(13) < t1[:, None])
    t2 = np.where(below_t1.any(axis=1), 12 - np.argmax(below_t1[:, ::-1], axis=1), -1)

    trips_set = np.where(pocket, ca >= 1, (ca == 2) | (cb == 2))
    two_pair = ~pocket & (ca == 1) & (cb == 1)
    pocket_free = pocket & (ca == 0)
    one_pair = ~pocket & (ca + cb == 1)
    pair_rank = np.where(ca == 1, r0, r1)
    kicker = np.where(ca == 1, r1, r0)
    above_kicker = 0x1FFF & ~((np.int64(2) << kicker) - 1)
    good_kicker = _POPCOUNT[above_kicker & ~board_mask & 0x1FFF] <= 1

    made = np.select(
        [
            strong & improves & (cat >= FULL_HOUSE),
            strong & improves & (cat == FLUSH),
            strong & improves,
            strong,
            trips_set,
            two_pair,
            pocket_free & (r0 > t1),
            pocket_free & (r0 < lowest),
            pocket_free & (r0 > t2),
            pocket_free,
            one_pair & (pair_rank == t1) & good_kicker,
            one_pair & (pair_rank == t1),
            one_pair & (pair_rank == t2),
            one_pair,
        ],
        [
            _FULL_HOUSE_PLUS, _FLUSH, _STRAIGHT, _NOTHING, _TRIPS_SET, _TWO_PAIR, _OVERPAIR,
            _UNDERPAIR, _MIDDLE_PAIR, _WEAK_PAIR, _TOP_PAIR_GOOD, _TOP_PAIR_WEAK, _MIDDLE_PAIR,
            _WEAK_PAIR,
        ],
        _NOTHING,
    ).astype(np.int8)

    if k == 5:
        return made, np.zeros(n, dtype=np.int8)
    s0, s1 = hole[:, 0] & 3, hole[:, 1] & 3
    suit_count = np.zeros((n, 4), dtype=np.int8)
    for j in range(k):
        suit_count[rows, board[:, j] & 3] += 1
    suit_count[rows, s0] += 1
    suit_count[rows, s1] += 1
    flush_draw = ((suit_count[rows, s0] == 4) | (suit_count[rows, s1] == 4)) & (made < _FLUSH)

    all_mask = board_mask | np.left_shift(1, r0) | np.left_shift(1, r1)
    bits = np.left_shift(1, np.arange(13, dtype=np.int64))
    is_new = (all_mask[:, None] & bits) == 0
    outs = is_new & (_STRAIGHT_TOP[all_mask[:, None] | bits] > _STRAIGHT_TOP[board_mask[:, None] | bits])
    n_outs = np.where(made < _STRAIGHT, outs.sum(axis=1), 0)
    draw = np.select(
        [flush_draw & (n_outs > 0), flush_draw, n_outs >= 2, n_outs == 1],
        [_COMBO_DRAW, _FLUSH_DRAW, _OESD, _GUTSHOT],
        _NO_DRAW,
    ).astype(np.int8)
    return made, draw


_COMBOS_INTP = COMBOS.astype(np.intp)


@lru_cache(maxsize=4096)
def _all_combos(board: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    b = np.array(board, dtype=np.intp)
    dead = np.zeros(52, dtype=bool)
    dead[b] = True
    live = np.flatnonzero(~(dead[_COMBOS_INTP[:, 0]] | dead[_COMBOS_INTP[:, 1]]))
    made = np.full(N_COMBOS, -1, dtype=np.int8)
    draw = np.full(N_COMBOS, -1, dtype=np.int8)
    made[live], draw[live] = _classify_rows(_COMBOS_INTP[live], np.broadcast_to(b, (live.size, b.size)))
    made.setflags(write=False)
    draw.setflags(write=False)
    return made, draw


def classes_for_all_combos(board: CardsLike) -> tuple[np.ndarray, np.ndarray]:
    """Made-hand and draw class of every holding on one board.

    Returns two int8 arrays of 1,326 entries in :data:`sudetect.cards.COMBOS`
    order: indices into :data:`MADE_CLASSES` and :data:`DRAW_CLASSES`, -1 for
    holdings that share a card with the board. Entry ``i`` equals
    ``made_hand(COMBOS[i], board)`` and ``draw_class(COMBOS[i], board)``.

    Results are cached per board (the order of the board cards does not
    matter), so the arrays are read-only; copy them before changing them.
    A new board takes about a millisecond.
    """
    b = _board_cards(board, (3, 4, 5))
    return _all_combos(tuple(sorted(b)))


def classify_batch(hole: np.ndarray, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Made-hand and draw class indices of many (holding, board) rows.

    Args:
        hole: (n, 2) card ids; a row with a negative id has unknown cards.
        board: (n, m) card ids, -1 for cards not dealt yet (at any position).

    Returns:
        Two int8 arrays of n indices into :data:`MADE_CLASSES` and
        :data:`DRAW_CLASSES`; -1 where the hole cards are unknown or fewer
        than three board cards are dealt.

    Raises:
        ValueError: a card id above 51, more than five board cards, or a card
            repeated within a row.
    """
    h = np.asarray(hole).astype(np.intp, copy=False)
    b = np.asarray(board).astype(np.intp, copy=False)
    if h.ndim != 2 or h.shape[1] != 2 or b.ndim != 2 or b.shape[0] != h.shape[0]:
        raise ValueError(f"expected hole (n, 2) and board (n, m), got {h.shape} and {b.shape}")
    if (h.size and h.max() > 51) or (b.size and b.max() > 51):
        raise ValueError("card ids must be at most 51")
    n = h.shape[0]
    made = np.full(n, -1, dtype=np.int8)
    draw = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return made, draw
    b = np.take_along_axis(b, np.argsort(b < 0, axis=1, kind="stable"), axis=1)  # dealt cards first
    n_board = (b >= 0).sum(axis=1)
    if n_board.max(initial=0) > 5:
        raise ValueError("a board has at most five cards")
    known = (h >= 0).all(axis=1) & (n_board >= 3)
    cards = np.concatenate([h, b], axis=1)
    cards = np.sort(np.where(cards < 0, -1 - np.arange(cards.shape[1]), cards), axis=1)
    if (np.diff(cards, axis=1) == 0).any():          # every row, also those left at -1
        raise ValueError("a card appears twice in a row")
    for k in (3, 4, 5):
        rows = np.flatnonzero(known & (n_board == k))
        if rows.size:
            made[rows], draw[rows] = _classify_rows(h[rows], b[rows, :k])
    return made, draw


# ----------------------------------------------------------------------------
# Boards
# ----------------------------------------------------------------------------

def _flop_parts(board: CardsLike) -> tuple[int, int, int, int]:
    """(number of suits, number of distinct ranks, high rank, connectivity) of the flop."""
    flop = _board_cards(board, (3, 4, 5))[:3]
    ranks = {c >> 2 for c in flop}
    n_suits = len({c & 3 for c in flop})
    connectivity = sum(1 for _, window in _WINDOWS if ranks <= window) if len(ranks) == 3 else 0
    return n_suits, len(ranks), max(ranks), connectivity


def _high_group(rank: int) -> str:
    if rank == _ACE:
        return "A"
    if rank >= _QUEEN:
        return "KQ"
    if rank >= _TEN:
        return "JT"
    return "96" if rank >= 4 else "52"


def board_texture(board: CardsLike) -> str:
    """Detailed flop texture, e.g. ``"twotone_unpaired_KQ_c12"`` (see the module docstring).

    Reads the first three cards, so a turn or river board gives its flop's texture.
    """
    n_suits, n_ranks, high, connectivity = _flop_parts(board)
    suits = {1: "monotone", 2: "twotone", 3: "rainbow"}[n_suits]
    pairing = {1: "trips", 2: "paired", 3: "unpaired"}[n_ranks]
    conn = "c0" if connectivity == 0 else ("c3" if connectivity >= 3 else "c12")
    return f"{suits}_{pairing}_{_high_group(high)}_{conn}"


def texture_class(board: CardsLike) -> str:
    """Coarse flop class, one of :data:`TEXTURE_CLASSES` (see the module docstring)."""
    n_suits, n_ranks, high, connectivity = _flop_parts(board)
    if n_ranks < 3:
        return "paired"
    if n_suits == 1:
        return "monotone"
    if high == _ACE:
        return "A_high"
    if high >= _QUEEN:
        return "KQ_high"
    shape = "connected" if connectivity else "dry"
    return f"{shape}_{'mid' if high >= _TEN else 'low'}"


def board_change(board: CardsLike) -> str:
    """What the last card of a turn or river board did; one of :data:`BOARD_CHANGES`.

    Raises:
        ValueError: the board does not have four or five cards.
    """
    b = _board_cards(board, (4, 5))
    before, card = b[:-1], b[-1]
    rank, suit = card >> 2, card & 3
    if sum(1 for c in b if c & 3 == suit) >= 3:
        return "flush_completes"
    old_ranks = {c >> 2 for c in before}
    if rank not in old_ranks:
        new_ranks = old_ranks | {rank}
        if any(rank in window and len(window & new_ranks) >= 3 for _, window in _WINDOWS):
            return "straight_completes"
    if rank == max(old_ranks):
        return "board_pairs_top"
    if rank in old_ranks:
        return "board_pairs_other"
    if rank > max(old_ranks):
        return "overcard"
    return "blank"


# ----------------------------------------------------------------------------
# Preflop
# ----------------------------------------------------------------------------

def _class_of(hand: CardsLike) -> int:
    """Class index of two cards, or of a class label such as ``"AKs"``."""
    if isinstance(hand, str) and len(hand.strip()) <= 3:
        return class_index(hand)
    return class_index(hand_class(hand))


def hand_class169(hole: CardsLike) -> str:
    """The 169-class label of two hole cards: ``"Kd Ah"`` -> ``"AKo"``."""
    return hand_class(hole)


def _bucket_of_label(label: str) -> str:
    hi, lo = RANKS.index(label[0]), RANKS.index(label[1])
    suited = label.endswith("s")
    if hi == lo:
        if hi >= _QUEEN:
            return "premium"
        if hi >= _TEN:
            return "strong"
        return "mid_pair" if hi >= 5 else "low_pair"     # 77-99 / 22-66
    if hi == _ACE and lo == _KING:
        return "premium"
    if hi == _ACE and lo == _QUEEN:
        return "strong"
    if suited and ((hi == _ACE and lo == _JACK) or (hi == _KING and lo in (_QUEEN, _JACK))):
        return "strong"
    if hi == _ACE:
        return "ax_suited" if suited else "ax_offsuit"
    if lo >= _TEN:
        return "broadway"
    if suited and 3 <= hi <= _TEN and hi - lo <= 2:      # five to ten high, at most one gap
        return "suited_connector"
    if hi == _KING or suited:
        return "semi_trash"
    return "trash"


#: Bucket of each of the 169 classes, in :data:`sudetect.cards.CLASS_LABELS` order.
BUCKET10_BY_CLASS: tuple[str, ...] = tuple(_bucket_of_label(label) for label in CLASS_LABELS)
#: (1326,) int8: index into :data:`BUCKET10_CLASSES` of each combo.
COMBO_BUCKET10: np.ndarray = np.array(
    [BUCKET10_CLASSES.index(BUCKET10_BY_CLASS[c]) for c in COMBO_TO_CLASS], dtype=np.int8
)
COMBO_BUCKET10.setflags(write=False)


def hand_bucket10(hole: CardsLike) -> str:
    """Preflop bucket of two hole cards (or of a class label such as ``"AJs"``).

    Example: ``hand_bucket10("Ah Js")`` is ``"ax_offsuit"``, ``hand_bucket10("AJs")`` is ``"strong"``.
    """
    return BUCKET10_BY_CLASS[_class_of(hole)]


def rank_classes(values: Sequence[float]) -> tuple[str, ...]:
    """Class labels ordered from the highest value to the lowest.

    ``values`` holds one strength per class in :data:`sudetect.cards.CLASS_LABELS`
    order; equal values keep that order.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (N_CLASSES,):
        raise ValueError(f"expected {N_CLASSES} values, got shape {v.shape}")
    return tuple(CLASS_LABELS[i] for i in np.argsort(-v, kind="stable"))


def strength_percentile(values: Sequence[float]) -> np.ndarray:
    """Percentile of each class under one strength value per class.

    ``(number of classes with a lower value + half the other classes with the
    same value) / 168``: 1 for the strongest class, 0 for the weakest.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (N_CLASSES,):
        raise ValueError(f"expected {N_CLASSES} values, got shape {v.shape}")
    lower = (v[None, :] < v[:, None]).sum(axis=1)
    equal = (v[None, :] == v[:, None]).sum(axis=1) - 1
    return (lower + 0.5 * equal) / (N_CLASSES - 1)


#: The 169 classes from the strongest to the weakest by exact equity against
#: one random hand (:data:`sudetect.equity.PREFLOP_VS_RANDOM`): AA, KK, QQ, JJ,
#: TT, 99, 88, AKs, 77, AQs, AJs, AKo, ... 72o, 52o, 62o, 42o, 32o.
PREFLOP_RANKING: tuple[str, ...] = rank_classes(PREFLOP_VS_RANDOM)


def _build_preflop_tables(ranking: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(percentile, top share) per class, in CLASS_LABELS order, from a strongest-first ranking."""
    if sorted(ranking) != sorted(CLASS_LABELS):
        raise ValueError("a ranking lists each of the 169 classes once")
    position = np.empty(N_CLASSES, dtype=np.int64)
    position[[class_index(label) for label in ranking]] = np.arange(N_CLASSES)
    percentile = (N_CLASSES - 1 - position) / (N_CLASSES - 1)
    sizes_in_order = CLASS_SIZE[[class_index(label) for label in ranking]].astype(np.int64)
    top_share = np.cumsum(sizes_in_order)[position] / N_COMBOS
    percentile.setflags(write=False)
    top_share.setflags(write=False)
    return percentile, top_share


#: (169,) preflop strength percentile per class in CLASS_LABELS order:
#: (number of weaker classes in :data:`PREFLOP_RANKING`) / 168.
#: PREFLOP_TOP_SHARE: (169,) share of the 1,326 combos in classes at least as strong.
CHEN_PERCENTILE, PREFLOP_TOP_SHARE = _build_preflop_tables(PREFLOP_RANKING)


def chen_percentile(hand: CardsLike) -> float:
    """Preflop strength percentile of two cards or a class label (``"AA"`` -> 1.0)."""
    return float(CHEN_PERCENTILE[_class_of(hand)])
