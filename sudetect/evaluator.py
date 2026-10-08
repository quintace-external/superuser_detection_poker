"""Vectorised hand evaluator for 5, 6 and 7 cards (numpy only).

:func:`eval_hands` takes an integer array of shape (n, k), k = 5, 6 or 7, of
card ids (see :mod:`sudetect.cards`) and returns one int32 *strength* per row:
higher wins, equal strengths split the pot. With 6 or 7 cards the strength is
that of the best five-card hand, as at a showdown.

Strength layout
---------------
The strength spells out the best five-card hand::

    strength = category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5

``category`` runs from 0 (high card) to 8 (straight flush), see
:data:`CATEGORY_NAMES`. ``r1 .. r5`` are ranks (0 = deuce .. 12 = ace) that
break ties inside the category, most important first, unused fields 0:

=================  =========================================
category           r1 .. r5
=================  =========================================
high card, flush   the five cards, high to low
pair               the pair, then three kickers
two pair           high pair, low pair, kicker
three of a kind    the trips, then two kickers
straight (flush)   the top card (a Five for A-2-3-4-5)
full house         the trips, then the pair
four of a kind     the quads, then the kicker
=================  =========================================

Comparing two strengths as integers therefore compares the hands exactly as
the rules do, and :func:`describe_value` turns a strength back into words.
There are 7,462 different strengths over all five-card hands.

How it is computed
------------------
Two small tables, both built once:

1. **Non-flush table.** Without flushes, a hand's value depends only on how
   many cards it holds of each rank. Each card adds ``5 ** rank`` to a *rank
   key*, so the key is the 13 rank counts written in base 5 (no count exceeds
   4). The key is split into the counts of the seven low ranks (2-8) and of
   the six high ranks (9-A); two lookup tables turn the halves into numbers
   whose sum is a dense index over every possible rank multiset, and a third
   table holds the best non-flush hand (pairs, trips, straights, full houses,
   quads) of each multiset: 6,175 entries for 5 cards, 18,395 for 6 and
   49,205 for 7.
2. **Flush table.** Each card also adds ``1 << 3 * suit`` to a *suit key*
   (suit counts in 3-bit fields). If some suit holds five or more cards, the
   13-bit mask of that suit's ranks indexes an 8,192-entry table with the
   flush or straight-flush value.

The result is the larger of the two values. That is exact for up to seven
cards: five cards of one suit have five different ranks, and the two cards
left over cannot turn them into a full house or quads, so whenever a flush
exists it is the best hand short of a straight flush.

Both keys are sums over cards, so code that deals many boards to the same
hole cards (the equity module) can add the cards' keys up front and reuse
them; see :func:`card_keys` and :func:`value_from_keys`.

:func:`eval_hands` works through its input 8,192 rows at a time so that the
intermediate arrays stay in the CPU cache. One laptop core evaluates about
ten million seven-card hands per second this way.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np

from .cards import CardsLike, parse_cards

__all__ = [
    "HIGH_CARD",
    "PAIR",
    "TWO_PAIR",
    "TRIPS",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "QUADS",
    "STRAIGHT_FLUSH",
    "CATEGORY_NAMES",
    "eval_hands",
    "eval_hand",
    "category",
    "decode",
    "describe_value",
    "describe_hand",
    "CARD_KEY",
    "FLUSH_BIT",
    "card_keys",
    "value_from_keys",
    "nonflush_strengths",
    "flush_strengths",
    "rank_index",
    "rank_multisets",
]

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

_RANK_NAMES = ("Deuce", "Three", "Four", "Five", "Six", "Seven", "Eight",
               "Nine", "Ten", "Jack", "Queen", "King", "Ace")
_RANK_PLURALS = ("Deuces", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights",
                 "Nines", "Tens", "Jacks", "Queens", "Kings", "Aces")

_LOW_RANKS = 7                       # ranks 0..6 (2..8) form the low half of the rank key
_LOW_BASE = 5 ** _LOW_RANKS          # 78,125 possible low halves
_HIGH_BASE = 5 ** (13 - _LOW_RANKS)  # 15,625 possible high halves
_RANKS = np.arange(13)


def _pack(cat: np.ndarray | int, *ranks: np.ndarray | int) -> np.ndarray:
    """category and up to five tie-break ranks -> strength (see module docstring)."""
    value = np.asarray(cat, dtype=np.int64) << 20
    for i, r in enumerate(ranks):
        value = value | (np.asarray(r, dtype=np.int64) << (16 - 4 * i))
    return value


# ----------------------------------------------------------------------------
# Tables over 13-bit rank masks: straights and flushes
# ----------------------------------------------------------------------------

def _build_mask_tables() -> tuple[np.ndarray, np.ndarray]:
    masks = np.arange(1 << 13)
    top = np.full(masks.size, -1, dtype=np.int64)
    for t in range(4, 13):                         # higher straights overwrite lower ones
        run = 0x1F << (t - 4)
        top[(masks & run) == run] = t
    wheel = (1 << 12) | 0xF                        # A-2-3-4-5
    top[(top < 0) & ((masks & wheel) == wheel)] = 3

    bits = (masks[:, None] >> _RANKS) & 1          # (8192, 13)
    n_bits = bits.sum(1)
    desc = np.sort(np.where(bits == 1, _RANKS, -1), axis=1)[:, ::-1]   # ranks high to low
    flush = np.where(top >= 0, _pack(STRAIGHT_FLUSH, top), _pack(FLUSH, *desc[:, :5].T))
    flush = np.where(n_bits >= 5, flush, 0)
    return top, flush.astype(np.int32)


#: _STRAIGHT_TOP[mask]: top rank of the best straight in a 13-bit rank mask, or -1.
#: _FLUSH_VALUE[mask]: flush / straight-flush strength of one suit's ranks, 0 below five cards.
_STRAIGHT_TOP, _FLUSH_VALUE = _build_mask_tables()


def _build_flush_suit() -> np.ndarray:
    """suit key (3-bit count per suit) -> the suit holding five or more cards, else -1."""
    keys = np.arange(1 << 12)
    out = np.full(keys.size, -1, dtype=np.int64)
    for suit in range(4):
        out[((keys >> (3 * suit)) & 7) >= 5] = suit
    return out


_FLUSH_SUIT = _build_flush_suit()


# ----------------------------------------------------------------------------
# Non-flush table over rank multisets
# ----------------------------------------------------------------------------

def _digit_table(n_digits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every base-5 number with ``n_digits`` digits: digit sum, rank among the
    numbers with the same digit sum (ascending), and the count per digit sum."""
    codes = np.arange(5 ** n_digits)
    size = ((codes[:, None] // 5 ** np.arange(n_digits)) % 5).sum(1)
    count = np.bincount(size, minlength=4 * n_digits + 1)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    order = np.argsort(size, kind="stable")
    rank = np.empty_like(codes)
    rank[order] = np.arange(codes.size) - start[size[order]]
    return size, rank, count


_LOW_SIZE, _LOW_RANK, _LOW_COUNT = _digit_table(_LOW_RANKS)
_HIGH_SIZE, _HIGH_RANK, _HIGH_COUNT = _digit_table(13 - _LOW_RANKS)
#: Second half of the perfect hash: the rank of the high half among the high
#: halves with the same number of cards. The first half depends on k.
_HIGH_PART = _HIGH_RANK.astype(np.int32)


def _nonflush_values(counts: np.ndarray) -> np.ndarray:
    """Best non-flush five-card strength for each row of rank counts (M, 13)."""
    m = counts.shape[0]
    rows = np.arange(m)
    present = counts > 0
    straight = _STRAIGHT_TOP[(present << _RANKS).sum(1)]
    # Ranks ordered by (count, rank), both descending: the "groups" of the hand.
    group = np.argsort(-np.where(present, counts * 16 + _RANKS, -1), axis=1, kind="stable")
    group_count = np.take_along_axis(counts, group, 1)
    g0, g1, g2, g3, g4 = group[:, :5].T
    c0, c1 = group_count[:, 0], group_count[:, 1]
    # Present ranks high to low; a kicker is the best one not already used.
    desc = np.sort(np.where(present, _RANKS, -1), axis=1)[:, ::-1][:, :3]

    def kicker(*used: np.ndarray) -> np.ndarray:
        ok = np.ones(desc.shape, dtype=bool)
        for u in used:
            ok &= desc != u[:, None]
        return desc[rows, ok.argmax(1)]

    conditions = [
        c0 == 4,
        (c0 == 3) & (c1 >= 2),
        straight >= 0,
        c0 == 3,
        (c0 == 2) & (c1 == 2),
        c0 == 2,
    ]
    choices = [
        _pack(QUADS, g0, kicker(g0)),
        _pack(FULL_HOUSE, g0, g1),
        _pack(STRAIGHT, straight),
        _pack(TRIPS, g0, g1, g2),
        _pack(TWO_PAIR, g0, g1, kicker(g0, g1)),
        _pack(PAIR, g0, g1, g2, g3),
    ]
    return np.select(conditions, choices, _pack(HIGH_CARD, g0, g1, g2, g3, g4))


@lru_cache(maxsize=None)
def _nonflush_table(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low-half part of the perfect hash, value per dense index, rank counts
    per dense index) for k-card hands."""
    # Number of rank multisets with a low cards and k - a high cards, laid out
    # one block per a; inside a block: low rank * (#high halves) + high rank.
    high = [int(_HIGH_COUNT[k - a]) for a in range(k + 1)]
    blocks = [int(_LOW_COUNT[a]) * high[a] for a in range(k + 1)]
    offset = np.concatenate([[0], np.cumsum(blocks)[:-1]])
    a = np.minimum(_LOW_SIZE, k)
    low_part = np.where(_LOW_SIZE <= k, offset[a] + _LOW_RANK * np.array(high)[a], 0).astype(np.int32)

    tuples = np.array(list(combinations_with_replacement(range(13), k)), dtype=np.int64)
    counts = np.zeros((len(tuples), 13), dtype=np.int64)
    np.add.at(counts, (np.arange(len(tuples))[:, None], tuples), 1)
    counts = counts[counts.max(1) <= 4]            # a deck has four cards of each rank
    key = (counts * 5 ** _RANKS).sum(1)
    index = low_part[key % _LOW_BASE] + _HIGH_PART[key // _LOW_BASE]
    if not np.array_equal(np.sort(index), np.arange(sum(blocks))):
        raise AssertionError("rank-multiset hash is not a bijection")  # pragma: no cover
    values = np.zeros(sum(blocks), dtype=np.int32)
    values[index] = _nonflush_values(counts)
    by_index = np.zeros_like(counts)
    by_index[index] = counts
    for table in (low_part, values, by_index):
        table.setflags(write=False)
    return low_part, values, by_index


# ----------------------------------------------------------------------------
# Per-card keys and evaluation from keys
# ----------------------------------------------------------------------------

_CARD_IDS = np.arange(52)
#: Key of each card: rank key ``5 ** rank`` in the low 32 bits and suit key
#: ``1 << 3 * suit`` (3-bit suit counts) from bit 32 up. Sums over up to seven
#: cards never carry from one part into the other.
CARD_KEY = (5 ** (_CARD_IDS >> 2)).astype(np.int64) | ((1 << (3 * (_CARD_IDS & 3))).astype(np.int64) << 32)
#: Flush bit of each card: bit ``13 * suit + rank``, so suit s's ranks sit in
#: bits 13s .. 13s + 12 of a sum.
FLUSH_BIT = np.int64(1) << (13 * (_CARD_IDS & 3) + (_CARD_IDS >> 2)).astype(np.int64)
CARD_KEY.setflags(write=False)
FLUSH_BIT.setflags(write=False)
_RANK_KEY_MASK = (1 << 32) - 1
_EVAL_BLOCK = 8192        # rows per step in eval_hands: the temporaries stay in cache


def _row_sum(a: np.ndarray) -> np.ndarray:
    """Sum over the last axis; for a few cards per row einsum is about twice
    as fast as ``a.sum(-1)``."""
    return np.einsum("...i->...", a)


def card_keys(cards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum the per-card keys along the last axis of an array of card ids.

    Returns ``(key, flush_bits)``, both int64 with the shape of ``cards``
    minus its last axis. Keys of disjoint card sets add up, so the keys of
    hole cards and of a board can be computed once and combined with ``+``.
    """
    c = np.asarray(cards).astype(np.intp, copy=False)
    return _row_sum(CARD_KEY[c]), _row_sum(FLUSH_BIT[c])


def value_from_keys(key: np.ndarray, flush_bits: np.ndarray, k: int = 7) -> np.ndarray:
    """Strengths of k-card hands from their summed keys (see :func:`card_keys`).

    Keys of any shape work, including the 0-d keys of a single hand, which
    give a single strength. A key that no real hand has (a rank five times)
    gives an arbitrary strength instead of an error; callers that evaluate a
    fixed grid of hands give such entries zero weight.
    """
    key = np.asarray(key)
    flush_bits = np.broadcast_to(flush_bits, key.shape)
    if key.ndim == 0:      # one hand, e.g. from card_keys of a 1-D array
        return _strengths(key[None], lambda hit: flush_bits[None][hit], k)[0]
    return _strengths(key, lambda hit: flush_bits[hit], k)


def nonflush_strengths(rank_key: np.ndarray, k: int = 7) -> np.ndarray:
    """Best strength of k-card hands when suits are ignored (no flushes).

    ``rank_key`` is the sum of ``5 ** rank`` over the cards, i.e. the low
    32 bits of a summed :data:`CARD_KEY`. The equity module uses this to
    value all 91 rank pairs an opponent can hold on one board at once.
    """
    _, values, _ = _nonflush_table(k)
    return values[np.minimum(rank_index(rank_key, k), len(values) - 1)]


def rank_index(rank_key: np.ndarray, k: int) -> np.ndarray:
    """Dense index of each k-card rank multiset (suits ignored), from its rank key.

    ``rank_key`` is the sum of ``5 ** rank`` over the k cards. The index runs
    from 0 to ``len(rank_multisets(k)) - 1`` (6,175 multisets of five cards,
    18,395 of six, 49,205 of seven), so it can address a table with one row
    per multiset. A key that no real hand has gives an arbitrary index.
    """
    low_part, _, _ = _nonflush_table(k)
    rank_key = np.asarray(rank_key)
    return low_part[rank_key % _LOW_BASE] + _HIGH_PART[np.minimum(rank_key // _LOW_BASE, _HIGH_BASE - 1)]


def rank_multisets(k: int) -> np.ndarray:
    """Rank counts (n, 13) of every k-card rank multiset, row i having :func:`rank_index` i."""
    return _nonflush_table(k)[2]


def flush_strengths(rank_mask: np.ndarray) -> np.ndarray:
    """Flush or straight-flush strength of a 13-bit mask of one suit's ranks
    (bit r = rank r); 0 when fewer than five bits are set."""
    return _FLUSH_VALUE[rank_mask]


def _strengths(key: np.ndarray, flush_bits_at, k: int) -> np.ndarray:
    """Core lookup. ``flush_bits_at(hit)`` returns the summed flush bits of the
    hands at the index tuple ``hit`` (from ``np.nonzero``); it is only called
    for hands holding a flush."""
    value = nonflush_strengths(key & _RANK_KEY_MASK, k)
    suit = _FLUSH_SUIT[(key >> 32) & 0xFFF]
    hit = np.nonzero(suit >= 0)
    if hit[0].size:
        # Index with the tuple, not through a flattened view: reshape(-1)
        # silently copies an array that is not C-contiguous.
        ranks = (flush_bits_at(hit) >> (13 * suit[hit])) & 0x1FFF
        value[hit] = np.maximum(value[hit], _FLUSH_VALUE[ranks])
    return value


_bit_count = getattr(np, "bitwise_count", None)   # numpy >= 2.0


# ----------------------------------------------------------------------------
# Public evaluation API
# ----------------------------------------------------------------------------

def eval_hands(cards: np.ndarray, *, check: bool = True) -> np.ndarray:
    """Strength of each hand in an (n, k) array of card ids, k = 5, 6 or 7.

    Returns an int32 array of n strengths: higher wins, equal is a tie. With
    6 or 7 cards each strength is that of the best five-card hand. See the
    module docstring for the layout of the value.

    Args:
        cards: integer array of shape (n, k) with distinct card ids per row.
        check: verify ids are in 0..51 and distinct within each row. Turning
            it off saves about a quarter of the time when the caller
            guarantees valid input.

    Raises:
        ValueError: wrong shape, a card id out of range or a repeated card.
    """
    c = np.asarray(cards)
    if c.ndim != 2 or not 5 <= c.shape[1] <= 7:
        raise ValueError(f"expected an (n, k) array with k in 5..7, got shape {c.shape}")
    if c.dtype.kind not in "iu":
        raise ValueError(f"card ids must be integers, got dtype {c.dtype}")
    k = c.shape[1]
    if c.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    if check and (c.min() < 0 or c.max() > 51):
        raise ValueError("card ids must be in 0..51")
    out = np.empty(c.shape[0], dtype=np.int32)
    for lo in range(0, c.shape[0], _EVAL_BLOCK):
        part = c[lo:lo + _EVAL_BLOCK].astype(np.intp)
        key = _row_sum(CARD_KEY[part])
        if check:
            # One bit per card, so the sum of the bits is their OR unless a
            # card repeats; then the sum carries and fewer than k bits are set.
            bits = _row_sum(FLUSH_BIT[part])
            if _bit_count is not None:
                repeated = _bit_count(bits) != k
            else:  # pragma: no cover - numpy < 2.0
                repeated = np.bitwise_or.reduce(FLUSH_BIT[part], axis=1) != bits
            if repeated.any():
                raise ValueError("a hand contains the same card twice")
            out[lo:lo + _EVAL_BLOCK] = _strengths(key, lambda hit, bits=bits: bits[hit], k)
        else:
            out[lo:lo + _EVAL_BLOCK] = _strengths(key, lambda hit, part=part: _row_sum(FLUSH_BIT[part[hit]]), k)
    return out


def eval_hand(cards: CardsLike) -> int:
    """Strength of one hand of 5-7 cards given as text or ids: ``eval_hand("Ah Kh Qh Jh Th")``."""
    ids = parse_cards(cards)
    return int(eval_hands(np.array([ids], dtype=np.int64))[0])


def category(strength: int | np.ndarray) -> int | np.ndarray:
    """Hand category 0..8 (high card .. straight flush) of a strength or an array of them."""
    out = np.asarray(strength) >> 20
    return int(out) if out.ndim == 0 else out


def decode(strength: int) -> tuple[int, tuple[int, ...]]:
    """Split a strength into ``(category, ranks)`` with only the meaningful ranks.

    ``decode(eval_hand("Kh Kd 7s 7c 2d"))`` is ``(2, (11, 5, 0))``: two pair,
    kings and sevens, deuce kicker.
    """
    s = int(strength)
    cat = s >> 20
    n = {HIGH_CARD: 5, PAIR: 4, TWO_PAIR: 3, TRIPS: 3, STRAIGHT: 1, FLUSH: 5,
         FULL_HOUSE: 2, QUADS: 2, STRAIGHT_FLUSH: 1}[cat]
    return cat, tuple((s >> (16 - 4 * i)) & 0xF for i in range(n))


def describe_value(strength: int) -> str:
    """Hand-history style text of a strength, e.g. ``"a full house, Kings full of Sevens"``."""
    cat, r = decode(strength)
    name, plural = _RANK_NAMES, _RANK_PLURALS
    if cat in (STRAIGHT, STRAIGHT_FLUSH):
        low = name[r[0] - 4] if r[0] >= 4 else name[12]          # the wheel runs Ace to Five
        if cat == STRAIGHT:
            return f"a straight, {low} to {name[r[0]]}"
        return "a Royal Flush" if r[0] == 12 else f"a straight flush, {low} to {name[r[0]]}"
    if cat == QUADS:
        return f"four of a kind, {plural[r[0]]}"
    if cat == FULL_HOUSE:
        return f"a full house, {plural[r[0]]} full of {plural[r[1]]}"
    if cat == FLUSH:
        return f"a flush, {name[r[0]]} high"
    if cat == TRIPS:
        return f"three of a kind, {plural[r[0]]}"
    if cat == TWO_PAIR:
        return f"two pair, {plural[r[0]]} and {plural[r[1]]}"
    if cat == PAIR:
        return f"a pair of {plural[r[0]]}"
    return f"high card {name[r[0]]}"


def describe_hand(cards: CardsLike | Iterable[int]) -> str:
    """Text of the best hand in 5-7 cards: ``describe_hand("Kh Qh 9h 4h 2h 2c 2d")`` -> ``"a flush, King high"``."""
    return describe_value(eval_hand(cards))
