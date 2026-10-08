"""Hold'em equity: the share of the pot a hand wins on average at showdown.

Equity answers one question: if nobody bet again and the board were dealt to
the end, what share of the pot would the hero take on average? A won pot
counts 1, a lost pot 0, and a pot split k ways 1/k. The functions differ in
what they assume about the opponent's cards.

Three kinds of equity
---------------------
Equity vs the opponent's real hand (:func:`equity_vs_hand`)
    Both players' hole cards are known and only the board cards still to come
    are not. The result averages over every way of finishing the board. A
    player who can see the opponent's cards is playing against this number,
    which makes it the yardstick for superuser detection.

Equity vs a random hand (:func:`equity_vs_random`)
    The opponent holds any two cards the hero cannot see, every holding
    equally likely. It needs no model of the opponent, so it is the simplest
    measure of how strong a hand is in a spot.

Equity vs a range (:func:`equity_vs_range`)
    The opponent's 1,326 possible holdings carry weights (in the order of
    :data:`sudetect.cards.COMBOS`), for example from a model of the hands that
    would have taken the actions seen so far. Equal weights give the equity
    vs a random hand; a single combo gives the equity vs that hand.

:func:`multiway_equity` returns every player's share when several hands are
known. The ``*_batch`` functions compute the same numbers for many rows of an
action table at once.

Exact or Monte Carlo
--------------------
*Exact* means every remaining board (and, when the opponent's hand is
unknown, every opponent holding) is enumerated, so the result is the true
fraction with no noise. The amount of work depends on the street:

========  ===================  =======================================
street    board cards to come  boards to enumerate (heads-up)
========  ===================  =======================================
river     0                    1
turn      1                    44 (known opponent) or 46
flop      2                    990 or 1,081
preflop   5                    1,712,304 or 2,118,760
========  ===================  =======================================

*Monte Carlo* deals ``n_samples`` random completions (and opponent holdings)
from a seeded generator instead. The result is reproducible for a given seed
but carries noise; its standard error is at most ``0.5 / sqrt(n_samples)``
(0.008 for the batch default of 4,096 samples).

What ``exact=None`` (the default) does, per function and street:

==========================  ==============  ======  ===========
function                    preflop         flop    turn, river
==========================  ==============  ======  ===========
:func:`equity_vs_hand`      exact           exact   exact
:func:`equity_vs_random`    exact (table)   exact   exact
:func:`equity_vs_range`     Monte Carlo     exact   exact
:func:`multiway_equity`     exact           exact   exact
equity_vs_hand_batch        Monte Carlo     exact   exact
equity_vs_random_batch      exact (table)   MC      exact
equity_vs_range_batch       Monte Carlo     MC      exact
==========================  ==============  ======  ===========

``exact=True`` turns every Monte Carlo cell into an exact enumeration and
``exact=False`` samples wherever the function supports sampling. "Table"
means :data:`PREFLOP_VS_RANDOM`, the exact equity of the 169 starting hands
against a random hand, produced by :func:`preflop_vs_random_exact`.

Rough costs on one laptop core: an exact flop spot takes a few milliseconds
(vs a known hand well under one), heads-up preflop against a known hand 0.3
seconds, preflop against a range about 15 seconds. The batch functions
handle 10,000 turn rows in about 0.1 s against known hands and about 1 s
against a random hand.

Batch Monte Carlo is reproducible row by row: a row's samples depend only on
its own cards and the seed, never on the other rows in the batch. Each sample
deals the unknown cards in an order read from a fixed table of 65,536
shuffles (one table per street and seed); a hash of the row's cards picks
where in that table the row starts reading and the order in which it lays
out its live cards. Two rows that read the same shuffles therefore still
deal unrelated cards, and the errors of many rows average out the way
independent errors do.

How exact equity against many holdings stays fast
-------------------------------------------------
On a complete board an opponent holding's strength depends only on its two
ranks, unless the holding completes a flush. So instead of evaluating up to
1,326 holdings per board, the code evaluates the 91 rank pairs (``22``,
``32``, ... ``AA``), counts each pair once with the total weight of its live
combos, and then corrects only the holdings that make a flush, which is
possible only when the board shows three or more cards of one suit. Holdings
that contain a card dealt on a later street are taken out again. A turn spot
against a full range costs 46 x 91 rank-pair lookups instead of 46 x 990
hand evaluations.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import lru_cache
from typing import Optional

import numpy as np

from .cards import (
    COMBO_INDEX,
    COMBO_TO_CLASS,
    COMBOS,
    N_COMBOS,
    CardsLike,
    class_combos,
    class_index,
    hand_class,
    parse_cards,
)
from .evaluator import (
    card_keys,
    flush_strengths,
    nonflush_strengths,
    rank_index,
    rank_multisets,
    value_from_keys,
)

__all__ = [
    "equity_vs_hand",
    "equity_vs_random",
    "equity_vs_range",
    "multiway_equity",
    "equity_vs_hand_batch",
    "equity_vs_random_batch",
    "equity_vs_range_batch",
    "preflop_vs_random_exact",
    "PREFLOP_VS_RANDOM",
]

_RANK_KEY_MASK = (1 << 32) - 1
_CHUNK = 1 << 13          # board completions scored per vectorised step (fits in cache)
_POOL_SIZE = 1 << 16      # shuffles per Monte Carlo table in the batch functions
_TICKS = 1 << 40          # resolution of the weighted draw in the batch functions
_ROW_BLOCK = 4096         # rows per block in the batch functions (bounds the memory used)


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------

def _hand(cards: CardsLike, who: str = "hero") -> list[int]:
    ids = parse_cards(cards)
    if len(ids) != 2:
        raise ValueError(f"{who} needs 2 hole cards, got {len(ids)}")
    return ids


def _board(cards: Optional[CardsLike]) -> list[int]:
    ids = [] if cards is None else parse_cards(cards)
    if len(ids) not in (0, 3, 4, 5):
        raise ValueError(f"a board has 0, 3, 4 or 5 cards, got {len(ids)}")
    return ids


def _all_distinct(*groups: list[int]) -> list[int]:
    cards = [c for g in groups for c in g]
    if len(set(cards)) != len(cards):
        raise ValueError("the same card is used twice")
    return cards


def _weights(weights: np.ndarray, rows: Optional[int] = None) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    shape = (N_COMBOS,) if rows is None else (rows, N_COMBOS)
    if w.shape != shape:
        raise ValueError(f"weights must have shape {shape}, got {w.shape}")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return w


def _scaled(weights: np.ndarray) -> np.ndarray:
    """Divide each range by its largest weight (rows of zeros stay zero).

    Only ratios matter, and with the largest weight at 1 the sums over
    holdings and board completions can neither overflow nor lose precision
    to subnormal numbers, whatever the scale of the input.
    """
    top = weights.max(axis=-1, keepdims=True)
    return weights / np.where(top > 0, top, 1.0)


def _check_samples(n_samples: int) -> None:
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")


def _check_rows(hero: np.ndarray, board: np.ndarray, villain: Optional[np.ndarray] = None) -> tuple:
    """Validate batch inputs; returns (hero, board, villain, board card count per row)."""
    hero = np.asarray(hero)
    board = np.asarray(board)
    if hero.ndim != 2 or hero.shape[1] != 2:
        raise ValueError(f"hero must have shape (n, 2), got {hero.shape}")
    n = hero.shape[0]
    if board.ndim != 2 or board.shape[0] != n or board.shape[1] > 5:
        raise ValueError(f"board must have shape (n, 5) with -1 for undealt cards, got {board.shape}")
    parts = [hero] if villain is None else [hero, np.asarray(villain)]
    if villain is not None and parts[1].shape != (n, 2):
        raise ValueError(f"villain must have shape (n, 2), got {parts[1].shape}")
    for p in parts + [board]:
        if p.dtype.kind not in "iu":
            raise ValueError("card ids must be integers")
    hole = np.concatenate(parts, axis=1).astype(np.int64)
    board = board.astype(np.int64)          # before padding: -1 does not fit an unsigned dtype
    if board.shape[1] < 5:
        board = np.concatenate([board, np.full((n, 5 - board.shape[1]), -1, np.int64)], axis=1)
    if hole.size and (hole.min() < 0 or hole.max() > 51):
        raise ValueError("hole card ids must be in 0..51")
    if board.size and (board.min() < -1 or board.max() > 51):
        raise ValueError("board ids must be in 0..51, or -1 for an undealt card")
    dealt = board >= 0
    n_board = dealt.sum(1)
    if not np.isin(n_board, (0, 3, 4, 5)).all() or (dealt[:, 1:] & ~dealt[:, :-1]).any():
        raise ValueError("each board must hold 0, 3, 4 or 5 cards, dealt from the left")
    cards = np.concatenate([hole, board], axis=1)
    bits = np.where(cards >= 0, np.int64(1) << np.where(cards >= 0, cards, 0), 0)
    if (np.bitwise_or.reduce(bits, axis=1) != bits.sum(1)).any():
        raise ValueError("a row uses the same card twice")
    villain_out = None if villain is None else hole[:, 2:]
    return hole[:, :2], board, villain_out, n_board


def _live_cards(known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """known (R, m) distinct cards per row -> (dead mask (R, 52), live cards (R, 52 - m) ascending)."""
    r = known.shape[0]
    dead = np.zeros((r, 52), dtype=bool)
    dead[np.arange(r)[:, None], known] = True
    live = np.nonzero(~dead)[1].reshape(r, 52 - known.shape[1])
    return dead, live


@lru_cache(maxsize=None)
def _combinations(n: int, k: int) -> np.ndarray:
    """All k-subsets of range(n) as rows of ascending positions (int8), lexicographic order."""
    if k == 0:
        out = np.zeros((1, 0), dtype=np.int8)
    elif k == 1:
        out = np.arange(n, dtype=np.int8)[:, None]
    else:
        parts = []
        for first in range(n - k + 1):
            rest = _combinations(n - first - 1, k - 1) + np.int8(first + 1)
            parts.append(np.concatenate([np.full((len(rest), 1), first, dtype=np.int8), rest], axis=1))
        out = np.concatenate(parts)
    out.setflags(write=False)
    return out


def _share(hero: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Hero's share of a heads-up pot: 1 win, 0.5 tie, 0 loss."""
    return 0.5 + 0.5 * np.sign(hero.astype(np.int64) - other)


# ----------------------------------------------------------------------------
# Tables for scoring a hero against all opponent holdings on one board
# ----------------------------------------------------------------------------

_HIGH = COMBOS[:, 0].astype(np.intp) >> 2
_LOW = COMBOS[:, 1].astype(np.intp) >> 2
#: Rank pair of each combo: (a, b) with a >= b -> a * (a + 1) / 2 + b, 91 values.
_PAIR_OF_COMBO = _HIGH * (_HIGH + 1) // 2 + _LOW
_PAIR_HIGH = np.repeat(np.arange(13), np.arange(1, 14))
_PAIR_LOW = np.concatenate([np.arange(a + 1) for a in range(13)])
_PAIR_RANK_KEY = (5 ** _PAIR_HIGH + 5 ** _PAIR_LOW).astype(np.int64)
_PAIR_IS_POCKET = _PAIR_HIGH == _PAIR_LOW
_POCKET_COLUMNS = np.flatnonzero(_PAIR_IS_POCKET)          # rank pair of 22, 33, ..., AA
#: One-hot (13, 91) maps from a rank to the rank pairs whose high / low rank it is.
_RANK_TO_PAIR_HIGH = (np.arange(13)[:, None] == _PAIR_HIGH).astype(np.float32)
_RANK_TO_PAIR_LOW = (np.arange(13)[:, None] == _PAIR_LOW).astype(np.float32)
_PAIR_SUM = np.zeros((N_COMBOS, 91))
_PAIR_SUM[np.arange(N_COMBOS), _PAIR_OF_COMBO] = 1.0
#: The 51 combos that hold each card.
_COMBOS_WITH = np.array([np.flatnonzero((COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)) for c in range(52)])
#: 64-bit mask of each combo's two cards.
_COMBO_MASK = (np.int64(1) << COMBOS[:, 0].astype(np.int64)) | (np.int64(1) << COMBOS[:, 1].astype(np.int64))
_POW5 = 5 ** np.arange(13, dtype=np.int64)
_LIVE_PAIRS_ON_RIVER = 45 * 44 // 2       # holdings left once 2 hole cards and 5 board cards are out


def _build_flush_candidates() -> dict[tuple[int, int], tuple[np.ndarray, ...]]:
    """(suit, cards needed) -> combos holding at least that many cards of the suit,
    the 13-bit rank mask of their cards in that suit, their card masks and rank pairs."""
    suit = COMBOS.astype(np.int64) & 3
    rank = COMBOS.astype(np.int64) >> 2
    out = {}
    for s in range(4):
        in_suit = suit == s
        count = in_suit.sum(1)
        bits = np.where(in_suit, 1 << rank, 0).sum(1)
        for need in (0, 1, 2):
            idx = np.flatnonzero(count >= need)
            out[s, need] = (idx, bits[idx], _COMBO_MASK[idx], _PAIR_OF_COMBO[idx])
    return out


_FLUSH_CANDIDATES = _build_flush_candidates()


@lru_cache(maxsize=None)
def _pair_value_table() -> np.ndarray:
    """(6175, 91) int32: suit-blind strength of each rank pair on each five-card
    board rank multiset (row = :func:`~sudetect.evaluator.rank_index` of the
    board). Pairs that would need a fifth card of a rank hold 0."""
    counts = rank_multisets(5)
    board_key = counts @ _POW5
    value = nonflush_strengths(board_key[:, None] + _PAIR_RANK_KEY, 7)
    over = (counts[:, _PAIR_HIGH] + 1 + _PAIR_IS_POCKET > 4) | (counts[:, _PAIR_LOW] + 1 > 4)
    value[over] = 0
    value.setflags(write=False)
    return value


def _score_completions(rows: np.ndarray, runout: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Score a chunk of (row, board completion) instances against all opponent holdings.

    ``rows`` (N,) gives each instance's row, ``runout`` (N, j) the cards that
    complete its board. ``ctx`` holds per-row arrays prepared by
    :func:`_exact_vs_holdings`. Returns ``(num, den)`` per instance: the
    opponent weight times the hero's pot share, and the opponent weight.
    """
    n, j = runout.shape
    weights = ctx["weights"]
    run_key, run_bits = card_keys(runout)
    board_key = ctx["board_key"][rows] + run_key
    board_bits = ctx["board_bits"][rows] + run_bits
    seven_key = ctx["hero_key"][rows] + run_key
    hero = value_from_keys(seven_key, ctx["hero_bits"][rows] + run_bits)
    dead_mask = ctx["dead_mask"][rows] | _sum_masks(runout)

    # Strength of every rank pair on this board, suits ignored, and the
    # hero's result against it: +1 ahead, 0 tie, -1 behind. The random
    # opponent only needs small integers, which float32 holds exactly.
    pair_value = _pair_value_table()[rank_index(board_key & _RANK_KEY_MASK, 5)]
    sign = np.subtract(hero[:, None], pair_value, dtype=np.float32 if weights is None else np.float64)
    np.sign(sign, out=sign)

    if weights is None:
        # Random opponent: every live combo has weight 1, so a rank pair is
        # worth hi * lo combos (hi, lo = live cards of its two ranks), or
        # l * (l - 1) / 2 for a pocket pair. There are always 990 in all.
        live = (4 - (seven_key & _RANK_KEY_MASK)[:, None] // _POW5 % 5).astype(np.float32)
        w = (live @ _RANK_TO_PAIR_HIGH) * (live @ _RANK_TO_PAIR_LOW)
        w[:, _POCKET_COLUMNS] = live * (live - 1) / 2
        den = np.full(n, float(_LIVE_PAIRS_ON_RIVER))
        num = 0.5 * (den + np.einsum("ij,ij->i", w, sign))
    else:
        w = ctx["pair_weights"][rows]
        den = w.sum(1)
        num = 0.5 * (den + np.einsum("ij,ij->i", w, sign))
        if j:
            # Combos holding a card that the completion put on the board are
            # impossible now: take them out. A combo made of two such cards is
            # listed twice by the first step and put back once by the second.
            touch = _COMBOS_WITH[runout].reshape(n, 51 * j)
            wt = weights[rows[:, None], touch]
            st = sign[np.arange(n)[:, None], _PAIR_OF_COMBO[touch]]
            den -= wt.sum(1)
            num -= 0.5 * (wt.sum(1) + np.einsum("ij,ij->i", wt, st))
            if j > 1:
                a, b = np.triu_indices(j, 1)
                both = COMBO_INDEX[runout[:, a], runout[:, b]]
                wb = weights[rows[:, None], both]
                sb = sign[np.arange(n)[:, None], _PAIR_OF_COMBO[both]]
                den += wb.sum(1)
                num += 0.5 * (wb.sum(1) + np.einsum("ij,ij->i", wb, sb))

    # Holdings that complete a flush: swap their rank-pair result for the
    # result against their flush. Only a suit with 3+ board cards qualifies,
    # and on a complete board at most one suit can have three.
    suit_count = ((board_key >> 32)[:, None] >> (3 * np.arange(4))) & 7
    flush_suit = suit_count.argmax(1)
    on_board = suit_count.max(1)
    for need in (2, 1, 0):                       # board shows 3, 4 or 5 of the suit
        for suit in range(4):
            idx = np.flatnonzero((on_board == 5 - need) & (flush_suit == suit))
            if not idx.size:
                continue
            cand, cand_bits, cand_mask, cand_pair = _FLUSH_CANDIDATES[suit, need]
            board_ranks = (board_bits[idx] >> (13 * suit)) & 0x1FFF
            flush_value = flush_strengths(board_ranks[:, None] | cand_bits)
            delta = np.subtract(hero[idx, None], flush_value, dtype=sign.dtype)
            np.sign(delta, out=delta)
            delta -= sign[idx[:, None], cand_pair]
            alive = (dead_mask[idx, None] & cand_mask) == 0
            if weights is None:
                wf = alive.astype(sign.dtype)
            else:
                wf = alive * weights[rows[idx, None], cand]
            num[idx] += 0.5 * np.einsum("ij,ij->i", wf, delta)
    return num, den


def _sum_masks(cards: np.ndarray) -> np.ndarray:
    """64-bit mask of the (distinct) cards in each row of ``cards``."""
    out = np.zeros(cards.shape[0], dtype=np.int64)
    for i in range(cards.shape[1]):
        out |= np.int64(1) << cards[:, i].astype(np.int64)
    return out


def _exact_vs_holdings(hero: np.ndarray, board: np.ndarray,
                       weights: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Exact (weighted pot share, total weight) per row, enumerating every completion.

    hero (R, 2) and board (R, b) hold card ids, all rows on the same street;
    weights (R, 1326) or None for a random opponent.
    """
    r, b = board.shape
    known = np.concatenate([hero, board], axis=1).astype(np.intp)
    dead, live = _live_cards(known)
    hero_key, hero_bits = card_keys(known)
    board_key, board_bits = card_keys(board.astype(np.intp))
    pair_weights = None
    if weights is not None:
        weights = _scaled(np.where(dead[:, COMBOS[:, 0]] | dead[:, COMBOS[:, 1]], 0.0, weights))
        pair_weights = weights @ _PAIR_SUM
    ctx = dict(hero_key=hero_key, hero_bits=hero_bits, board_key=board_key, board_bits=board_bits,
               dead_mask=_sum_masks(known), weights=weights, pair_weights=pair_weights)
    template = _combinations(live.shape[1], 5 - b)
    n_run = len(template)
    num = np.zeros(r)
    den = np.zeros(r)
    total = r * n_run
    for start in range(0, total, _CHUNK):
        rows, run = np.divmod(np.arange(start, min(start + _CHUNK, total)), n_run)
        runout = live[rows[:, None], template[run]]
        nm, dn = _score_completions(rows, runout, ctx)
        num += np.bincount(rows, nm, r)
        den += np.bincount(rows, dn, r)
    return num, den


# ----------------------------------------------------------------------------
# Monte Carlo helpers
# ----------------------------------------------------------------------------

def _sample_subsets(cards: np.ndarray, k: int, n: int, rng: np.random.Generator,
                    exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """n random k-subsets of ``cards`` (ascending card ids), each uniform.

    ``exclude`` (n, m) lists cards that sample i must avoid.
    """
    if k == 0:
        return np.zeros((n, 0), dtype=np.intp)
    keys = rng.random((n, cards.size))
    if exclude is not None:
        position = np.full(52, -1)
        position[cards] = np.arange(cards.size)
        keys[np.arange(n)[:, None], position[exclude]] = 2.0          # never among the k smallest
    return cards[np.argpartition(keys, k - 1, axis=1)[:, :k]]


def _mc_vs_holdings(hero: list[int], board: list[int], weights: np.ndarray,
                    n_samples: int, seed: int) -> float:
    """Monte Carlo equity vs weighted holdings (weights already zero where blocked)."""
    _check_samples(n_samples)
    rng = np.random.default_rng(seed)
    live = np.setdiff1d(np.arange(52), hero + board)
    hero_key, hero_bits = card_keys(np.array(hero + board))
    board_key, board_bits = card_keys(np.array(board, dtype=np.intp))
    p = weights / weights.sum()
    total = 0.0
    for start in range(0, n_samples, 1 << 16):
        n = min(1 << 16, n_samples - start)
        villain = COMBOS[rng.choice(N_COMBOS, size=n, p=p)].astype(np.intp)
        runout = _sample_subsets(live, 5 - len(board), n, rng, exclude=villain)
        run_key, run_bits = card_keys(runout)
        vil_key, vil_bits = card_keys(villain)
        hv = value_from_keys(hero_key + run_key, hero_bits + run_bits)
        vv = value_from_keys(vil_key + board_key + run_key, vil_bits + board_bits + run_bits)
        total += _share(hv, vv).sum()
    return float(total / n_samples)


def _mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser: a well-spread 64-bit hash of each element."""
    x = np.asarray(x, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _row_hash(seed: int, *card_groups: np.ndarray) -> np.ndarray:
    """64-bit hash per row of the given card groups (role-aware) and the seed."""
    n = card_groups[0].shape[0]
    h = _mix64(np.full(n, seed & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    for group in card_groups:
        g = np.asarray(group, dtype=np.int64)
        bits = np.where(g >= 0, np.int64(1) << np.maximum(g, 0), 0).sum(1)
        h = _mix64(h ^ bits.astype(np.uint64))
    return h


@lru_cache(maxsize=16)
def _shuffle_pool(n_live: int, width: int, seed: int) -> np.ndarray:
    """(65,536, width) int8: the first ``width`` positions of independent
    uniform shuffles of range(n_live). Fixed for a given (n_live, width, seed)."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, n_live, width])
    base = np.broadcast_to(np.arange(n_live, dtype=np.int8), (_POOL_SIZE, n_live))
    pool = np.ascontiguousarray(rng.permuted(base, axis=1)[:, :width])
    pool.setflags(write=False)
    return pool


def _mc_rows(hero: np.ndarray, board: np.ndarray, villain: Optional[np.ndarray],
             weights: Optional[np.ndarray], n_samples: int, seed: int) -> np.ndarray:
    """Monte Carlo pot share per row, all rows on the same street.

    villain (R, 2) known hands, or None; then the opponent is random
    (weights None) or drawn from weights (R, 1326).
    """
    if not 1 <= n_samples <= _POOL_SIZE:
        raise ValueError(f"n_samples must be between 1 and {_POOL_SIZE} for the batch functions")
    seed = operator.index(seed)          # numpy integers too; the hashing below needs a Python int
    r, b = board.shape
    j = 5 - b
    known_parts = [hero, board] if villain is None else [hero, villain, board]
    known = np.concatenate(known_parts, axis=1).astype(np.intp)
    dead, live = _live_cards(known)
    hero_key, hero_bits = card_keys(np.concatenate([hero, board], axis=1).astype(np.intp))
    board_key, board_bits = card_keys(board.astype(np.intp))
    if villain is not None:
        vil_key, vil_bits = card_keys(np.concatenate([villain, board], axis=1).astype(np.intp))
    width = j if villain is not None else j + 2
    pool = _shuffle_pool(live.shape[1], width, seed)
    row_hash = _row_hash(seed, *known_parts)
    start = (row_hash % np.uint64(_POOL_SIZE)).astype(np.int64)
    # Each row lays its live cards out in its own hashed order before it
    # reads the shared shuffles. In ascending order, rows with similar dead
    # cards would turn the same shuffle into nearly the same cards, so their
    # errors would be shared and would not average out over many rows.
    order_key = _mix64(_mix64(row_hash ^ np.uint64(0xD6E8FEB86659FD93))[:, None]
                       + np.arange(live.shape[1], dtype=np.uint64))
    live = np.take_along_axis(live, np.argsort(order_key, axis=1), axis=1)
    if weights is not None:
        # Each row's cumulative weights as integer ticks (2**40 per row), so
        # one vectorised search draws combos for many rows with exactly the
        # result a row would get on its own.
        weights = _scaled(np.where(dead[:, COMBOS[:, 0]] | dead[:, COMBOS[:, 1]], 0.0, weights))
        cum = np.cumsum(weights, axis=1)
        total = cum[:, -1]
        ticks = np.floor(cum / np.where(total > 0, total, 1.0)[:, None] * _TICKS).astype(np.int64)
        ticks[:, -1] = _TICKS
    out = np.full(r, np.nan)
    sample = np.arange(n_samples)
    step = max(1, (1 << 18) // n_samples)
    for lo in range(0, r, step):
        rows = np.arange(lo, min(lo + step, r))
        pos = pool[(start[rows, None] + sample) & (_POOL_SIZE - 1)]           # (Rc, n, width)
        cards = live[rows[:, None, None], pos]
        if villain is not None:
            run_key, run_bits = card_keys(cards)
            hv = value_from_keys(hero_key[rows, None] + run_key, hero_bits[rows, None] + run_bits)
            vv = value_from_keys(vil_key[rows, None] + run_key, vil_bits[rows, None] + run_bits)
            out[rows] = _share(hv, vv).mean(1)
            continue
        if weights is None:
            opp, runout = cards[..., :2], cards[..., 2:]
        else:
            # Draw the opponent's combo from the weights (inverse CDF on
            # hashed 40-bit uniforms), then deal the first j shuffled cards
            # it does not hold.
            u = (_mix64(row_hash[rows, None] ^ (sample + 1).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
                 >> np.uint64(24)).astype(np.int64)
            shift = (np.arange(rows.size, dtype=np.int64) * (_TICKS + 1))[:, None]
            found = np.searchsorted((ticks[rows] + shift).ravel(), (u + shift).ravel(), side="right")
            combo = found.reshape(rows.size, n_samples) - np.arange(rows.size)[:, None] * N_COMBOS
            opp = COMBOS[np.clip(combo, 0, N_COMBOS - 1)].astype(np.intp)
            # The shuffle holds j + 2 cards and the opponent's two cards can
            # sit among them. A runout card the opponent holds is replaced by
            # the first spare (positions j, j + 1) the opponent does not hold,
            # a second one by the next: the same set of cards as the first j
            # cards of the shuffle that the opponent does not hold.
            clash = (cards == opp[..., :1]) | (cards == opp[..., 1:])
            runout = cards[..., :j]
            hit = clash[..., :j]
            spare_0 = np.where(clash[..., j], cards[..., j + 1], cards[..., j])
            first_hit = np.cumsum(hit, axis=-1) == 1
            runout = np.where(hit, np.where(first_hit, spare_0[..., None], cards[..., j + 1:j + 2]), runout)
        run_key, run_bits = card_keys(runout)
        opp_key, opp_bits = card_keys(opp)
        hv = value_from_keys(hero_key[rows, None] + run_key, hero_bits[rows, None] + run_bits)
        vv = value_from_keys(opp_key + board_key[rows, None] + run_key, opp_bits + board_bits[rows, None] + run_bits)
        out[rows] = _share(hv, vv).mean(1)
    if weights is not None:
        out[total <= 0] = np.nan
    return out


# ----------------------------------------------------------------------------
# Public API: one spot
# ----------------------------------------------------------------------------

def equity_vs_hand(hero: CardsLike, villain: CardsLike, board: Optional[CardsLike] = None, *,
                   exact: Optional[bool] = None, n_samples: int = 200_000, seed: int = 0) -> float:
    """Hero's equity against the opponent's known hole cards.

    Every way to finish the board is dealt (exact, the default on every
    street) or ``n_samples`` random ones (``exact=False``). Ties count 1/2.

    Example::

        >>> round(equity_vs_hand("Ah Ad", "Ks Kc"), 4)        # preflop, 1,712,304 boards
        0.8126
        >>> round(equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h 2c Js"), 4)   # 10 of 44 rivers win
        0.2273

    Args:
        hero, villain: two cards each, as text (``"Ah Kd"``) or ids.
        board: 0, 3, 4 or 5 cards already dealt.
        exact: ``None`` or ``True`` enumerates, ``False`` samples.
        n_samples, seed: Monte Carlo settings when ``exact=False``.
    """
    h, v, b = _hand(hero), _hand(villain, "villain"), _board(board)
    live = np.setdiff1d(np.arange(52), _all_distinct(h, v, b))
    j = 5 - len(b)
    hero_key, hero_bits = card_keys(np.array(h + b))
    vil_key, vil_bits = card_keys(np.array(v + b))
    if exact is None or exact:
        template = _combinations(live.size, j)
        batches = (live[template[i:i + (1 << 18)]] for i in range(0, len(template), 1 << 18))
        n = len(template)
    else:
        _check_samples(n_samples)
        rng = np.random.default_rng(seed)
        batches = (_sample_subsets(live, j, min(1 << 18, n_samples - i), rng) for i in range(0, n_samples, 1 << 18))
        n = n_samples
    total = 0.0
    for runout in batches:
        run_key, run_bits = card_keys(runout)
        hv = value_from_keys(hero_key + run_key, hero_bits + run_bits)
        vv = value_from_keys(vil_key + run_key, vil_bits + run_bits)
        total += _share(hv, vv).sum()
    return float(total / n)


def equity_vs_random(hero: CardsLike, board: Optional[CardsLike] = None, *,
                     exact: Optional[bool] = None, n_samples: int = 100_000, seed: int = 0) -> float:
    """Hero's equity against a random hand: every holding of two cards the hero
    cannot see is equally likely.

    Exact by default on every street: the flop, turn and river by enumerating
    every opponent holding and every completion of the board, preflop from
    :data:`PREFLOP_VS_RANDOM`. ``exact=False`` uses Monte Carlo.

    Example::

        >>> round(equity_vs_random("Ah Ad"), 4)
        0.852
        >>> round(equity_vs_random("7h 2d"), 4)
        0.3458
    """
    h, b = _hand(hero), _board(board)
    _all_distinct(h, b)
    if exact is None or exact:
        if not b:
            return PREFLOP_VS_RANDOM[class_index(hand_class(h))]
        num, den = _exact_vs_holdings(np.array([h]), np.array([b], dtype=np.intp), None)
        return float(num[0] / den[0])
    w = np.ones(N_COMBOS)
    blocked = np.zeros(52, dtype=bool)
    blocked[h + b] = True
    w[blocked[COMBOS[:, 0]] | blocked[COMBOS[:, 1]]] = 0.0
    return _mc_vs_holdings(h, b, w, n_samples, seed)


def equity_vs_range(hero: CardsLike, board: Optional[CardsLike], weights: np.ndarray, *,
                    exact: Optional[bool] = None, n_samples: int = 100_000, seed: int = 0) -> float:
    """Hero's equity against a weighted range of opponent holdings.

    ``weights`` holds 1,326 non-negative numbers in :data:`sudetect.cards.COMBOS`
    order; only their ratios matter. Holdings that share a card with the hero
    or the board are ignored (weight 0). Returns ``nan`` when no weight is left.

    The opponent's holding and the rest of the board are drawn together: a
    holding is picked in proportion to its weight, then the board is finished
    from the cards that are left. Exact by default on the flop, turn and
    river; Monte Carlo preflop unless ``exact=True`` (about 15 seconds).

    Example::

        >>> from sudetect.cards import range_weights
        >>> w = range_weights({"QQ": 1, "AK": 1})            # 6 + 16 combos
        >>> round(equity_vs_range("Jh Jd", "Td 7c 2h", w), 4)
        0.5693
    """
    h, b = _hand(hero), _board(board)
    _all_distinct(h, b)
    w = _weights(weights).copy()
    blocked = np.zeros(52, dtype=bool)
    blocked[h + b] = True
    w[blocked[COMBOS[:, 0]] | blocked[COMBOS[:, 1]]] = 0.0
    w = _scaled(w)
    if w.sum() <= 0:
        return float("nan")
    if (len(b) >= 3) if exact is None else exact:
        num, den = _exact_vs_holdings(np.array([h]), np.array(b, dtype=np.intp).reshape(1, len(b)), w[None, :])
        return float(num[0] / den[0])
    return _mc_vs_holdings(h, b, w, n_samples, seed)


def multiway_equity(hands: Sequence[CardsLike], board: Optional[CardsLike] = None, *,
                    exact: Optional[bool] = None, n_samples: int = 200_000, seed: int = 0) -> list[float]:
    """Each player's equity when all hole cards are known (2 to 10 hands).

    A pot won alone counts 1, a pot split k ways 1/k each, so the results sum
    to 1. Exact by default: even the largest case, heads-up preflop, has
    1.7 million boards. ``exact=False`` samples ``n_samples`` boards.

    Example::

        >>> [round(x, 4) for x in multiway_equity(["Ah Ad", "Kh Kd", "Qh Qd"])]
        [0.6767, 0.1723, 0.151]
    """
    hs = [_hand(x, f"hand {i + 1}") for i, x in enumerate(hands)]
    if not 2 <= len(hs) <= 10:
        raise ValueError("multiway_equity needs 2 to 10 hands")
    b = _board(board)
    live = np.setdiff1d(np.arange(52), _all_distinct(*hs, b))
    j = 5 - len(b)
    keys = [card_keys(np.array(x + b)) for x in hs]
    if exact is None or exact:
        template = _combinations(live.size, j)
        batches = (live[template[i:i + (1 << 18)]] for i in range(0, len(template), 1 << 18))
        n = len(template)
    else:
        _check_samples(n_samples)
        rng = np.random.default_rng(seed)
        batches = (_sample_subsets(live, j, min(1 << 18, n_samples - i), rng) for i in range(0, n_samples, 1 << 18))
        n = n_samples
    total = np.zeros(len(hs))
    for runout in batches:
        run_key, run_bits = card_keys(runout)
        values = np.stack([value_from_keys(k + run_key, f + run_bits) for k, f in keys], axis=1)
        best = values == values.max(1, keepdims=True)
        total += (best / best.sum(1, keepdims=True)).sum(0)
    return (total / n).tolist()


# ----------------------------------------------------------------------------
# Public API: batches of action rows
# ----------------------------------------------------------------------------

def _street_blocks(n_board: np.ndarray):
    """Yield ``(board cards, rows)`` per street, at most :data:`_ROW_BLOCK` rows
    at a time, so the working memory stays bounded however many rows come in."""
    for b in (0, 3, 4, 5):
        rows = np.flatnonzero(n_board == b)
        for lo in range(0, rows.size, _ROW_BLOCK):
            yield b, rows[lo:lo + _ROW_BLOCK]


def equity_vs_hand_batch(hero: np.ndarray, villain: np.ndarray, board: np.ndarray, *,
                         exact: Optional[bool] = None, n_samples: int = 4096, seed: int = 0) -> np.ndarray:
    """:func:`equity_vs_hand` for many rows at once.

    Rows are grouped by street and each group is computed in vectorised
    blocks, so a table with millions of rows is processed in one call.

    Args:
        hero, villain: (n, 2) card ids.
        board: (n, 5) card ids with -1 for cards not dealt yet (0, 3, 4 or 5
            dealt per row, from the left).
        exact: flop, turn and river rows are always exact (990, 44 and 1
            boards per row). Preflop rows use Monte Carlo with ``n_samples``
            boards unless ``exact=True`` (1,712,304 boards, about 0.3 s per row).
        n_samples, seed: preflop Monte Carlo settings (``n_samples`` at most
            65,536); see the module docstring for how rows stay reproducible.

    Returns:
        float64 array of n equities.
    """
    hero, board, villain, n_board = _check_rows(hero, board, villain)
    out = np.empty(hero.shape[0])
    for b, rows in _street_blocks(n_board):
        if b == 0 and not exact:
            out[rows] = _mc_rows(hero[rows], board[rows, :0], villain[rows], None, n_samples, seed)
            continue
        step = max(1, (1 << 18) // len(_combinations(48 - b, 5 - b)))
        for lo in range(0, rows.size, step):
            part = rows[lo:lo + step]
            out[part] = _exact_vs_hand_rows(hero[part], villain[part], board[part, :b])
    return out


def _exact_vs_hand_rows(hero: np.ndarray, villain: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Exact equity per row for known hands, rows on the same street."""
    b = board.shape[1]
    known = np.concatenate([hero, villain, board], axis=1).astype(np.intp)
    _, live = _live_cards(known)
    hero_key, hero_bits = card_keys(np.concatenate([hero, board], axis=1).astype(np.intp))
    vil_key, vil_bits = card_keys(np.concatenate([villain, board], axis=1).astype(np.intp))
    template = _combinations(live.shape[1], 5 - b)
    total = np.zeros(hero.shape[0])
    for i in range(0, len(template), 1 << 18):
        runout = np.ascontiguousarray(live[:, template[i:i + (1 << 18)]])  # (R, n_run, j)
        run_key, run_bits = card_keys(runout)
        hv = value_from_keys(hero_key[:, None] + run_key, hero_bits[:, None] + run_bits)
        vv = value_from_keys(vil_key[:, None] + run_key, vil_bits[:, None] + run_bits)
        total += _share(hv, vv).sum(1)
    return total / len(template)


def equity_vs_random_batch(hero: np.ndarray, board: np.ndarray, *, exact: Optional[bool] = None,
                           n_samples: int = 4096, seed: int = 0) -> np.ndarray:
    """:func:`equity_vs_random` for many rows at once.

    Args:
        hero: (n, 2) card ids.
        board: (n, 5) card ids with -1 for cards not dealt yet.
        exact: turn and river rows are always exact and preflop rows are read
            from :data:`PREFLOP_VS_RANDOM`. Flop rows use Monte Carlo with
            ``n_samples`` deals unless ``exact=True`` (1,081 board completions
            x 990 opponent holdings per row, a few milliseconds).
            ``exact=False`` samples the preflop rows too.
        n_samples, seed: Monte Carlo settings (``n_samples`` at most 65,536).

    Returns:
        float64 array of n equities.
    """
    hero, board, _, n_board = _check_rows(hero, board)
    out = np.empty(hero.shape[0])
    for b, rows in _street_blocks(n_board):
        if b == 0 and exact is not False:
            out[rows] = _PREFLOP_TABLE[_class_of_cards(hero[rows])]
        elif (b == 3 and not exact) or b == 0:
            out[rows] = _mc_rows(hero[rows], board[rows, :b], None, None, n_samples, seed)
        else:
            num, den = _exact_vs_holdings(hero[rows], board[rows, :b], None)
            out[rows] = num / den
    return out


def equity_vs_range_batch(hero: np.ndarray, board: np.ndarray, weights: np.ndarray, *,
                          exact: Optional[bool] = None, n_samples: int = 4096, seed: int = 0) -> np.ndarray:
    """:func:`equity_vs_range` for many rows at once, one range per row.

    Args:
        hero: (n, 2) card ids.
        board: (n, 5) card ids with -1 for cards not dealt yet.
        weights: (n, 1326) non-negative weights per row.
        exact: turn and river rows are always exact. Flop and preflop rows use
            Monte Carlo with ``n_samples`` deals unless ``exact=True`` (a few
            milliseconds per flop row, several seconds per preflop row).
        n_samples, seed: Monte Carlo settings (``n_samples`` at most 65,536).

    Returns:
        float64 array of n equities; ``nan`` where a row has no weight left.
    """
    hero, board, _, n_board = _check_rows(hero, board)
    w = _weights(weights, hero.shape[0])
    out = np.empty(hero.shape[0])
    for b, rows in _street_blocks(n_board):
        if b in (0, 3) and not exact:
            out[rows] = _mc_rows(hero[rows], board[rows, :b], None, w[rows], n_samples, seed)
        else:
            num, den = _exact_vs_holdings(hero[rows], board[rows, :b], w[rows])
            with np.errstate(invalid="ignore", divide="ignore"):
                out[rows] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def _class_of_cards(hero: np.ndarray) -> np.ndarray:
    """Preflop class index of each row's two cards."""
    return COMBO_TO_CLASS[COMBO_INDEX[hero[:, 0], hero[:, 1]]]


# ----------------------------------------------------------------------------
# Preflop table
# ----------------------------------------------------------------------------

def preflop_vs_random_exact(label: str) -> float:
    """Exact equity of a starting hand class (``"AKs"``) against a random hand.

    Enumerates all 2,118,760 boards and every opponent holding (about 2.1
    billion showdowns, scored 91 rank pairs at a time); takes about five
    seconds. This is how :data:`PREFLOP_VS_RANDOM` was produced.
    """
    hi, lo = COMBOS[class_combos(label)[0]]
    num, den = _exact_vs_holdings(np.array([[hi, lo]]), np.zeros((1, 0), dtype=np.intp), None)
    return float(num[0] / den[0])


#: Exact equity against a random hand of each starting hand, in
#: :data:`sudetect.cards.CLASS_LABELS` order: AA 0.8520, AKs 0.6704, 72o 0.3458.
#: Produced by :func:`preflop_vs_random_exact`. The values average exactly 0.5
#: when each class is weighted by its number of combos, as they must.
PREFLOP_VS_RANDOM: tuple[float, ...] = (
    # pairs: AA, KK, ..., 22
    0.8520371330210104, 0.823956797867859, 0.7992516406108319, 0.7746947290114992,
    0.7501177995095664, 0.7205725194515336, 0.6916303546900217, 0.6623602279473166,
    0.6328474821655738, 0.6032492051287479, 0.5702282119082039, 0.536930763867793,
    0.5033401907843562,
    # suited: AKs .. A2s
    0.6704463230923519, 0.6620886239731224, 0.6539267903219932, 0.6460238678769801,
    0.6278121391662095, 0.6194381064033833, 0.6098395855132343, 0.5990582804197844,
    0.5992292561629815, 0.5903363597842916, 0.5822032059537016, 0.5737889786307256,
    # suited: KQs .. K2s
    0.6340040329478019, 0.6256734013090561, 0.6178855816848086, 0.5998847551102408,
    0.5831234993366617, 0.5753773750550875, 0.5664073552359862, 0.5579291763659743,
    0.5488463656844456, 0.5405497645754683, 0.5321172832937733,
    # suited: QJs .. Q2s
    0.6025920514114316, 0.5946755930808395, 0.5766432171781055, 0.5601773297551017,
    0.5430226320197576, 0.536125664363242, 0.5276941089137138, 0.5185530203868053,
    0.5101924648703425, 0.5016903526190561,
    # suited: JTs .. J2s
    0.5752785710757826, 0.5566247055405573, 0.5401564417991007, 0.5232478118514526,
    0.5060590714294295, 0.49986849512321957, 0.49070453396507313, 0.4823162401927104,
    0.473781524060862,
    # suited: T9s .. T2s
    0.540275286564602, 0.5233437072303202, 0.5063903753691649, 0.48940675682994306,
    0.47216258971561603, 0.4653049360775342, 0.45692512020085696, 0.4483948272774756,
    # suited: 98s .. 92s
    0.5080075538751369, 0.49117731097148304, 0.4742829077079771, 0.457218745584181,
    0.438619708192194, 0.4326425779153082, 0.4241517186724997,
    # suited: 87s .. 82s
    0.4793634024265384, 0.46243269266891573, 0.4454499270203975, 0.42701627295439243,
    0.40873504104077646, 0.40271634437981735,
    # suited: 76s .. 72s
    0.45371766643191913, 0.43675536634635354, 0.4184931187595718, 0.4003593587520507,
    0.3815589347476159,
    # suited: 65s .. 62s
    0.4313338621827785, 0.41333319031085647, 0.3953355993814564, 0.3766896410822339,
    # suited: 54s .. 52s
    0.4145342036823139, 0.39692962397865267, 0.378493279898229,
    # suited: 43s .. 42s
    0.386419483780393, 0.3682901472197098,
    # suited: 32s
    0.3598443059700824,
    # offsuit: AKo .. A2o
    0.6532007178870203, 0.6443183937298184, 0.6356325791186039, 0.6272165463752288,
    0.6077280638799405, 0.5987260508862531, 0.5884119542190772, 0.5768245229580633,
    0.5769653436038727, 0.5672967762638372, 0.5584460233649146, 0.5492855867573391,
    # offsuit: KQo .. K2o
    0.614558000477123, 0.6056868516195197, 0.597389151382808, 0.5781192465633129,
    0.5602017255757179, 0.5518735017203698, 0.5422327894379235, 0.5331397283354796,
    0.5232747210537286, 0.5142568964484849, 0.5050872377516028,
    # offsuit: QJo .. Q2o
    0.5813468967745762, 0.5729078259706316, 0.5536043492467769, 0.5359979207392317,
    0.5176566594316363, 0.5102405230446396, 0.5012008279189791, 0.4912768410282286,
    0.48219436025187973, 0.47295436882178654,
    # offsuit: JTo .. J2o
    0.5524770308285902, 0.5325119688359744, 0.5149016300939123, 0.4968193360095699,
    0.47844273051075614, 0.47180888225836687, 0.46186384531947505, 0.45275544887032265,
    0.4434846761427639,
    # offsuit: T9o .. T2o
    0.5153167239900753, 0.4972127367331874, 0.47908135518945616, 0.4609200328436816,
    0.44250949550060825, 0.4350410801076521, 0.425945508483998, 0.41668350589471903,
    # offsuit: 98o .. 92o
    0.48097032765114567, 0.4629780640706371, 0.4449134525702188, 0.42669142814808203,
    0.4067105352358755, 0.4001951434429629, 0.39097936285774926,
    # offsuit: 87o .. 82o
    0.45050812262785306, 0.4324090186350659, 0.41427525981939883, 0.39446791467126474,
    0.37483812596885807, 0.3682767410078432,
    # offsuit: 76o .. 72o
    0.42322746881108847, 0.40511968859811465, 0.3854982783907721, 0.36602256994800275,
    0.34583647315344157,
    # offsuit: 65o .. 62o
    0.39944302303939544, 0.3801048824345706, 0.3607763095567047, 0.34075138336107014,
    # offsuit: 54o .. 52o
    0.3815528708329686, 0.36264771123037276, 0.3428464550258194,
    # offsuit: 43o .. 42o
    0.35145893390855065, 0.3319975012543071,
    # offsuit: 32o
    0.32303228126952854,
)
_PREFLOP_TABLE = np.array(PREFLOP_VS_RANDOM)
_PREFLOP_TABLE.setflags(write=False)
