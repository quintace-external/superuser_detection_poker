"""Equity backends: the equity references of step 02 from interchangeable engines.

Step 02 stores two equities per decision row: the actor's equity against one
random hand and against the real hole cards of every live opponent. A
*backend* computes both for whole arrays of rows. Three backends exist:

* :class:`NumpyBackend`, the default. It calls the batch functions of
  :mod:`sudetect.equity` and, against two or more known hands, the exact
  enumeration :func:`multiway_exact_batch` written here. It needs numpy only.
* :class:`PyrustBackend` calls the optional ``pyrust_poker`` engine (MIT, Rust)
  once per row with ``n_threads=1``. Every value it returns is exact.
* :class:`ReferenceBackend` calls the single-spot functions of
  :mod:`sudetect.equity`, one row at a time. It is slow and serves the
  cross-checks (:func:`cross_check`).

What each backend does per street (``mc`` = Monte Carlo with a seed):

=============  ==========================  ========================  ==============
backend        street                      vs one random hand        vs known hands
=============  ==========================  ========================  ==============
numpy          preflop                     exact (table)             mc
numpy          flop                        mc, or exact on request   exact
numpy          turn, river                 exact                     exact
pyrust         preflop                     exact (table)             exact
pyrust         flop, turn, river           exact                     exact
reference      preflop                     exact (table)             mc
reference      flop, turn, river           exact                     exact
=============  ==========================  ========================  ==============

"Table" is :data:`sudetect.equity.PREFLOP_VS_RANDOM`, the exact preflop equity of
the 169 starting hands against a random hand. ``pyrust_poker`` accepts at most
six hands per call; rows with more than five opponents go to the numpy code.

The module also holds the backend-independent part of step 02: card text
parsing, ``ahead_now``, the scoring of a batch of decision rows
(:func:`score_decisions`) or heads-up hands (:func:`score_hu_hands`), and the
process-pool worker (:func:`run_task`). The worker lives at module level so
that pools started with ``spawn`` (Windows, macOS) can import it.

Monte Carlo standard errors
---------------------------
A pot share X takes values in [0, 1], so ``Var X = E[X^2] - p^2 <= p - p^2``.
The stored standard error of a Monte Carlo value is therefore
``sqrt(p (1 - p) / n_samples)``. It is exact when no deal splits the pot and an
upper bound otherwise. Exact values carry a standard error of 0.
"""

from __future__ import annotations

import itertools
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .cards import format_card
from .equity import (
    equity_vs_hand,
    equity_vs_hand_batch,
    equity_vs_random,
    equity_vs_random_batch,
    multiway_equity,
)
from .evaluator import card_keys, eval_hands, value_from_keys

__all__ = [
    "CODE_VERSION",
    "BACKENDS",
    "EquityBatch",
    "Backend",
    "NumpyBackend",
    "PyrustBackend",
    "ReferenceBackend",
    "get_backend",
    "pyrust_available",
    "mc_se",
    "cards_from_text",
    "multiway_exact_batch",
    "ahead_now_batch",
    "DECISION_COLUMNS",
    "HAND_COLUMNS",
    "score_decisions",
    "score_hu_hands",
    "init_worker",
    "run_task",
    "CrossCheck",
    "cross_check",
]

CODE_VERSION = "1"
"""Changes whenever a number this module produces would change; step 02 stores it with every staged part."""

BACKENDS = ("numpy", "pyrust", "reference")
"""Names accepted by :func:`get_backend`."""

_MAX_PYRUST_HANDS = 6          # pyrust_poker refuses more than six entries per call
_BLOCK_EVALS = 1 << 18         # hand evaluations per vectorised block of the multiway enumeration


# ----------------------------------------------------------------------------
# Results and the backend protocol
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EquityBatch:
    """Equities of a batch of rows.

    Attributes:
        value: float64, the hero's pot share per row; ``nan`` where the engine gave no value.
        exact: bool, True where the value was enumerated, False where it was sampled.
        se: float64 standard error per row: 0 where exact, ``sqrt(p (1 - p) / n)`` where sampled.
    """

    value: np.ndarray
    exact: np.ndarray
    se: np.ndarray


@runtime_checkable
class Backend(Protocol):
    """What step 02 needs from an equity engine.

    ``hero`` is an ``(n, 2)`` integer array of card ids, ``board`` an ``(n, 5)``
    array with -1 for cards not dealt yet (0, 3, 4 or 5 dealt from the left)
    and ``opponents`` an ``(n, k, 2)`` array, the same k >= 1 for every row of
    one call. Cards within a row are distinct. Ties count 1/k of the pot.
    """

    name: str

    def vs_random(self, hero: np.ndarray, board: np.ndarray) -> EquityBatch:
        """Equity against ONE random hand (any two cards the hero cannot see)."""
        ...

    def vs_hands(self, hero: np.ndarray, opponents: np.ndarray, board: np.ndarray) -> EquityBatch:
        """Equity against the known hole cards of every opponent, all at once."""
        ...


def mc_se(p: np.ndarray, n_samples: int) -> np.ndarray:
    """Standard error of a Monte Carlo pot share: ``sqrt(p (1 - p) / n_samples)`` (an upper bound with ties)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return np.sqrt(p * (1.0 - p) / n_samples)


def _result(value: np.ndarray, exact: np.ndarray, n_samples: int) -> EquityBatch:
    value = np.asarray(value, dtype=np.float64)
    exact = np.asarray(exact, dtype=bool)
    se = np.where(exact, 0.0, mc_se(value, n_samples))
    se[np.isnan(value)] = np.nan
    return EquityBatch(value, exact, se)


def _n_board(board: np.ndarray) -> np.ndarray:
    return (np.asarray(board) >= 0).sum(1)


def _row_seed(seed: int, cards: np.ndarray) -> int:
    """A seed for one row that depends only on the base seed and the row's cards."""
    data = np.ascontiguousarray(cards, dtype=np.int16).tobytes()
    return (int(seed) & 0x7FFFFFFF) << 32 | zlib.crc32(data)


# ----------------------------------------------------------------------------
# Card text
# ----------------------------------------------------------------------------

_RANK_CODE = np.full(128, -1, dtype=np.int16)
_SUIT_CODE = np.full(128, -1, dtype=np.int16)
for _i, _ch in enumerate("23456789TJQKA"):
    _RANK_CODE[ord(_ch)] = _i
for _i, _ch in enumerate("cdhs"):
    _SUIT_CODE[ord(_ch)] = _i
_CARD_TEXT = [format_card(c) for c in range(52)]


def cards_from_text(texts: Sequence[Any], n_cards: int) -> np.ndarray:
    """Concatenated card text per row -> an ``(n, n_cards)`` int16 array, -1 padded on the right.

    ``"AhKd"`` gives ``[50, 45]``; ``""``, None and NaN give a row of -1. Rank
    letters are upper case and suits lower case, the way step 01 writes them.

    Raises:
        ValueError: a row is longer than ``n_cards`` cards or holds a token that is not a card.
    """
    n = len(texts)
    if n_cards <= 0 or n == 0:
        return np.full((n, max(n_cards, 0)), -1, dtype=np.int16)
    clean = [t if isinstance(t, str) else "" for t in texts]
    arr = np.array(clean, dtype=str)
    width = 2 * n_cards
    if arr.dtype.itemsize // 4 > width:
        longest = max(clean, key=len)
        raise ValueError(f"more than {n_cards} cards in {longest!r}")
    codes = arr.astype(f"<U{width}").view(np.uint32).reshape(n, width)
    if ((codes != 0).sum(1) % 2).any():
        bad = int(np.flatnonzero((codes != 0).sum(1) % 2)[0])
        raise ValueError(f"not a card list: {clean[bad]!r}")
    codes = np.minimum(codes, 127)
    present = codes[:, 0::2] != 0
    ranks = _RANK_CODE[codes[:, 0::2]]
    suits = _SUIT_CODE[codes[:, 1::2]]
    bad_rows = (present & ((ranks < 0) | (suits < 0))).any(1)
    if bad_rows.any():
        raise ValueError(f"not a card list: {clean[int(np.flatnonzero(bad_rows)[0])]!r}")
    return np.where(present, ranks * 4 + suits, -1).astype(np.int16)


def _card_text_rows(cards: np.ndarray) -> list[str]:
    """Inverse of :func:`cards_from_text` for one array of rows."""
    return ["".join(_CARD_TEXT[c] for c in row if c >= 0) for row in np.asarray(cards).tolist()]


def _distinct_rows(*groups: np.ndarray) -> np.ndarray:
    """True for rows whose dealt cards (ids >= 0) are all different."""
    n = groups[0].shape[0]
    cards = np.concatenate([np.asarray(g).reshape(n, -1) for g in groups], axis=1).astype(np.int64)
    bits = np.where(cards >= 0, np.int64(1) << np.clip(cards, 0, 62), np.int64(0))
    return np.bitwise_or.reduce(bits, axis=1) == bits.sum(1)


def _valid_board(board: np.ndarray) -> np.ndarray:
    dealt = np.asarray(board) >= 0
    return np.isin(dealt.sum(1), (0, 3, 4, 5)) & ~(dealt[:, 1:] & ~dealt[:, :-1]).any(1)


# ----------------------------------------------------------------------------
# Exact multiway equity and the made-hand comparison
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _combinations(n: int, k: int) -> np.ndarray:
    """All k-subsets of ``range(n)`` as rows of positions; one empty row when k is 0."""
    out = np.array(list(itertools.combinations(range(n), k)), dtype=np.intp).reshape(-1, k) if k else \
        np.zeros((1, 0), dtype=np.intp)
    out.setflags(write=False)
    return out


def _live_cards(known: np.ndarray) -> np.ndarray:
    """``(r, m)`` distinct card ids per row -> the ``(r, 52 - m)`` cards not among them, ascending."""
    r = known.shape[0]
    dead = np.zeros((r, 52), dtype=bool)
    dead[np.arange(r)[:, None], known] = True
    return np.nonzero(~dead)[1].reshape(r, 52 - known.shape[1])


def multiway_exact_batch(hands: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Exact pot share of player 0 when every player's hole cards are known, flop to river.

    Every completion of the board is dealt: ``C(52 - 2m - b, 5 - b)`` boards
    for m players and b board cards (903 for three players on the flop). The
    best hand wins and a pot split k ways counts 1/k for each.

    Args:
        hands: ``(n, m, 2)`` card ids, m >= 2 players, player 0 is the hero.
        board: ``(n, 5)`` card ids, 3, 4 or 5 dealt from the left, -1 padded.

    Returns:
        float64 array of n shares.
    """
    hands = np.asarray(hands)
    board = np.asarray(board)
    n, m, _ = hands.shape
    out = np.empty(n)
    n_board = _n_board(board)
    if not np.isin(n_board, (3, 4, 5)).all():
        raise ValueError("multiway_exact_batch needs 3, 4 or 5 board cards per row")
    for b in (3, 4, 5):
        rows = np.flatnonzero(n_board == b)
        if rows.size == 0:
            continue
        j = 5 - b
        template = _combinations(52 - 2 * m - b, j)
        per_block = max(1, _BLOCK_EVALS // (len(template) * m))
        for lo in range(0, rows.size, per_block):
            r = rows[lo:lo + per_block]
            h = hands[r].astype(np.intp)
            bd = board[r, :b].astype(np.intp)
            keys = [card_keys(np.concatenate([h[:, p], bd], axis=1)) for p in range(m)]
            if j == 0:
                values = np.stack([value_from_keys(k, f) for k, f in keys], axis=1)            # (R, m)
                best = values == values.max(1, keepdims=True)
                out[r] = best[:, 0] / best.sum(1)
                continue
            live = _live_cards(np.concatenate([h.reshape(len(r), -1), bd], axis=1))
            run_key, run_bits = card_keys(live[:, template])                                 # (R, T)
            values = np.stack([value_from_keys(k[:, None] + run_key, f[:, None] + run_bits) for k, f in keys],
                              axis=2)                                                         # (R, T, m)
            best = values == values.max(2, keepdims=True)
            out[r] = (best[:, :, 0] / best.sum(2)).mean(1)
    return out


def ahead_now_batch(hero: np.ndarray, opponents: np.ndarray, board: np.ndarray) -> np.ndarray:
    """True where the hero's best hand on the current board beats every opponent's best hand.

    The hands are compared as they stand (hole cards plus the 3, 4 or 5 board
    cards dealt), with no cards to come. A tie with the best opponent is not
    ahead. On the river this is the same as ``equity == 1``.

    Args:
        hero: ``(n, 2)``; opponents: ``(n, k, 2)``; board: ``(n, 5)`` with 3 to 5 dealt cards.
    """
    hero = np.asarray(hero, dtype=np.intp)
    opponents = np.asarray(opponents, dtype=np.intp)
    board = np.asarray(board, dtype=np.intp)
    n_board = _n_board(board)
    if not np.isin(n_board, (3, 4, 5)).all():
        raise ValueError("ahead_now_batch needs 3, 4 or 5 board cards per row")
    out = np.zeros(hero.shape[0], dtype=bool)
    for b in (3, 4, 5):
        rows = np.flatnonzero(n_board == b)
        if rows.size == 0:
            continue
        bd = board[rows, :b]
        mine = eval_hands(np.concatenate([hero[rows], bd], axis=1), check=False)
        theirs = np.max([eval_hands(np.concatenate([opponents[rows, p], bd], axis=1), check=False)
                         for p in range(opponents.shape[1])], axis=0)
        out[rows] = mine > theirs
    return out


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

@dataclass
class NumpyBackend:
    """Batch equity with numpy (the default backend).

    Args:
        seed: base seed of every Monte Carlo estimate (a row's samples depend
            only on its own cards and this seed).
        mc_samples: deals per Monte Carlo row (at most 65,536).
        flop_random_exact: enumerate the flop against a random hand instead of
            sampling it (1,081 boards x 990 holdings per row, about 2.7 times the time).
    """

    seed: int = 0
    mc_samples: int = 4096
    flop_random_exact: bool = False
    name: str = field(default="numpy", init=False)

    def vs_random(self, hero: np.ndarray, board: np.ndarray) -> EquityBatch:
        hero, board = np.asarray(hero), np.asarray(board)
        value = equity_vs_random_batch(hero, board, exact=True if self.flop_random_exact else None,
                                       n_samples=self.mc_samples, seed=self.seed)
        exact = (_n_board(board) != 3) | self.flop_random_exact
        return _result(value, exact, self.mc_samples)

    def vs_hands(self, hero: np.ndarray, opponents: np.ndarray, board: np.ndarray) -> EquityBatch:
        hero, opponents, board = np.asarray(hero), np.asarray(opponents), np.asarray(board)
        n_board = _n_board(board)
        if opponents.shape[1] == 1:
            value = equity_vs_hand_batch(hero, opponents[:, 0], board, n_samples=self.mc_samples, seed=self.seed)
            return _result(value, n_board > 0, self.mc_samples)
        hands = np.concatenate([hero[:, None, :], opponents], axis=1)
        value = np.empty(hero.shape[0])
        post = np.flatnonzero(n_board > 0)
        if post.size:
            value[post] = multiway_exact_batch(hands[post], board[post])
        for i in np.flatnonzero(n_board == 0):
            value[i] = multiway_equity([h.tolist() for h in hands[i]], None, exact=False,
                                       n_samples=self.mc_samples, seed=_row_seed(self.seed, hands[i]))[0]
        return _result(value, n_board > 0, self.mc_samples)


class PyrustBackend:
    """Exact equity from the optional ``pyrust_poker`` engine, one call per row.

    Routing (every call single-threaded, every value exact): a random
    opponent and any number of concrete hands up to five opponents go to
    ``exact_equity``; preflop against a random hand reads the exact table of
    :mod:`sudetect.equity` (the engine would take seconds per row); rows with
    more than five opponents go to :class:`NumpyBackend`. A Rust panic arrives
    as a ``BaseException``; it is caught, counted in :attr:`failures`, and the
    row gets ``nan`` (which step 02's checks then report).

    The arguments mirror :class:`NumpyBackend`; ``seed`` and ``mc_samples``
    matter only for the preflop rows with more than five opponents that the
    numpy code takes over, and ``flop_random_exact`` is ignored (every value
    of this backend is exact).
    """

    name = "pyrust"

    def __init__(self, seed: int = 0, mc_samples: int = 4096, flop_random_exact: bool = False) -> None:
        try:
            import pyrust_poker
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ImportError("the pyrust backend needs the optional package pyrust_poker "
                              "(see third_party/pyrust_poker)") from exc
        self._engine = pyrust_poker
        self._fallback = NumpyBackend(seed=seed, mc_samples=mc_samples, flop_random_exact=True)
        self.failures = 0

    def _exact(self, entries: list[str], board: str) -> float:
        try:
            value = self._engine.exact_equity(entries, board, 1)[0]
        except BaseException as exc:  # a Rust panic is a BaseException
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            self.failures += 1
            return math.nan
        return float(value) if value == value else math.nan

    def vs_random(self, hero: np.ndarray, board: np.ndarray) -> EquityBatch:
        hero, board = np.asarray(hero), np.asarray(board)
        value = np.empty(hero.shape[0])
        pre = _n_board(board) == 0
        if pre.any():
            value[pre] = equity_vs_random_batch(hero[pre], board[pre])
        rows = np.flatnonzero(~pre)
        h_text, b_text = _card_text_rows(hero[rows]), _card_text_rows(board[rows])
        for i, h, b in zip(rows, h_text, b_text):
            value[i] = self._exact([h, "random"], b)
        return _result(value, np.ones(hero.shape[0], dtype=bool), 1)

    def vs_hands(self, hero: np.ndarray, opponents: np.ndarray, board: np.ndarray) -> EquityBatch:
        hero, opponents, board = np.asarray(hero), np.asarray(opponents), np.asarray(board)
        if opponents.shape[1] + 1 > _MAX_PYRUST_HANDS:
            return self._fallback.vs_hands(hero, opponents, board)
        h_text, b_text = _card_text_rows(hero), _card_text_rows(board)
        o_text = [_card_text_rows(opponents[:, p]) for p in range(opponents.shape[1])]
        value = np.array([self._exact([h_text[i]] + [o[i] for o in o_text], b_text[i])
                          for i in range(hero.shape[0])], dtype=np.float64)
        return _result(value, np.ones(hero.shape[0], dtype=bool), 1)


@dataclass
class ReferenceBackend:
    """Single-spot calls of :mod:`sudetect.equity`, one row at a time (for cross-checks).

    Exact on the flop, turn and river; preflop against known hands uses Monte
    Carlo with ``mc_samples`` deals and a per-row seed.
    """

    seed: int = 0
    mc_samples: int = 4096
    name: str = field(default="reference", init=False)

    def vs_random(self, hero: np.ndarray, board: np.ndarray) -> EquityBatch:
        value = np.array([equity_vs_random(h.tolist(), b[b >= 0].tolist())
                          for h, b in zip(np.asarray(hero), np.asarray(board))], dtype=np.float64)
        return _result(value, np.ones(len(value), dtype=bool), self.mc_samples)

    def vs_hands(self, hero: np.ndarray, opponents: np.ndarray, board: np.ndarray) -> EquityBatch:
        hero, opponents, board = np.asarray(hero), np.asarray(opponents), np.asarray(board)
        value = np.empty(hero.shape[0])
        exact = _n_board(board) > 0
        for i in range(hero.shape[0]):
            b = board[i][board[i] >= 0].tolist()
            hands = [hero[i].tolist()] + [o.tolist() for o in opponents[i]]
            mc = {} if exact[i] else {"exact": False, "n_samples": self.mc_samples,
                                      "seed": _row_seed(self.seed, np.asarray(hands))}
            if len(hands) == 2:
                value[i] = equity_vs_hand(hands[0], hands[1], b, **mc)
            else:
                value[i] = multiway_equity(hands, b, **mc)[0]
        return _result(value, exact, self.mc_samples)


def pyrust_available() -> bool:
    """True when the optional ``pyrust_poker`` package imports."""
    try:
        import pyrust_poker  # noqa: F401
    except ImportError:
        return False
    return True


def get_backend(name: str, *, seed: int = 0, mc_samples: int = 4096, flop_random_exact: bool = False) -> Backend:
    """A backend by name: ``numpy``, ``pyrust`` or ``reference`` (see the module docstring)."""
    if name == "numpy":
        return NumpyBackend(seed=seed, mc_samples=mc_samples, flop_random_exact=flop_random_exact)
    if name == "pyrust":
        return PyrustBackend(seed=seed, mc_samples=mc_samples, flop_random_exact=flop_random_exact)
    if name == "reference":
        return ReferenceBackend(seed=seed, mc_samples=mc_samples)
    raise ValueError(f"unknown equity backend {name!r}; choose one of {', '.join(BACKENDS)}")


# ----------------------------------------------------------------------------
# Scoring batches of rows (step 02)
# ----------------------------------------------------------------------------

DECISION_COLUMNS = ("hole_cards", "board", "n_live_opp", "n_opp_known", "opp_cards")
"""Input columns of :func:`score_decisions`, as ``02_equity/sql/inputs_decisions.sql`` names them."""

HAND_COLUMNS = ("cards_a", "cards_b")
"""Input columns of :func:`score_hu_hands`, as ``02_equity/sql/inputs_hands.sql`` names them."""


def score_decisions(cols: Mapping[str, Sequence[Any]], backend: Backend) -> dict[str, np.ndarray]:
    """Equity references for a batch of decision rows.

    Args:
        cols: per row ``hole_cards`` (the actor's two cards, e.g. ``"AhKd"``),
            ``board`` (the board at the decision, ``""`` preflop),
            ``n_live_opp`` (live opponents), ``n_opp_known`` (how many of them
            have known cards) and ``opp_cards`` (their cards concatenated,
            4 characters per opponent; read only when all are known).
        backend: the engine.

    Returns:
        Arrays of the batch's length: ``n_live_opp``; ``eq_vs_random``,
        ``eq_random_exact``, ``eq_se``; ``eq_real_known``, ``eq_vs_real``,
        ``eq_real_exact``, ``eq_real_se`` (``nan`` unless known); ``ahead_now``
        as int8 (1 ahead, 0 not ahead, -1 unknown or preflop); ``valid`` (False
        for a row whose own cards or board are unusable; it gets no equity).
    """
    hero = cards_from_text(cols["hole_cards"], 2)
    board = cards_from_text(cols["board"], 5)
    n = hero.shape[0]
    k = np.asarray(cols["n_live_opp"], dtype=np.int64).reshape(n)
    n_known = np.asarray(cols["n_opp_known"], dtype=np.int64).reshape(n)
    n_board = _n_board(board)
    valid = (hero >= 0).all(1) & _valid_board(board) & _distinct_rows(hero, board)
    known = valid & (k >= 1) & (n_known == k)
    k_max = int(k[known].max()) if known.any() else 0
    opp_text = [t if ok else "" for t, ok in zip(cols["opp_cards"], known.tolist())]
    opponents = cards_from_text(opp_text, 2 * k_max).reshape(n, k_max, 2)
    if k_max:
        known &= ((opponents >= 0).sum((1, 2)) == 2 * k) & _distinct_rows(hero, board, opponents)

    eq_random = np.full(n, np.nan)
    random_exact = np.zeros(n, dtype=bool)
    random_se = np.full(n, np.nan)
    rows = np.flatnonzero(valid)
    if rows.size:
        res = backend.vs_random(hero[rows], board[rows])
        eq_random[rows], random_exact[rows], random_se[rows] = res.value, res.exact, res.se

    eq_real = np.full(n, np.nan)
    real_exact = np.zeros(n, dtype=bool)
    real_se = np.full(n, np.nan)
    ahead = np.full(n, -1, dtype=np.int8)
    for kk in np.unique(k[known]).tolist():
        rows = np.flatnonzero(known & (k == kk))
        res = backend.vs_hands(hero[rows], opponents[rows, :kk], board[rows])
        eq_real[rows], real_exact[rows], real_se[rows] = res.value, res.exact, res.se
        post = rows[n_board[rows] >= 3]
        if post.size:
            ahead[post] = ahead_now_batch(hero[post], opponents[post, :kk], board[post])
    return {
        "n_live_opp": k.astype(np.int16),
        "eq_vs_random": eq_random,
        "eq_random_exact": random_exact,
        "eq_se": random_se,
        "eq_real_known": known,
        "eq_vs_real": eq_real,
        "eq_real_exact": real_exact,
        "eq_real_se": real_se,
        "ahead_now": ahead,
        "valid": valid,
    }


def score_hu_hands(cols: Mapping[str, Sequence[Any]], backend: Backend) -> dict[str, np.ndarray]:
    """Preflop equity of player A against player B's real hand, for heads-up hands.

    Player B's equity is ``1 - eq_a`` (the same deals, ties split), so the two
    rows of a hand sum to exactly 1 and share one standard error.

    Returns:
        ``eq_a``, ``exact``, ``se``; ``nan`` where the cards are unusable.
    """
    a = cards_from_text(cols["cards_a"], 2)
    b = cards_from_text(cols["cards_b"], 2)
    n = a.shape[0]
    ok = (a >= 0).all(1) & (b >= 0).all(1) & _distinct_rows(a, b)
    eq_a = np.full(n, np.nan)
    exact = np.zeros(n, dtype=bool)
    se = np.full(n, np.nan)
    rows = np.flatnonzero(ok)
    if rows.size:
        res = backend.vs_hands(a[rows], b[rows][:, None, :], np.full((rows.size, 5), -1, dtype=np.int16))
        eq_a[rows], exact[rows], se[rows] = res.value, res.exact, res.se
    return {"eq_a": eq_a, "exact": exact, "se": se}


# ----------------------------------------------------------------------------
# Process-pool worker (module level: importable by spawned processes)
# ----------------------------------------------------------------------------

_WORKER_BACKENDS: dict[tuple, Backend] = {}


def _backend_for(name: str, settings: Mapping[str, Any]) -> Backend:
    key = (name, tuple(sorted(settings.items())))
    if key not in _WORKER_BACKENDS:
        _WORKER_BACKENDS[key] = get_backend(name, **settings)
    return _WORKER_BACKENDS[key]


def init_worker() -> None:
    """Pool initializer: build the lookup tables and warm the optional engine once per process."""
    eval_hands(np.array([[0, 5, 10, 15, 20]]))
    if pyrust_available():
        import pyrust_poker
        try:
            pyrust_poker.exact_equity(["AhKd", "QsQc"], "2c7d9s", 1)
        except BaseException as exc:  # pragma: no cover - a broken engine shows up in the run instead
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise


def run_task(task: tuple[str, str, Mapping[str, Any], Mapping[str, Sequence[Any]]]) -> dict[str, np.ndarray]:
    """Score one chunk: ``(kind, backend name, backend settings, columns)``, kind ``decisions`` or ``hands``."""
    kind, name, settings, cols = task
    backend = _backend_for(name, settings)
    if kind == "decisions":
        return score_decisions(cols, backend)
    if kind == "hands":
        return score_hu_hands(cols, backend)
    raise ValueError(f"unknown task kind {kind!r}")


# ----------------------------------------------------------------------------
# Cross-check of two backends
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossCheck:
    """Outcome of :func:`cross_check`.

    Attributes:
        n_spots: random spots dealt.
        n_values: equities compared (two per spot: vs a random hand and vs the known hands).
        max_abs_diff: largest absolute difference over all compared values.
        max_z: largest ``|difference| / sqrt(se_a^2 + se_b^2)`` among pairs with a Monte Carlo side (0 if none).
        by_kind: largest absolute difference per ``"<street> vs random"`` / ``"<street> vs <k> hands"``.
        failures: one line per value outside the tolerance.
    """

    n_spots: int
    n_values: int
    max_abs_diff: float
    max_z: float
    by_kind: dict[str, float]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def cross_check(a: Backend, b: Backend, n_spots: int = 200, *, seed: int = 0,
                streets: Sequence[str] = ("flop", "turn", "river"), max_opponents: int = 3,
                exact_tol: float = 1e-9, z_tol: float = 4.5) -> CrossCheck:
    """Compare two backends on random spots.

    Spot i is dealt on ``streets[i % len(streets)]`` with
    ``1 + (i // len(streets)) % max_opponents`` opponents. Both equities of
    every spot are compared. Two exact values must agree within
    ``exact_tol``; when either side is Monte Carlo the difference must stay
    within ``z_tol`` combined standard errors.
    """
    board_cards = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for i in range(n_spots):
        street = streets[i % len(streets)]
        k = 1 + (i // len(streets)) % max_opponents
        deck = rng.permutation(52)
        board = np.full(5, -1)
        nb = board_cards[street]
        board[:nb] = deck[2 + 2 * k:2 + 2 * k + nb]
        groups.setdefault((street, k), []).append((deck[:2], deck[2:2 + 2 * k].reshape(k, 2), board))

    by_kind: dict[str, float] = {}
    failures: list[str] = []
    max_diff, max_z, n_values = 0.0, 0.0, 0
    for (street, k), spots in groups.items():
        hero = np.array([s[0] for s in spots])
        opps = np.array([s[1] for s in spots])
        board = np.array([s[2] for s in spots])
        pairs = {f"{street} vs random": (a.vs_random(hero, board), b.vs_random(hero, board)),
                 f"{street} vs {k} hand{'s' if k > 1 else ''}": (a.vs_hands(hero, opps, board),
                                                                    b.vs_hands(hero, opps, board))}
        for kind, (ra, rb) in pairs.items():
            diff = np.abs(ra.value - rb.value)
            both_exact = ra.exact & rb.exact
            se = np.sqrt(np.nan_to_num(ra.se) ** 2 + np.nan_to_num(rb.se) ** 2)
            z = np.where(both_exact, 0.0, diff / np.maximum(se, 1e-12))
            bad = np.isnan(diff) | np.where(both_exact, diff > exact_tol, z > z_tol)
            for i in np.flatnonzero(bad):
                failures.append(f"{kind}: hero {_card_text_rows(hero[i:i + 1])[0]} board "
                                f"{_card_text_rows(board[i:i + 1])[0] or '-'}: {ra.value[i]!r} vs {rb.value[i]!r}")
            finite = diff[~np.isnan(diff)]
            worst = float(finite.max()) if finite.size else math.nan
            by_kind[kind] = max(by_kind.get(kind, 0.0), worst)
            max_diff = max(max_diff, worst)
            max_z = max(max_z, float(np.nan_to_num(z).max(initial=0.0)))
            n_values += diff.size
    return CrossCheck(n_spots=n_spots, n_values=n_values, max_abs_diff=max_diff, max_z=max_z,
                      by_kind=dict(sorted(by_kind.items())), failures=failures)
