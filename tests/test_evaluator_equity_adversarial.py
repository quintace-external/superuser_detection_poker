"""Adversarial tests for sudetect.cards, sudetect.evaluator and sudetect.equity.

Three references that do not share the library's shortcuts:

* ``_bf_classify`` / ``_bf_best``: a brute-force evaluator written here from
  scratch. It classifies every five-card subset of a hand by counting ranks
  and checking suits and runs directly, and keeps the best subset.
* ``_bf_equity``: equity found by listing every (board completion, opponent
  holding) pair and scoring each pair with an evaluator. With ``_bf_best`` it
  shares no code with the library; with ``eval_hands`` (checked here against
  ``_bf_best``) it still bypasses every trick of the equity module: rank-pair
  tables, flush corrections, inclusion-exclusion and Monte Carlo pools.
* ``pyrust_poker.exact_equity``, an independent enumerator.

The spots are chosen to be hard: wheels and steel wheels inside seven cards,
six and seven suited cards, two trips, three pairs, quads with a pair or
trips, boards that play for everyone, monotone, four- and five-flush boards,
paired, trips and quads boards, ranges with blocked and zero-weight combos,
mixed streets in one batch, unsigned and narrow integer dtypes, extreme
weights, and seeds given as numpy integers.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from sudetect.cards import (
    CLASS_LABELS,
    COMBO_INDEX,
    COMBO_TO_CLASS,
    COMBOS,
    combo_index,
    combos_blocked,
    format_cards,
    parse_cards,
    range_weights,
)
from sudetect.equity import (
    PREFLOP_VS_RANDOM,
    equity_vs_hand,
    equity_vs_hand_batch,
    equity_vs_random,
    equity_vs_random_batch,
    equity_vs_range,
    equity_vs_range_batch,
    multiway_equity,
)
from sudetect.evaluator import (
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
    category,
    decode,
    describe_hand,
    describe_value,
    eval_hand,
    eval_hands,
    value_from_keys,
)

pyrust_poker = pytest.importorskip("pyrust_poker")

EXACT = 1e-9                      # two exact enumerations may differ by rounding only
P = parse_cards


# ---------------------------------------------------------------------------
# Brute-force evaluator, written from scratch
# ---------------------------------------------------------------------------

def _bf_classify(cards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Category, ranks ordered by (count, rank) descending, and straight top of
    each row of five card ids."""
    c = np.asarray(cards, dtype=np.int64)
    rank, suit = c >> 2, c & 3
    count = (rank[:, :, None] == rank[:, None, :]).sum(2)
    order = np.argsort(-(count * 16 + rank), axis=1, kind="stable")
    g = np.take_along_axis(rank, order, 1)
    gc = np.take_along_axis(count, order, 1)
    flush = (suit == suit[:, :1]).all(1)
    distinct = gc[:, 0] == 1
    wheel = distinct & (np.sort(rank, 1) == [0, 1, 2, 3, 12]).all(1)
    straight = distinct & ((rank.max(1) - rank.min(1) == 4) | wheel)
    top = np.where(wheel, 3, rank.max(1))
    cat = np.select(
        [straight & flush, gc[:, 0] == 4, (gc[:, 0] == 3) & (gc[:, 3] == 2), flush, straight,
         gc[:, 0] == 3, (gc[:, 0] == 2) & (gc[:, 2] == 2), gc[:, 0] == 2],
        [8, 7, 6, 5, 4, 3, 2, 1], 0)
    return cat, g, top


def _bf_five(cards: np.ndarray) -> np.ndarray:
    """Strength of five cards in the layout the evaluator documents."""
    cat, g, top = _bf_classify(cards)
    # positions in g of the ranks that break ties, per category
    fields = {0: (0, 1, 2, 3, 4), 1: (0, 2, 3, 4), 2: (0, 2, 4), 3: (0, 3, 4), 5: (0, 1, 2, 3, 4),
              6: (0, 3), 7: (0, 4)}
    out = np.zeros(len(cat), dtype=np.int64)
    for c, pos in fields.items():
        sel = cat == c
        value = np.full(sel.sum(), c << 20, dtype=np.int64)
        for i, p in enumerate(pos):
            value |= g[sel, p] << (16 - 4 * i)
        out[sel] = value
    for c in (4, 8):
        sel = cat == c
        out[sel] = (c << 20) | (top[sel] << 16)
    return out


def _bf_best(cards: np.ndarray) -> np.ndarray:
    """Best five of the 5 to 7 cards of each row, trying every subset."""
    c = np.asarray(cards, dtype=np.int64)
    subsets = np.array(list(itertools.combinations(range(c.shape[1]), 5)))
    return _bf_five(c[:, subsets].reshape(-1, 5)).reshape(len(c), len(subsets)).max(1)


_NAME = ("Deuce", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace")
_PLURAL = ("Deuces", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights", "Nines", "Tens", "Jacks",
           "Queens", "Kings", "Aces")


def _bf_text(cat: int, g: list[int], top: int) -> str:
    """Hand-history wording of a five-card hand from its brute-force classification."""
    low = _NAME[top - 4] if top >= 4 else "Ace"
    return {
        0: f"high card {_NAME[g[0]]}",
        1: f"a pair of {_PLURAL[g[0]]}",
        2: f"two pair, {_PLURAL[g[0]]} and {_PLURAL[g[2]]}",
        3: f"three of a kind, {_PLURAL[g[0]]}",
        4: f"a straight, {low} to {_NAME[top]}",
        5: f"a flush, {_NAME[g[0]]} high",
        6: f"a full house, {_PLURAL[g[0]]} full of {_PLURAL[g[3]]}",
        7: f"four of a kind, {_PLURAL[g[0]]}",
        8: "a Royal Flush" if top == 12 else f"a straight flush, {low} to {_NAME[top]}",
    }[cat]


def _restricted_hands(n: int, k: int, n_ranks: int, n_suits: int, seed: int, run: bool = False) -> np.ndarray:
    """n hands of k cards, each dealt from its own small deck of n_ranks ranks
    (a run of ranks, ace low too, when ``run``) times n_suits suits. Small decks
    produce the rare hands: quads, full houses, flushes of 6-7 cards, wheels."""
    rng = np.random.default_rng(seed)
    if run:
        start = rng.integers(-1, 13 - n_ranks + 1, n)
        ranks = (start[:, None] + np.arange(n_ranks)) % 13
    else:
        ranks = np.argsort(rng.random((n, 13)), 1)[:, :n_ranks]
    suits = np.argsort(rng.random((n, 4)), 1)[:, :n_suits]
    deck = (ranks[:, :, None] * 4 + suits[:, None, :]).reshape(n, -1)
    return np.take_along_axis(deck, np.argsort(rng.random(deck.shape), 1)[:, :k], 1)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_five() -> np.ndarray:
    flat = itertools.chain.from_iterable(itertools.combinations(range(52), 5))
    return np.fromiter(flat, dtype=np.int8, count=5 * 2_598_960).reshape(-1, 5)


def test_brute_force_layout_on_all_five_card_hands(all_five):
    values = eval_hands(all_five)
    for lo in range(0, len(all_five), 400_000):
        np.testing.assert_array_equal(values[lo:lo + 400_000], _bf_five(all_five[lo:lo + 400_000]))


def test_describe_value_on_every_distinct_strength(all_five):
    values, first = np.unique(eval_hands(all_five), return_index=True)
    assert len(values) == 7462
    cat, g, top = _bf_classify(all_five[first])
    for v, c, gg, t in zip(values.tolist(), cat.tolist(), g.tolist(), top.tolist()):
        assert describe_value(v) == _bf_text(c, gg, t), v


@pytest.mark.parametrize("k", [5, 6, 7])
@pytest.mark.parametrize("n_ranks, n_suits, run", [
    (2, 4, False), (3, 4, False), (4, 4, False), (3, 3, False),       # quads, full houses, two trips, three pairs
    (5, 4, True), (6, 4, True), (7, 2, True), (8, 2, True),           # straights with pairs, wheels
    (7, 1, True), (9, 1, True), (13, 1, False), (10, 1, False),       # straight flushes, 6-7 suited cards
    (6, 2, False), (4, 2, False), (7, 4, False),
])
def test_structured_hands_match_brute_force(k, n_ranks, n_suits, run):
    if n_ranks * n_suits < k:
        pytest.skip("deck smaller than the hand")
    hands = _restricted_hands(5000, k, n_ranks, n_suits, seed=100 * n_ranks + 10 * n_suits + k, run=run)
    np.testing.assert_array_equal(eval_hands(hands), _bf_best(hands))
    np.testing.assert_array_equal(eval_hands(hands, check=False), _bf_best(hands))


def test_structured_decks_reach_every_category():
    hands = np.concatenate([_restricted_hands(4000, 7, r, s, seed=r * 7 + s, run=run)
                            for r, s, run in [(2, 4, False), (3, 4, False), (9, 1, True), (13, 1, False),
                                              (6, 4, True), (7, 4, False)]])
    counts = np.bincount(category(eval_hands(hands)), minlength=9)
    assert (counts > 0).all(), counts


@pytest.mark.parametrize("cards, cat, ranks, text", [
    # wheel and steel wheel inside seven cards
    ("Ah 2c 3d 4h 5s Kc Kd", STRAIGHT, (3,), "a straight, Ace to Five"),
    ("Ah 2c 3d 4h 5s Ad Ac", STRAIGHT, (3,), "a straight, Ace to Five"),                 # wheel beats trip aces
    ("Ah 2c 3d 4h 5s 6c Kd", STRAIGHT, (4,), "a straight, Deuce to Six"),                # six-high beats the wheel
    ("Ah 2h 3h 4h 5h Kc Qd", STRAIGHT_FLUSH, (3,), "a straight flush, Ace to Five"),
    ("Ah 2h 3h 4h 5h 6c 7d", STRAIGHT_FLUSH, (3,), "a straight flush, Ace to Five"),     # steel wheel beats 7-high
    ("Ah 2h 3h 4h 5h 6h Kh", STRAIGHT_FLUSH, (4,), "a straight flush, Deuce to Six"),
    ("Ah 2h 3h 4h 5h Kh Qh", STRAIGHT_FLUSH, (3,), "a straight flush, Ace to Five"),     # seven hearts
    ("Ac Kc Qc Jc Tc 9c 8c", STRAIGHT_FLUSH, (12,), "a Royal Flush"),
    ("Kd Qd Jd Td 9d Ad 2s", STRAIGHT_FLUSH, (12,), "a Royal Flush"),
    ("Ts Js Qs Ks As", STRAIGHT_FLUSH, (12,), "a Royal Flush"),
    ("9s 8s 7s 6s 5s 4s 3s", STRAIGHT_FLUSH, (7,), "a straight flush, Five to Nine"),
    # six and seven suited cards: the five highest count
    ("Ah Kh 9h 8h 6h 4h 2h", FLUSH, (12, 11, 7, 6, 4), "a flush, Ace high"),
    ("Kh 9h 8h 6h 4h 2h Ac", FLUSH, (11, 7, 6, 4, 2), "a flush, King high"),
    ("Jh 9h 8h 7h 6h 2h Th", STRAIGHT_FLUSH, (9,), "a straight flush, Seven to Jack"),
    ("Ah Kh Qh Jh 9h 8d 8c", FLUSH, (12, 11, 10, 9, 7), "a flush, Ace high"),             # flush beats trips
    ("Ah Kh Qh Jh 9h Tc 2d", FLUSH, (12, 11, 10, 9, 7), "a flush, Ace high"),             # flush beats broadway
    # two trips, three pairs, quads with a pair or trips
    ("2c 2d 2h 3c 3d 3h 4s", FULL_HOUSE, (1, 0), "a full house, Threes full of Deuces"),
    ("Ac Ad Ah 2c 2d 2h 3s", FULL_HOUSE, (12, 0), "a full house, Aces full of Deuces"),
    ("7c 7d 7h Kc Kd 2c 2d", FULL_HOUSE, (5, 11), "a full house, Sevens full of Kings"),
    ("Ac Ad Kc Kd Qc Qd Js", TWO_PAIR, (12, 11, 10), "two pair, Aces and Kings"),          # third pair is the kicker
    ("2c 2d 3h 3s 4c 4d Ah", TWO_PAIR, (2, 1, 12), "two pair, Fours and Threes"),
    ("5c 5d 6h 6s 9c 9d 2h", TWO_PAIR, (7, 4, 3), "two pair, Nines and Sixes"),
    ("Kd Kh Kc Ks 2c 2d 2h", QUADS, (11, 0), "four of a kind, Kings"),
    ("2c 2d 2h 2s Ac Ad Ah", QUADS, (0, 12), "four of a kind, Deuces"),
    ("9c 9d 9h 9s Kc Kd 3h", QUADS, (7, 11), "four of a kind, Nines"),
    ("9c 9d 9h 9s 3c 3d Kh", QUADS, (7, 11), "four of a kind, Nines"),
    # straights with pairs and trips
    ("5c 6d 7h 8s 9c 9d 9h", STRAIGHT, (7,), "a straight, Five to Nine"),
    ("Tc Jd Qh Ks Ac Ad Kh", STRAIGHT, (12,), "a straight, Ten to Ace"),
    ("6c 6d 7h 8s 9c Td 2h", STRAIGHT, (8,), "a straight, Six to Ten"),
    # the rest of the ladder
    ("Ac Ad Ah Kc Qd 7h 2s", TRIPS, (12, 11, 10), "three of a kind, Aces"),
    ("2c 2d 7h 8s 9c Jd Kh", PAIR, (0, 11, 9, 7), "a pair of Deuces"),
    ("Ah Kd 9c 8s 6h 4d 2c", HIGH_CARD, (12, 11, 7, 6, 4), "high card Ace"),
    ("7h 5d 4c 3s 2h", HIGH_CARD, (5, 3, 2, 1, 0), "high card Seven"),
    ("6s 6h 5d 4c 3s 2h", STRAIGHT, (4,), "a straight, Deuce to Six"),
])
def test_edge_case_hands(cards, cat, ranks, text):
    value = eval_hand(cards)
    assert decode(value) == (cat, ranks)
    assert describe_hand(cards) == text
    assert value == int(_bf_best(np.array([P(cards)]))[0])


def test_ladder_and_ties_between_hands():
    # equal strengths for equal hands in different suits; the board plays for both
    a = eval_hand("2c 3d Ah Kh Qh Jh Th")
    b = eval_hand("4c 5d Ah Kh Qh Jh Th")
    assert a == b
    assert eval_hand("Ac Ad Ah As Kc 2d 3h") == eval_hand("Ac Ad Ah As Kc 7d 8h")
    # kicker beyond the fifth card does not count
    assert eval_hand("Ac Ad Kh Qs Js 3d 2c") == eval_hand("Ac Ad Kh Qs Js 4d 2c")
    # but the fifth one does
    assert eval_hand("Ac Ad Kh Qs Js 3d 2c") > eval_hand("Ac Ad Kh Qs Ts 3d 2c")
    # wheel < six-high straight < ... < broadway; steel wheel < six-high straight flush
    assert eval_hand("Ah 2c 3d 4h 5s") < eval_hand("2c 3d 4h 5s 6c") < eval_hand("Tc Jd Qh Ks Ac")
    assert eval_hand("Ah 2h 3h 4h 5h") < eval_hand("2h 3h 4h 5h 6h") < eval_hand("Th Jh Qh Kh Ah")


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_eval_hands_any_dtype_and_memory_layout(dtype):
    hands = _restricted_hands(3000, 7, 9, 2, seed=5, run=True)
    expected = _bf_best(hands)
    np.testing.assert_array_equal(eval_hands(hands.astype(dtype)), expected)
    np.testing.assert_array_equal(eval_hands(np.asfortranarray(hands.astype(dtype))), expected)
    np.testing.assert_array_equal(eval_hands(hands.astype(dtype)[::-1]), expected[::-1])
    np.testing.assert_array_equal(eval_hands(hands.astype(dtype)[:, ::-1]), expected)        # card order irrelevant
    wide = np.concatenate([hands, hands], axis=1).astype(dtype)                               # a strided view
    np.testing.assert_array_equal(eval_hands(wide[:, 7:]), expected)


@pytest.mark.parametrize("bad", [
    np.array([[0, 1, 2, 3, -1]]),                     # undealt slot
    np.array([[0, 1, 2, 3, 52]]),
    np.array([[0, 1, 2, 3, 3]]),                      # repeated card
    np.array([[51, 51, 0, 1, 2, 3, 4]]),
    np.array([[12, 25, 38, 51, 12, 0]]),
    np.array([[0, 1, 2, 3, 4]], dtype=bool),
    np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]),
    np.array([0, 1, 2, 3, 4]),                        # one-dimensional
    np.array([[0, 1, 2, 3]]),
    np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
    np.array([[255, 1, 2, 3, 4]], dtype=np.uint8),
])
def test_eval_hands_rejects(bad):
    with pytest.raises(ValueError):
        eval_hands(bad)


def test_repeated_card_detected_in_any_row_of_a_large_batch():
    hands = np.argsort(np.random.default_rng(3).random((20_000, 52)), 1)[:, :7]
    hands[17_123, 6] = hands[17_123, 0]
    with pytest.raises(ValueError):
        eval_hands(hands)


def test_value_from_keys_of_one_hand():
    # card_keys of a 1-D array gives 0-d keys; that used to crash (np.nonzero on a 0-d array)
    for text in ("Ah Kh Qh Jh 9h 2c 3d", "Ah Kd Qh Jh 9h 2c 3d", "Ah Kh Qh Jh Th", "2c 2d 2h 3c 3d 3h"):
        cards = np.array(P(text))
        key, bits = card_keys(cards)
        assert np.ndim(key) == 0
        assert value_from_keys(key, bits, k=cards.size) == eval_hand(text)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def test_combo_index_unknown_cards_are_not_wrapped():
    # -1 marks an unknown card; a bare COMBO_INDEX[-1, x] would be the combo (As, x)
    assert combo_index(-1, 5) == -1
    assert combo_index(5, -1) == -1
    assert combo_index(-1, -1) == -1
    assert combo_index(-52, 0) == -1
    out = combo_index(np.array([-1, 50, 3, 7]), np.array([5, 45, 3, -1]))
    assert out.tolist() == [-1, int(COMBO_INDEX[50, 45]), -1, -1]
    with pytest.raises(IndexError):
        combo_index(52, 0)
    every = combo_index(COMBOS[:, 0], COMBOS[:, 1])
    np.testing.assert_array_equal(every, np.arange(1326))


def test_combos_blocked_counts_on_textured_boards():
    for text, blocked in (("Ah Kh Qh Jh Th", 5 * 51 - 10), ("Ac Ad Ah As", 4 * 51 - 6), ("", 0)):
        assert combos_blocked(text).sum() == blocked


# ---------------------------------------------------------------------------
# Brute-force equity
# ---------------------------------------------------------------------------

def _runouts(live: list[int], j: int) -> np.ndarray:
    runs = list(itertools.combinations(live, j))
    return np.array(runs, dtype=np.int64).reshape(len(runs), j)


def _bf_equity(hero, board, weights, evaluate=eval_hands) -> float:
    """Exact equity vs weighted opponent holdings, every (completion, holding) pair listed."""
    hero, board = list(hero), list(board)
    dead = set(hero + board)
    runouts = _runouts([c for c in range(52) if c not in dead], 5 - len(board))
    w = np.asarray(weights, dtype=np.float64)
    keep = (w > 0) & ~np.isin(COMBOS, list(dead)).any(1)
    combos, cw = COMBOS[keep].astype(np.int64), w[keep]
    if not len(combos):
        return float("nan")
    full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (len(runouts), len(board))), runouts], 1)
    mine = evaluate(np.concatenate([np.broadcast_to(np.array(hero), (len(full), 2)), full], 1)).astype(np.int64)
    clash = (combos[None, :, :, None] == runouts[:, None, None, :]).any((2, 3))
    ri, ci = np.nonzero(~clash)
    theirs = evaluate(np.concatenate([combos[ci], full[ri]], 1)).astype(np.int64)
    share = 0.5 + 0.5 * np.sign(mine[ri] - theirs)
    return float((cw[ci] * share).sum() / cw[ci].sum())


def _bf_vs_hand(hero, villain, board, evaluate=eval_hands) -> float:
    w = np.zeros(1326)
    w[combo_index(*villain)] = 1.0
    return _bf_equity(hero, board, w, evaluate)


_TEXTURES = ("random", "monotone", "flush4", "flush5", "paired", "trips", "quads", "twopair", "run", "wheel")


def _textured_board(kind: str, n: int, rng: np.random.Generator) -> list[int]:
    """n board cards; the first ones carry the texture (as far as n allows)."""
    deck = [int(c) for c in rng.permutation(52)]
    suit, rank = int(rng.integers(4)), int(rng.integers(13))
    other = (rank + 1 + int(rng.integers(12))) % 13
    of_suit = [c for c in deck if c & 3 == suit]
    of_rank = [c for c in deck if c >> 2 == rank]
    head = {
        "monotone": of_suit[:3], "flush4": of_suit[:4], "flush5": of_suit[:5],
        "paired": of_rank[:2], "trips": of_rank[:3], "quads": of_rank[:4],
        "twopair": of_rank[:2] + [c for c in deck if c >> 2 == other][:2],
        "run": [next(c for c in deck if c >> 2 == (rank % 9) + 4 - i) for i in range(5)],
        "wheel": [next(c for c in deck if c >> 2 == r) for r in (12, 0, 1, 2, 3)],
    }.get(kind, [])
    board = (head + [c for c in deck if c not in head])[:n]
    rng.shuffle(board)
    return board


def _holes(board: list[int], rng: np.random.Generator, n: int = 2, bias: float = 0.4) -> list[list[int]]:
    """n disjoint hole-card pairs off the board, often suited to the board's main suit."""
    used = set(board)
    suits = [c & 3 for c in board]
    main = max(range(4), key=suits.count) if board else 0
    hands = []
    for _ in range(n):
        hand = []
        for _ in range(2):
            live = [c for c in range(52) if c not in used]
            same = [c for c in live if c & 3 == main]
            card = int(rng.choice(same if same and rng.random() < bias else live))
            hand.append(card)
            used.add(card)
        hands.append(hand)
    return hands


def _padded(board: list[int]) -> np.ndarray:
    return np.array([board + [-1] * (5 - len(board))])


def test_exact_equity_vs_brute_force_on_textured_boards():
    """Every exact path (single and batch, hand / random / weighted range) on
    boards built to stress the flush correction and the rank-pair tables."""
    rng = np.random.default_rng(2026)
    suited = (COMBOS[:, 0] & 3) == (COMBOS[:, 1] & 3)
    worst = 0.0
    for i in range(90):
        kind = _TEXTURES[i % len(_TEXTURES)]
        n_board = (4, 5, 3)[(i // len(_TEXTURES)) % 3]
        board = _textured_board(kind, n_board, rng)
        hero, vil = _holes(board, rng)
        weights = [np.ones(1326),
                   np.where(rng.random(1326) < 0.2, rng.random(1326), 0.0),
                   np.where(suited, 3 * rng.random(1326), 0.1 * rng.random(1326)),
                   rng.random(1326) ** 4][i % 4]
        expected = _bf_equity(hero, board, weights)
        got = [equity_vs_range(hero, board, weights),
               equity_vs_range_batch(np.array([hero]), _padded(board), weights[None], exact=True)[0]]
        if i % 4 == 0:
            got += [equity_vs_random(hero, board),
                    equity_vs_random_batch(np.array([hero]), _padded(board), exact=True)[0]]
        known = _bf_vs_hand(hero, vil, board)
        got_known = [equity_vs_hand(hero, vil, board),
                     equity_vs_hand_batch(np.array([hero]), np.array([vil]), _padded(board))[0]]
        worst = max(worst, *(abs(g - expected) for g in got), *(abs(g - known) for g in got_known))
        assert worst < EXACT, (kind, format_cards(hero), format_cards(vil), format_cards(board))


def test_river_and_turn_vs_scratch_evaluator():
    """Equity scored with the from-scratch evaluator only (no library code in the reference)."""
    rng = np.random.default_rng(77)
    for i in range(24):
        n_board = 5 if i < 20 else 4
        board = _textured_board(_TEXTURES[i % len(_TEXTURES)], n_board, rng)
        hero, vil = _holes(board, rng)
        assert equity_vs_hand(hero, vil, board) == pytest.approx(_bf_vs_hand(hero, vil, board, _bf_best), abs=EXACT)
        if n_board == 5 or i == 20:
            weights = np.where(rng.random(1326) < 0.5, rng.random(1326), 0.0)
            expected = _bf_equity(hero, board, np.ones(1326), _bf_best)
            assert equity_vs_random(hero, board) == pytest.approx(expected, abs=EXACT)
            expected = _bf_equity(hero, board, weights, _bf_best)
            assert equity_vs_range(hero, board, weights) == pytest.approx(expected, abs=EXACT)


def _pyrust(ranges: list[str], board: list[int]) -> float:
    return pyrust_poker.exact_equity(ranges, format_cards(board, ""), 4)[0]


def test_1000_random_spots_match_pyrust_poker():
    rng = np.random.default_rng(1000)
    deviations = {}
    for i in range(1000):
        n_board = (3, 4, 5)[i % 3]
        kind = ("hand", "random", "range")[(i // 3) % 3]
        texture = _TEXTURES[(i // 9) % len(_TEXTURES)] if i % 2 else "random"
        board = _textured_board(texture, n_board, rng)
        hero, vil = _holes(board, rng, bias=0.3)
        h = format_cards(hero, "")
        if kind == "hand":
            ours, theirs = equity_vs_hand(hero, vil, board), _pyrust([h, format_cards(vil, "")], board)
        elif kind == "random":
            ours, theirs = equity_vs_random(hero, board), _pyrust([h, "random"], board)
        else:
            labels = rng.choice(CLASS_LABELS, size=int(rng.integers(3, 40)), replace=False).tolist()
            dead = set(hero + board)
            spare = [c for c in np.flatnonzero(~np.isin(COMBOS, list(dead)).any(1))
                     if CLASS_LABELS[COMBO_TO_CLASS[c]] not in labels]
            picked = rng.choice(spare, size=int(rng.integers(0, 4)), replace=False)
            combos = [format_cards(COMBOS[j], "") for j in picked]
            weights = range_weights(labels + combos)
            if not (weights * ~combos_blocked(hero + board)).any():
                continue
            ours, theirs = equity_vs_range(hero, board, weights), _pyrust([h, ",".join(labels + combos)], board)
        deviations[i] = abs(ours - theirs)
    assert len(deviations) > 990
    assert max(deviations.values()) < EXACT


# ---------------------------------------------------------------------------
# Ties, symmetry and global identities
# ---------------------------------------------------------------------------

def test_board_plays_for_everyone():
    weights = range_weights(["AA", "KK", "72o", "T9s", "AhKh"])
    for board in ("Ah Kh Qh Jh Th", "Ac Ad Ah As Kc"):
        hero, vil = "2c 3d", "4c 5d"
        b = _padded(P(board))
        for value in (equity_vs_random(hero, board), equity_vs_hand(hero, vil, board),
                      equity_vs_range(hero, board, weights), equity_vs_range(hero, board, np.ones(1326))):
            assert value == 0.5
        assert multiway_equity([hero, vil, "6c 7d", "8c 9d"], board) == [0.25] * 4
        assert equity_vs_random_batch(np.array([P(hero)]), b)[0] == 0.5
        assert equity_vs_range_batch(np.array([P(hero)]), b, weights[None])[0] == 0.5
    # the flop already gives hero a royal flush: every completion and holding loses to it
    for exact in (None, True, False):
        assert equity_vs_random("Jh Th", "Ah Kh Qh", exact=exact) == 1.0
        assert equity_vs_range("Jh Th", "Ah Kh Qh", weights, exact=exact) == 1.0
    hero, board = np.array([P("Jh Th")] * 3), np.array([P("Ah Kh Qh") + [-1, -1]] * 3)
    assert (equity_vs_random_batch(hero, board) == 1.0).all()                      # Monte Carlo rows
    assert (equity_vs_range_batch(hero, board, np.ones((3, 1326))) == 1.0).all()


def test_split_pots_and_shared_hands():
    # the same hand in other suits: only flushes break the tie
    assert equity_vs_hand("Ah Kd", "Ac Ks", "Qh Jd 2c 3s 9h") == 0.5
    e = equity_vs_hand("Ah Kd", "Ac Ks", "Qh Jh 2c")
    assert e == pytest.approx(_pyrust(["AhKd", "AcKs"], P("Qh Jh 2c")), abs=EXACT)
    hands = [P(h) for h in ("As Ks", "Ah Kh", "Ad Kd", "Ac Kc")]
    shares = multiway_equity(hands, "Qs Jh 2c 3d")
    assert sum(shares) == pytest.approx(1.0, abs=1e-12)
    assert shares == pytest.approx([float(x) for x in _bf_multiway(hands, P("Qs Jh 2c 3d"))], abs=1e-12)
    assert shares[0] == pytest.approx(shares[1], abs=1e-12) and shares[2] == pytest.approx(shares[3], abs=1e-12)
    # three players on the same broadway straight, a fourth drawing dead
    assert multiway_equity(["Ts 2c", "Td 3c", "Th 4c", "9s 9d"], "Js Qd Kh Ac 5c") == pytest.approx([1 / 3] * 3 + [0.0])


def _bf_multiway(hands: list[list[int]], board: list[int]) -> list[Fraction]:
    """Exact shares as fractions: every completion listed, a pot split k ways pays 1/k."""
    dead = set(sum(hands, []) + board)
    runouts = _runouts([c for c in range(52) if c not in dead], 5 - len(board))
    full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (len(runouts), len(board))), runouts], 1)
    values = np.stack([eval_hands(np.concatenate([np.broadcast_to(np.array(h), (len(full), 2)), full], 1))
                       for h in hands], 1)
    best = values == values.max(1, keepdims=True)
    unit = 60                                        # divisible by any number of winners from 1 to 6
    points = (best * (unit // best.sum(1, keepdims=True))).sum(0)
    return [Fraction(int(p), unit * len(runouts)) for p in points]


def test_multiway_is_exact_on_random_spots():
    """2 to 6 players against exact fractions. pyrust_poker agrees only to
    about 1e-8 when three players split often (quads on board): its result
    for 6c2c / Tc3c / Qc5c on 7s7h7d7c is 0.08730157... for the first two
    players, while the exact share is 11/126 = 0.08730158..."""
    rng = np.random.default_rng(66)
    for i in range(40):
        k = 2 + i % 5
        board = _textured_board(_TEXTURES[i % len(_TEXTURES)], (3, 4, 5)[i % 3], rng)
        hands = _holes(board, rng, n=k)
        ours = multiway_equity(hands, board)
        assert sum(ours) == pytest.approx(1.0, abs=1e-12)
        assert ours == pytest.approx([float(x) for x in _bf_multiway(hands, board)], abs=1e-12)
        theirs = pyrust_poker.exact_equity([format_cards(h, "") for h in hands], format_cards(board, ""), 4)
        assert ours == pytest.approx(theirs, abs=1e-7)
    assert multiway_equity(["6c 2c", "Tc 3c", "Qc 5c"], "7s 7h 7d 7c") == pytest.approx([11 / 126, 11 / 126, 52 / 63],
                                                                                          abs=1e-15)
    # preflop, three players: only the reference is fast enough to enumerate
    hands = ["Ah Kh", "Qs Qd", "7c 6c"]
    expected = pyrust_poker.exact_equity([h.replace(" ", "") for h in hands], "", 4)
    assert multiway_equity(hands) == pytest.approx(expected, abs=1e-7)


def test_exact_preflop_range():
    """The only path that removes five dealt cards from a range per board
    (inclusion-exclusion over ten card pairs): about 12 s."""
    entries = ["QQ", "AKs", "AKo", "JTs", "76s", "A5s", "JcJd"]
    ours = equity_vs_range("Jh Js", None, range_weights(entries), exact=True)
    assert ours == pytest.approx(pyrust_poker.exact_equity(["JhJs", ",".join(entries)], "", 4)[0], abs=EXACT)


def test_single_spot_monte_carlo_on_every_street():
    rng = np.random.default_rng(23)
    n = 20_000
    tol = 4.5 * 0.5 / np.sqrt(n)
    for n_board in (0, 3, 4, 5):
        board = _textured_board("monotone", n_board, rng) if n_board else []
        hero, vil, third = _holes(board, rng, n=3)
        w = np.where(rng.random(1326) < 0.3, rng.random(1326), 0.0)
        assert equity_vs_hand(hero, vil, board, exact=False, n_samples=n) == pytest.approx(
            equity_vs_hand(hero, vil, board), abs=tol)
        assert equity_vs_random(hero, board, exact=False, n_samples=n) == pytest.approx(
            equity_vs_random(hero, board), abs=tol)
        if n_board:
            assert equity_vs_range(hero, board, w, exact=False, n_samples=n) == pytest.approx(
                equity_vs_range(hero, board, w), abs=tol)
        exact = multiway_equity([hero, vil, third], board)
        assert multiway_equity([hero, vil, third], board, exact=False, n_samples=n) == pytest.approx(exact, abs=tol)


def test_equity_vs_hand_is_antisymmetric():
    rng = np.random.default_rng(9)
    deck = np.argsort(rng.random((300, 52)), 1)
    board = np.full((300, 5), -1)
    n_board = np.array([3, 4, 5])[np.arange(300) % 3]
    for i in range(300):
        board[i, :n_board[i]] = deck[i, 4:4 + n_board[i]]
    a = equity_vs_hand_batch(deck[:, :2], deck[:, 2:4], board)
    b = equity_vs_hand_batch(deck[:, 2:4], deck[:, :2], board)
    np.testing.assert_allclose(a + b, 1.0, rtol=0, atol=1e-12)
    assert equity_vs_hand("Ah Kd", "7c 7s") + equity_vs_hand("7c 7s", "Ah Kd") == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("board", ["Kh 8h 2h", "Ah 2h 3h 4d", "9c 9d 9h 9s", "Kh Qh Jh Th 2h", "7d 7c 2s 2h 7h"])
def test_average_equity_vs_random_over_all_hero_hands_is_one_half(board):
    """Every (hero, opponent, completion) triple is also counted with the roles
    swapped, so the mean over all hero holdings must be exactly 1/2."""
    b = P(board)
    live = ~combos_blocked(b)
    hero = COMBOS[live].astype(np.int64)
    out = equity_vs_random_batch(hero, np.broadcast_to(np.array(b + [-1] * (5 - len(b))), (len(hero), 5)), exact=True)
    assert out.mean() == pytest.approx(0.5, abs=1e-12)
    # the same holds for the preflop table, each class weighted by its combos
    assert np.array(PREFLOP_VS_RANDOM)[COMBO_TO_CLASS].mean() == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Ranges: blocked combos, zero weight, extreme scales
# ---------------------------------------------------------------------------

def test_ranges_with_blocked_combos_and_zero_weight():
    hero, board = P("Ah Kd"), P("Ac 7d 2s 9h")
    blocked = combos_blocked(hero + board)
    w = range_weights({"AA": 1, "KK": 2, "77": 0.5, "AK": 1, "T9s": 3, "22": 1})
    base = equity_vs_range(hero, board, w)
    # heavy weight on impossible combos changes nothing; only impossible combos -> nan
    assert equity_vs_range(hero, board, w + 1e12 * blocked) == pytest.approx(base, abs=1e-12)
    assert np.isnan(equity_vs_range(hero, board, 1e12 * blocked))
    # a zero weight is the same as leaving the combo out
    w0 = w.copy()
    w0[COMBO_TO_CLASS == CLASS_LABELS.index("T9s")] = 0.0
    assert equity_vs_range(hero, board, w0) == pytest.approx(_bf_equity(hero, board, w0), abs=EXACT)
    # a range whose live combos all hold one card: completions that deal it leave no weight
    single_card = np.zeros(1326)
    single_card[[combo_index(P("Qs")[0], c) for c in P("Jc Jd 3h")]] = [1.0, 2.0, 0.5]
    assert equity_vs_range(hero, board, single_card) == pytest.approx(_bf_equity(hero, board, single_card), abs=EXACT)
    # batch: rows with no weight left are nan on every street, the others are untouched
    rows_board = np.array([board + [-1], board[:3] + [-1, -1], [-1] * 5, board + [-1], board[:3] + [-1, -1], [-1] * 5])
    rows_hero = np.array([hero] * 6)
    rows_w = np.stack([w, w, w, blocked * 1.0, combos_blocked(hero + board[:3]) * 3.0, combos_blocked(hero) * 1.0])
    out = equity_vs_range_batch(rows_hero, rows_board, rows_w)
    assert np.isnan(out[3:]).all() and np.isfinite(out[:3]).all()
    assert out[0] == pytest.approx(base, abs=EXACT)
    np.testing.assert_array_equal(out[:3], equity_vs_range_batch(rows_hero[:3], rows_board[:3], rows_w[:3]))


@pytest.mark.parametrize("scale", [5e-324, 1e-310, 1e-200, 1e200, 1e305, 1.7e308])
def test_extreme_weight_scales(scale):
    """Only ratios matter. Huge weights used to overflow the sums (nan), and
    subnormal ones lost their precision (0.7267 instead of 0.7268)."""
    hero, flop = P("Ah Kh"), P("Qh 7h 2c")
    w = range_weights({"QQ": 1, "77": 1, "AK": 1, "T9s": 1, "JTs": 1})
    rows_hero, rows_board = np.array([hero, hero]), np.array([flop + [-1, -1], flop + [5, -1]])
    rows_w = np.stack([w, w])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        exact = equity_vs_range(hero, flop, w * scale)
        sampled = equity_vs_range(hero, [], w * scale, n_samples=4000)
        batch_exact = equity_vs_range_batch(rows_hero, rows_board, rows_w * scale, exact=True)
        batch_default = equity_vs_range_batch(rows_hero, rows_board, rows_w * scale)
    assert exact == pytest.approx(equity_vs_range(hero, flop, w), abs=1e-12)
    assert sampled == pytest.approx(equity_vs_range(hero, [], w, n_samples=4000), abs=1e-3)
    expected = equity_vs_range_batch(rows_hero, rows_board, rows_w, exact=True)
    np.testing.assert_allclose(batch_exact, expected, atol=1e-12)
    np.testing.assert_allclose(batch_default, equity_vs_range_batch(rows_hero, rows_board, rows_w), atol=1e-3)


# ---------------------------------------------------------------------------
# Batch API: mixed streets, padding, shapes, dtypes
# ---------------------------------------------------------------------------

def _mixed_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    deck = np.argsort(rng.random((n, 52)), 1)
    board = np.full((n, 5), -1)
    streets = rng.choice([0, 3, 4, 5], size=n)
    for i, b in enumerate(streets):
        board[i, :b] = deck[i, 4:4 + b]
    return deck[:, :2], deck[:, 2:4], board


def test_batch_with_mixed_streets_equals_scalar():
    hero, vil, board = _mixed_rows(48, seed=3)
    pre = (board < 0).all(1)
    w = np.where(np.random.default_rng(4).random((48, 1326)) < 0.3, 1.0, 0.0)
    exact_hand = equity_vs_hand_batch(hero[~pre], vil[~pre], board[~pre])
    np.testing.assert_allclose(exact_hand, [equity_vs_hand(h, v, b) for h, v, b in
                                            zip(hero[~pre], vil[~pre], board[~pre])], atol=EXACT)
    rand = equity_vs_random_batch(hero, board, exact=True)
    np.testing.assert_allclose(rand, [equity_vs_random(h, b) for h, b in zip(hero, board)], atol=EXACT)
    post = equity_vs_range_batch(hero[~pre], board[~pre], w[~pre], exact=True)
    np.testing.assert_allclose(post, [equity_vs_range(h, b, ww) for h, b, ww in zip(hero[~pre], board[~pre], w[~pre])],
                               atol=EXACT)
    # a mixed batch gives each row what a batch of its street alone gives
    mixed = equity_vs_range_batch(hero, board, w)
    for street in (0, 3, 4, 5):
        sel = (board >= 0).sum(1) == street
        np.testing.assert_array_equal(mixed[sel], equity_vs_range_batch(hero[sel], board[sel], w[sel]))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_batch_short_boards_in_every_integer_dtype(dtype):
    # A short board is padded with -1, which an unsigned dtype cannot hold:
    # uint8 / uint16 / ... flop boards used to raise OverflowError.
    deck = np.argsort(np.random.default_rng(1).random((6, 52)), 1)
    h, v, flop = deck[:, :2], deck[:, 2:4], deck[:, 4:7]
    padded = np.concatenate([flop, np.full((6, 2), -1)], 1)
    ones = np.ones((6, 1326))
    np.testing.assert_array_equal(equity_vs_hand_batch(h.astype(dtype), v.astype(dtype), flop.astype(dtype)),
                                  equity_vs_hand_batch(h, v, padded))
    np.testing.assert_array_equal(equity_vs_random_batch(h.astype(dtype), flop.astype(dtype)),
                                  equity_vs_random_batch(h, padded))
    np.testing.assert_array_equal(equity_vs_range_batch(h.astype(dtype), flop.astype(dtype), ones),
                                  equity_vs_range_batch(h, padded, ones))
    expected = np.array(PREFLOP_VS_RANDOM)[COMBO_TO_CLASS[COMBO_INDEX[h[:, 0], h[:, 1]]]]
    np.testing.assert_array_equal(equity_vs_random_batch(h.astype(dtype), np.zeros((6, 0), dtype=dtype)), expected)


def test_batch_empty_and_list_inputs():
    empty2, empty5 = np.zeros((0, 2), dtype=int), np.zeros((0, 5), dtype=int)
    assert equity_vs_hand_batch(empty2, empty2, empty5).shape == (0,)
    assert equity_vs_random_batch(empty2, empty5).shape == (0,)
    assert equity_vs_range_batch(empty2, empty5, np.zeros((0, 1326))).shape == (0,)
    out = equity_vs_random_batch([[50, 45], [0, 1]], [[10, 11, 12, 13, 14], [-1, -1, -1, -1, -1]])
    assert out[0] == pytest.approx(equity_vs_random([50, 45], [10, 11, 12, 13, 14]), abs=EXACT)
    assert out[1] == PREFLOP_VS_RANDOM[CLASS_LABELS.index("22")]


@pytest.mark.parametrize("hero, board", [
    ([[50, 45]], [[10, -1, 12, -1, -1]]),            # a hole in the board
    ([[50, 45]], [[-1, 10, 11, 12, -1]]),            # dealt from the right
    ([[50, 45]], [[10, 11, 12, 13, -2]]),            # not the -1 sentinel
    ([[50, 45]], [[10, -1, -1, -1, -1]]),            # one card
    ([[50, 45]], [[10, 11, -1, -1, -1]]),            # two cards
    ([[50, -1]], [[10, 11, 12, -1, -1]]),            # unknown hole card
    ([[50, 45]], [[10, 11, 12, 45, -1]]),            # board repeats a hole card
    ([[50, 50]], [[10, 11, 12, -1, -1]]),            # pocket of one card twice
    ([[50, 45]], [[10, 11, 12, 13, 14, -1]]),        # six columns
    ([[50, 45]], [10, 11, 12, -1, -1]),              # one-dimensional board
    ([[50, 45], [1, 2]], [[10, 11, 12, -1, -1]]),    # row counts differ
    ([[50, 45, 3]], [[10, 11, 12, -1, -1]]),         # three hole cards
    ([[50, 45]], [[255, 11, 12, -1, -1]]),           # 255 is not -1
])
def test_batch_rejects_bad_rows(hero, board):
    for call in (lambda: equity_vs_random_batch(np.array(hero), np.array(board)),
                 lambda: equity_vs_range_batch(np.array(hero), np.array(board), np.ones((len(hero), 1326))),
                 lambda: equity_vs_hand_batch(np.array(hero), np.array([[30, 31]] * len(hero)), np.array(board))):
        with pytest.raises(ValueError):
            call()


def test_batch_rejects_bad_weights_and_samples():
    h, b = np.array([[50, 45]]), np.array([[10, 11, 12, -1, -1]])
    for bad in (np.ones(1326), np.ones((1, 1325)), -np.ones((1, 1326)), np.full((1, 1326), np.inf),
                np.full((1, 1326), np.nan)):
        with pytest.raises(ValueError):
            equity_vs_range_batch(h, b, bad)
    for n in (0, -1, 65_537):
        with pytest.raises(ValueError):
            equity_vs_random_batch(h, b, n_samples=n)
        with pytest.raises(ValueError):
            equity_vs_range_batch(h, b, np.ones((1, 1326)), n_samples=n)
        with pytest.raises(ValueError):
            equity_vs_hand_batch(h, np.array([[0, 1]]), np.full((1, 5), -1), n_samples=n)


def test_single_spot_monte_carlo_rejects_bad_sample_counts():
    # n_samples=0 used to divide by zero, and a negative count returned -0.0 as an equity
    for n in (0, -5):
        for call in (lambda: equity_vs_hand("Ah Kh", "Qs Qd", exact=False, n_samples=n),
                     lambda: equity_vs_random("Ah Kh", exact=False, n_samples=n),
                     lambda: equity_vs_random("Ah Kh", "Qh 7h 2c", exact=False, n_samples=n),
                     lambda: equity_vs_range("Ah Kh", None, np.ones(1326), n_samples=n),
                     lambda: multiway_equity(["Ah Kh", "Qs Qd", "7c 7d"], exact=False, n_samples=n)):
            with pytest.raises(ValueError):
                call()


@pytest.mark.parametrize("n, n_samples, probe", [
    (4300, 8, (0, 1, 4095, 4096, 4097, 4299)),         # two row blocks of 4,096
    (150, 4096, (0, 63, 64, 65, 127, 128, 149)),         # inner chunks of 64 rows
])
def test_batch_rows_are_independent_of_blocks_and_chunks(n, n_samples, probe):
    """Each row gets exactly what it gets alone, wherever block and chunk
    boundaries fall."""
    rng = np.random.default_rng(12)
    deck = np.argsort(rng.random((n, 52)), 1)
    flop = np.concatenate([deck[:, 4:7], np.full((n, 2), -1)], 1)
    pre = np.full((n, 5), -1)
    w = np.where(rng.random((n, 1326)) < 0.1, rng.random((n, 1326)), 0.0)
    kw = dict(n_samples=n_samples, seed=5)
    full_random = equity_vs_random_batch(deck[:, :2], flop, **kw)
    full_range = equity_vs_range_batch(deck[:, :2], flop, w, **kw)
    full_hand = equity_vs_hand_batch(deck[:, :2], deck[:, 2:4], pre, **kw)
    for i in probe:
        one = slice(i, i + 1)
        assert equity_vs_random_batch(deck[one, :2], flop[one], **kw)[0] == full_random[i]
        assert equity_vs_range_batch(deck[one, :2], flop[one], w[one], **kw)[0] == full_range[i]
        assert equity_vs_hand_batch(deck[one, :2], deck[one, 2:4], pre[one], **kw)[0] == full_hand[i]
    # duplicated rows give duplicated results
    twice = equity_vs_random_batch(np.concatenate([deck[:50, :2]] * 2), np.concatenate([flop[:50]] * 2), n_samples=64)
    np.testing.assert_array_equal(twice[:50], twice[50:])


# ---------------------------------------------------------------------------
# Monte Carlo: tolerances, seeds, independence
# ---------------------------------------------------------------------------

def _se(n_samples: int) -> float:
    return 0.5 / np.sqrt(n_samples)


def test_preflop_monte_carlo_vs_exact():
    rng = np.random.default_rng(31)
    deck = np.argsort(rng.random((150, 52)), 1)
    hero, vil = deck[:, :2], deck[:, 2:4]
    mc = equity_vs_hand_batch(hero, vil, np.full((150, 5), -1))
    exact = np.array([_pyrust([format_cards(h, ""), format_cards(v, "")], []) for h, v in zip(hero, vil)])
    z = (mc - exact) / _se(4096)
    assert np.abs(z).max() < 4.5
    assert abs(z.mean()) * np.sqrt(len(z)) < 4.5            # no bias across rows
    # exact=True preflop rows are exact
    enumerated = equity_vs_hand_batch(hero[:2], vil[:2], np.full((2, 5), -1), exact=True)
    assert enumerated == pytest.approx(exact[:2], abs=EXACT)
    # single-spot Monte Carlo
    sampled = equity_vs_hand(hero[0], vil[0], exact=False, n_samples=40_000, seed=2)
    assert sampled == pytest.approx(exact[0], abs=4.5 * _se(40_000))
    # ranges, preflop: batch and single spot against an exact enumeration
    labels = [["AA", "KK", "AKs"], ["22", "T9s", "76s", "AQo"], ["72o", "J5s"], ["QQ", "JJ", "TT", "AKo", "KQs"]]
    heroes = ["Kh Kd", "9c 8c", "As 2s", "Jh Td"]
    w = np.stack([range_weights(x) for x in labels])
    out = equity_vs_range_batch(np.array([P(h) for h in heroes]), np.full((4, 5), -1), w)
    for i, (h, lab) in enumerate(zip(heroes, labels)):
        ref = pyrust_poker.exact_equity([h.replace(" ", ""), ",".join(lab)], "", 4)[0]
        assert out[i] == pytest.approx(ref, abs=4.5 * _se(4096))
        assert equity_vs_range(h, None, w[i], n_samples=30_000, seed=i) == pytest.approx(ref, abs=4.5 * _se(30_000))
    # the random-hand table against Monte Carlo, all 169 classes
    first = COMBOS[[np.flatnonzero(COMBO_TO_CLASS == c)[0] for c in range(169)]].astype(np.int64)
    sampled = equity_vs_random_batch(first, np.full((169, 5), -1), exact=False)
    assert np.abs(sampled - np.array(PREFLOP_VS_RANDOM)).max() < 4.5 * _se(4096)
    # multiway
    hands = ["Ah Ad", "Kh Kd", "7c 6c"]
    assert multiway_equity(hands, exact=False, n_samples=40_000, seed=1) == pytest.approx(
        multiway_equity(hands), abs=4.5 * _se(40_000))


def test_flop_monte_carlo_with_narrow_ranges():
    """One- to three-combo ranges make the opponent's cards collide with the
    shuffled board cards often; the replacement logic must stay unbiased."""
    rng = np.random.default_rng(41)
    n = 60
    deck = np.argsort(rng.random((n, 52)), 1)
    hero, flop = deck[:, :2], np.concatenate([deck[:, 4:7], np.full((n, 2), -1)], 1)
    w = np.zeros((n, 1326))
    for i in range(n):
        live = np.flatnonzero(~combos_blocked(np.concatenate([hero[i], flop[i, :3]])))
        w[i, rng.choice(live, size=1 + i % 3, replace=False)] = rng.random(1 + i % 3) + 0.1
    exact = equity_vs_range_batch(hero, flop, w, exact=True)
    mc = equity_vs_range_batch(hero, flop, w, n_samples=65_536)
    z = (mc - exact) / _se(65_536)
    assert np.abs(z).max() < 4.5
    assert abs(z.mean()) * np.sqrt(n) < 4.5


def test_batch_monte_carlo_errors_average_out_across_rows():
    """Rows read shuffles from a shared table. Laid out in ascending order,
    rows with similar dead cards drew nearly the same boards from it, so their
    errors were shared: averaged over the 1,326 preflop hands they were up to
    7 standard errors off for some seeds. Each row now orders its live cards
    by its own hash, and the errors average out like independent ones."""
    hero = COMBOS.astype(np.int64)
    board = np.full((1326, 5), -1)
    exact = np.array(PREFLOP_VS_RANDOM)[COMBO_TO_CLASS]
    z = []
    for seed in range(4):
        err = equity_vs_random_batch(hero, board, exact=False, n_samples=4096, seed=seed) - exact
        z.append(err.mean() / (err.std() / np.sqrt(len(err))))
    assert (np.array(z) ** 2).sum() < 30          # chi-square, 4 degrees of freedom: p < 1e-5 under independence


def test_seed_determinism_and_types():
    h = np.array([P("Ah Kh"), P("7c 2d"), P("Ts 9s")])
    flop = np.array([P("Qh 7h 2c") + [-1, -1], P("Ad Kd 5s") + [-1, -1], P("8s 7d 2h") + [-1, -1]])
    w = np.stack([range_weights(["QQ", "AK", "T9s"])] * 3)
    vil = np.array([P("Qs Qd"), P("As Ks"), P("3c 3d")])
    pre = np.full((3, 5), -1)
    calls = {
        "random": lambda s: equity_vs_random_batch(h, flop, seed=s),
        "range": lambda s: equity_vs_range_batch(h, flop, w, seed=s),
        "hand": lambda s: equity_vs_hand_batch(h, vil, pre, seed=s),
        "random preflop": lambda s: equity_vs_random_batch(h, pre, exact=False, seed=s),
        "range preflop": lambda s: equity_vs_range_batch(h, pre, w, seed=s),
    }
    for name, call in calls.items():
        base = call(7)
        np.testing.assert_array_equal(call(7), base)
        # numpy integer seeds used to raise OverflowError in the row hash
        for s in (np.int64(7), np.int32(7), np.uint64(7), np.uint8(7)):
            np.testing.assert_array_equal(call(s), base, err_msg=f"{name} {type(s).__name__}")
        assert not np.array_equal(call(8), base), name
        assert np.isfinite(call(-3)).all()
    with pytest.raises(TypeError):
        equity_vs_random_batch(h, flop, seed=7.0)
    # single spots
    for call in (lambda s: equity_vs_random("Ah Kh", "Qh 7h 2c", exact=False, n_samples=5000, seed=s),
                 lambda s: equity_vs_hand("Ah Kh", "Qs Qd", exact=False, n_samples=5000, seed=s),
                 lambda s: equity_vs_range("Ah Kh", None, w[0], n_samples=5000, seed=s),
                 lambda s: multiway_equity(["Ah Kh", "Qs Qd", "7c 7d"], exact=False, n_samples=5000, seed=s)):
        assert call(3) == call(3) == call(np.int64(3))
        assert call(3) != call(4)
