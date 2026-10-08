"""Tests for sudetect.evaluator.

The evaluator is checked three ways: exhaustively over all 2,598,960
five-card hands (7,462 distinct values, the textbook category counts),
against the independent ``phevaluator`` package on millions of random 5, 6
and 7-card hands, and on hand-picked edge cases (wheel, steel wheel, best five
of seven). ``phevaluator`` ranks hands 1 (royal flush) to 7,462 (seven high),
so LOWER is better there while HIGHER is better here.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from sudetect.cards import parse_cards
from sudetect.evaluator import (
    CATEGORY_NAMES,
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
    flush_strengths,
    nonflush_strengths,
    rank_index,
    rank_multisets,
    value_from_keys,
)

phevaluator = pytest.importorskip("phevaluator")


def _ph(hands: np.ndarray) -> np.ndarray:
    """phevaluator rank of each row (1 = best); same card ids as sudetect."""
    evaluate = phevaluator.evaluate_cards
    return np.array([evaluate(*row) for row in hands.tolist()], dtype=np.int64)


def _random_hands(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for lo in range(0, n, 250_000):
        m = min(250_000, n - lo)
        parts.append(np.argsort(rng.random((m, 52)), axis=1)[:, :k].astype(np.int8))
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def all_five_card_hands() -> np.ndarray:
    flat = itertools.chain.from_iterable(itertools.combinations(range(52), 5))
    return np.fromiter(flat, dtype=np.int8, count=5 * 2_598_960).reshape(-1, 5)


@pytest.fixture(scope="module")
def five_card_values(all_five_card_hands) -> np.ndarray:
    return eval_hands(all_five_card_hands)


@pytest.fixture(scope="module")
def ph_rank_to_value(all_five_card_hands, five_card_values) -> np.ndarray:
    """Map phevaluator rank -> sudetect strength, built from every 5-card hand."""
    ph = _ph(all_five_card_hands)
    pairs = np.unique(np.stack([ph, five_card_values], axis=1), axis=0)
    # a bijection between the 7,462 ranks and the 7,462 strengths ...
    assert len(pairs) == 7462
    assert len(np.unique(pairs[:, 0])) == 7462 and len(np.unique(pairs[:, 1])) == 7462
    # ... that reverses the order: rank 1 (best) holds the largest strength
    by_rank = pairs[np.argsort(pairs[:, 0])]
    assert (np.diff(by_rank[:, 1]) < 0).all()
    table = np.full(7463, -1, dtype=np.int64)
    table[pairs[:, 0]] = pairs[:, 1]
    return table


# ---------------------------------------------------------------------------
# Exhaustive five-card checks
# ---------------------------------------------------------------------------

def test_exactly_7462_distinct_five_card_values(five_card_values):
    assert five_card_values.dtype == np.int32
    assert len(np.unique(five_card_values)) == 7462


def test_five_card_category_counts(five_card_values):
    cats = category(five_card_values)
    counts = np.bincount(cats, minlength=9).tolist()
    assert counts == [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]
    distinct = [len(np.unique(five_card_values[cats == c])) for c in range(9)]
    assert distinct == [1277, 2860, 858, 858, 10, 1277, 156, 156, 10]


def test_five_card_order_matches_phevaluator(ph_rank_to_value):
    # the fixture asserts the bijection and the order; here only its range
    assert (ph_rank_to_value[1:] > 0).all()


# ---------------------------------------------------------------------------
# Random six- and seven-card hands against phevaluator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k, n", [(7, 2_000_000), (6, 500_000), (5, 500_000)])
def test_random_hands_match_phevaluator(k, n, ph_rank_to_value):
    hands = _random_hands(n, k, seed=1000 + k)
    ours = eval_hands(hands)
    ph = _ph(hands)
    # Same best-five-card hand => same phevaluator rank => the same strength
    # that the exhaustive five-card pass assigned to that rank. This checks
    # the exact value, not only the order.
    np.testing.assert_array_equal(ours, ph_rank_to_value[ph])
    # and therefore every pairwise comparison agrees
    i, j = np.arange(n), np.random.default_rng(k).permutation(n)
    np.testing.assert_array_equal(np.sign(ours[i] - ours[j]), np.sign(ph[j] - ph[i]))


def test_check_flag_does_not_change_values():
    hands = _random_hands(100_000, 7, seed=3)
    np.testing.assert_array_equal(eval_hands(hands), eval_hands(hands, check=False))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64, np.intp])
def test_any_integer_dtype(dtype):
    hands = _random_hands(1000, 7, seed=4)
    np.testing.assert_array_equal(eval_hands(hands.astype(dtype)), eval_hands(hands))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

ASCENDING = [
    "7c 5d 4h 3s 2c",    # worst high card
    "Ah Kd Qc Js 9h",    # best high card
    "2c 2d 3h 4s 5d",    # worst pair
    "Ac Ad Kh Qs Jd",    # best pair
    "3c 3d 2h 2s 4d",    # worst two pair
    "Ac Ad Kh Ks Qd",    # best two pair
    "2c 2d 2h 3s 4d",    # worst three of a kind
    "Ac Ad Ah Ks Qd",    # best three of a kind
    "Ac 2d 3h 4s 5c",    # wheel: the lowest straight
    "2c 3d 4h 5s 6c",    # six-high straight
    "Tc Jd Qh Ks Ac",    # broadway
    "7h 5h 4h 3h 2h",    # worst flush
    "Ah Kh Qh Jh 9h",    # best flush
    "2c 2d 2h 3s 3d",    # worst full house
    "Ac Ad Ah Ks Kd",    # best full house
    "2c 2d 2h 2s 3d",    # worst quads
    "Ac Ad Ah As Kd",    # best quads
    "Ah 2h 3h 4h 5h",    # steel wheel: the lowest straight flush
    "2h 3h 4h 5h 6h",    # six-high straight flush
    "Ah Kh Qh Jh Th",    # royal flush
]


def test_hand_ladder_is_strictly_increasing():
    values = [eval_hand(h) for h in ASCENDING]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert [category(v) for v in values] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]


def test_wheel_details():
    wheel = eval_hand("Ac 2d 3h 4s 5c")
    assert decode(wheel) == (STRAIGHT, (3,))                     # top card is the Five
    assert wheel < eval_hand("2c 3d 4h 5s 6c")
    assert wheel > eval_hand("Ac Ad Ah Ks Qd")
    assert decode(eval_hand("Ah 2h 3h 4h 5h")) == (STRAIGHT_FLUSH, (3,))
    # A-K-Q-J-T plus wheel cards: the ace plays high
    assert decode(eval_hand("Ac Kd Qh Js Tc 2d 3h")) == (STRAIGHT, (12,))
    # no wrap-around straight
    assert category(eval_hand("Qc Kd Ah 2s 3c")) == HIGH_CARD


@pytest.mark.parametrize("cards, expected", [
    ("Kh Kd 7s 7c Ad Ac 2d", (TWO_PAIR, (12, 11, 5))),         # three pairs: best two + best kicker
    ("Kh Kd Ks 7c 7d 7h 2d", (FULL_HOUSE, (11, 5))),           # two sets: higher set full of the lower
    ("Qh Qd Qs Qc Kd Kc Ks", (QUADS, (10, 11))),               # quads, kicker from the set
    ("9h 8h 7c 6h 5h 2h Ts", (FLUSH, (7, 6, 4, 3, 0))),        # flush beats the ten-high straight
    ("5h 6h 7h 8h 9h Tc Jd", (STRAIGHT_FLUSH, (7,))),          # straight flush beats a higher straight
    ("4h 5h 6h 7h 8h 9h 2c", (STRAIGHT_FLUSH, (7,))),          # six suited in a row: the top five
    ("Ah Kh Qh Jh 9h 8h 7h", (FLUSH, (12, 11, 10, 9, 7))),     # seven suited: best five
    ("2c 3d 4h 5s 6c 7d 8h", (STRAIGHT, (6,))),                # seven in a row
    ("Ac Ad Ah 2c 2d 3h 3s", (FULL_HOUSE, (12, 1))),           # trips + two pairs: the higher pair
    ("Ac Kd 9h 7s 5c 3d 2h", (HIGH_CARD, (12, 11, 7, 5, 3))),
    ("Ac Ad 9h 7s 5c 3d 2h", (PAIR, (12, 7, 5, 3))),
    ("Ac Ad Ah 9s 7c 3d 2h", (TRIPS, (12, 7, 5))),
])
def test_best_five_of_seven(cards, expected):
    assert decode(eval_hand(cards)) == expected


def test_ties_and_kickers():
    assert eval_hand("Ah Kd Qc Js 9h") == eval_hand("As Kc Qd Jh 9c")
    assert eval_hand("Ah Ad Kc Qs Jh") > eval_hand("Ac As Kd Qh Th")
    # the board plays for both players
    board = "Ah Kh Qh Jh Th"
    assert eval_hand(board + " 2c 3d") == eval_hand(board + " 4c 5d")
    # a fourth kicker matters in a pair, a sixth card never does
    assert eval_hand("8c 8d Ah 5s 3c") > eval_hand("8h 8s Ac 5d 2c")
    assert eval_hand("8c 8d Ah 5s 3c 2d") == eval_hand("8c 8d Ah 5s 3c")


@pytest.mark.parametrize("cards, text", [
    ("Ah 9c 7d 4s 2h", "high card Ace"),
    ("Kh Kd 9c 7s 2h", "a pair of Kings"),
    ("Ah Ad 7s 7c 2d", "two pair, Aces and Sevens"),
    ("5h 5d 5c Ks 2d", "three of a kind, Fives"),
    ("6c 7d 8h 9s Tc", "a straight, Six to Ten"),
    ("Ac 2d 3h 4s 5c", "a straight, Ace to Five"),
    ("Tc Jd Qh Ks Ac", "a straight, Ten to Ace"),
    ("Kh Qh 9h 4h 2h", "a flush, King high"),
    ("Kh Kd Ks 7c 7d", "a full house, Kings full of Sevens"),
    ("Qh Qd Qs Qc 2d", "four of a kind, Queens"),
    ("5h 6h 7h 8h 9h", "a straight flush, Five to Nine"),
    ("Ah 2h 3h 4h 5h", "a straight flush, Ace to Five"),
    ("Ah Kh Qh Jh Th", "a Royal Flush"),
    ("Kh Qh 9h 4h 2h 2c 2d", "a flush, King high"),
    ("2c 2d Ah 9s 5c 4d 3h", "a straight, Ace to Five"),
    ("6s 6d 6h 9s 9c 9d 2h", "a full house, Nines full of Sixes"),
])
def test_describe_hand(cards, text):
    assert describe_hand(cards) == text
    assert describe_hand(parse_cards(cards)) == text
    assert describe_value(eval_hand(cards)) == text


def test_category_names_and_decode():
    assert CATEGORY_NAMES[HIGH_CARD] == "high card" and CATEGORY_NAMES[STRAIGHT_FLUSH] == "straight flush"
    value = eval_hand("Kh Kd 7s 7c 2d")
    assert isinstance(category(value), int) and category(value) == TWO_PAIR
    assert decode(value) == (TWO_PAIR, (11, 5, 0))
    arr = eval_hands(np.array([parse_cards(h) for h in ASCENDING]))
    assert category(arr).tolist() == [category(int(v)) for v in arr]
    assert category(arr).dtype.kind == "i"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.zeros((3, 4), dtype=int),
    np.zeros((3, 8), dtype=int),
    np.arange(7),
    np.zeros((2, 7), dtype=float),
])
def test_rejects_bad_shapes_and_dtypes(bad):
    with pytest.raises(ValueError):
        eval_hands(bad)


def test_rejects_bad_cards():
    with pytest.raises(ValueError, match="0..51"):
        eval_hands(np.array([[0, 1, 2, 3, 52]]))
    with pytest.raises(ValueError, match="0..51"):
        eval_hands(np.array([[0, 1, 2, 3, -1]]))
    for row in ([0, 1, 2, 3, 3], [51, 50, 49, 48, 47, 46, 51], [5, 9, 13, 5, 30, 31]):
        with pytest.raises(ValueError, match="twice"):
            eval_hands(np.array([row]))
    with pytest.raises(ValueError):
        eval_hand("Ah Kh")


def test_empty_input():
    out = eval_hands(np.zeros((0, 7), dtype=np.int64))
    assert out.shape == (0,) and out.dtype == np.int32


# ---------------------------------------------------------------------------
# Building blocks used by the equity module
# ---------------------------------------------------------------------------

def test_keys_add_up_across_disjoint_card_sets():
    hands = _random_hands(200_000, 7, seed=5).astype(np.int64)
    hole_key, hole_bits = card_keys(hands[:, :2])
    board_key, board_bits = card_keys(hands[:, 2:])
    full = eval_hands(hands)
    np.testing.assert_array_equal(value_from_keys(hole_key + board_key, hole_bits + board_bits), full)
    six_key, six_bits = card_keys(hands[:, :6])
    np.testing.assert_array_equal(value_from_keys(six_key, six_bits, k=6), eval_hands(hands[:, :6]))


def test_value_from_keys_any_memory_layout():
    # Regression: flush hands used to be updated through a flattened view,
    # which silently copies a non-C-contiguous array and loses the flushes.
    hands = _random_hands(6000, 7, seed=8).astype(np.int64)
    key, bits = card_keys(hands)
    expected = eval_hands(hands).reshape(60, 100)
    assert (category(expected) == FLUSH).any()
    k2, b2 = key.reshape(60, 100), bits.reshape(60, 100)
    np.testing.assert_array_equal(value_from_keys(k2.T, b2.T), expected.T)
    np.testing.assert_array_equal(value_from_keys(np.asfortranarray(k2), np.asfortranarray(b2)), expected)
    np.testing.assert_array_equal(value_from_keys(k2[:, ::2], b2[:, ::2]), expected[:, ::2])
    # card ids gathered through a slice-then-index pattern (an F-ordered result)
    live = np.arange(52)[None, :].repeat(4, 0)
    pick = live[:, np.array([[0, 13, 26, 39, 4, 8, 12], [1, 2, 3, 5, 9, 17, 25]])]
    np.testing.assert_array_equal(value_from_keys(*card_keys(pick)), eval_hands(pick.reshape(-1, 7)).reshape(4, 2))


def test_nonflush_and_flush_parts():
    hands = _random_hands(200_000, 7, seed=6).astype(np.int64)
    key, bits = card_keys(hands)
    suits = (hands & 3)
    no_flush = (np.stack([(suits == s).sum(1) for s in range(4)], 1) < 5).all(1)
    np.testing.assert_array_equal(nonflush_strengths(key[no_flush] & ((1 << 32) - 1)), eval_hands(hands[no_flush]))
    assert flush_strengths(np.array([0b1111, 0b10111])).tolist() == [0, 0]
    assert category(int(flush_strengths(np.array(0b1111100000000)))) == STRAIGHT_FLUSH
    assert describe_value(int(flush_strengths(np.array(0b1000000001111)))) == "a straight flush, Ace to Five"


@pytest.mark.parametrize("k, count", [(5, 6175), (6, 18395), (7, 49205)])
def test_rank_multisets_are_densely_indexed(k, count):
    counts = rank_multisets(k)
    assert counts.shape == (count, 13)
    assert (counts.sum(1) == k).all() and counts.max() <= 4
    keys = counts @ (5 ** np.arange(13))
    np.testing.assert_array_equal(rank_index(keys, k), np.arange(count))


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def test_speed_one_million_seven_card_hands(capsys):
    hands = _random_hands(1_000_000, 7, seed=7)
    eval_hands(hands[:1000])
    best = {}
    for check in (True, False):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            eval_hands(hands, check=check)
            times.append(time.perf_counter() - t0)
        best[check] = 1_000_000 / min(times)
    with capsys.disabled():
        print(f"\n[evaluator] 1M x 7 cards: {best[True] / 1e6:.1f}M hands/s with input checks, "
              f"{best[False] / 1e6:.1f}M/s without")
    assert best[True] >= 3_000_000
