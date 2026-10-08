"""Tests for sudetect.equity.

Reference numbers come from the independent ``pyrust_poker`` package, whose
``exact_equity`` enumerates every board (and every holding of a range) the
same way, so exact results must agree to floating-point rounding. Monte Carlo
results are checked against exact ones with a tolerance of four standard
errors.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sudetect.cards import (
    CLASS_LABELS,
    CLASS_SIZE,
    COMBOS,
    class_combos,
    combo_index,
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
    preflop_vs_random_exact,
)

pyrust_poker = pytest.importorskip("pyrust_poker")

EXACT = 1e-9           # exact vs exact: only floating-point rounding may differ


def ref(ranges: list[str], board: str = "") -> list[float]:
    """pyrust_poker exact equities; hands as text, ranges in its syntax."""
    return pyrust_poker.exact_equity([r.replace(" ", "") for r in ranges], board.replace(" ", ""), 4)


def mc_tol(n_samples: int) -> float:
    """Four standard errors of a Monte Carlo share (worst case p = 1/2)."""
    return 4 * 0.5 / np.sqrt(n_samples)


def deal(rng: np.random.Generator, n_board: int, villain: bool = True) -> tuple[list[int], list[int], list[int]]:
    deck = rng.permutation(52).tolist()
    hero, vil = deck[:2], deck[2:4] if villain else []
    board = deck[4:4 + n_board]
    return hero, vil, board


def rows(n: int, n_board: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n random (hero, villain, board) rows on one street, board padded with -1."""
    rng = np.random.default_rng(seed)
    deck = np.argsort(rng.random((n, 52)), axis=1)
    board = np.full((n, 5), -1)
    board[:, :n_board] = deck[:, 4:4 + n_board]
    return deck[:, :2], deck[:, 2:4], board


# ---------------------------------------------------------------------------
# Textbook spots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hero, villain, textbook", [
    ("Ah Ad", "Ks Kc", 0.81),     # AA vs KK, no shared suit: about 81 %
    ("Ah Kh", "Qs Qd", 0.46),     # AKs vs QQ: about 46 %
    ("Ah Kd", "2s 2c", 0.47),     # AKo vs 22: a coin flip, the pair slightly ahead
])
def test_preflop_matchups(hero, villain, textbook):
    ours = equity_vs_hand(hero, villain)
    assert ours == pytest.approx(ref([hero, villain])[0], abs=EXACT)
    assert ours == pytest.approx(textbook, abs=0.01)
    assert equity_vs_hand(villain, hero) == pytest.approx(1 - ours, abs=EXACT)


@pytest.mark.parametrize("hero, textbook", [("Ah Ad", 0.852), ("7h 2d", 0.346), ("Ah Kh", 0.670)])
def test_preflop_vs_random(hero, textbook):
    ours = equity_vs_random(hero)
    assert ours == pytest.approx(textbook, abs=0.001)
    assert ours == pytest.approx(ref([hero, "random"])[0], abs=EXACT)


def test_flush_draw_vs_top_pair_on_the_flop():
    # nut flush draw (9 hearts) + 3 aces against top pair: about 12 outs twice
    ours = equity_vs_hand("Ah 5h", "Kc Qd", "Kh 8h 2c")
    assert ours == pytest.approx(ref(["Ah5h", "KcQd"], "Kh8h2c")[0], abs=EXACT)
    assert 0.40 < ours < 0.48
    # on the turn, one card to come: 9 hearts + 3 aces of 44 rivers win
    assert equity_vs_hand("Ah 5h", "Kc Qd", "Kh 8h 2c Jc") == pytest.approx(12 / 44, abs=EXACT)
    # a 3 on the turn adds a wheel draw: the three non-heart fours win too
    assert equity_vs_hand("Ah 5h", "Kc Qd", "Kh 8h 2c 3s") == pytest.approx(15 / 44, abs=EXACT)


def test_set_vs_overpair():
    ours = equity_vs_hand("7c 7d", "Ac Ad", "7h Ks 2d")
    assert ours == pytest.approx(ref(["7c7d", "AcAd"], "7hKs2d")[0], abs=EXACT)
    assert 0.88 < ours < 0.95
    # turn: AA needs one of the two remaining aces (2 of 44)
    assert equity_vs_hand("7c 7d", "Ac Ad", "7h Ks 2d 9c") == pytest.approx(42 / 44, abs=EXACT)


def test_river_is_a_showdown():
    assert equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h 2c Js 3h") == 1.0
    assert equity_vs_hand("Qs Qd", "Ah Kh", "Qh 7h 2c Js 3h") == 0.0
    assert equity_vs_hand("Ah 2c", "As 3d", "Kh Kd Qc Qs Jh") == 0.5    # both play A-K-K-Q-Q
    # against a random hand on the river only the 990 opponent holdings vary
    river_vs_random = ref(["AhKh", "random"], "Qh7h2cJs3h")[0]
    assert equity_vs_random("Ah Kh", "Qh 7h 2c Js 3h") == pytest.approx(river_vs_random, abs=EXACT)


def test_docstring_examples():
    assert round(equity_vs_hand("Ah Ad", "Ks Kc"), 4) == 0.8126
    assert round(equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h 2c Js"), 4) == 0.2273
    assert round(equity_vs_random("Ah Ad"), 4) == 0.852
    assert round(equity_vs_random("7h 2d"), 4) == 0.3458
    assert round(equity_vs_range("Jh Jd", "Td 7c 2h", range_weights({"QQ": 1, "AK": 1})), 4) == 0.5693
    assert [round(x, 4) for x in multiway_equity(["Ah Ad", "Kh Kd", "Qh Qd"])] == [0.6767, 0.1723, 0.151]


# ---------------------------------------------------------------------------
# 200 random spots against pyrust_poker
# ---------------------------------------------------------------------------

def test_200_random_spots_match_pyrust_poker():
    rng = np.random.default_rng(2024)
    worst = 0.0
    for i in range(200):
        n_board = (3, 4, 5)[i % 3]
        kind = ("hand", "random", "range")[(i // 3) % 3]
        hero, vil, board = deal(rng, n_board)
        h, b = format_cards(hero, ""), format_cards(board, "")
        if kind == "hand":
            ours = equity_vs_hand(hero, vil, board)
            theirs = ref([h, format_cards(vil, "")], b)[0]
        elif kind == "random":
            ours = equity_vs_random(hero, board)
            theirs = ref([h, "random"], b)[0]
        else:
            labels = rng.choice(CLASS_LABELS, size=int(rng.integers(8, 60)), replace=False).tolist()
            ours = equity_vs_range(hero, board, range_weights(labels))
            theirs = ref([h, ",".join(labels)], b)[0]
        worst = max(worst, abs(ours - theirs))
    assert worst < 1e-6
    assert worst < EXACT       # in practice they agree to rounding


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------

def test_uniform_range_is_the_random_hand():
    rng = np.random.default_rng(1)
    ones = np.ones(1326)
    for n_board in (3, 4, 5):
        hero, _, board = deal(rng, n_board, villain=False)
        assert equity_vs_range(hero, board, ones) == pytest.approx(equity_vs_random(hero, board), abs=EXACT)


def test_single_combo_range_is_the_known_hand():
    rng = np.random.default_rng(2)
    for n_board in (3, 4, 5):
        hero, vil, board = deal(rng, n_board)
        w = np.zeros(1326)
        w[combo_index(*vil)] = 1.0
        assert equity_vs_range(hero, board, w) == pytest.approx(equity_vs_hand(hero, vil, board), abs=EXACT)


def test_weights_scale_and_blocked_combos_do_not_matter():
    hero, board = parse_cards("Ah Kd"), parse_cards("Ac 7d 2s 9h")
    w = range_weights({"AA": 1, "KK": 2, "77": 0.5, "AK": 1, "T9s": 3})
    base = equity_vs_range(hero, board, w)
    assert equity_vs_range(hero, board, w * 7.25) == pytest.approx(base, abs=EXACT)
    blocked = np.zeros(1326)
    for i, (a, b) in enumerate(COMBOS.tolist()):
        if {a, b} & set(hero + board):
            blocked[i] = 1.0
    assert equity_vs_range(hero, board, w + 5.0 * blocked) == pytest.approx(base, abs=EXACT)
    # only blocked combos left -> no opponent hand is possible
    assert np.isnan(equity_vs_range(hero, board, blocked))
    assert np.isnan(equity_vs_range(hero, board, np.zeros(1326)))
    # preflop only the hero's own cards block: a range of hands holding them is empty
    assert np.isnan(equity_vs_range(hero, [], range_weights(["AhAs", "KdKs", "AhKd"])))


def test_range_preflop_monte_carlo_matches_exact():
    w = range_weights(["QQ", "KK", "AA", "AKs", "AKo", "JTs"])
    exact = ref(["JhJd", "QQ,KK,AA,AKs,AKo,JTs"])[0]
    mc = equity_vs_range("Jh Jd", None, w, n_samples=100_000, seed=3)
    assert mc == pytest.approx(exact, abs=mc_tol(100_000))
    assert mc == equity_vs_range("Jh Jd", None, w, n_samples=100_000, seed=3)     # reproducible


def test_range_flop_monte_carlo_option_matches_exact():
    hero, board = "Ah 5h", "Kh 8h 2c"
    w = range_weights(["KK", "88", "AK", "KQ", "KJs", "QJs", "T9s"])
    exact = equity_vs_range(hero, board, w)
    assert exact == pytest.approx(ref(["Ah5h", "KK,88,AK,KQ,KJs,QJs,T9s"], "Kh8h2c")[0], abs=EXACT)
    mc = equity_vs_range(hero, board, w, exact=False, n_samples=50_000, seed=9)
    assert mc == pytest.approx(exact, abs=mc_tol(50_000))


# ---------------------------------------------------------------------------
# Monte Carlo options of the single-spot functions
# ---------------------------------------------------------------------------

def test_monte_carlo_options_are_reproducible_and_close():
    exact = equity_vs_random("Ah Ad")
    a = equity_vs_random("Ah Ad", exact=False, n_samples=100_000, seed=7)
    assert a == equity_vs_random("Ah Ad", exact=False, n_samples=100_000, seed=7)
    assert a == pytest.approx(exact, abs=mc_tol(100_000))
    assert a != equity_vs_random("Ah Ad", exact=False, n_samples=100_000, seed=8)
    flop = equity_vs_random("Ah Kh", "Qh 7h 2c")
    flop_mc = equity_vs_random("Ah Kh", "Qh 7h 2c", exact=False, n_samples=100_000)
    assert flop_mc == pytest.approx(flop, abs=mc_tol(100_000))
    known = equity_vs_hand("Ah Ad", "Ks Kc")
    known_mc = equity_vs_hand("Ah Ad", "Ks Kc", exact=False, n_samples=100_000, seed=1)
    assert known_mc == pytest.approx(known, abs=mc_tol(100_000))


# ---------------------------------------------------------------------------
# Multiway
# ---------------------------------------------------------------------------

def test_multiway_heads_up_equals_equity_vs_hand():
    shares = multiway_equity(["Ah Kh", "Qs Qd"], "Qh 7h 2c")
    e = equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h 2c")
    assert shares == pytest.approx([e, 1 - e], abs=EXACT)


@pytest.mark.parametrize("hands, board", [
    (["Ah Ad", "Kh Kd", "Qh Qd"], ""),
    (["Ah Kh", "Qs Qd", "Jc Tc", "7s 7d"], "Qh 7h 2c"),
    (["As Ks", "Ah Kh", "Ad Kd", "Ac Kc"], "Qs Jh 2c 3d"),
])
def test_multiway_matches_pyrust_poker(hands, board):
    shares = multiway_equity(hands, board)
    assert shares == pytest.approx(ref(hands, board), abs=EXACT)
    assert sum(shares) == pytest.approx(1.0, abs=1e-12)


def test_multiway_split_pots():
    # the board is a royal flush: everybody splits
    assert multiway_equity(["2c 3d", "4c 5d", "7s 8s"], "Ah Kh Qh Jh Th") == pytest.approx([1 / 3] * 3)
    # two players hold the same straight (broadway), the third has less
    shares = multiway_equity(["Th 2c", "Tc 3d", "9s 9d"], "Js Qd Kh Ac 5s")
    assert shares == pytest.approx([0.5, 0.5, 0.0])


def test_multiway_monte_carlo_option():
    hands = ["Ah Ad", "Kh Kd", "Qh Qd"]
    exact = multiway_equity(hands)
    mc = multiway_equity(hands, exact=False, n_samples=100_000, seed=4)
    assert mc == pytest.approx(exact, abs=mc_tol(100_000))
    assert sum(mc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Preflop table
# ---------------------------------------------------------------------------

def test_preflop_table_is_consistent():
    table = np.array(PREFLOP_VS_RANDOM)
    assert table.shape == (169,)
    # every hand against every other hand once from each side: the average is 1/2
    assert float((table * CLASS_SIZE).sum() / 1326) == pytest.approx(0.5, abs=1e-12)
    assert CLASS_LABELS[int(table.argmax())] == "AA"
    assert CLASS_LABELS[int(table.argmin())] == "32o"
    for label in ("AKs", "AKo", "22", "72o"):
        hand = format_cards(COMBOS[class_combos(label)[0]])
        assert equity_vs_random(hand) == table[CLASS_LABELS.index(label)]


def test_preflop_table_matches_pyrust_poker():
    # a spot check: all 169 entries were compared when the table was built
    for label in ("AA", "22", "AKs", "T9s", "72o", "32o"):
        hand = format_cards(COMBOS[class_combos(label)[0]], "")
        assert PREFLOP_VS_RANDOM[CLASS_LABELS.index(label)] == pytest.approx(ref([hand, "random"])[0], abs=EXACT)


def test_preflop_table_is_reproducible():
    assert preflop_vs_random_exact("72o") == PREFLOP_VS_RANDOM[CLASS_LABELS.index("72o")]


# ---------------------------------------------------------------------------
# Batch functions
# ---------------------------------------------------------------------------

def test_vs_hand_batch_equals_scalar():
    parts = [rows(2, 0, 10), rows(20, 3, 11), rows(20, 4, 12), rows(20, 5, 13)]
    hero, vil, board = (np.concatenate(x) for x in zip(*parts))
    order = np.random.default_rng(0).permutation(len(hero))           # streets mixed
    hero, vil, board = hero[order], vil[order], board[order]
    out = equity_vs_hand_batch(hero, vil, board, exact=True)
    expected = [equity_vs_hand(h, v, b) for h, v, b in zip(hero, vil, board)]
    np.testing.assert_allclose(out, expected, rtol=0, atol=EXACT)
    # default: preflop rows by Monte Carlo, the rest unchanged
    default = equity_vs_hand_batch(hero, vil, board)
    pre = (board < 0).all(1)
    np.testing.assert_array_equal(default[~pre], out[~pre])
    np.testing.assert_allclose(default[pre], out[pre], atol=mc_tol(4096))


def test_vs_random_batch_equals_scalar():
    parts = [rows(30, 0, 20), rows(6, 3, 21), rows(20, 4, 22), rows(20, 5, 23)]
    hero, _, board = (np.concatenate(x) for x in zip(*parts))
    out = equity_vs_random_batch(hero, board, exact=True)
    expected = [equity_vs_random(h, b) for h, b in zip(hero, board)]
    np.testing.assert_allclose(out, expected, rtol=0, atol=EXACT)
    default = equity_vs_random_batch(hero, board)
    flop = (board >= 0).sum(1) == 3
    np.testing.assert_array_equal(default[~flop], out[~flop])
    np.testing.assert_allclose(default[flop], out[flop], atol=mc_tol(4096))
    sampled = equity_vs_random_batch(hero, board, exact=False)
    pre = (board < 0).all(1)
    np.testing.assert_allclose(sampled[pre], out[pre], atol=mc_tol(4096))


def test_vs_range_batch_equals_scalar():
    parts = [rows(6, 3, 31), rows(12, 4, 32), rows(12, 5, 33)]
    hero, _, board = (np.concatenate(x) for x in zip(*parts))
    rng = np.random.default_rng(5)
    weights = np.where(rng.random((len(hero), 1326)) < 0.15, rng.random((len(hero), 1326)), 0.0)
    weights[-1] = 0.0                                                    # a row with no range left
    out = equity_vs_range_batch(hero, board, weights)
    exact_out = equity_vs_range_batch(hero, board, weights, exact=True)
    n_board = (board >= 0).sum(1)
    for i in range(len(hero) - 1):
        scalar = equity_vs_range(hero[i], board[i], weights[i])          # exact from the flop on
        assert exact_out[i] == pytest.approx(scalar, abs=EXACT)
        if n_board[i] >= 4:
            assert out[i] == exact_out[i]                                 # turn and river: always exact
        else:
            assert out[i] == pytest.approx(scalar, abs=mc_tol(4096))      # flop: Monte Carlo by default
    assert np.isnan(out[-1]) and np.isnan(exact_out[-1])
    assert np.isnan(equity_vs_range(hero[-1], board[-1], weights[-1]))


def test_vs_range_batch_preflop_rows():
    labels = [["AA", "KK", "QQ", "AKs"], ["22", "33", "T9s", "AKo", "KQo"], ["72o", "J5s", "88"]]
    heroes = ["Jh Jd", "Ah Kh", "9c 8c"]
    hero = np.array([parse_cards(h) for h in heroes])
    board = np.full((3, 5), -1)
    weights = np.stack([range_weights(x) for x in labels])
    out = equity_vs_range_batch(hero, board, weights)
    for i in range(3):
        exact = ref([heroes[i], ",".join(labels[i])])[0]
        assert out[i] == pytest.approx(exact, abs=mc_tol(4096))
        single = equity_vs_range(heroes[i], None, weights[i], n_samples=50_000)
        assert single == pytest.approx(exact, abs=mc_tol(50_000))


def test_batch_monte_carlo_is_reproducible_row_by_row():
    hero, vil, board = rows(40, 3, 40)
    w = np.broadcast_to(range_weights(["AA", "KK", "AK", "QJs", "T9s", "55"]), (40, 1326))
    for fn in (lambda h, b, v, ww: equity_vs_random_batch(h, b, seed=11),
               lambda h, b, v, ww: equity_vs_range_batch(h, b, ww, seed=11),
               lambda h, b, v, ww: equity_vs_hand_batch(h, v, np.full_like(b, -1), seed=11)):
        full = fn(hero, board, vil, w)
        assert np.array_equal(full, fn(hero, board, vil, w))
        # a row gives the same number alone, in another order or in another batch
        alone = np.array([fn(hero[i:i + 1], board[i:i + 1], vil[i:i + 1], w[i:i + 1])[0] for i in range(5)])
        np.testing.assert_array_equal(alone, full[:5])
        rev = fn(hero[::-1], board[::-1], vil[::-1], w[::-1])[::-1]
        np.testing.assert_array_equal(rev, full)
    other_seed = equity_vs_random_batch(hero, board, seed=12)
    assert not np.array_equal(other_seed, equity_vs_random_batch(hero, board, seed=11))


def test_batch_accepts_short_boards_and_any_int_dtype():
    hero, vil, board = rows(10, 4, 50)
    expected = equity_vs_hand_batch(hero, vil, board)
    short = equity_vs_hand_batch(hero.astype(np.int8), vil.astype(np.int16), board[:, :4])
    np.testing.assert_array_equal(short, expected)
    assert equity_vs_hand_batch(np.zeros((0, 2), int), np.zeros((0, 2), int), np.zeros((0, 5), int)).shape == (0,)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: equity_vs_hand("Ah Kh", "Ah Qd"),                    # shared card
    lambda: equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h"),           # two-card board
    lambda: equity_vs_hand("Ah Kh Qc", "Qs Qd"),                 # three hole cards
    lambda: equity_vs_hand("Ah Kh", "Qs Qd", "Qh 7h 2c Js 3h 4h"),
    lambda: equity_vs_random("Ah Kh", "Kh 7h 2c"),
    lambda: equity_vs_range("Ah Kh", "Qh 7h 2c", np.ones(1325)),
    lambda: equity_vs_range("Ah Kh", "Qh 7h 2c", -np.ones(1326)),
    lambda: equity_vs_range("Ah Kh", "Qh 7h 2c", np.full(1326, np.nan)),
    lambda: multiway_equity(["Ah Kh"]),
    lambda: multiway_equity(["Ah Kh", "Kh Qd"]),
])
def test_single_spot_validation(call):
    with pytest.raises(ValueError):
        call()


def _batch(board, hero=None, vil=None):
    hero = np.array([[50, 45]]) if hero is None else hero
    vil = np.array([[0, 1]]) if vil is None else vil
    return equity_vs_hand_batch(hero, vil, np.array([board]))


@pytest.mark.parametrize("call", [
    lambda: _batch([10, 11, 12, 13, 14, 15]),                          # six board cards
    lambda: _batch([10, 11, -1, -1, -1]),                              # two dealt
    lambda: _batch([10, -1, 12, 13, -1]),                              # not dealt from the left
    lambda: _batch([10, 11, 50, -1, -1]),                              # board repeats a hole card
    lambda: _batch([10, 11, 52, -1, -1]),                              # id out of range
    lambda: _batch([10, 11, 12, -1, -1], vil=np.array([[0, 1], [2, 3]])),
    lambda: _batch([10, 11, 12, -1, -1], hero=np.array([[50.0, 45.0]])),
    lambda: equity_vs_random_batch(np.array([[50, 45]]), np.array([[10, 11, 12, -1, -1]]), n_samples=70_000),
    lambda: equity_vs_range_batch(np.array([[50, 45]]), np.array([[10, 11, 12, 13, -1]]), np.ones((2, 1326))),
])
def test_batch_validation(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def test_batch_speed_per_street(capsys):
    """Report per-street timings; check the turn target (10,000 rows in under 2 s).

    Other processes can slow a single run down, so each turn timing is the
    best of three runs.
    """
    timings = []

    def timed(label, n, fn, repeats=1):
        best = min(_seconds(fn) for _ in range(repeats))
        timings.append((label, n, best))
        return best

    for n_board, street, n in ((0, "preflop", 2000), (3, "flop", 2000), (4, "turn", 10_000), (5, "river", 10_000)):
        hero, vil, board = rows(n, n_board, 60 + n_board)
        repeats = 3 if street == "turn" else 1
        t_hand = timed(f"{street:7s} vs hand  ", n, lambda: equity_vs_hand_batch(hero, vil, board), repeats)
        t_rand = timed(f"{street:7s} vs random", n, lambda: equity_vs_random_batch(hero, board), repeats)
        if street == "turn":
            assert t_hand < 2.0
            assert t_rand < 2.0
    with capsys.disabled():
        print("\n[equity batch] default settings (exact turn/river, 4,096 Monte Carlo samples on the flop "
              "vs random and preflop vs a hand; preflop vs random from the table)")
        for label, n, dt in timings:
            print(f"  {label}  {n:6d} rows  {dt:6.2f} s  ({dt / n * 1e3:.3f} ms/row)")


def _seconds(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
