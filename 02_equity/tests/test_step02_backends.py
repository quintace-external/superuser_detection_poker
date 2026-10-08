"""Equity backends: card text, exact multiway equity, ahead_now, and cross-checks on random spots."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sudetect import equity_backends as eb
from sudetect.cards import class_index, hand_class, parse_cards
from sudetect.equity import PREFLOP_VS_RANDOM, equity_vs_hand, equity_vs_random, multiway_equity

SPOTS = 200


def rows(*specs: tuple[str, list[str], str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hero, [opponents], board) as text -> hero (n, 2), opponents (n, k, 2), board (n, 5)."""
    hero = np.array([parse_cards(h) for h, _, _ in specs])
    opps = np.array([[parse_cards(o) for o in os_] for _, os_, _ in specs])
    board = np.full((len(specs), 5), -1)
    for i, (_, _, b) in enumerate(specs):
        cards = parse_cards(b)
        board[i, :len(cards)] = cards
    return hero, opps, board


# ---------------------------------------------------------------------------- card text


def test_cards_from_text_reads_step01_strings():
    out = eb.cards_from_text(["AhKd", "", None, float("nan"), "2c3d4h5s6c"], 5)
    assert out.dtype == np.int16
    assert out[0].tolist() == [50, 45, -1, -1, -1]
    assert out[1].tolist() == out[2].tolist() == out[3].tolist() == [-1] * 5
    assert out[4].tolist() == parse_cards("2c3d4h5s6c")


@pytest.mark.parametrize("bad", ["Ah K", "AhKdQ", "1hKd", "AHKd", "Ah10"])
def test_cards_from_text_rejects_malformed_text(bad):
    with pytest.raises(ValueError):
        eb.cards_from_text([bad], 2)


def test_cards_from_text_rejects_rows_that_are_too_long():
    with pytest.raises(ValueError):
        eb.cards_from_text(["AhKdQs"], 2)


# ---------------------------------------------------------------------------- multiway and ahead_now


def test_multiway_exact_matches_single_spot_function():
    specs = [("Ah Ad", ["Kh Kd", "Qh Qd"], "2c 7s 9d"),
             ("Ts 9s", ["Ac Kd", "8h 8d", "2s 3s"], "Js 8s 2d"),
             ("5c 5d", ["Ah Kh", "Qc Jc"], "Th 9h 2c 3d"),
             ("Kc Qd", ["Ac 2d", "7h 7s"], "Ad Ks Qc 7d 2h")]
    hero, opps, board = rows(*[(h, o[:2], b) for h, o, b in specs if len(o) == 2])
    got = eb.multiway_exact_batch(np.concatenate([hero[:, None], opps], axis=1), board)
    for g, (h, o, b) in zip(got, [s for s in specs if len(s[1]) == 2]):
        assert g == pytest.approx(multiway_equity([h] + o, b)[0], abs=1e-12)
    hero, opps, board = rows(specs[1])
    got = eb.multiway_exact_batch(np.concatenate([hero[:, None], opps], axis=1), board)
    assert got[0] == pytest.approx(multiway_equity([specs[1][0]] + specs[1][1], specs[1][2])[0], abs=1e-12)


def test_multiway_split_pots_count_one_kth():
    # A royal flush on the board: three players split every pot, each gets exactly 1/3.
    hero, opps, board = rows(("2c 3d", ["4c 5d", "6c 7d"], "Ah Kh Qh Jh Th"))
    got = eb.multiway_exact_batch(np.concatenate([hero[:, None], opps], axis=1), board)
    assert got[0] == pytest.approx(1 / 3, abs=1e-15)


def test_ahead_now_compares_made_hands_and_ties_are_not_ahead():
    hero, opps, board = rows(
        ("Ah Kd", ["Qs Qc"], "Ac 7d 2h"),          # top pair vs an underpair: ahead
        ("Ah Kd", ["Qs Qc"], "Qd 7d 2h"),          # top pair vs a set: behind
        ("Ah Kd", ["As Kc"], "Ac Kh 2h"),          # the same two pair: a tie, not ahead
        ("9h 8h", ["Ac Ad"], "Th Jh 2c 3c 4d"),     # a flush draw that missed, behind on the river
    )
    assert eb.ahead_now_batch(hero, opps, board).tolist() == [True, False, False, False]
    hero, opps, board = rows(("Jh Jd", ["Th Td", "Ac Kc"], "2s 7d 9c"))
    assert eb.ahead_now_batch(hero, opps, board).tolist() == [True]
    hero, opps, board = rows(("Jh Jd", ["Th Td", "Ac 9s"], "2s 7d 9c"))
    assert eb.ahead_now_batch(hero, opps, board).tolist() == [True]      # 9s pair below jacks
    hero, opps, board = rows(("Jh Jd", ["Th Td", "9h 9s"], "2s 7d 9c"))
    assert eb.ahead_now_batch(hero, opps, board).tolist() == [False]     # one opponent has a set


# ---------------------------------------------------------------------------- backends


def test_numpy_backend_methods_and_standard_errors():
    backend = eb.NumpyBackend(seed=3, mc_samples=4096)
    hero, opps, board = rows(("Ah Kd", ["Qs Qc"], ""), ("Ah Kd", ["Qs Qc"], "2c 7s 9d"),
                             ("Ah Kd", ["Qs Qc"], "2c 7s 9d Th"), ("Ah Kd", ["Qs Qc"], "2c 7s 9d Th 3h"))
    r = backend.vs_random(hero, board)
    assert r.exact.tolist() == [True, False, True, True]
    assert r.value[0] == PREFLOP_VS_RANDOM[class_index(hand_class("Ah Kd"))]
    assert r.se[1] == pytest.approx(math.sqrt(r.value[1] * (1 - r.value[1]) / 4096))
    assert r.se[[0, 2, 3]].tolist() == [0.0, 0.0, 0.0]
    h = backend.vs_hands(hero, opps, board)
    assert h.exact.tolist() == [False, True, True, True]
    assert h.value[0] == pytest.approx(equity_vs_hand("Ah Kd", "Qs Qc"), abs=4.5 * h.se[0])
    for i, b in ((1, "2c 7s 9d"), (2, "2c 7s 9d Th"), (3, "2c 7s 9d Th 3h")):
        assert h.value[i] == pytest.approx(equity_vs_hand("Ah Kd", "Qs Qc", b), abs=1e-12)
        assert r.value[i] == pytest.approx(equity_vs_random("Ah Kd", b), abs=4.5 * r.se[i] + 1e-12)


def test_numpy_backend_preflop_multiway_is_seeded_monte_carlo():
    hero, opps, board = rows(("Ah Kd", ["Qs Qc", "7h 6h"], ""))
    a = eb.NumpyBackend(seed=5, mc_samples=8192).vs_hands(hero, opps, board)
    b = eb.NumpyBackend(seed=5, mc_samples=8192).vs_hands(hero, opps, board)
    assert a.value.tolist() == b.value.tolist() and not a.exact[0]
    assert a.value[0] == pytest.approx(multiway_equity(["Ah Kd", "Qs Qc", "7h 6h"])[0], abs=4.5 * a.se[0])


def test_mc_se_is_the_binomial_bound():
    assert eb.mc_se(np.array([0.0, 0.5, 1.0]), 4096).tolist() == [0.0, 0.5 / 64, 0.0]


def test_numpy_exact_flop_matches_reference_on_random_spots():
    check = eb.cross_check(eb.NumpyBackend(flop_random_exact=True), eb.ReferenceBackend(), SPOTS, seed=21)
    assert check.n_values == 2 * SPOTS
    assert check.passed, check.failures[:5]
    assert check.max_abs_diff <= 1e-9


def test_numpy_default_matches_reference_within_monte_carlo_error():
    check = eb.cross_check(eb.NumpyBackend(seed=7), eb.ReferenceBackend(), SPOTS, seed=22, max_opponents=4)
    assert check.passed, check.failures[:5]
    assert 0 < check.max_z < 4.5
    exact_kinds = {k: v for k, v in check.by_kind.items() if not k.startswith("flop vs random")}
    assert max(exact_kinds.values()) <= 1e-9


def test_pyrust_backend_matches_numpy_on_random_spots():
    pytest.importorskip("pyrust_poker")
    check = eb.cross_check(eb.NumpyBackend(flop_random_exact=True), eb.PyrustBackend(), SPOTS, seed=23,
                           max_opponents=5, exact_tol=1e-7)
    assert check.passed, check.failures[:5]
    # heads-up the two engines agree to rounding; with several opponents pyrust_poker's integer division of
    # split pots leaves differences of about 1e-8
    heads_up = [v for k, v in check.by_kind.items() if k.endswith("1 hand") or k.endswith("random")]
    assert max(heads_up) <= 1e-9


def test_pyrust_backend_turns_engine_errors_into_nan():
    pytest.importorskip("pyrust_poker")
    backend = eb.PyrustBackend()
    assert math.isnan(backend._exact(["AhKd", "AhQs"], ""))      # the engine refuses a duplicated card
    assert backend.failures == 1


def test_pyrust_backend_sends_more_than_five_opponents_to_numpy():
    pytest.importorskip("pyrust_poker")
    hero, opps, board = rows(("Ah Kd", ["Qs Qc", "2c 2d", "3c 3d", "4c 4d", "5c 5d", "6c 6d"], "7h 8h 9s"))
    got = eb.PyrustBackend().vs_hands(hero, opps, board)
    want = multiway_equity(["Ah Kd", "Qs Qc", "2c 2d", "3c 3d", "4c 4d", "5c 5d", "6c 6d"], "7h 8h 9s")[0]
    assert got.exact[0] and got.value[0] == pytest.approx(want, abs=1e-12)


def test_get_backend_names():
    assert isinstance(eb.get_backend("numpy"), eb.NumpyBackend)
    assert isinstance(eb.get_backend("reference"), eb.ReferenceBackend)
    assert isinstance(eb.get_backend("numpy"), eb.Backend)
    with pytest.raises(ValueError):
        eb.get_backend("gpu")


# ---------------------------------------------------------------------------- scoring rows


def test_score_decisions_null_rules_and_bad_rows():
    cols = {
        "hole_cards": ["AhKd", "AhKd", "AhKd", "AhKd", "AhAh", "AhKd"],
        "board": ["Ac7s9d", "Ac7s9d", "Ac7s9dTh3h", "", "Ac7s9d", "Ac7s9d"],
        "n_live_opp": [1, 2, 1, 1, 1, 1],
        "n_opp_known": [1, 1, 1, 1, 1, 1],
        "opp_cards": ["QsQc", "QsQc", "QsQc", "QsQc", "QsQc", "QsKd"],
    }
    out = eb.score_decisions(cols, eb.NumpyBackend())
    assert out["valid"].tolist() == [True, True, True, True, False, True]
    # row 1: one of two opponents unknown; row 4: duplicated hero card; row 5: opponent shares the hero's Kd
    assert out["eq_real_known"].tolist() == [True, False, True, True, False, False]
    assert np.isnan(out["eq_vs_real"][[1, 4, 5]]).all() and not np.isnan(out["eq_vs_real"][[0, 2, 3]]).any()
    assert out["ahead_now"].tolist() == [1, -1, 1, -1, -1, -1]           # preflop (row 3) has no ahead_now
    assert np.isnan(out["eq_vs_random"][4]) and not np.isnan(out["eq_vs_random"][[0, 1, 2, 3, 5]]).any()
    assert out["eq_vs_real"][2] == 1.0


def test_score_hu_hands_rows_sum_to_one_by_construction():
    out = eb.score_hu_hands({"cards_a": ["AhAd", "7h2c"], "cards_b": ["KsKc", "AhKh"]},
                            eb.NumpyBackend(seed=1, mc_samples=8192))
    assert out["eq_a"][0] == pytest.approx(equity_vs_hand("Ah Ad", "Ks Kc"), abs=4.5 * out["se"][0])
    assert out["se"][0] == pytest.approx(math.sqrt(out["eq_a"][0] * (1 - out["eq_a"][0]) / 8192))
    assert not out["exact"].any()
