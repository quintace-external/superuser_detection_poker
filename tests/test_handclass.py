"""Tests for sudetect.handclass: hand-checked tables for every class, and the
vectorised forms against the scalar reference on random boards."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pytest

from sudetect import handclass as hc
from sudetect.cards import CLASS_LABELS, COMBO_INDEX, COMBOS, N_COMBOS, class_index, parse_cards
from sudetect.evaluator import eval_hands

# ----------------------------------------------------------------------------
# Made hand: (hole, board, expected class)
# ----------------------------------------------------------------------------

MADE_CASES = [
    # nothing: no pair with a hole card, or the hero plays the board
    ("Ah Kd", "Qs 7c 2d", "nothing"),
    ("9h 8h", "Kd 7c 2s", "nothing"),
    ("Kc Kd", "Ah Kh 9h 5h 2h", "nothing"),        # set of kings, but the board flush plays
    ("9c 9d", "5h 6s 7d 8c 9h", "nothing"),        # set of nines, but the board straight plays
    ("Ac Kd", "7c 7d 7h 7s 2c", "nothing"),        # four of a kind on the board, ace only a kicker
    ("Qd Jc", "8c 8d 8h 8s", "nothing"),           # same on the turn
    ("4c 3c", "Ah Kh 9h 5h 2h", "nothing"),        # wheel with both hole cards, board flush is better
    ("Ad 2s", "Kc Kd Qh Qs 7c", "nothing"),        # ace kicker on a double-paired board
    # underpair: pocket pair below every board card
    ("2c 2d", "Kh 9s 5d", "underpair"),
    ("6h 6s", "Ac Td 8c 7h", "underpair"),
    ("3c 3d", "Kh Kd 7s", "underpair"),
    ("4c 4d", "Jh 9s 8d 6c 5h", "underpair"),
    ("7s 7d", "Qc 9h 8d", "underpair"),
    ("8c 8h", "Ad Kd Td 9c 9s", "underpair"),
    # weak_pair: pairs a board rank below t2, or pocket pair below t2 above the lowest card
    ("Ac 2d", "Kh 9s 2h", "weak_pair"),
    ("8c 8d", "Ks Th 5c", "weak_pair"),
    ("9h 5h", "Ad Qc 5s 3d", "weak_pair"),
    ("Ad 2s", "Kc Kd Qh Qs 2c", "weak_pair"),      # counterfeited on the river, still counts
    ("4c 4d", "Jh 9s 3d 2c", "weak_pair"),
    ("Jc 6d", "As Th 6s", "weak_pair"),
    # middle_pair: pairs t2, or pocket pair between t2 and t1
    ("Ac 9d", "Kh 9s 4c", "middle_pair"),
    ("Th Tc", "Qd 8s 3h", "middle_pair"),
    ("Qs Qh", "Kc Kd 7h", "middle_pair"),          # pocket pair on a paired board is one pair
    ("Ah 7s", "Kc Kd 7h", "middle_pair"),          # the board's kings are ignored
    ("Jd 8c", "Ac 8h 5s 2d 3c", "middle_pair"),
    ("Jc Jd", "Qh 9s 4d 2c", "middle_pair"),
    # top_pair_weak: pairs t1, kicker not one of the two best kickers
    ("Ah Js", "Ad 7c 2s", "top_pair_weak"),        # K and Q are the better kickers
    ("Kd 9c", "Kh 7s 2d", "top_pair_weak"),
    ("7h Qd", "7c 5s 2h", "top_pair_weak"),        # A and K are the better kickers
    ("Qs 4s", "Qd Tc 3h 2c", "top_pair_weak"),
    ("Td 5h", "Tc 6s 2d", "top_pair_weak"),
    ("Kc Jd", "Kh 9s 5d 3c 2h", "top_pair_weak"),  # A and Q are the better kickers
    # top_pair_good: pairs t1 with one of the two best kickers
    ("Ah Qs", "Ad 7c 2s", "top_pair_good"),
    ("Ah Ks", "Ad 7c 2s", "top_pair_good"),
    ("Kd Ac", "Kh 7s 2d", "top_pair_good"),
    ("Kd Qc", "Kh 7s 2d", "top_pair_good"),
    ("Ac Td", "Ah Kd Qs", "top_pair_good"),        # only J is a better kicker
    ("9s Kd", "9h 6c 2d", "top_pair_good"),
    ("Jc Kh", "Kc Qd 5s", "top_pair_good"),
    # overpair
    ("Qh Qs", "9d 7c 2s", "overpair"),
    ("As Ad", "Kc Kd 7h", "overpair"),
    ("Tc Td", "8h 6s 3d 2c", "overpair"),
    ("Kd Kc", "Qh Jh 4s 4d 2c", "overpair"),
    ("9s 9h", "8c 5d 2h", "overpair"),
    # two_pair: each hole card pairs a different board rank
    ("Ah 7s", "Ad 7c 2s", "two_pair"),
    ("Kd Qc", "Kh Qs 5d", "two_pair"),
    ("7h 2c", "Kc Kd 7d 2s", "two_pair"),
    ("9c 8c", "9h 8d 3s Js 2h", "two_pair"),
    ("Td 4h", "Tc 4s Ad", "two_pair"),
    ("As Kd", "Ac Kh 5d 5s", "two_pair"),
    # trips_set
    ("7h 7s", "7d Qd 2c", "trips_set"),
    ("Ah 7s", "7d 7c 2c", "trips_set"),
    ("Kd Qs", "Kh Kc 9d 4s", "trips_set"),
    ("3c 3d", "Ah 3s Jd 9c Tc", "trips_set"),
    ("Jc Th", "Js Jd 4h", "trips_set"),
    ("2s 2h", "2d 9c Kh", "trips_set"),
    # straight: higher than any straight on the board
    ("5h 4c", "2d 3c Ah", "straight"),             # the wheel
    ("6c 5d", "4h 3s 2c", "straight"),
    ("Jh Tc", "9d 8s 7c", "straight"),
    ("Ts 2c", "5h 6d 7s 8c 9h", "straight"),       # beats the straight on the board
    ("Kd Qd", "Jc Ts 9h 4d", "straight"),
    ("8c 7d", "6h 5s 4c 4d 2h", "straight"),
    # flush: a hole card among the five flush cards
    ("Kh Qh", "Ah Jh 2h", "flush"),
    ("6h 5h", "Ah Jh 2h", "flush"),
    ("3h 4c", "Ah Kh 9h 5h 2h", "flush"),          # the three improves the board flush
    ("Qd 2d", "9d 7d 4c 3d", "flush"),
    ("Ac 5s", "Kc 9c 7c 2c 2d", "flush"),
    ("Jc Tc", "Qc 9c 3c", "flush"),
    # full_house_plus
    ("Kh Kd", "Ks 7c 7d", "full_house_plus"),
    ("7h 7s", "7d 7c Ah", "full_house_plus"),      # four of a kind with a pocket pair
    ("Kd 2c", "7c 7d 7h Ks 2s", "full_house_plus"),
    ("9h 8h", "Th Jh Qh", "full_house_plus"),      # straight flush
    ("Ac 4s", "Ad Ah 4d 9c", "full_house_plus"),
    ("Kc Qd", "7s 7d 7h Kh Kd", "full_house_plus"),  # kings full beats the board's sevens full
    ("Qs 3c", "Qh Qd Qc 5s", "full_house_plus"),
]


@pytest.mark.parametrize("hole,board,expected", MADE_CASES)
def test_made_hand_table(hole, board, expected):
    assert hc.made_hand(hole, board) == expected
    made, _ = hc.classes_for_all_combos(board)
    h = parse_cards(hole)
    assert hc.MADE_CLASSES[made[COMBO_INDEX[h[0], h[1]]]] == expected


def test_made_table_covers_every_class_five_times():
    counts = Counter(expected for _, _, expected in MADE_CASES)
    assert set(counts) == set(hc.MADE_CLASSES)
    assert min(counts.values()) >= 5


# ----------------------------------------------------------------------------
# Draw: (hole, board, expected class)
# ----------------------------------------------------------------------------

DRAW_CASES = [
    # none
    ("9h Th", "Jh Qh 2c 3d 4s", "none"),           # river
    ("Ah Kd", "7s 4c 2d", "none"),
    ("8s 7d", "5s 6c 9h", "none"),                 # made straight, no draw reported
    ("Ah Kc", "5d 6c 7h 8s", "none"),              # the 4 and the 9 complete the board alone
    ("3c 2d", "5s 6h 7d 8c", "none"),              # a 4 gives everyone the same straight
    ("7c 6d", "Ah Kh 9h 2h", "none"),              # four to a flush on the board, no heart held
    ("Ah Kh", "Qh Jh 2h", "none"),                 # made flush: neither draw counts
    ("Qh Jd", "8c 4s 2h", "none"),
    # gutshot: one out rank
    ("As 2s", "3d 5c Qh", "gutshot"),              # a four makes the wheel
    ("8s 9s", "5s 7d 2c", "gutshot"),
    ("Jh Tc", "Kd 9s 2c", "gutshot"),
    ("Kd Qc", "Ah Js 4d", "gutshot"),
    ("Td 2c", "5s 6d 7h 8d", "gutshot"),           # only a nine, which lets the ten play
    ("Jc Tc", "5s 6d 7h 8d", "gutshot"),
    ("Ac Kc", "Qd Jh 3s 3h", "gutshot"),
    # oesd: two or more out ranks
    ("8h 7h", "6c 5d Ks", "oesd"),
    ("Jd 8c", "9s 5h 7d", "oesd"),                 # double gutshot (6 or T)
    ("Kc Qd", "Js Th 4c", "oesd"),
    ("5h 4h", "3c 2d Ks", "oesd"),                 # ace (wheel) or six
    ("9c 8d", "Td 7s 2h 2c", "oesd"),
    ("Ts 9d", "Jh 8c Ac 3d", "oesd"),
    # flush_draw
    ("Ah Kh", "7h 4h 2c", "flush_draw"),
    ("Qs 9s", "Ks 5s 2d", "flush_draw"),
    ("3d 2c", "Ad Kd 8d", "flush_draw"),           # one hole card on a monotone flop
    ("Ac Jc", "Kc 7c 2d 2s", "flush_draw"),
    ("8h 8s", "Kh 9h 4h", "flush_draw"),
    ("8h 7h", "6h 5c 4h", "flush_draw"),           # made straight plus a flush draw
    # combo_draw: flush draw and an out rank
    ("9h Th", "Jh Qh 2c", "combo_draw"),
    ("Ah Kh", "Qh Jd 2h", "combo_draw"),
    ("8s 9s", "5s 7s 2c", "combo_draw"),
    ("6c 5c", "4c 3d Kc", "combo_draw"),
    ("Jd Td", "9d 8c 2d 3s", "combo_draw"),
    ("As 4s", "5s 3c Ks", "combo_draw"),
]


@pytest.mark.parametrize("hole,board,expected", DRAW_CASES)
def test_draw_table(hole, board, expected):
    assert hc.draw_class(hole, board) == expected
    _, draw = hc.classes_for_all_combos(board)
    h = parse_cards(hole)
    assert hc.DRAW_CLASSES[draw[COMBO_INDEX[h[0], h[1]]]] == expected


def test_draw_table_covers_every_class_five_times():
    counts = Counter(expected for _, _, expected in DRAW_CASES)
    assert set(counts) == set(hc.DRAW_CLASSES)
    assert min(counts.values()) >= 5


def test_classify_returns_both_classes():
    assert hc.classify("9h Th", "Jh Qh 2c") == ("nothing", "combo_draw")
    assert hc.classify("8h 7h", "6h 5c 4h") == ("straight", "flush_draw")
    assert hc.classify("7h 7s", "7d Qd 2c 3c 4s") == ("trips_set", "none")


@pytest.mark.parametrize(
    "hole,board,expected",
    [
        ("8s 9s", "5s 7d 2c", True),               # spades 8-9-5
        ("Ah Kd", "7h 4h 2c", True),               # one hole card with two on the board
        ("Ah Kh", "7c 4d 2h", True),               # two hole cards with one on the board
        ("Ah Kd", "7s 4c 2d", False),
        ("Ah Kh", "7h 4h 2c", False),              # four hearts is a flush draw, not a backdoor
        ("Ah Kd", "7h 4h 2c 9s", False),           # turn
        ("2c 3d", "Ah Kh 9h", False),              # three to a flush on the board alone
    ],
)
def test_backdoor_flush_draw(hole, board, expected):
    assert hc.backdoor_flush_draw(hole, board) is expected


# ----------------------------------------------------------------------------
# Boards
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "board,expected",
    [
        ("Kh Qd 9h", "twotone_unpaired_KQ_c12"),
        ("As Kd Qh", "rainbow_unpaired_A_c12"),
        ("9c 8c 7c", "monotone_unpaired_96_c3"),
        ("Kh Kd 9c", "rainbow_paired_KQ_c0"),
        ("7h 7d 7c", "rainbow_trips_96_c0"),
        ("5s 4s 2d", "twotone_unpaired_52_c12"),   # wheel and 2-6
        ("Jd 7c 2h", "rainbow_unpaired_JT_c0"),
        ("Ah 2d 3c", "rainbow_unpaired_A_c12"),    # the wheel only
        ("Td 9d 6s", "twotone_unpaired_JT_c12"),
        ("6h 5h 4h", "monotone_unpaired_96_c3"),
        ("Jh Td 9c", "rainbow_unpaired_JT_c3"),
        ("Kh Qd 9h 2c", "twotone_unpaired_KQ_c12"),  # a turn board reads its flop
        ("Kh Qd 9h 2c 2d", "twotone_unpaired_KQ_c12"),
    ],
)
def test_board_texture(board, expected):
    assert hc.board_texture(board) == expected


TEXTURE_CASES = [
    ("Kh Kd 9c", "paired"),
    ("7h 7d 7c", "paired"),
    ("2c 2d As", "paired"),
    ("Jh 9c 9d", "paired"),
    ("4s 8h 4h", "paired"),
    ("Ah Kh 2h", "monotone"),
    ("9c 8c 7c", "monotone"),
    ("Qd 7d 3d", "monotone"),
    ("6s 5s 4s", "monotone"),
    ("Jh 8h 2h", "monotone"),
    ("As Kd Qh", "A_high"),
    ("Ah 7d 2c", "A_high"),
    ("Ac 5d 4h", "A_high"),
    ("Ad Jd 8c", "A_high"),
    ("Ah 2d 3c", "A_high"),
    ("Kh Qd 9h", "KQ_high"),
    ("Kc 7d 2h", "KQ_high"),
    ("Qs Js Td", "KQ_high"),
    ("Qh 5d 2c", "KQ_high"),
    ("Kd 8d 4s", "KQ_high"),
    ("Jh Td 9c", "connected_mid"),
    ("Jd 8c 7h", "connected_mid"),
    ("Ts 9h 6d", "connected_mid"),
    ("Tc 7d 6s", "connected_mid"),
    ("Jc 9d 7h", "connected_mid"),
    ("Jh 7d 2c", "dry_mid"),
    ("Ts 5h 2d", "dry_mid"),
    ("Jc 4d 3h", "dry_mid"),
    ("Td 4c 2s", "dry_mid"),
    ("Jd 6s 5c", "dry_mid"),
    ("9c 8d 7h", "connected_low"),
    ("6h 5d 4c", "connected_low"),
    ("5s 4d 2h", "connected_low"),
    ("8h 6d 5c", "connected_low"),
    ("9d 7c 6s", "connected_low"),
    ("9h 4d 2c", "dry_low"),
    ("8c 3d 2h", "dry_low"),
    ("9s 5d 3h", "dry_low"),
    ("8h 7d 2c", "dry_low"),
    ("9c 8d 2h", "dry_low"),
]


@pytest.mark.parametrize("board,expected", TEXTURE_CASES)
def test_texture_class(board, expected):
    assert hc.texture_class(board) == expected


def test_texture_table_covers_every_class_five_times():
    counts = Counter(expected for _, expected in TEXTURE_CASES)
    assert set(counts) == set(hc.TEXTURE_CLASSES)
    assert min(counts.values()) >= 5


def test_texture_class_takes_eight_values_over_all_flops():
    seen = Counter()
    for i in range(52):
        for j in range(i):
            for k in range(j):
                seen[hc.texture_class([i, j, k])] += 1
    assert set(seen) == set(hc.TEXTURE_CLASSES)
    assert sum(seen.values()) == 22100


CHANGE_CASES = [
    # flush_completes
    ("Ah 7h 2c Kh", "flush_completes"),
    ("9s 8s 3d 2s", "flush_completes"),
    ("Kc Qc Jc 2c", "flush_completes"),            # fourth club
    ("Ad 7d 2c 5h Jd", "flush_completes"),
    ("Ah Kh 9h 2c 3h", "flush_completes"),
    ("Td 9d 2c Jd", "flush_completes"),            # also a straight card: flush comes first
    # straight_completes
    ("Th 9d 2c Js", "straight_completes"),
    ("Ah 2d 9c 4s", "straight_completes"),         # wheel window A-2-4
    ("9c 8d 2h 6s", "straight_completes"),
    ("Kc 7d 2h 6s 5c", "straight_completes"),
    ("Qh Jd 3c 3s Td", "straight_completes"),
    ("8h 7c 6d 9s", "straight_completes"),         # four to a straight
    # board_pairs_top
    ("Kh 7d 2c Ks", "board_pairs_top"),
    ("Ah 9c 4d Ac", "board_pairs_top"),
    ("Qd Jc 5s Qh", "board_pairs_top"),
    ("9h 6c 2d Ts Td", "board_pairs_top"),         # pairs the turn card, now the top card
    ("Jc 8d 4h 2s Js", "board_pairs_top"),
    # board_pairs_other
    ("Kh 7d 2c 7s", "board_pairs_other"),
    ("Ah 9c 4d 4s", "board_pairs_other"),
    ("Qd Jc 5s Jh", "board_pairs_other"),
    ("Kh 7d 2c 7s 7c", "board_pairs_other"),       # trips
    ("Th 9c 5d 3s 5h", "board_pairs_other"),
    # overcard
    ("9h 6c 2d Ks", "overcard"),
    ("Th 8c 3d As", "overcard"),
    ("7h 5c 2d Js", "overcard"),
    ("8h 5c 2d 3s Qd", "overcard"),
    ("Jd 9c 6h 2s Ah", "overcard"),
    ("Tc 8d 4s Kh", "overcard"),
    # blank
    ("Kh 9c 4d 2s", "blank"),
    ("As Jd 7c 3h", "blank"),
    ("Qh 9c 5d 5s 2c", "blank"),
    ("Kc Jd 6h 3s 8d", "blank"),
    ("Ah Kd 7c 2h", "blank"),
    ("Qc 8d 3h 2s", "blank"),
]


@pytest.mark.parametrize("board,expected", CHANGE_CASES)
def test_board_change(board, expected):
    assert hc.board_change(board) == expected


def test_change_table_covers_every_class_on_turn_and_river():
    counts = Counter(expected for _, expected in CHANGE_CASES)
    assert set(counts) == set(hc.BOARD_CHANGES)
    assert min(counts.values()) >= 5
    for change in hc.BOARD_CHANGES:
        sizes = {len(parse_cards(b)) for b, e in CHANGE_CASES if e == change}
        assert sizes == {4, 5}, change


def test_board_functions_reject_wrong_sizes():
    with pytest.raises(ValueError):
        hc.board_change("Kh 7d 2c")
    with pytest.raises(ValueError):
        hc.board_texture("Kh 7d")


# ----------------------------------------------------------------------------
# Preflop
# ----------------------------------------------------------------------------

BUCKET_CASES = [
    ("As Ad", "premium"), ("Kh Kc", "premium"), ("Qd Qs", "premium"), ("Ah Kh", "premium"),
    ("Ac Kd", "premium"), ("AKo", "premium"),
    ("Jh Jd", "strong"), ("Ts Tc", "strong"), ("Ah Qh", "strong"), ("Ad Qc", "strong"),
    ("Ac Jc", "strong"), ("Kh Qh", "strong"), ("Ks Js", "strong"),
    ("9h 9d", "mid_pair"), ("8s 8c", "mid_pair"), ("7d 7h", "mid_pair"), ("99", "mid_pair"),
    ("77", "mid_pair"),
    ("6h 6d", "low_pair"), ("5s 5c", "low_pair"), ("4d 4h", "low_pair"), ("3c 3s", "low_pair"),
    ("2h 2d", "low_pair"),
    ("Ah Th", "ax_suited"), ("As 9s", "ax_suited"), ("Ad 5d", "ax_suited"), ("Ac 2c", "ax_suited"),
    ("A7s", "ax_suited"),
    ("Ah Jc", "ax_offsuit"), ("As Td", "ax_offsuit"), ("Ad 5c", "ax_offsuit"),
    ("Ac 2h", "ax_offsuit"), ("A9o", "ax_offsuit"),
    ("Kh Qd", "broadway"), ("Ks Jc", "broadway"), ("Kd Td", "broadway"), ("Qh Jh", "broadway"),
    ("Jc Ts", "broadway"), ("QTo", "broadway"),
    ("5h 4h", "suited_connector"), ("Ts 9s", "suited_connector"), ("8d 6d", "suited_connector"),
    ("5c 3c", "suited_connector"), ("Th 8h", "suited_connector"), ("76s", "suited_connector"),
    ("Kh 9d", "semi_trash"), ("Ks 2s", "semi_trash"), ("Qh 8h", "semi_trash"),
    ("4d 3d", "semi_trash"), ("Jc 9c", "semi_trash"), ("K5o", "semi_trash"), ("52s", "semi_trash"),
    ("Qh 9d", "trash"), ("9s 8c", "trash"), ("7d 2h", "trash"), ("Jc 3s", "trash"),
    ("6h 4d", "trash"), ("32o", "trash"),
]


@pytest.mark.parametrize("hand,expected", BUCKET_CASES)
def test_hand_bucket10(hand, expected):
    assert hc.hand_bucket10(hand) == expected


def test_bucket_table_and_class_counts():
    counts = Counter(expected for _, expected in BUCKET_CASES)
    assert set(counts) == set(hc.BUCKET10_CLASSES)
    assert min(counts.values()) >= 5
    per_bucket = Counter(hc.BUCKET10_BY_CLASS)
    assert per_bucket == {
        "premium": 5, "strong": 7, "mid_pair": 3, "low_pair": 5, "ax_suited": 9,
        "ax_offsuit": 10, "broadway": 10, "suited_connector": 12, "semi_trash": 56, "trash": 52,
    }
    for i in range(N_COMBOS):
        label = CLASS_LABELS[class_index(hc.hand_class169(COMBOS[i].tolist()))]
        assert hc.BUCKET10_CLASSES[hc.COMBO_BUCKET10[i]] == hc.hand_bucket10(label)


def test_hand_class169():
    assert hc.hand_class169("Kd Ah") == "AKo"
    assert hc.hand_class169([51, 47]) == "AKs"          # As Ks
    assert hc.hand_class169("7h 7d") == "77"


def test_preflop_ranking_and_percentile():
    assert sorted(hc.PREFLOP_RANKING) == sorted(CLASS_LABELS)
    assert hc.PREFLOP_RANKING[:4] == ("AA", "KK", "QQ", "JJ")
    assert hc.PREFLOP_RANKING[-1] == "32o"
    assert hc.chen_percentile("AA") == 1.0
    assert hc.chen_percentile("2c 3d") == 0.0
    assert hc.chen_percentile("As Ah") == 1.0
    pct = np.array([hc.CHEN_PERCENTILE[class_index(c)] for c in hc.PREFLOP_RANKING])
    assert np.allclose(pct, np.arange(168, -1, -1) / 168)
    top = np.array([hc.PREFLOP_TOP_SHARE[class_index(c)] for c in hc.PREFLOP_RANKING])
    assert top[0] == pytest.approx(6 / 1326) and top[-1] == pytest.approx(1.0)
    assert np.all(np.diff(top) > 0)
    assert hc.rank_classes(hc.CHEN_PERCENTILE) == hc.PREFLOP_RANKING
    assert np.allclose(hc.strength_percentile(hc.CHEN_PERCENTILE), hc.CHEN_PERCENTILE)


def test_strength_percentile_ties():
    values = np.zeros(169)
    values[0] = 1.0
    pct = hc.strength_percentile(values)
    assert pct[0] == 1.0
    assert np.allclose(pct[1:], 0.5 * 167 / 168)


def test_preflop_ranking_follows_exact_equity():
    from sudetect.equity import PREFLOP_VS_RANDOM

    eq = np.array([PREFLOP_VS_RANDOM[class_index(c)] for c in hc.PREFLOP_RANKING])
    assert np.all(np.diff(eq) < 0)                       # no two classes tie
    assert hc.PREFLOP_RANKING[:12] == (
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "77", "AQs", "AJs", "AKo")
    assert hc.PREFLOP_RANKING[-5:] == ("72o", "52o", "62o", "42o", "32o")
    known = {"AA": 0.8520, "KK": 0.8240, "AKs": 0.6704, "72o": 0.3458, "32o": 0.3230}
    for label, value in known.items():
        assert PREFLOP_VS_RANDOM[class_index(label)] == pytest.approx(value, abs=1e-4)


# ----------------------------------------------------------------------------
# Vectorised forms against the scalar reference
# ----------------------------------------------------------------------------

def test_scalar_strength_matches_evaluator():
    rng = np.random.default_rng(11)
    for k in (5, 6, 7):
        hands = np.array([rng.choice(52, k, replace=False) for _ in range(4000)])
        expected = eval_hands(hands)
        got = np.array([hc._strength(h.tolist()) for h in hands])
        assert np.array_equal(got, expected), k


@pytest.mark.parametrize("street", [3, 4, 5])
def test_all_combos_equal_scalar(street):
    rng = np.random.default_rng(100 + street)
    combos = COMBOS.tolist()
    for _ in range(200):
        board = rng.choice(52, street, replace=False).tolist()
        made, draw = hc.classes_for_all_combos(board)
        assert (made == -1).sum() == N_COMBOS - (52 - street) * (51 - street) // 2
        dead = set(board)
        for i, hole in enumerate(combos):
            if dead.intersection(hole):
                assert made[i] == -1 and draw[i] == -1
                continue
            m = hc._made_index(hole, board)
            d = hc._draw_index(hole, board, m)
            assert (made[i], draw[i]) == (m, d), (hole, board)


def test_scalar_public_functions_agree_with_private_ones():
    rng = np.random.default_rng(5)
    for _ in range(300):
        cards = rng.choice(52, 2 + rng.integers(3, 6), replace=False).tolist()
        hole, board = cards[:2], cards[2:]
        m = hc._made_index(hole, board)
        assert hc.made_hand(hole, board) == hc.MADE_CLASSES[m]
        assert hc.draw_class(hole, board) == hc.DRAW_CLASSES[hc._draw_index(hole, board, m)]


def test_all_combos_ignores_board_order_and_is_read_only():
    a = hc.classes_for_all_combos("Kh Qd 9h 2c")
    b = hc.classes_for_all_combos("2c 9h Kh Qd")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        a[0][0] = 3


def test_draws_are_none_on_the_river_for_every_combo():
    _, draw = hc.classes_for_all_combos("9h Th Jh 2c 3d")
    assert set(np.unique(draw)) <= {-1, 0}


def test_classify_batch_matches_scalar():
    rng = np.random.default_rng(3)
    n = 3000
    hole = np.empty((n, 2), dtype=np.int64)
    board = np.full((n, 5), -1, dtype=np.int64)
    sizes = rng.integers(0, 6, size=n)
    for i in range(n):
        cards = rng.choice(52, 7, replace=False)
        hole[i] = cards[:2]
        k = sizes[i]
        slots = np.sort(rng.choice(5, k, replace=False))   # padding anywhere in the row
        board[i, slots] = cards[2:2 + k]
    hole[::17] = -1                                      # unknown cards
    made, draw = hc.classify_batch(hole, board)
    for i in range(n):
        b = [c for c in board[i].tolist() if c >= 0]
        if hole[i, 0] < 0 or len(b) < 3:
            assert made[i] == -1 and draw[i] == -1
            continue
        h = hole[i].tolist()
        m = hc._made_index(h, b)
        assert (made[i], draw[i]) == (m, hc._draw_index(h, b, m))


def test_classify_batch_rejects_repeated_cards():
    with pytest.raises(ValueError):
        hc.classify_batch(np.array([[50, 45]]), np.array([[50, 3, 7, -1, -1]]))
    made, draw = hc.classify_batch(np.zeros((0, 2), int), np.zeros((0, 5), int))
    assert made.shape == (0,) and draw.shape == (0,)


def test_input_errors():
    with pytest.raises(ValueError):
        hc.made_hand("Ah", "Kd 7c 2s")
    with pytest.raises(ValueError):
        hc.made_hand("Ah Kd", "Kd 7c 2s")
    with pytest.raises(ValueError):
        hc.made_hand("Ah Kd", "7c 2s")
    with pytest.raises(ValueError):
        hc.draw_class("Ah Kd", "7c 2s 3d 4h 5s 6c")


def test_all_combos_speed():
    """A new board takes well under the 3 ms budget of the range model (about 0.5-1 ms here).

    Each board keeps the fastest of three runs, so that other processes sharing
    the machine do not decide the result.
    """
    rng = np.random.default_rng(9)
    raw = hc._all_combos.__wrapped__
    for street in (3, 4, 5):
        boards = [tuple(sorted(rng.choice(52, street, replace=False).tolist())) for _ in range(100)]
        raw(boards[0])
        total = 0.0
        for b in boards:
            runs = []
            for _ in range(3):
                start = time.perf_counter()
                raw(b)
                runs.append(time.perf_counter() - start)
            total += min(runs)
        per_board = total / len(boards)
        assert per_board < 0.005, f"street {street}: {per_board * 1000:.2f} ms per board"
