"""Tests for sudetect.cards: card ids, combos and the 169 preflop classes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sudetect.cards import (
    CLASS_LABELS,
    CLASS_SIZE,
    COMBO_INDEX,
    COMBO_TO_CLASS,
    COMBOS,
    N_CARDS,
    N_CLASSES,
    N_COMBOS,
    RANKS,
    SUITS,
    class_combos,
    class_index,
    combo_index,
    combos_blocked,
    format_card,
    format_cards,
    hand_class,
    parse_card,
    parse_cards,
    range_weights,
)


# ---------------------------------------------------------------------------
# Single cards
# ---------------------------------------------------------------------------

def test_constants():
    assert RANKS == "23456789TJQKA"
    assert SUITS == "cdhs"
    assert (N_CARDS, N_COMBOS, N_CLASSES) == (52, 1326, 169)


def test_card_id_is_rank_times_four_plus_suit():
    for r, rank in enumerate(RANKS):
        for s, suit in enumerate(SUITS):
            assert parse_card(rank + suit) == r * 4 + s
    assert parse_card("2c") == 0
    assert parse_card("2d") == 1
    assert parse_card("Ah") == 50
    assert parse_card("As") == 51


def test_every_card_round_trips():
    for card in range(52):
        text = format_card(card)
        assert parse_card(text) == card
        assert len(text) == 2


@pytest.mark.parametrize("text, expected", [
    ("td", parse_card("Td")),
    ("10s", parse_card("Ts")),
    (" Ah ", 50),
    ("aH", 50),
    ("KC", parse_card("Kc")),
])
def test_parse_card_is_lenient_about_case_tens_and_spaces(text, expected):
    assert parse_card(text) == expected


@pytest.mark.parametrize("bad", ["", "A", "1h", "Ax", "Zh", "AhK", "11h", "hA"])
def test_parse_card_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_card(bad)


def test_format_card_rejects_out_of_range():
    for bad in (-1, 52, 100):
        with pytest.raises(ValueError):
            format_card(bad)


# ---------------------------------------------------------------------------
# Several cards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cards", [
    "Ah Kd", "AhKd", "[Ah Kd]", "Ah,Kd", "  ah   kd ", ["Ah", "Kd"], ("Ah", "Kd"),
    [50, 45], np.array([50, 45]), np.array([[50, 45]]), [np.int8(50), np.int64(45)],
])
def test_parse_cards_accepts_every_documented_form(cards):
    assert parse_cards(cards) == [50, 45]


def test_parse_cards_board_forms():
    assert parse_cards("Qh 7h 2c") == parse_cards("Qh7h2c") == [parse_card(c) for c in ("Qh", "7h", "2c")]
    assert parse_cards("10h 9h") == [parse_card("Th"), parse_card("9h")]
    assert parse_cards("") == []
    assert parse_cards([]) == []


def test_parse_cards_drops_undealt_slots():
    assert parse_cards(np.array([12, 30, 51, -1, -1])) == [12, 30, 51]
    assert parse_cards([-1, -1]) == []


@pytest.mark.parametrize("bad", ["Ah Ah", "AhKdAh", [50, 50], ["Ah", 50]])
def test_parse_cards_rejects_duplicates(bad):
    with pytest.raises(ValueError, match="duplicate"):
        parse_cards(bad)


@pytest.mark.parametrize("bad", ["Ah Kx", [52], [True, 3], [1.5], "AhK"])
def test_parse_cards_rejects_bad_entries(bad):
    with pytest.raises(ValueError):
        parse_cards(bad)


def test_format_cards():
    assert format_cards([50, 45]) == "Ah Kd"
    assert format_cards(np.array([50, 45, -1])) == "Ah Kd"
    assert format_cards([50, 45], sep="") == "AhKd"
    assert format_cards([]) == ""
    deck = list(range(52))
    assert parse_cards(format_cards(deck)) == deck


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------

def test_combos_layout():
    assert COMBOS.shape == (1326, 2)
    hi, lo = COMBOS[:, 0].astype(int), COMBOS[:, 1].astype(int)
    assert (hi > lo).all()
    assert (hi * (hi - 1) // 2 + lo == np.arange(1326)).all()
    # sorted by the higher card, then the lower card, with every pair once
    assert [tuple(r) for r in COMBOS.tolist()] == [(h, l) for h in range(52) for l in range(h)]
    assert format_cards(COMBOS[0]) == "2d 2c"
    assert format_cards(COMBOS[1]) == "2h 2c"
    assert format_cards(COMBOS[2]) == "2h 2d"
    assert format_cards(COMBOS[-1]) == "As Ah"


def test_combo_index_is_the_inverse_of_combos():
    assert COMBO_INDEX.shape == (52, 52)
    assert (np.diag(COMBO_INDEX) == -1).all()
    assert (COMBO_INDEX == COMBO_INDEX.T).all()
    rows = np.arange(1326)
    assert (COMBO_INDEX[COMBOS[:, 0], COMBOS[:, 1]] == rows).all()
    assert (COMBO_INDEX[COMBOS[:, 1], COMBOS[:, 0]] == rows).all()
    off_diagonal = COMBO_INDEX[~np.eye(52, dtype=bool)]
    assert sorted(set(off_diagonal.tolist())) == list(range(1326))


def test_combo_index_scalar_and_array():
    ah, kd = parse_cards("Ah Kd")
    i = combo_index(ah, kd)
    assert isinstance(i, int)
    assert i == combo_index(kd, ah)
    assert sorted(COMBOS[i].tolist()) == sorted([ah, kd])
    assert combo_index(7, 7) == -1
    arr = combo_index(np.array([50, 0, 3]), np.array([45, 1, 3]))
    assert arr.tolist() == [combo_index(50, 45), 0, -1]


def test_shared_tables_are_read_only():
    for table in (COMBOS, COMBO_INDEX, COMBO_TO_CLASS, CLASS_SIZE):
        with pytest.raises(ValueError):
            table[0] = 0


def test_combos_blocked_counts():
    assert not combos_blocked([]).any()
    assert combos_blocked("Ah").sum() == 51
    assert combos_blocked("Ah Kd").sum() == 101                 # 51 + 51 - 1
    assert combos_blocked("Ah Kd Qh 7h 2c").sum() == 245        # 5 * 51 - C(5, 2)
    assert combos_blocked(np.array([50, 45, -1, -1, -1])).sum() == 101


def test_combos_blocked_matches_brute_force():
    dead = parse_cards("Ah Kd 7c")
    mask = combos_blocked(dead)
    expected = [bool(set(c) & set(dead)) for c in COMBOS.tolist()]
    assert mask.tolist() == expected


def test_combos_blocked_two_dimensional():
    rows = np.array([[50, 45, -1], [0, 1, 2], [-1, -1, -1]])
    masks = combos_blocked(rows)
    assert masks.shape == (3, 1326)
    for row, mask in zip(rows, masks):
        assert (mask == combos_blocked(row)).all()


# ---------------------------------------------------------------------------
# Preflop classes
# ---------------------------------------------------------------------------

def test_class_label_order():
    assert len(CLASS_LABELS) == 169 == len(set(CLASS_LABELS))
    assert CLASS_LABELS[:13] == ("AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22")
    assert CLASS_LABELS[13:16] == ("AKs", "AQs", "AJs")
    assert CLASS_LABELS[13 + 11] == "A2s"
    assert CLASS_LABELS[13 + 12] == "KQs"
    assert CLASS_LABELS[90] == "32s"
    assert CLASS_LABELS[91] == "AKo"
    assert CLASS_LABELS[168] == "32o"
    assert all(x.endswith("s") for x in CLASS_LABELS[13:91])
    assert all(x.endswith("o") for x in CLASS_LABELS[91:])
    assert [x[:2] for x in CLASS_LABELS[13:91]] == [x[:2] for x in CLASS_LABELS[91:]]


def test_class_sizes():
    assert CLASS_SIZE.sum() == 1326
    assert (CLASS_SIZE[:13] == 6).all()
    assert (CLASS_SIZE[13:91] == 4).all()
    assert (CLASS_SIZE[91:] == 12).all()
    assert (np.bincount(COMBO_TO_CLASS, minlength=169) == CLASS_SIZE).all()


def test_combo_to_class_agrees_with_hand_class():
    for i, (a, b) in enumerate(COMBOS.tolist()):
        assert CLASS_LABELS[COMBO_TO_CLASS[i]] == hand_class([a, b])


@pytest.mark.parametrize("cards, label", [
    ("Kd Ah", "AKo"), ("Ah Kh", "AKs"), ("7h 7d", "77"), ("2c 3c", "32s"), ("Ts 9d", "T9o"),
])
def test_hand_class(cards, label):
    assert hand_class(cards) == label


def test_hand_class_needs_two_cards():
    with pytest.raises(ValueError):
        hand_class("Ah Kh Qh")


def test_class_index_is_lenient_and_strict():
    assert class_index("AKs") == 13
    assert class_index("kas") == class_index("KAs") == 13
    assert class_index("aa") == 0
    assert class_index("t9O") == class_index("T9o")
    for bad in ("AK", "AAs", "AKx", "XY", "", "AKs+", "A"):
        with pytest.raises(ValueError):
            class_index(bad)


def test_class_combos():
    aks = class_combos("AKs")
    assert len(aks) == 4
    assert sorted(format_cards(COMBOS[i]) for i in aks) == ["Ac Kc", "Ad Kd", "Ah Kh", "As Ks"]
    assert len(class_combos("AKo")) == 12
    pairs = class_combos("QQ")
    assert len(pairs) == 6 and all(c[0] >> 2 == c[1] >> 2 == 10 for c in COMBOS[pairs].tolist())
    assert (np.diff(aks) > 0).all()
    every = np.concatenate([class_combos(x) for x in CLASS_LABELS])
    assert sorted(every.tolist()) == list(range(1326))


def test_range_weights():
    w = range_weights({"QQ": 1.0, "AK": 1.0, "AhKh": 0.5})
    assert w.shape == (1326,)
    assert w.sum() == pytest.approx(6 + 15 + 0.5)
    assert w[combo_index(*parse_cards("Ah Kh"))] == 0.5
    assert w[combo_index(*parse_cards("Ad Kc"))] == 1.0
    w2 = range_weights(["AA", "72o"])
    assert w2.sum() == 18 and set(np.unique(w2).tolist()) == {0.0, 1.0}
    assert range_weights([]).sum() == 0


def test_every_two_card_hand_maps_to_one_combo():
    seen = set()
    for a, b in itertools.combinations(range(52), 2):
        seen.add(combo_index(a, b))
    assert seen == set(range(1326))
