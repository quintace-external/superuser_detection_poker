"""Cards, two-card combos and the 169 preflop classes.

Card ids
--------
Every module in the package identifies a card by an integer from 0 to 51::

    card = rank * 4 + suit
    rank:  2  3  4  5  6  7  8  9  T  J  Q  K  A   ->  0 .. 12
    suit:  c  d  h  s                               ->  0 .. 3

so ``2c`` is 0, ``2d`` is 1, ``Ah`` is 50 and ``As`` is 51. Arrays that hold a
board use -1 for a card that has not been dealt yet. ``rank = card >> 2`` and
``suit = card & 3``.

Combos
------
A two-card holding is a *combo*. There are C(52, 2) = 1,326 of them and
:data:`COMBOS` lists them in a fixed order. Row ``i`` holds
``(higher card, lower card)`` and::

    i = hi * (hi - 1) // 2 + lo          (hi > lo, both card ids)

so the list is sorted by the higher card first and the lower card second:
``2d2c, 2h2c, 2h2d, 2s2c, 2s2d, 2s2h, 3c2c, ..., AsAh``. A range over the
opponent's holdings is a float array of 1,326 weights in this order.
:data:`COMBO_INDEX` (or :func:`combo_index`) goes from two cards back to the row.

Preflop classes
---------------
Ignoring suits, the 1,326 combos fall into 169 classes: 13 pairs (6 combos
each), 78 suited hands (4 combos each) and 78 offsuit hands (12 combos each).
:data:`CLASS_LABELS` lists them in this order::

    AA, KK, ..., 22,                     pairs, high to low      (index 0 .. 12)
    AKs, AQs, ..., A2s, KQs, ..., 32s    suited, high card first (index 13 .. 90)
    AKo, AQo, ..., A2o, KQo, ..., 32o    offsuit, same order     (index 91 .. 168)

:data:`COMBO_TO_CLASS` maps each combo to its class index and :data:`CLASS_SIZE`
holds the number of combos per class (6, 4 or 12).

Example::

    >>> parse_cards("Ah Kd")
    [50, 45]
    >>> format_cards([50, 45])
    'Ah Kd'
    >>> CLASS_LABELS[COMBO_TO_CLASS[combo_index(50, 45)]]
    'AKo'
    >>> int(CLASS_SIZE[class_index("AKs")])
    4
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Union

import numpy as np

__all__ = [
    "RANKS",
    "SUITS",
    "N_CARDS",
    "N_COMBOS",
    "N_CLASSES",
    "COMBOS",
    "COMBO_INDEX",
    "CLASS_LABELS",
    "COMBO_TO_CLASS",
    "CLASS_SIZE",
    "parse_card",
    "parse_cards",
    "format_card",
    "format_cards",
    "combo_index",
    "combos_blocked",
    "class_index",
    "class_combos",
    "hand_class",
    "range_weights",
]

RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_CARDS = 52
N_COMBOS = 1326
N_CLASSES = 169

CardsLike = Union[str, Iterable[Union[str, int, np.integer]], np.ndarray]


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


# ----------------------------------------------------------------------------
# Single cards
# ----------------------------------------------------------------------------

_CARD_TEXT = tuple(r + s for r in RANKS for s in SUITS)


def parse_card(text: str) -> int:
    """Parse one card such as ``"Ah"``, ``"td"`` or ``"10s"`` into its id (0..51).

    The rank is case-insensitive and ``10`` is accepted for a ten; the suit is
    one of ``c d h s`` in either case.

    Raises:
        ValueError: The text is not a card.
    """
    s = str(text).strip()
    if s[:2] == "10":
        s = "T" + s[2:]
    if len(s) != 2:
        raise ValueError(f"not a card: {text!r}")
    rank = RANKS.find(s[0].upper())
    suit = SUITS.find(s[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"not a card: {text!r}")
    return rank * 4 + suit


def _split_card_text(text: str) -> list[str]:
    """``"[Ah Kd]"``, ``"AhKd"`` or ``"Ah,Kd"`` -> ``["Ah", "Kd"]``."""
    out: list[str] = []
    for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split():
        while token:
            size = 3 if token.startswith("10") else 2
            out.append(token[:size])
            token = token[size:]
    return out


def parse_cards(cards: CardsLike) -> list[int]:
    """Parse several cards into a list of ids.

    Accepts a string (``"Ah Kd"``, ``"AhKd"``, ``"[Ah Kd]"``, ``"Ah,Kd"``), a
    sequence of card strings (``["Ah", "Kd"]``) or a sequence of card ids.
    Negative ids (undealt board slots) are dropped, so a padded board array
    can be passed as is. An empty string gives an empty list.

    Raises:
        ValueError: A token is not a card, or a card appears twice.
    """
    if isinstance(cards, str):
        out = [parse_card(t) for t in _split_card_text(cards)]
    else:
        out = []
        for c in cards.ravel().tolist() if isinstance(cards, np.ndarray) else cards:
            if isinstance(c, str):
                out.append(parse_card(c))
            elif isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)):
                c = int(c)
                if c >= N_CARDS:
                    raise ValueError(f"card id out of range 0..51: {c}")
                if c >= 0:
                    out.append(c)
            else:
                raise ValueError(f"not a card: {c!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate card in {format_cards(out)!r}")
    return out


def format_card(card: int) -> str:
    """Card id -> text, e.g. ``50 -> "Ah"``."""
    c = int(card)
    if not 0 <= c < N_CARDS:
        raise ValueError(f"card id out of range 0..51: {card!r}")
    return _CARD_TEXT[c]


def format_cards(cards: Iterable[int] | np.ndarray, sep: str = " ") -> str:
    """Card ids -> text, e.g. ``[50, 45] -> "Ah Kd"``. Negative ids are skipped."""
    seq = np.asarray(cards).ravel().tolist() if isinstance(cards, np.ndarray) else cards
    return sep.join(format_card(c) for c in seq if int(c) >= 0)


# ----------------------------------------------------------------------------
# Combos
# ----------------------------------------------------------------------------

def _build_combos() -> tuple[np.ndarray, np.ndarray]:
    combos = np.array([(hi, lo) for hi in range(N_CARDS) for lo in range(hi)], dtype=np.int8)
    index = np.full((N_CARDS, N_CARDS), -1, dtype=np.int16)
    rows = np.arange(N_COMBOS, dtype=np.int16)
    index[combos[:, 0], combos[:, 1]] = rows
    index[combos[:, 1], combos[:, 0]] = rows
    return _readonly(combos), _readonly(index)


#: (1326, 2) int8: row ``i`` = (higher card id, lower card id); see the module docstring.
#: COMBO_INDEX: (52, 52) int16 lookup from two card ids to the combo row; -1 on the diagonal.
COMBOS, COMBO_INDEX = _build_combos()


def combo_index(card1: int | np.ndarray, card2: int | np.ndarray) -> int | np.ndarray:
    """Row of the combo made of two cards (in either order); -1 if they are
    equal or either one is negative (an unknown card).

    Works element-wise on arrays of card ids.
    """
    c1, c2 = np.asarray(card1), np.asarray(card2)
    known = (c1 >= 0) & (c2 >= 0)
    # A bare COMBO_INDEX[c1, c2] would wrap a negative id round to a real card.
    out = np.where(known, COMBO_INDEX[np.where(known, c1, 0), np.where(known, c2, 0)], -1)
    return int(out) if np.ndim(out) == 0 else out


def combos_blocked(dead_cards: CardsLike) -> np.ndarray:
    """Boolean mask of the combos that share a card with ``dead_cards``.

    ``dead_cards`` is anything :func:`parse_cards` accepts; negative ids are
    ignored. A 2-D integer array of shape (n, m) gives one mask per row,
    shape (n, 1326).

    Example: two hole cards block 101 combos, hole cards plus a flop 245.
    """
    if isinstance(dead_cards, np.ndarray) and dead_cards.ndim == 2:
        d = dead_cards.astype(np.int64)
        dead = np.zeros((d.shape[0], N_CARDS + 1), dtype=bool)
        dead[np.arange(d.shape[0])[:, None], np.where(d < 0, N_CARDS, d)] = True
        dead = dead[:, :N_CARDS]
        return dead[:, COMBOS[:, 0]] | dead[:, COMBOS[:, 1]]
    dead = np.zeros(N_CARDS, dtype=bool)
    ids = parse_cards(dead_cards)
    if ids:
        dead[ids] = True
    return dead[COMBOS[:, 0]] | dead[COMBOS[:, 1]]


# ----------------------------------------------------------------------------
# Preflop classes
# ----------------------------------------------------------------------------

def _build_classes() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    high_first = RANKS[::-1]
    pairs = [r + r for r in high_first]
    unpaired = [high_first[i] + high_first[j] for i in range(13) for j in range(i + 1, 13)]
    labels = tuple(pairs + [h + "s" for h in unpaired] + [h + "o" for h in unpaired])
    position = {label: i for i, label in enumerate(labels)}
    to_class = np.empty(N_COMBOS, dtype=np.int16)
    for i, (hi, lo) in enumerate(COMBOS.tolist()):
        to_class[i] = position[_class_label(hi, lo)]
    size = np.array([6 if len(x) == 2 else (4 if x[2] == "s" else 12) for x in labels], dtype=np.int16)
    return labels, _readonly(to_class), _readonly(size)


def _class_label(card1: int, card2: int) -> str:
    r1, r2 = card1 >> 2, card2 >> 2
    if r1 == r2:
        return RANKS[r1] * 2
    hi, lo = (r1, r2) if r1 > r2 else (r2, r1)
    return RANKS[hi] + RANKS[lo] + ("s" if (card1 & 3) == (card2 & 3) else "o")


#: The 169 class labels in the order given in the module docstring.
#: COMBO_TO_CLASS: (1326,) int16 class index of each combo.
#: CLASS_SIZE: (169,) int16 number of combos in each class (6, 4 or 12).
CLASS_LABELS, COMBO_TO_CLASS, CLASS_SIZE = _build_classes()
_CLASS_POSITION = {label: i for i, label in enumerate(CLASS_LABELS)}


def class_index(label: str) -> int:
    """Index of a class label such as ``"AKs"``, ``"T9o"`` or ``"77"``.

    The two ranks may come in either order and in either case (``"kas"`` is
    ``"AKs"``).

    Raises:
        ValueError: Not one of the 169 labels.
    """
    s = str(label).strip()
    if len(s) in (2, 3) and s[0].upper() in RANKS and s[1].upper() in RANKS:
        r1, r2 = s[0].upper(), s[1].upper()
        if RANKS.index(r1) < RANKS.index(r2):
            r1, r2 = r2, r1
        key = r1 + r2 + s[2:].lower()
        if key in _CLASS_POSITION:
            return _CLASS_POSITION[key]
    raise ValueError(f"not a preflop class: {label!r} (examples: 'AA', 'AKs', 'T9o')")


def class_combos(label: str) -> np.ndarray:
    """Combo rows (indices into :data:`COMBOS`) of one preflop class, ascending.

    ``COMBOS[class_combos("AKs")]`` gives the four suited ace-king holdings.
    """
    return np.flatnonzero(COMBO_TO_CLASS == class_index(label))


def hand_class(cards: CardsLike) -> str:
    """Preflop class label of a two-card hand: ``"Kd Ah" -> "AKo"``."""
    ids = parse_cards(cards)
    if len(ids) != 2:
        raise ValueError(f"a hand has two cards, got {len(ids)}")
    return _class_label(ids[0], ids[1])


def range_weights(spec: Mapping[str, float] | Iterable[str]) -> np.ndarray:
    """Build a 1,326-weight range from class labels and/or explicit combos.

    ``spec`` maps entries to weights, or lists entries that all get weight 1.
    An entry is a class label (``"QQ"``, ``"AKs"``, ``"T9o"``), a rank pair
    without suffix meaning suited plus offsuit (``"AK"``), or one exact combo
    (``"AhKh"``). Later entries overwrite earlier ones on shared combos.

    Example::

        w = range_weights({"QQ": 1.0, "AK": 1.0, "AhKh": 0.5})
    """
    items = spec.items() if isinstance(spec, Mapping) else ((s, 1.0) for s in spec)
    w = np.zeros(N_COMBOS, dtype=np.float64)
    for entry, weight in items:
        text = str(entry).strip()
        if len(text) == 4 and text[1].lower() in SUITS and text[3].lower() in SUITS:
            c1, c2 = parse_cards(text)
            w[COMBO_INDEX[c1, c2]] = weight
        elif len(text) == 2 and text[0].upper() != text[1].upper():
            w[class_combos(text + "s")] = weight
            w[class_combos(text + "o")] = weight
        else:
            w[class_combos(text)] = weight
    return w
