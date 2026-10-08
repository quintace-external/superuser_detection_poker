"""Card encoding and a dependency-free 5-7 card hand evaluator.

Cards are ints: rank * 4 + suit, rank 0..12 = 2..A, suit 0 diamonds, 1 clubs, 2 hearts, 3 spades
(the replay format's encoding; keys whose value is 0 are omitted in the JSON).
"""
from __future__ import annotations

RANKS = "23456789TJQKA"
SUITS = "dchs"
SUIT_SYMBOLS = "♦♣♥♠"
CATEGORY_NAMES = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                  "Flush", "Full House", "Four of a Kind", "Straight Flush"]


class InvalidCard(ValueError):
    pass


def decode(card: dict) -> int:
    """{'rank': 9, 'suit': 3} -> int (rank 0-12 for 2 ... A, suit 0-3). Missing keys mean 0."""
    n, s = card.get("rank", 0), card.get("suit", 0)
    if not (0 <= n <= 12 and 0 <= s <= 3):
        raise InvalidCard(f"invalid card {card}")
    return n * 4 + s


def to_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def to_pretty(cards) -> str:
    return " ".join(SUIT_SYMBOLS[c & 3] + ("10" if (c >> 2) == 8 else RANKS[c >> 2]) for c in cards)


def _straight_high(rset: set) -> int | None:
    for hi in range(12, 3, -1):
        if all((hi - k) in rset for k in range(5)):
            return hi
    if {12, 0, 1, 2, 3} <= rset:          # wheel
        return 3
    return None


def evaluate(cards) -> tuple:
    """Best hand from 5-7 cards. Returns a tuple, higher is better; element 0 is the category (0..8)."""
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    flush = None
    for s in range(4):
        if suits.count(s) >= 5:
            fr = sorted({r for r, su in zip(ranks, suits) if su == s}, reverse=True)
            sh = _straight_high(set(fr))
            if sh is not None:
                return (8, sh)
            flush = (5,) + tuple(fr[:5])
            break
    cnt: dict[int, int] = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if groups[0][1] == 4:
        return (7, groups[0][0], max(r for r in cnt if r != groups[0][0]))
    if groups[0][1] == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return flush
    sh = _straight_high(set(ranks))
    if sh is not None:
        return (4, sh)
    if groups[0][1] == 3:
        return (3, groups[0][0], *sorted((r for r in cnt if r != groups[0][0]), reverse=True)[:2])
    pairs = [r for r, c in groups if c == 2]
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1], max(r for r in cnt if r not in pairs[:2]))
    if len(pairs) == 1:
        return (1, pairs[0], *sorted((r for r in cnt if r != pairs[0]), reverse=True)[:3])
    return (0, *sorted(ranks, reverse=True)[:5])


def category_name(cards) -> str:
    return CATEGORY_NAMES[evaluate(cards)[0]] if len(cards) >= 5 else ""
