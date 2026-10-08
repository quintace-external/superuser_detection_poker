"""Replay one parsed hand chip by chip and produce the rows of the three warehouse tables.

Input is a :class:`sudetect.hhparse.ParsedHand`; output a :class:`HandRows`
with one row for ``hands``, one per dealt player for ``hand_players`` and one
per post or betting line for ``actions``. Amounts are integer units inside the
replay and become big blinds (rounded to 1e-6) when a batch is converted to
Arrow (:class:`Batch`).

Rules (the numbers refer to the action-table conventions documented in
``01_action_table/README.md``):

* Every row stores the chips it moved (``amount``). Bets and raises are capped
  at the largest amount a live opponent can still match, computed with what is
  known at that moment and never revised (R7, R8)::

      cap          = max over live opponents j of (street_put_j + behind_j)
      raise_to_eff = min(street_put_i + amount, cap)
      amount_eff   = raise_to_eff - street_put_i

  An aggressive line whose capped total does not exceed the current level is
  a call (R9). ``stack_before`` is the actual stack; street puts, totals and
  the running pot are effective (R11).
* ``pot_before`` = every chip committed before the row, posts included (R3);
  ``to_call`` = level - own street put, not capped by the stack;
  ``to_call_eff`` = what a call puts in: ``to_call`` capped by the stack and,
  when every live opponent is all-in, by the largest of their street totals
  (R4). While another player can still act a call matches the full level, even
  the full big blind facing a short all-in blind; the part nobody matches comes
  back as the uncalled excess.
* ``size_pot = (amount_eff - to_call) / (pot_before + to_call)`` for bets and
  raises; ``size_x = raise_to_eff / level_before`` for raises (R2).
* ``pot_odds = to_call_eff / W`` with ``W = sum_j min(C_j, C_i + to_call_eff)``
  over every player, folded ones included, where ``C`` are the chips committed
  so far: the share of the pot the caller can win that he has to pay (R5).
  Dead money (the dead part of a missed-blind post, a dead button blind) sits in
  the main pot, so it is added whole and only the rest of ``C`` is capped.
* ``eff_stack = min(behind, cap - street_put)``, ``spr = eff_stack / pot_before`` (R10).
* The uncalled excess is the part of the last bet or raise that nobody matched:
  on the street where it was made, the maker's actual street put minus the
  largest other street put. Dead posts never count toward a street put, so a
  dead blind is never "returned" (R12). The final pot and the side pots follow
  from the actual contributions (R13); ``net = won + returned - committed``
  with the printed payouts, so the hand's nets plus rake and other drops sum to
  zero (R14).
* All-in is a flag (``amount == behind``), not an action (R15). Posts are rows;
  live posts count toward the street bet, dead posts only toward the pot (R17).
* End-of-hand values repeated on action rows (``went_to_showdown``,
  ``cards_shown``, ``net_bb``) are hindsight and never decision features (R25).

Hand classes (``made_hand``, ``draw``, ``hand_bucket10``, ``board_texture``,
``texture_class``, ``board_change``) come from :mod:`sudetect.handclass`,
imported on first use; when it cannot be imported the columns stay NULL. The
board columns and the preflop bucket are filled per hand; ``made_hand`` and
``draw`` are filled for a whole batch at once (:meth:`Batch.to_arrow`), with
the vectorised classifier.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pyarrow as pa

from . import situations as sit
from .cards import RANKS, SUITS
from .hhparse import ACT, DEAL, POST, PARSER_VERSION, ParsedHand
from .positions import assign_positions

__all__ = [
    "STR", "I16", "I32", "F64", "BOOL", "TS", "DATE", "UNITS", "OPPS",
    "HAND_COLUMNS",
    "PLAYER_COLUMNS",
    "ACTION_COLUMNS",
    "HARD_ERRORS",
    "IdMapper",
    "ReplayContext",
    "HandRows",
    "Batch",
    "replay_hand",
    "hand_class_columns",
    "made_and_draw",
    "handclass_available",
    "format_hole",
]

# column kinds
STR, I16, I32, F64, BOOL, TS, DATE, UNITS, OPPS = range(9)

STREET_NAMES = ("preflop", "flop", "turn", "river")

HAND_COLUMNS: tuple[tuple[str, int], ...] = (
    ("hand_id", STR), ("source", STR), ("table_id", STR), ("table_seq", I32), ("game", STR),
    ("variant", STR), ("table_size", I16), ("n_dealt", I16), ("started_at", TS), ("hand_date", DATE),
    ("sb_bb", UNITS), ("ante_bb", UNITS), ("straddle_bb", UNITS), ("n_straddles", I16),
    ("button_blind_bb", UNITS), ("button_seat", I16), ("sb_seat", I16), ("bb_seat", I16),
    ("board", STR), ("board_run2", STR), ("n_runs", I16), ("runout_street", STR),
    ("pot_flop_bb", UNITS), ("pot_turn_bb", UNITS), ("pot_river_bb", UNITS), ("final_pot_bb", UNITS),
    ("uncalled_bb", UNITS), ("uncalled_seat", I16), ("rake_bb", UNITS), ("fees_bb", UNITS),
    ("n_side_pots", I16), ("preflop_raises", I16), ("pot_type", STR), ("last_street", STR),
    ("players_flop", I16), ("players_turn", I16), ("players_river", I16),
    ("went_to_showdown", BOOL), ("n_showdown", I16), ("n_shown", I16), ("cards_visibility", STR),
    ("error_flag", BOOL), ("error_detail", STR), ("parser_version", STR),
)

PLAYER_COLUMNS: tuple[tuple[str, int], ...] = (
    ("hand_id", STR), ("player_id", STR), ("seat", I16), ("hand_date", DATE),
    ("position", STR), ("position_class", STR), ("is_sb", BOOL), ("is_bb", BOOL), ("is_straddle", BOOL),
    ("preflop_order", I16), ("postflop_order", I16),
    ("stack_start_bb", UNITS), ("eff_stack_start_bb", UNITS),
    ("ante_bb", UNITS), ("blind_bb", UNITS), ("dead_bb", UNITS), ("post_kind", STR),
    ("hole_cards", STR), ("hand_class", STR), ("hand_bucket10", STR), ("cards_known", BOOL),
    ("cards_source", STR),
    ("vpip", BOOL), ("pfr", BOOL), ("preflop_raises_made", I16),
    ("saw_flop", BOOL), ("saw_turn", BOOL), ("saw_river", BOOL),
    ("folded_street", STR), ("allin_street", STR), ("n_decisions", I16),
    ("committed_bb", UNITS), ("returned_bb", UNITS), ("won_bb", UNITS), ("net_bb", UNITS),
    ("rake_share_bb", UNITS), ("net_pre_rake_bb", UNITS),
    ("went_to_showdown", BOOL), ("cards_shown", BOOL), ("won_at_showdown", BOOL),
    ("error_flag", BOOL), ("error_detail", STR),
)

ACTION_COLUMNS: tuple[tuple[str, int], ...] = (
    ("hand_id", STR), ("action_index", I16), ("player_id", STR), ("seat", I16), ("hand_date", DATE),
    ("acted_at", TS), ("decision_s", F64), ("auto_kind", STR),
    ("street", STR), ("street_action_index", I16), ("player_street_round", I16),
    ("is_post", BOOL), ("post_kind", STR), ("action", STR), ("action_raw", STR),
    ("is_allin", BOOL), ("is_capped", BOOL), ("free_fold", BOOL), ("is_decision", BOOL),
    ("position", STR), ("position_class", STR), ("n_dealt", I16), ("table_players", I16),
    ("is_hu_table", BOOL), ("n_live", I16), ("n_live_start_street", I16), ("live_opponents", OPPS),
    ("n_behind", I16), ("is_ip", BOOL),
    ("last_aggressor_id", STR), ("prev_street_aggressor_id", STR), ("preflop_aggressor_id", STR),
    ("faced_bettor_id", STR),
    ("bets_before", I16), ("calls_before", I16), ("checks_before", I16), ("n_limpers", I16),
    ("pot_type_now", STR), ("situation", STR), ("line", STR),
    ("hole_cards", STR), ("hand_class", STR), ("hand_bucket10", STR), ("made_hand", STR), ("draw", STR),
    ("board_at_action", STR), ("board_texture", STR), ("texture_class", STR), ("board_change", STR),
    ("stack_before_bb", UNITS), ("amount_bb", UNITS), ("amount_eff_bb", UNITS),
    ("street_put_before_bb", UNITS), ("level_before_bb", UNITS), ("to_call_bb", UNITS),
    ("to_call_eff_bb", UNITS), ("raise_to_eff_bb", UNITS), ("pot_before_bb", UNITS), ("eff_stack_bb", UNITS),
    ("spr", F64), ("size_pot", F64), ("size_x", F64), ("faced_size_pot", F64), ("pot_odds", F64),
    ("went_to_showdown", BOOL), ("cards_shown", BOOL), ("net_bb", UNITS),
    ("error_flag", BOOL), ("error_detail", STR),
)

HARD_ERRORS: frozenset[str] = frozenset({
    "CALL_MISMATCH",          # a call moved a different amount than the capped amount to call
    "CHECK_FACING_BET",       # a check while chips were owed
    "HIDDEN_POST_INFERRED",   # an unprinted post was added to make the money reconcile
    "CONTRIBUTION_MISMATCH",  # the replay's actual contributions differ from the parser's
})
"""Error codes that set ``error_flag``; every other code in ``error_detail`` is informational."""

_INFO_CODE = {
    "run_it_multiple": "RIT",
    "negative_rake": "NEGATIVE_RAKE",
    "other_drops": "OTHER_DROPS",
    "uncalled_text_differs": "UNCALLED_TEXT_DIFFERS",
    "total_pot_differs": "TOTAL_POT_DIFFERS",
    "board_text_differs": "BOARD_TEXT_DIFFERS",
    "dead_button": "DEAD_BUTTON",
    "straddle": "STRADDLE",
    "blind_all_in_from_ante": "BLIND_ALL_IN_FROM_ANTE",
    "allin_marker_missing": "ALLIN_MARKER_MISSING",
    "time_zone_not_utc": "TIME_ZONE_NOT_UTC",
    "hidden_post": "HIDDEN_POST_INFERRED",
    "sb_from_seat_order": "SB_FROM_SEAT_ORDER",
    "bb_from_seat_order": "BB_FROM_SEAT_ORDER",
    "hu_button_not_sb": "HU_BUTTON_NOT_SB",
}

_POST_RANK = {"ante": 0, "dead": 1, "button_blind": 1, "post": 2, "straddle": 3, "sb": 4, "bb": 5}
"""Which post names a player's ``post_kind`` when he made several (the highest wins; inferred posts never)."""

_OPP_TYPE = pa.list_(pa.struct([
    ("player_id", pa.string()), ("seat", pa.int16()), ("behind_bb", pa.float64()),
    ("street_put_bb", pa.float64()), ("is_allin", pa.bool_()),
]))
_ARROW_TYPES = {
    STR: pa.string(), I16: pa.int16(), I32: pa.int32(), F64: pa.float64(), BOOL: pa.bool_(),
    TS: pa.timestamp("us"), DATE: pa.date32(), UNITS: pa.float64(), OPPS: _OPP_TYPE,
}

_NUMPY_KINDS = {BOOL: np.bool_, I16: np.int16, I32: np.int32, F64: np.float64}
"""Kinds converted through numpy when a column has no NULL (much faster than from Python objects)."""

_CARD_TEXT = tuple(r + s for r in RANKS for s in SUITS)


def format_hole(cards: Sequence[int]) -> str:
    """Two hole cards as four characters, higher card first: ``(45, 50) -> "AhKd"``."""
    a, b = (cards[0], cards[1]) if cards[0] > cards[1] else (cards[1], cards[0])
    return _CARD_TEXT[a] + _CARD_TEXT[b]


def _class169(cards: Sequence[int]) -> str:
    a, b = cards
    ra, rb = a >> 2, b >> 2
    if ra == rb:
        return RANKS[ra] * 2
    hi, lo = (ra, rb) if ra > rb else (rb, ra)
    return RANKS[hi] + RANKS[lo] + ("s" if (a & 3) == (b & 3) else "o")


# ------------------------------------------------------------------ hand classes

_HANDCLASS: Any = None


def handclass_available() -> bool:
    """True when :mod:`sudetect.handclass` can be imported."""
    return _load_handclass() is not None


def _load_handclass() -> Any:
    global _HANDCLASS
    if _HANDCLASS is None:
        try:
            from . import handclass as module
            for name in ("made_hand", "draw_class", "hand_bucket10", "board_texture", "texture_class",
                         "board_change", "classify_batch", "MADE_CLASSES", "DRAW_CLASSES"):
                getattr(module, name)
        except (ImportError, AttributeError):
            _HANDCLASS = False
        else:
            _HANDCLASS = module
    return _HANDCLASS or None


_BUCKET_CACHE: dict[str, str] = {}


def _bucket_of(label: str) -> Optional[str]:
    """10-bucket of a 169-class label (cached), or None without :mod:`sudetect.handclass`."""
    got = _BUCKET_CACHE.get(label)
    if got is None:
        hc = _load_handclass()
        if hc is None:
            return None
        got = _BUCKET_CACHE[label] = hc.hand_bucket10(label)
    return got


_FLOP_CACHE: dict[tuple[int, ...], tuple[str, str]] = {}
_CHANGE_CACHE: dict[tuple[int, ...], str] = {}


def _board_columns(board: tuple[int, ...]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(board_texture, texture_class, board_change)`` of a board of 3-5 cards (cached)."""
    hc = _load_handclass()
    if hc is None or len(board) < 3:
        return None, None, None
    flop = board[:3]
    tex = _FLOP_CACHE.get(flop)
    if tex is None:
        tex = _FLOP_CACHE[flop] = (hc.board_texture(list(flop)), hc.texture_class(list(flop)))
    change = None
    if len(board) >= 4:
        change = _CHANGE_CACHE.get(board)
        if change is None:
            if len(_CHANGE_CACHE) > 1_000_000:
                _CHANGE_CACHE.clear()
            change = _CHANGE_CACHE[board] = hc.board_change(list(board))
    return tex[0], tex[1], change


def made_and_draw(hole: "np.ndarray", board: "np.ndarray") -> tuple[list[Optional[str]], list[Optional[str]]]:
    """Made-hand and draw names for many rows (``hole`` (n, 2), ``board`` (n, 5) with -1 padding).

    Rows with unknown cards or fewer than three board cards get None; all
    rows get None when :mod:`sudetect.handclass` is not available.
    """
    hc = _load_handclass()
    n = len(hole)
    if hc is None or n == 0:
        return [None] * n, [None] * n
    made, draw = hc.classify_batch(hole, board)
    made_names = (None,) + tuple(hc.MADE_CLASSES)
    draw_names = (None,) + tuple(hc.DRAW_CLASSES)
    return [made_names[i + 1] for i in made.tolist()], [draw_names[i + 1] for i in draw.tolist()]


def hand_class_columns(hole: Optional[Sequence[int]], board: Sequence[int]) -> tuple[Optional[str], ...]:
    """``(hand_bucket10, made_hand, draw, board_texture, texture_class, board_change)`` for one row.

    ``hole`` is None when the actor's cards are unknown (the first three come
    back NULL); ``board`` is the board seen at the row (0, 3, 4 or 5 cards).
    Every value is NULL when :mod:`sudetect.handclass` is not available.
    """
    hc = _load_handclass()
    if hc is None:
        return (None,) * 6
    k = len(board)
    bucket = hc.hand_bucket10(list(hole)) if hole is not None else None
    made = draw = texture = tclass = change = None
    if k >= 3:
        flop = list(board[:3])
        texture = hc.board_texture(flop)
        tclass = hc.texture_class(flop)
        if k >= 4:
            change = hc.board_change(list(board))
        if hole is not None:
            made = hc.made_hand(list(hole), list(board))
            draw = hc.draw_class(list(hole), list(board))
    return bucket, made, draw, texture, tclass, change


# ------------------------------------------------------------------ ids


class IdMapper:
    """Player, table and hand ids: as written, or salted HMAC-SHA256 hashes.

    With a salt, ``player("name")`` is ``"P_"`` + the first ``player_hex``
    hex digits of ``HMAC-SHA256(salt, "player:" + name)``; tables use ``"T_"``
    and hands ``"H_"``. The salt never leaves the process.
    """

    def __init__(self, salt: Optional[bytes] = None, player_hex: int = 12, table_hex: int = 12,
                 hand_hex: int = 16) -> None:
        self.salt = salt
        self._widths = {"player": player_hex, "table": table_hex, "hand": hand_hex}
        self._prefix = {"player": "P_", "table": "T_", "hand": "H_"}
        self._cache: dict[tuple[str, str], str] = {}

    def _digest(self, kind: str, text: str) -> str:
        assert self.salt is not None
        digest = hmac.new(self.salt, f"{kind}:{text}".encode("utf-8"), hashlib.sha256).hexdigest()
        return self._prefix[kind] + digest[: self._widths[kind]]

    def _hash(self, kind: str, text: str) -> str:
        """Hash of a name that repeats across hands (players, tables), cached."""
        if self.salt is None:
            return text
        key = (kind, text)
        out = self._cache.get(key)
        if out is None:
            out = self._digest(kind, text)
            if len(self._cache) > 2_000_000:
                self._cache.clear()
            self._cache[key] = out
        return out

    def player(self, name: str) -> str:
        return self._hash("player", name)

    def table(self, name: str) -> str:
        return self._hash("table", name)

    def hand(self, hand_id: str) -> str:
        return hand_id if self.salt is None else self._digest("hand", hand_id)


@dataclass
class ReplayContext:
    """What the replay needs besides the hand: the data source label and the id mapper."""

    source: str = "sim"
    ids: IdMapper = field(default_factory=IdMapper)
    hand_classes: bool = True


@dataclass
class HandRows:
    """The rows of one hand; every row tuple ends with the big blind in units (dropped on insert).

    ``class_keys`` is aligned with ``actions``: ``(hole1, hole2, board...)`` card ids of a
    postflop betting row whose actor's cards are known, else None. :class:`Batch` uses them
    to fill ``made_hand`` and ``draw``.
    """

    hand: tuple
    players: list[tuple]
    actions: list[tuple]
    class_keys: list[Optional[tuple]] = field(default_factory=list)


# ------------------------------------------------------------------ replay


def _board_text(cards: Sequence[int]) -> str:
    return "".join([_CARD_TEXT[c] for c in cards])


def _street_excess(street_put: Sequence[int]) -> tuple[int, int]:
    """``(player, chips)``: the largest street put and the part of it nobody matched (0 on a tie)."""
    top = max(range(len(street_put)), key=street_put.__getitem__)
    second = max((x for q, x in enumerate(street_put) if q != top), default=0)
    return top, street_put[top] - second


def _runout_street(run1: Sequence[int], run2: Sequence[int]) -> Optional[str]:
    for k, name in ((3, "flop"), (4, "turn"), (5, "river")):
        if list(run1[:k]) != list(run2[:k]):
            return name
    return None


def replay_hand(ph: ParsedHand, ctx: Optional[ReplayContext] = None) -> HandRows:
    """Replay one hand into the rows of ``hands``, ``hand_players`` and ``actions``."""
    ctx = ctx or ReplayContext()
    ids = ctx.ids
    n = len(ph.players)
    bb = ph.bb
    pids = [ids.player(x) for x in ph.players]
    hid = ids.hand(ph.hand_id)
    seats = ph.seats
    hand_date = ph.started_at.date()
    pos = assign_positions(seats, ph.button_seat, ph.max_seats, ph.sb_player, ph.bb_player, ph.straddlers)
    labels, classes = pos.labels, pos.classes
    pre_rank, post_rank = pos.preflop_order, pos.postflop_order
    hu_table = n == 2
    use_classes = ctx.hand_classes and _load_handclass() is not None

    # ---- cards
    hole: dict[int, tuple[int, int]] = dict(ph.dealt_cards)
    cards_source: dict[int, str] = {p: "dealt" for p in hole}
    for p, cs in ph.shown_cards.items():
        if p not in hole:
            hole[p] = cs
            cards_source[p] = "showdown" if p in ph.shown_at_showdown else "shown"
    hole_txt = {p: format_hole(cs) for p, cs in hole.items()}
    hole_cls = {p: _class169(cs) for p, cs in hole.items()}
    board = ph.boards[0]
    full_txt = _board_text(board)
    board_txt = ["", full_txt[:6], full_txt[:8], full_txt[:10]]

    # ---- end-of-hand facts that the text already settles (hindsight columns)
    folded_end = {e[1] for e in ph.events if e[0] == ACT and e[2] == "fold"}
    n_live_end = n - len(folded_end)
    went_sd = [p not in folded_end and n_live_end >= 2 for p in range(n)]
    shown_any = [p in ph.shown_cards for p in range(n)]
    net_u = [ph.collected[p] + ph.returned[p] - ph.contributed[p] for p in range(n)]

    # ---- hand classes: preflop bucket per player, board columns per street; made hand and draw
    # are filled per batch from class_keys (Batch.to_arrow)
    if use_classes:
        hole_bucket = {p: _bucket_of(label) for p, label in hole_cls.items()}
        board_cols = [(None, None, None)] + [
            _board_columns(tuple(board[:k])) if len(board) >= k else (None, None, None) for k in (3, 4, 5)
        ]
    else:
        hole_bucket = {}
        board_cols = [(None, None, None)] * 4
    class_keys: list[Optional[tuple]] = []

    # ---- state
    stack = list(ph.stacks)          # actual chips behind
    put_st = [0] * n                 # effective street put
    put_tot = [0] * n                # effective chips committed in the hand
    put_act = [0] * n                # actual chips committed in the hand
    st_act = [0] * n                 # actual street put (live posts and betting lines; dead posts never)
    uncalled = 0                     # chips of the last bet or raise that nobody matched
    uncalled_player: Optional[int] = None
    folded = [False] * n
    n_folded = 0
    level = 0
    pot = 0
    street = 0
    floor_done = False
    codes: set[str] = set()

    st_aggr = 0                      # effective bets and raises on this street
    st_last_aggr = -1
    st_last_size: Optional[float] = None
    st_first_spot: Optional[str] = None
    st_checked = [False] * n
    st_acted = [False] * n
    st_calls = st_checks = st_limps = st_calls_after = 0
    st_vol = 0
    st_rounds = [0] * n
    had_bet = [False, False, False, False]
    prev_aggr = -1
    pf_aggr = -1
    pf_raises = 0
    pf_pot_type = "limped"
    n_live_start = n
    ip_player = -1
    street_pot: dict[int, int] = {}
    players_at = [n, 0, 0, 0]
    saw = [[True] * n, [False] * n, [False] * n, [False] * n]

    vpip = [False] * n
    pfr = [False] * n
    raises_made = [0] * n
    folded_street: list[Optional[str]] = [None] * n
    allin_street: list[Optional[str]] = [None] * n
    n_dec = [0] * n
    ante_u = [0] * n
    blind_u = [0] * n
    dead_u = [0] * n
    post_kind_of: list[Optional[str]] = [None] * n
    is_sb = [False] * n
    is_bb = [False] * n
    is_straddle = [False] * n
    straddle_max = 0
    n_straddles = 0
    button_blind = 0
    ante_posted = 0
    pre_voluntary = 0

    rows: list[tuple] = []
    aidx = 0
    STREET = STREET_NAMES

    for ev in ph.events:
        etype = ev[0]
        if etype == DEAL:
            had_bet[street] = st_aggr > 0
            prev_aggr = st_last_aggr
            if street == 0:
                pf_pot_type = sit.pot_type(pf_raises)
            street = ev[1]
            excess_player, excess = _street_excess(st_act)
            if excess > 0:
                uncalled += excess
                uncalled_player = excess_player
            st_act = [0] * n
            street_pot[street] = sum(put_act) - uncalled
            live_now = [p for p in range(n) if not folded[p]]
            players_at[street] = len(live_now)
            row_saw = saw[street]
            for p in live_now:
                row_saw[p] = True
            n_live_start = len(live_now)
            active = [p for p in live_now if stack[p] > 0]
            ip_player = max(active, key=lambda q: post_rank[q]) if active else -1
            put_st = [0] * n
            level = 0
            st_aggr = 0
            st_last_aggr = -1
            st_last_size = None
            st_first_spot = None
            st_checked = [False] * n
            st_acted = [False] * n
            st_calls = st_checks = st_limps = st_calls_after = 0
            st_vol = 0
            st_rounds = [0] * n
            continue

        aidx += 1
        p = ev[1]
        behind = stack[p]
        c_i = put_st[p]
        if etype == POST:
            _, _, kind, amt, live, _mark = ev
            is_allin = amt > 0 and amt == behind
            bucket = hole_bucket.get(p)
            class_keys.append(None)
            rows.append((
                hid, aidx, pids[p], seats[p], hand_date,
                None, None, None,
                STREET[street], None, None,
                True, kind, "post", "post",
                is_allin, False, False, False,
                labels[p], classes[p], n, n, hu_table,
                n - n_folded, n_live_start, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None,
                hole_txt.get(p), hole_cls.get(p), bucket, None, None,
                None, None, None, None,
                behind, amt, amt, c_i, level, None,
                None, None, pot, None,
                None, None, None, None, None,
                went_sd[p], shown_any[p], net_u[p],
            ))
            stack[p] = behind - amt
            put_tot[p] += amt
            put_act[p] += amt
            pot += amt
            if live:
                st_act[p] += live
                put_st[p] = c_i + live
                if put_st[p] > level:
                    level = put_st[p]
            if is_allin:
                allin_street[p] = STREET[street]
            if kind == "ante":
                ante_u[p] += amt
                ante_posted = max(ante_posted, amt)
            elif kind in ("dead", "button_blind"):
                dead_u[p] += amt
                if kind == "button_blind":
                    button_blind += amt
            else:
                blind_u[p] += live
                if amt > live:
                    dead_u[p] += amt - live
            if kind == "sb":
                is_sb[p] = True
            elif kind == "bb":
                is_bb[p] = True
            elif kind == "straddle":
                is_straddle[p] = True
                n_straddles += 1
                straddle_max = max(straddle_max, amt)
            if _POST_RANK.get(kind, -1) > _POST_RANK.get(post_kind_of[p], -1):
                post_kind_of[p] = kind
            continue

        # ---- a betting line
        if not floor_done:
            floor_done = True
            if street == 0 and ph.preflop_level > level:
                level = ph.preflop_level
        _, _, verb, added, _mark, auto, _tamt, _tto = ev
        to_call = level - c_i if level > c_i else 0
        cap = 0
        n_live = 1
        n_behind = 0
        others_active = False
        rank = pre_rank if street == 0 else post_rank
        my_rank = rank[p]
        opps = []
        for q in range(n):
            if q == p or folded[q]:
                continue
            n_live += 1
            sq = stack[q]
            cq = put_st[q] + sq
            if cq > cap:
                cap = cq
            opps.append((pids[q], seats[q], sq, put_st[q], sq == 0))
            if sq > 0:
                others_active = True
                if rank[q] > my_rank:
                    n_behind += 1
        room = cap - c_i if cap > c_i else 0
        to_call_eff = to_call
        if behind < to_call_eff:
            to_call_eff = behind
        if not others_active and room < to_call_eff:
            # every live opponent is all-in: only their largest street total has to be matched
            to_call_eff = room
        eff_stack = behind if behind < room else room
        pot_before = pot
        if to_call_eff > 0:
            t_i = put_tot[p] + to_call_eff
            reach = t_i - dead_u[p]      # what the caller matches; dead money is won whole
            w = t_i
            for q in range(n):
                if q != p:
                    cq = put_tot[q] - dead_u[q]
                    w += (cq if cq < reach else reach) + dead_u[q]
            pot_odds: Optional[float] = to_call_eff / w
        else:
            pot_odds = None
        spr = eff_stack / pot_before if pot_before > 0 else None
        facing = to_call > 0

        # situation before the row
        if street == 0:
            situation = sit.preflop_situation(pf_raises, st_limps, st_calls_after)
            pot_now = sit.pot_type(pf_raises, st_limps, True)
            line = None
            is_ip = None
        else:
            if facing:
                situation = sit.facing_situation(st_aggr, st_first_spot)
            else:
                agg = prev_aggr
                active = agg >= 0 and not folded[agg] and stack[agg] > 0
                situation = sit.postflop_spot(
                    agg == p,
                    active,
                    active and st_checked[agg],
                    active and post_rank[agg] > post_rank[p] and not st_acted[agg],
                )
            pot_now = pf_pot_type
            line = sit.line_prefix(street, had_bet)
            is_ip = p == ip_player

        # the action, effective
        action = verb
        amount_eff = added
        raise_to_eff: Optional[int] = None
        size_pot: Optional[float] = None
        size_x: Optional[float] = None
        if verb == "bet" or verb == "raise":
            rt = c_i + added
            if rt > cap:
                rt = cap
            if rt <= level:
                action = "call" if to_call_eff > 0 else "check"
                amount_eff = to_call_eff
            else:
                raise_to_eff = rt
                amount_eff = rt - c_i
                action = "bet" if level == 0 else "raise"
                denom = pot_before + to_call
                size_pot = (amount_eff - to_call) / denom if denom > 0 else None
                size_x = rt / level if level > 0 else None
        elif verb == "call":
            if added != to_call_eff:
                codes.add("CALL_MISMATCH")
        elif verb == "check":
            if to_call_eff > 0:
                codes.add("CHECK_FACING_BET")
        is_allin = added > 0 and added == behind
        is_decision = auto != "timeout"
        free_fold = verb == "fold" and to_call == 0
        st_vol += 1
        st_rounds[p] += 1
        bucket = hole_bucket.get(p)
        texture, tclass, change = board_cols[street]
        if street > 0 and use_classes and p in hole:
            cs = hole[p]
            class_keys.append((cs[0], cs[1]) + tuple(board[: (0, 3, 4, 5)[street]]))
        else:
            class_keys.append(None)
        rows.append((
            hid, aidx, pids[p], seats[p], hand_date,
            None, None, auto,
            STREET[street], st_vol, st_rounds[p],
            False, None, action, verb,
            is_allin, amount_eff < added, free_fold, is_decision,
            labels[p], classes[p], n, n, hu_table,
            n_live, n_live_start, opps, n_behind, is_ip,
            pids[st_last_aggr] if st_last_aggr >= 0 else None,
            pids[prev_aggr] if street > 0 and prev_aggr >= 0 else None,
            pids[pf_aggr] if pf_aggr >= 0 else None,
            pids[st_last_aggr] if facing and st_last_aggr >= 0 else None,
            st_aggr, st_calls, st_checks, st_limps,
            pot_now, situation, line,
            hole_txt.get(p), hole_cls.get(p), bucket, None, None,
            board_txt[street], texture, tclass, change,
            behind, added, amount_eff, c_i, level, to_call,
            to_call_eff, raise_to_eff, pot_before, eff_stack,
            spr, size_pot, size_x, st_last_size if facing else None, pot_odds,
            went_sd[p], shown_any[p], net_u[p],
        ))

        # apply
        stack[p] = behind - added
        put_st[p] = c_i + amount_eff
        put_tot[p] += amount_eff
        put_act[p] += added
        st_act[p] += added
        pot += amount_eff
        if action == "bet" or action == "raise":
            level = c_i + amount_eff
            if st_aggr == 0:
                st_first_spot = situation
            st_aggr += 1
            st_last_aggr = p
            st_last_size = size_pot
            st_calls_after = 0
            if street == 0:
                pf_raises += 1
                pf_aggr = p
                pfr[p] = True
                raises_made[p] += 1
        elif action == "call":
            st_calls += 1
            if st_aggr == 0:
                st_limps += 1
            else:
                st_calls_after += 1
        elif action == "check":
            st_checks += 1
            st_checked[p] = True
        else:  # fold
            folded[p] = True
            n_folded += 1
            folded_street[p] = STREET[street]
        if street == 0 and action in ("call", "bet", "raise"):
            vpip[p] = True
            pre_voluntary += 1
        st_acted[p] = True
        if is_allin:
            allin_street[p] = STREET[street]
        if is_decision:
            n_dec[p] += 1

    had_bet[street] = st_aggr > 0
    if street == 0:
        pf_pot_type = sit.pot_type(pf_raises)
    excess_player, excess = _street_excess(st_act)
    if excess > 0:
        uncalled += excess
        uncalled_player = excess_player

    # ---- hand level
    if put_act != list(ph.contributed):
        codes.add("CONTRIBUTION_MISMATCH")
    final_pot = sum(put_act) - uncalled
    # side-pot layers come from the matched chips; dead money goes to the main pot
    matched = [put_act[q] - dead_u[q] for q in range(n)]
    if uncalled_player is not None:
        matched[uncalled_player] -= uncalled
    live_end = [p for p in range(n) if not folded[p]]
    levels = sorted({matched[p] for p in live_end})
    n_pots = 0
    previous = 0
    for lv in levels:
        if sum(min(c, lv) - previous for c in matched if c > previous) > 0:
            n_pots += 1
        previous = lv
    if sum(ph.returned) != uncalled:
        codes.add("UNCALLED_TEXT_DIFFERS")
    for flag in ph.flags | pos.flags:
        code = _INFO_CODE.get(flag)
        if code is not None:
            codes.add(code)
    error_flag = bool(codes & HARD_ERRORS)
    error_detail = ",".join(sorted(codes)) if codes else None
    tail = (error_flag, error_detail, bb)
    actions = [r + tail for r in rows]

    won_total = sum(ph.collected)
    rake = ph.rake
    players: list[tuple] = []
    max_other = [max(ph.stacks[q] for q in range(n) if q != p) for p in range(n)]
    for p in range(n):
        won = ph.collected[p]
        share = rake * won / won_total if won_total else 0.0
        cards = hole.get(p)
        bucket = hole_bucket.get(p)
        players.append((
            hid, pids[p], seats[p], hand_date,
            labels[p], classes[p], is_sb[p] or (hu_table and p == pos.button and is_sb[p]), is_bb[p],
            is_straddle[p], pre_rank[p], post_rank[p],
            ph.stacks[p], min(ph.stacks[p], max_other[p]),
            ante_u[p], blind_u[p], dead_u[p], post_kind_of[p],
            hole_txt.get(p), hole_cls.get(p), bucket, cards is not None, cards_source.get(p),
            vpip[p], pfr[p], raises_made[p],
            saw[1][p], saw[2][p], saw[3][p],
            folded_street[p], allin_street[p], n_dec[p],
            put_act[p], ph.returned[p], won, net_u[p],
            share, net_u[p] + share,
            went_sd[p], shown_any[p], went_sd[p] and won > 0,
            error_flag, error_detail, bb,
        ))

    if street == 0 and n_live_end == 1 and pre_voluntary == 0:
        pot_type = "walk"
    else:
        pot_type = pf_pot_type
    runs = [b for b in ph.boards if b]
    run2 = ph.boards[1] if len(ph.boards) > 1 and ph.boards[1] else None
    hand = (
        hid, ctx.source, ids.table(ph.table_name), None, "NLHE",
        ph.variant, ph.max_seats, n, ph.started_at, hand_date,
        ph.sb, ph.ante if ph.ante > 0 else ante_posted, straddle_max or None, n_straddles,
        button_blind or None, ph.button_seat,
        seats[pos.sb] if pos.sb is not None else None, seats[pos.bb] if pos.bb is not None else None,
        _board_text(board), _board_text(run2) if run2 else None, max(1, len(runs)),
        _runout_street(board, run2) if run2 else None,
        street_pot.get(1), street_pot.get(2), street_pot.get(3), final_pot,
        uncalled, seats[uncalled_player] if uncalled_player is not None else None, rake, ph.fees,
        max(0, n_pots - 1), pf_raises, pot_type, STREET[street],
        players_at[1], players_at[2], players_at[3],
        n_live_end >= 2, n_live_end if n_live_end >= 2 else 0, len(ph.shown_cards),
        "full" if len(ph.dealt_cards) == n else "showdown",
        error_flag, error_detail, PARSER_VERSION, bb,
    )
    return HandRows(hand, players, actions, class_keys)


# ------------------------------------------------------------------ batches


def _opponents_array(values: Sequence[Optional[list]], bb: np.ndarray) -> pa.Array:
    """List-of-struct column from per-row lists of ``(player_id, seat, behind, street_put, all_in)``
    with amounts in units; amounts become big blinds here, vectorised."""
    offsets = np.zeros(len(values) + 1, dtype=np.int32)
    valid = np.ones(len(values), dtype=bool)
    flat: list[tuple] = []
    row_bb: list[float] = []
    for i, entries in enumerate(values):
        if entries is None:
            valid[i] = False
        else:
            flat.extend(entries)
            row_bb.extend([bb[i]] * len(entries))
        offsets[i + 1] = len(flat)
    if flat:
        pid, seat, behind, put, allin = zip(*flat)
    else:
        pid = seat = behind = put = allin = ()
    scale = np.asarray(row_bb, dtype=np.float64)
    struct = pa.StructArray.from_arrays(
        [
            pa.array(pid, type=pa.string()),
            pa.array(seat, type=pa.int16()),
            pa.array(np.round(np.asarray(behind, dtype=np.float64) / scale, 6) if flat else [], type=pa.float64()),
            pa.array(np.round(np.asarray(put, dtype=np.float64) / scale, 6) if flat else [], type=pa.float64()),
            pa.array(allin, type=pa.bool_()),
        ],
        fields=list(_OPP_TYPE.value_type),
    )
    mask = None if valid.all() else pa.array(~valid)
    return pa.ListArray.from_arrays(pa.array(offsets), struct, type=_OPP_TYPE, mask=mask)


def _column_arrays(columns: Sequence[tuple[str, int]], rows: list[tuple],
                   replace: Optional[dict[str, list]] = None) -> pa.Table:
    names = [c[0] for c in columns]
    if not rows:
        return pa.table({name: pa.array([], type=_ARROW_TYPES[kind]) for name, kind in columns})
    cols = list(zip(*rows))
    if len(cols) != len(columns) + 1:
        raise ValueError(f"rows have {len(cols) - 1} values, expected {len(columns)}")
    bb = np.asarray(cols[-1], dtype=np.float64)
    arrays = []
    for k, (name, kind) in enumerate(columns):
        values = replace[name] if replace and name in replace else cols[k]
        if kind == UNITS:
            arr = np.array(values, dtype=np.float64)
            arr = np.round(arr / bb, 6)
            mask = np.isnan(arr)
            arrays.append(pa.array(arr, type=pa.float64(), mask=mask if mask.any() else None))
        elif kind == OPPS:
            arrays.append(_opponents_array(values, bb))
        elif kind in _NUMPY_KINDS and None not in values:
            arrays.append(pa.array(np.array(values, dtype=_NUMPY_KINDS[kind])))
        else:
            arrays.append(pa.array(values, type=_ARROW_TYPES[kind]))
    return pa.Table.from_arrays(arrays, names=names)


class Batch:
    """Rows of many hands, converted to Arrow tables for a bulk insert."""

    def __init__(self) -> None:
        self.hands: list[tuple] = []
        self.players: list[tuple] = []
        self.actions: list[tuple] = []
        self.class_keys: list[Optional[tuple]] = []

    def add(self, rows: HandRows) -> None:
        self.hands.append(rows.hand)
        self.players.extend(rows.players)
        self.actions.extend(rows.actions)
        keys = rows.class_keys or [None] * len(rows.actions)
        if len(keys) != len(rows.actions):
            raise ValueError("class_keys must align with the action rows")
        self.class_keys.extend(keys)

    def __len__(self) -> int:
        return len(self.hands)

    def dates(self) -> set:
        """UTC dates of the hands in the batch."""
        k = [c[0] for c in HAND_COLUMNS].index("hand_date")
        return {h[k] for h in self.hands}

    def made_draw_columns(self) -> tuple[list[Optional[str]], list[Optional[str]]]:
        """``made_hand`` and ``draw`` of every action row, classified in one vectorised call."""
        n = len(self.actions)
        made: list[Optional[str]] = [None] * n
        draw: list[Optional[str]] = [None] * n
        rows = [i for i, k in enumerate(self.class_keys) if k is not None]
        if not rows:
            return made, draw
        hole = np.empty((len(rows), 2), dtype=np.int16)
        board = np.full((len(rows), 5), -1, dtype=np.int16)
        for j, i in enumerate(rows):
            key = self.class_keys[i]
            hole[j, 0], hole[j, 1] = key[0], key[1]
            board[j, : len(key) - 2] = key[2:]
        m_names, d_names = made_and_draw(hole, board)
        for j, i in enumerate(rows):
            made[i] = m_names[j]
            draw[i] = d_names[j]
        return made, draw

    def to_arrow(self) -> dict[str, pa.Table]:
        """``{"hands": ..., "hand_players": ..., "actions": ...}`` as Arrow tables."""
        made, draw = self.made_draw_columns()
        return {
            "hands": _column_arrays(HAND_COLUMNS, self.hands),
            "hand_players": _column_arrays(PLAYER_COLUMNS, self.players),
            "actions": _column_arrays(ACTION_COLUMNS, self.actions, {"made_hand": made, "draw": draw}),
        }

    def clear(self) -> None:
        self.hands.clear()
        self.players.clear()
        self.actions.clear()
        self.class_keys.clear()
