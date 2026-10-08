"""Parse no-limit hold'em hand histories written in the PokerStars-style text family.

One hand at a time, pure Python. :func:`parse_hand` turns the text of one hand
into a :class:`ParsedHand` (players, posts, betting lines normalised to the
chips each line moved, boards, shown cards, payouts) or, when the hand fails
a consistency check, into a :class:`GateReject` that names the reason. The
loader counts every reason; no hand is ever dropped silently.

Grammar
-------
=====================  ==========================================================
header                 ``[<site>]Hand #<id>: Hold'em No Limit (<sb>/<bb>[ - Ante <a>| ante <a>][ <CUR>]) - <yyyy/mm/dd hh:mm:ss>[ <TZ>]``
table                  ``Table '<name>' <N>-max[ (<extra>)] Seat #<b> is the button``
seat                   ``Seat <s>: <name> (<stack> in chips[, ...])[ is sitting out]``
posts                  ``posts the ante <a>`` (or ``posts ante``), ``posts small blind``,
                       ``posts big blind``, ``posts straddle``, ``posts small blind (dead) <x>``,
                       ``posts dead blind <x>``, ``posts small blind (dead) and big blind <x>``,
                       ``posts small & big blinds <x>``, ``posts button blind <x>``; any of them
                       may end with ``and is all-in``
hole cards             ``*** HOLE CARDS ***`` and ``Dealt to <name> [Xx Yy]`` (one line per seat
                       in a full-information export, the hero only otherwise)
betting                ``folds``, ``checks``, ``calls <x>``, ``bets <x>``, ``raises <inc> to <total>``,
                       and the older ``raises <x> and is all-in`` (no ``to``: x = chips put in now)
streets                ``*** FLOP *** [a b c]``, ``*** TURN *** [a b c] [d]``, ``*** RIVER *** [a b c d] [e]``;
                       a run-it-twice hand prefixes ``FIRST`` / ``SECOND`` / ``THIRD``
showdown               ``*** SHOW DOWN ***`` (or ``SHOWDOWN``), ``<name>: shows [..]``,
                       ``mucks hand``, ``doesn't show hand``
returns, payouts       ``Uncalled bet (<x>) returned to <name>``, ``<name> collected <x> from pot``
                       (``main pot``, ``side pot``, ``side pot-N``, ``side-pot N``)
summary                ``Total pot <x> | Rake <y>[ | <other drop> <z> ...]``, ``Board [..]``,
                       ``Seat <s>: <name> ... showed [..] ...`` / ``mucked [..]``
cancelled              ``*** HAND CANCELLED ***`` at the start of a line (the hand is skipped)
informational          timeouts, time bank, disconnects, chat, joins and leaves (ignored;
                       a fold or check right after a timeout line is marked as automatic;
                       chat text is never read as an action, a marker or a new hand)
=====================  ==========================================================

Amounts may carry a leading currency glyph and thousands separators. They are
parsed to integers of the smallest unit (``10 ** amount_decimals`` units per
currency unit) and every check below is exact. Player names may contain
spaces and colons: the subject of a line is the longest seated name followed
by ``": "``.

Gate (the first failing check names the reason; the order is fixed)
-------------------------------------------------------------------
``cancelled`` · ``bad_header`` · ``unsupported_game`` · ``bad_table_line`` ·
``unsupported_structure_no_sb`` (small blind 0: ante-only tables) ·
``bomb_pot_no_blinds`` (no big blind, no preflop betting, a flop) ·
``no_big_blind_post`` · ``no_actions`` (nobody acted although the betting
was still open) · ``unknown_grammar`` (unknown line, post, verb or marker,
negative amount, unparsable seat line, a name or seat listed twice, a line
before the hole cards) · ``bad_card`` (unreadable, repeated, or two different
hands for one player) · ``payee_not_dealt`` · ``stack_inconsistent`` (a line puts in
more than the stack, an all-in marker with chips behind) ·
``action_order_invalid`` (acting after folding or while all-in) ·
``amount_inconsistent`` (a call that does not match the amount to call, a
check facing a bet after the flop, a bet facing a bet) ·
``money_unreconciled`` (collected + returned != put in - rake - other drops,
exact) · ``hidden_post_inferred`` (a preflop check or short call below the
level, one player acting twice in a row, a raise increment that contradicts
the level: an unprinted post; dropped unless ``drop_hidden_post_hands`` is
off).

Hands kept with a remark carry flags instead (see :data:`INFO_FLAGS`).

Events
------
:attr:`ParsedHand.events` lists the hand in text order as tuples:

* ``(POST, player, kind, amount, live, all_in_marker)`` — ``kind`` is one of
  ``ante``, ``sb``, ``bb``, ``straddle``, ``post`` (an extra live big blind),
  ``dead``, ``button_blind`` or ``hidden`` (an inferred unprinted post);
  ``live`` is the part that counts toward the street bet.
* ``(DEAL, street, board)`` — ``street`` 1..3, ``board`` the run-1 board so far.
* ``(ACT, player, verb, added, all_in_marker, auto, text_amount, text_to)`` —
  ``verb`` as written (``fold``, ``check``, ``call``, ``bet``, ``raise``),
  ``added`` the chips this line moved from the stack to the pot, ``auto``
  ``"timeout"`` or ``"unknown"``.

Players are indices into :attr:`ParsedHand.players` (dealt players in seat
order); cards are ids from :mod:`sudetect.cards`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

from .cards import parse_card

__all__ = [
    "PARSER_VERSION",
    "POST",
    "DEAL",
    "ACT",
    "GATE_REASONS",
    "INFO_FLAGS",
    "ParseOptions",
    "GateReject",
    "ParsedHand",
    "parse_amount",
    "split_hands",
    "parse_hand",
]

PARSER_VERSION = "1.0"

POST, DEAL, ACT = 0, 1, 2

GATE_REASONS: tuple[str, ...] = (
    "cancelled",
    "bad_header",
    "unsupported_game",
    "bad_table_line",
    "unsupported_structure_no_sb",
    "bomb_pot_no_blinds",
    "no_big_blind_post",
    "no_actions",
    "unknown_grammar",
    "bad_card",
    "payee_not_dealt",
    "stack_inconsistent",
    "action_order_invalid",
    "amount_inconsistent",
    "money_unreconciled",
    "hidden_post_inferred",
)
"""Reasons a hand is dropped, in the order they are checked."""

INFO_FLAGS: tuple[str, ...] = (
    "run_it_multiple",       # several boards were dealt
    "negative_rake",         # the rake field is negative (a bonus booked as rake)
    "other_drops",           # the summary lists drops besides the rake (they count in the money check)
    "uncalled_text_differs", # the printed uncalled amount differs from the betting
    "total_pot_differs",     # the printed total pot differs from the pot recomputed from the betting
    "board_text_differs",    # the summary board differs from the street lines
    "dead_button",           # the button seat is empty
    "straddle",
    "blind_all_in_from_ante",  # a blind seat was all-in from the ante and posted nothing
    "allin_marker_missing",  # a stack reached zero without the all-in marker
    "showdown_marker",       # the text has a showdown section
    "time_zone_not_utc",     # the header time zone is not UTC/GMT; the time is kept as written
    "hidden_post",           # an unprinted post was inferred (kept only if not dropped)
)
"""Flags of kept hands (they become informational codes of ``hands.error_detail``)."""

_HIDDEN_FLAGS = frozenset({"hidden_post_check", "hidden_post_call", "same_player_twice",
                           "raise_increment_mismatch"})


@dataclass(frozen=True)
class ParseOptions:
    """Switches for the quirks some hand-history writers have.

    Attributes:
        amount_decimals: Decimal places of amounts; amounts become integers of
            ``10 ** amount_decimals`` units per currency unit.
        infer_hidden_posts: When a preflop check or short call shows that a
            post was not printed, add the missing post so the money reconciles.
        drop_hidden_post_hands: Drop such hands at the gate (their per-line
            amounts are shifted even when the money reconciles).
        keep_bomb_pots: Keep hands with antes only, no preflop betting and a
            flop as ``variant = 'bomb_pot'`` instead of dropping them.
        timeout_marks_auto_action: A fold or check that follows a
            ``has timed out`` line of the same player is the system's action.
        money_tolerance: Allowed difference of the money check, in units.
    """

    amount_decimals: int = 2
    infer_hidden_posts: bool = True
    drop_hidden_post_hands: bool = True
    keep_bomb_pots: bool = False
    timeout_marks_auto_action: bool = True
    money_tolerance: int = 0


DEFAULT_OPTIONS = ParseOptions()


@dataclass(slots=True)
class GateReject:
    """A hand that failed the gate: the reason (one of :data:`GATE_REASONS`) and its id."""

    reason: str
    hand_id: str
    detail: str = ""


@dataclass(slots=True)
class ParsedHand:
    """One parsed hand. Amounts are integer units; players are indices into ``players``.

    Attributes:
        hand_id: Hand number as written.
        started_at: Header time (naive; UTC unless ``time_zone_not_utc`` is flagged).
        time_zone: Time zone text of the header ("" if none).
        sb, bb: Blinds from the header.
        ante: Ante from the header (0 when the header has none; antes may still be posted).
        table_name, max_seats, button_seat: From the table line.
        players: Dealt players (seat order); ``seats`` and ``stacks`` (before any post) align.
        events: The hand as POST / DEAL / ACT tuples (module docstring).
        dealt_cards: Player -> cards from ``Dealt to`` lines.
        shown_cards: Player -> cards revealed at the table (showdown or a voluntary show).
        shown_at_showdown: Players among ``shown_cards`` who showed in the showdown section.
        mucked: Players who mucked at the showdown.
        collected: Chips collected per player (all pots, all runs).
        returned: Uncalled chips returned per player, as printed.
        returned_calc: The same recomputed from the betting.
        contributed: Chips each player put in (posts and betting lines, before returns).
        boards: Board of each run; run 1 first. A hand without a flop has ``[[]]``.
        total_pot: Printed total pot, or None.
        rake: Printed rake (may be negative), 0 when the summary has none.
        fees: Other drops printed next to the rake.
        variant: ``standard`` or ``bomb_pot``.
        preflop_level: The amount to call before the flop for a player who has
            not posted (the big blind, or the largest live post when larger).
        sb_player, bb_player: First live small-blind / big-blind posters.
        straddlers: Players who posted a straddle.
        flags: Informational flags (:data:`INFO_FLAGS`).
    """

    hand_id: str
    started_at: datetime
    time_zone: str
    sb: int
    bb: int
    ante: int
    table_name: str
    max_seats: int
    button_seat: int
    players: list[str]
    seats: list[int]
    stacks: list[int]
    events: list[tuple]
    dealt_cards: dict[int, tuple[int, int]]
    shown_cards: dict[int, tuple[int, int]]
    shown_at_showdown: set[int]
    mucked: set[int]
    collected: list[int]
    returned: list[int]
    returned_calc: list[int]
    contributed: list[int]
    boards: list[list[int]]
    total_pot: Optional[int]
    rake: int
    fees: int
    variant: str
    preflop_level: int
    sb_player: Optional[int]
    bb_player: Optional[int]
    straddlers: list[int]
    flags: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ grammar

_A = r"[^\d\s(\-]?(-?[\d,]*\.?\d+)"
_HEADER_RE = re.compile(
    r"^(?P<site>.*?)Hand #(?P<hid>[\w-]+):\s+(?P<game>.+?)\s+\((?P<stakes>.+?)\)\s+-\s+"
    r"(?P<y>\d{4})/(?P<mo>\d{1,2})/(?P<d>\d{1,2}) (?P<h>\d{1,2}):(?P<mi>\d{2}):(?P<s>\d{2})"
    r"(?:\s+(?P<tz>[A-Za-z]+))?")
_STAKES_RE = re.compile(
    r"^" + _A + r"/" + _A + r"(?:\s*-?\s*[Aa]nte\s+" + _A + r")?(?:\s+(?P<cur>[A-Za-z]+))?$")
_TABLE_RE = re.compile(r"^Table '(?P<table>.*)' (?P<max>\d+)-max (?:\(.*?\) )?Seat #(?P<btn>\d+) is the button")
_SEAT_RE = re.compile(r"^Seat (?P<seat>\d+): (?P<name>.+?) \(" + _A + r" in chips[^)]*\)(?P<tail>.*)$")
_STREET_RE = re.compile(r"^\*\*\* (?:(?P<run>FIRST|SECOND|THIRD) )?(?P<street>FLOP|TURN|RIVER) \*\*\*(?P<cards>.*)$")
_CARDS_RE = re.compile(r"\[([^\]]*)\]")
_COLLECT_RE = re.compile(r"^(?P<name>.+?) collected " + _A + r"(?: from (?P<src>.+))?$")
_TOTAL_RE = re.compile(r"^Total pot " + _A + r"(?P<rest>.*)$")
_DROP_RE = re.compile(r"\|\s*(?P<name>[A-Za-z][A-Za-z ]*?):?\s+" + _A)
_SUMMARY_CARDS_RE = re.compile(r"^Seat \d+: (?P<name>.+?) (?:\([^)]*\) )*(?P<what>showed|mucked) \[(?P<cards>[^\]]+)\]")
# no double quote before "Hand #": a chat line quoting a header must not start a new hand
_HEADER_PREFIX = r"[^\n\"]{0,80}?Hand #[\w-]+: [^\n]*? - \d{4}/\d{1,2}/\d{1,2} "
_SPLIT_RE = re.compile(r"(?m)^(?=" + _HEADER_PREFIX + r")")

_RUNS = {None: 0, "FIRST": 0, "SECOND": 1, "THIRD": 2}
_STREETS = {"FLOP": 1, "TURN": 2, "RIVER": 3}
_BOARD_LEN = (0, 3, 4, 5)
_UTC_ZONES = frozenset({"", "UTC", "GMT", "Z"})
_ACTION_STARTS = ("posts ", "folds", "checks", "calls ", "bets ", "raises ")
_INFO_INFIXES = (" has timed out", " activated time-bank", " activated time bank", " is disconnected",
                 " is connected", " has reconnected", " has returned", " joins the table",
                 " leaves the table", " was removed from the table", " will be allowed to play",
                 " is sitting out", " sits out", " said, \"", " stands up", " sits down")
_INFO_PREFIXES = ("Game ended", "Hand was run", "All bets returned")
_ALLIN = " and is all-in"

_CARD_IDS: dict[str, int] = {r + s: i * 4 + j for i, r in enumerate("23456789TJQKA") for j, s in enumerate("cdhs")}


def _card(token: str) -> int:
    c = _CARD_IDS.get(token)
    return c if c is not None else parse_card(token)


def _cards(text: str) -> list[int]:
    get = _CARD_IDS.get
    out = []
    for t in text.split():
        c = get(t)
        out.append(c if c is not None else parse_card(t))
    return out


def parse_amount(text: str, decimals: int = 2) -> int:
    """Parse an amount such as ``"1.50"``, ``"$1,234.5"`` or ``"-0.20"`` into integer units.

    Raises:
        ValueError: Not an amount, or more decimal places than ``decimals``
            with a non-zero digit beyond them.
    """
    s = text.strip()
    if s and not (s[0].isdigit() or s[0] in ".-"):
        s = s[1:]  # a currency glyph
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    s = s.replace(",", "")
    whole, dot, frac = s.partition(".")
    if not (whole or frac) or (whole and not whole.isdigit()) or (frac and not frac.isdigit()):
        raise ValueError(f"not an amount: {text!r}")
    if len(frac) > decimals:
        if frac[decimals:].strip("0"):
            raise ValueError(f"amount {text!r} has more than {decimals} decimal places")
        frac = frac[:decimals]
    value = int(whole or "0") * 10 ** decimals + int(frac.ljust(decimals, "0") or "0")
    return -value if neg else value


_AMOUNT_CACHE: dict[tuple[str, int], int] = {}


def _amount_cached(text: str, decimals: int) -> int:
    key = (text, decimals)
    got = _AMOUNT_CACHE.get(key)
    if got is None:
        got = parse_amount(text, decimals)
        if len(_AMOUNT_CACHE) > 200_000:
            _AMOUNT_CACHE.clear()
        _AMOUNT_CACHE[key] = got
    return got


def split_hands(text: str) -> list[str]:
    """Split a file's text into the texts of its hands (a hand starts at a header line)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    return [p.strip("\n") for p in _SPLIT_RE.split(text) if p.strip() and _SPLIT_RE.match(p.lstrip("\n"))]


def _is_nlhe(game: str) -> bool:
    g = game.lower().replace("-", " ").replace("holdem", "hold'em")
    return "hold'em no limit" in g or "hold'em nl" in g


def _post_kind(desc: str) -> Optional[str]:
    if desc in ("the ante", "ante", "an ante"):
        return "ante"
    if desc.startswith("small blind (dead) and big blind") or desc in ("small & big blinds", "small and big blinds"):
        return "dead_and_live"
    if desc in ("small blind (dead)", "dead blind", "dead small blind", "a dead blind"):
        return "dead"
    if desc == "small blind":
        return "sb"
    if desc == "big blind":
        return "bb"
    if desc.startswith("straddle"):
        return "straddle"
    if desc == "button blind":
        return "button_blind"
    return None


# ------------------------------------------------------------------ parser


def parse_hand(text: str, options: Optional[ParseOptions] = None) -> Union[ParsedHand, GateReject]:
    """Parse one hand; return the hand or the reason it is dropped."""
    opt = options or DEFAULT_OPTIONS
    dec = opt.amount_decimals
    lines = [ln.rstrip() for ln in text.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return GateReject("bad_header", "")
    m = _HEADER_RE.match(lines[0])
    if m is None:
        return GateReject("bad_header", "", lines[0][:80])
    hid = m["hid"]
    # the marker must start a line: the same words typed in the chat do not cancel a hand
    if "*** HAND CANCELLED" in text and any(ln.startswith("*** HAND CANCELLED") for ln in lines):
        return GateReject("cancelled", hid)
    sm = _STAKES_RE.match(m["stakes"].strip())
    if sm is None:
        return GateReject("bad_header", hid, "stakes")
    try:
        sb = parse_amount(sm.group(1), dec)
        bb = parse_amount(sm.group(2), dec)
        ante = parse_amount(sm.group(3), dec) if sm.group(3) else 0
        started_at = datetime(int(m["y"]), int(m["mo"]), int(m["d"]), int(m["h"]), int(m["mi"]), int(m["s"]))
    except ValueError as exc:
        return GateReject("bad_header", hid, str(exc))
    if bb <= 0 or sb < 0 or ante < 0:
        return GateReject("bad_header", hid, "blinds")
    if not _is_nlhe(m["game"]):
        return GateReject("unsupported_game", hid, m["game"][:40])
    tz = m["tz"] or ""
    tm = _TABLE_RE.match(lines[1]) if len(lines) > 1 else None
    if tm is None:
        return GateReject("bad_table_line", hid)
    flags: set[str] = set()
    if tz.upper() not in _UTC_ZONES:
        flags.add("time_zone_not_utc")

    # seats
    seat_no: dict[str, int] = {}
    stack0: dict[str, int] = {}
    sitting_out: set[str] = set()
    bad_seat = False
    i = 2
    while i < len(lines) and lines[i].startswith("Seat ") and " in chips" in lines[i]:
        s = _SEAT_RE.match(lines[i])
        if s is None:
            bad_seat = True
        else:
            try:
                st = parse_amount(s.group(3), dec)
            except ValueError:
                bad_seat = True
            else:
                name = s["name"]
                if name in seat_no or int(s["seat"]) in seat_no.values():
                    bad_seat = True  # a name or a seat listed twice
                seat_no[name] = int(s["seat"])
                stack0[name] = st
                if "sitting out" in s["tail"]:
                    sitting_out.add(name)
        i += 1
    body = lines[i:]
    seated = stack0.keys()

    def subject(line: str) -> tuple[Optional[str], str]:
        """Longest seated name followed by ': ' at the start of the line."""
        found: Optional[str] = None
        rest = ""
        k = line.find(": ")
        while k > 0:
            cand = line[:k]
            if cand in seated:
                found, rest = cand, line[k + 2:]
            k = line.find(": ", k + 1)
        return found, rest

    # players dealt in: subjects of a post or betting line, and Dealt-to lines
    dealt_names: set[str] = set()
    subjects: dict[int, tuple[Optional[str], str]] = {}
    for li, ln in enumerate(body):
        c0 = ln[0]
        if c0 == "D" and ln.startswith("Dealt to "):
            j = ln.rfind(" [")
            if j > 9 and ln[9:j] in seated:
                dealt_names.add(ln[9:j])
            continue
        if c0 == "*" and ln.startswith("***"):
            if ln.startswith("*** SUMMARY"):
                break
            continue
        name, rest = subjects[li] = subject(ln)
        if name is not None and rest.startswith(_ACTION_STARTS):
            dealt_names.add(name)
    players = sorted(dealt_names, key=lambda x: seat_no[x])
    n = len(players)
    if n < 2:
        return GateReject("no_actions", hid, "fewer than two players dealt")
    pidx = {name: k for k, name in enumerate(players)}
    seats = [seat_no[x] for x in players]
    stacks = [stack0[x] for x in players]

    contrib = [0] * n
    st = [0] * n                 # actual chips put in on the current street
    folded = [False] * n
    returned = [0] * n
    returned_calc = [0] * n
    collected = [0] * n
    events: list[tuple] = []
    dealt_cards: dict[int, tuple[int, int]] = {}
    shown: dict[int, tuple[int, int]] = {}
    shown_sd: set[int] = set()
    mucked: set[int] = set()
    boards: list[list[int]] = [[]]
    errors: set[str] = set()     # gate-level problems found while reading
    cur = 0                      # actual amount to match on this street
    street = 0
    in_hole = in_summary = in_showdown = False
    last_actor = -1
    sb_done = bb_done = False
    sb_player: Optional[int] = None
    bb_player: Optional[int] = None
    straddlers: list[int] = []
    timed_out: set[int] = set()
    n_acts = 0
    pre_acts = 0
    total_pot: Optional[int] = None
    rake = 0
    fees = 0
    summary_board: Optional[list[int]] = None
    infer = opt.infer_hidden_posts

    def amount(tok: str) -> int:
        try:
            return _amount_cached(tok, dec)
        except ValueError:
            errors.add("unknown_grammar")
            return 0

    for li, ln in enumerate(body):
        if ln[0] == "*" and ln.startswith("***"):
            if ln.startswith("*** HOLE CARDS"):
                in_hole = True
                street = 0
                cur = max(max(st), bb)
                continue
            sm2 = _STREET_RE.match(ln)
            if sm2 is not None:
                run = _RUNS[sm2["run"]]
                try:
                    cards = _cards(" ".join(_CARDS_RE.findall(sm2["cards"])))
                except ValueError:
                    errors.add("bad_card")
                    continue
                s_idx = _STREETS[sm2["street"]]
                if len(cards) != _BOARD_LEN[s_idx]:
                    errors.add("bad_card")
                    continue
                if run == 0:
                    if s_idx != street + 1:
                        errors.add("unknown_grammar")
                    street = s_idx
                    boards[0] = cards
                    st = [0] * n
                    cur = 0
                    last_actor = -1
                    events.append((DEAL, s_idx, tuple(cards)))
                else:
                    flags.add("run_it_multiple")
                    while len(boards) <= run:
                        boards.append([])
                    boards[run] = cards
                continue
            if "SHOW DOWN" in ln or "SHOWDOWN" in ln:
                in_showdown = True
                flags.add("showdown_marker")
                continue
            if ln.startswith("*** SUMMARY"):
                in_summary = True
                continue
            errors.add("unknown_grammar")
            continue

        if in_summary:
            if ln.startswith("Total pot"):
                t = _TOTAL_RE.match(ln)
                if t is None:
                    errors.add("unknown_grammar")
                    continue
                total_pot = amount(t.group(1))
                rest_t = t["rest"]
                for d in _DROP_RE.finditer(rest_t):
                    value = amount(d.group(2))
                    if d["name"].strip().lower() == "rake":
                        rake += value
                    else:
                        fees += value
                        flags.add("other_drops")
                if rake < 0:
                    flags.add("negative_rake")
            elif ln.startswith("Board ") or ln.startswith("FIRST Board"):
                try:
                    summary_board = _cards(" ".join(_CARDS_RE.findall(ln)))
                except ValueError:
                    errors.add("bad_card")
            elif ln.startswith("Seat "):
                sc = _SUMMARY_CARDS_RE.match(ln)
                if sc is not None and sc["name"] in pidx:
                    p = pidx[sc["name"]]
                    try:
                        cs = _cards(sc["cards"])
                    except ValueError:
                        errors.add("bad_card")
                        continue
                    if len(cs) == 2 and p not in shown:
                        shown[p] = (cs[0], cs[1])
            elif ln.startswith("Hand was run"):
                flags.add("run_it_multiple")
            continue

        if ln[0] == "D" and ln.startswith("Dealt to "):
            j = ln.rfind(" [")
            name = ln[9:j] if j > 9 else ""
            if name not in pidx:
                if name not in seated:
                    errors.add("unknown_grammar")
                continue
            try:
                cs = _cards(ln[j + 2:].split("]")[0])
            except ValueError:
                errors.add("bad_card")
                continue
            if len(cs) != 2:
                errors.add("bad_card")
                continue
            p = pidx[name]
            if p in dealt_cards and set(dealt_cards[p]) != set(cs):
                errors.add("bad_card")  # two different hands dealt to one player
            dealt_cards[p] = (cs[0], cs[1])
            continue

        if ln[0] == "U" and ln.startswith("Uncalled bet ("):
            close = ln.find(")")
            marker = ") returned to "
            if close < 0 or not ln.startswith(marker, close):
                errors.add("unknown_grammar")
                continue
            name = ln[close + len(marker):]
            if name not in pidx:
                errors.add("payee_not_dealt" if name in seated else "unknown_grammar")
                continue
            p = pidx[name]
            amt = amount(ln[14:close])
            others = max((st[q] for q in range(n) if q != p), default=0)
            calc = max(0, st[p] - others)
            returned[p] += amt
            returned_calc[p] += calc
            if calc != amt:
                flags.add("uncalled_text_differs")
            st[p] -= calc
            continue

        got = subjects.get(li)
        name, rest = got if got is not None else subject(ln)
        if name is None:
            if " collected " in ln:
                cm = _COLLECT_RE.match(ln)
                if cm is not None and cm["name"] in seated:
                    if cm["name"] not in pidx:
                        errors.add("payee_not_dealt")
                    else:
                        collected[pidx[cm["name"]]] += amount(cm.group(2))
                    continue
            if ln.startswith(_INFO_PREFIXES):
                if ln.startswith("Hand was run"):
                    flags.add("run_it_multiple")
                continue
            if any(x in ln for x in _INFO_INFIXES):
                if " has timed out" in ln:
                    who = ln.split(" has timed out")[0]
                    if who in pidx:
                        timed_out.add(pidx[who])
                continue
            errors.add("unknown_grammar")
            continue

        if name not in pidx:
            # a seated player who was not dealt in: only non-action lines are allowed
            if rest.startswith(("didn't post", "sits out", "is sitting out", "has returned", "mucks", "doesn't show")):
                continue
            errors.add("unknown_grammar")
            continue
        p = pidx[name]

        if rest.startswith("posts "):
            body_p = rest[6:]
            allin_mark = body_p.endswith(_ALLIN)
            if allin_mark:
                body_p = body_p[: -len(_ALLIN)]
            desc, _, tok = body_p.rpartition(" ")
            kind = _post_kind(desc)
            amt = amount(tok)
            if kind is None or amt < 0:
                errors.add("unknown_grammar")
                continue
            if in_hole:
                errors.add("unknown_grammar")  # posts after the hole cards are not supported
                continue
            behind = stacks[p] - contrib[p]
            if amt > behind:
                errors.add("stack_inconsistent")
            if kind == "dead_and_live":
                live = min(bb, amt)
                if amt > live:
                    events.append((POST, p, "dead", amt - live, 0, False))
                events.append((POST, p, "post", live, live, allin_mark))
                st[p] += live
                if not bb_done:
                    bb_done = True
                    bb_player = p
                    events[-1] = (POST, p, "bb", live, live, allin_mark)
            else:
                if kind == "sb":
                    if sb_done:
                        kind = "dead"
                    else:
                        sb_done = True
                        sb_player = p
                elif kind == "bb":
                    if bb_done:
                        kind = "post"
                    else:
                        bb_done = True
                        bb_player = p
                elif kind == "straddle":
                    straddlers.append(p)
                    flags.add("straddle")
                live = amt if kind in ("sb", "bb", "straddle", "post") else 0
                events.append((POST, p, kind, amt, live, allin_mark))
                st[p] += live
            contrib[p] += amt
            if contrib[p] == stacks[p] and amt > 0 and not allin_mark:
                flags.add("allin_marker_missing")
            if allin_mark and contrib[p] != stacks[p]:
                errors.add("stack_inconsistent")
            continue

        if rest.startswith(("shows", "mucks", "doesn't show", "show ")):
            if rest.startswith("shows"):
                cm2 = _CARDS_RE.search(rest)
                if cm2 is not None:
                    try:
                        cs = _cards(cm2.group(1))
                    except ValueError:
                        errors.add("bad_card")
                        continue
                    if len(cs) == 2:
                        if p not in shown:
                            shown[p] = (cs[0], cs[1])
                        if in_showdown:
                            shown_sd.add(p)
            elif rest.startswith("mucks"):
                mucked.add(p)
            continue
        if rest.startswith(("didn't post", "sits out", "is sitting out", "has returned")):
            continue

        # ---- betting line
        if not in_hole:
            errors.add("unknown_grammar")
            continue
        allin_mark = rest.endswith(_ALLIN)
        core = rest[: -len(_ALLIN)] if allin_mark else rest
        parts = core.split(" ")
        verb = parts[0]
        behind = stacks[p] - contrib[p]
        if folded[p] or (behind <= 0 and verb != "folds"):
            errors.add("action_order_invalid")
        if last_actor == p:
            flags.add("same_player_twice")
        # chips owed: the level while another live player can still act; when every live opponent is
        # all-in, only the largest of their street totals
        top_live = 0
        others_active = False
        for q in range(n):
            if q != p and not folded[q]:
                if st[q] > top_live:
                    top_live = st[q]
                if stacks[q] - contrib[q] > 0:
                    others_active = True
        owed = max(0, (cur if others_active else min(cur, top_live)) - st[p])
        text_amt = 0
        text_to: Optional[int] = None
        hidden = 0
        if verb == "folds":
            act = "fold"
            added = 0
            if len(parts) > 1 and not core.startswith("folds ["):
                errors.add("unknown_grammar")
        elif verb == "checks" and len(parts) == 1:
            act = "check"
            added = 0
            if owed > 0:
                if street == 0:
                    flags.add("hidden_post_check")
                    if infer:
                        hidden = owed
                else:
                    errors.add("amount_inconsistent")
        elif verb in ("calls", "bets") and len(parts) == 2:
            text_amt = amount(parts[1])
            added = text_amt
            if verb == "calls":
                act = "call"
                if text_amt > owed:
                    errors.add("amount_inconsistent")
                elif text_amt < owed and not (allin_mark or text_amt == behind):
                    if street == 0:
                        flags.add("hidden_post_call")
                        if infer:
                            hidden = owed - text_amt
                    else:
                        errors.add("amount_inconsistent")
            else:
                act = "bet"
                if cur > st[p] and street > 0:
                    errors.add("amount_inconsistent")
        elif verb == "raises" and len(parts) == 4 and parts[2] == "to":
            act = "raise"
            text_amt = amount(parts[1])
            text_to = amount(parts[3])
            added = text_to - st[p]
            if text_to - text_amt != cur:
                flags.add("raise_increment_mismatch")
            if text_to <= cur:
                errors.add("amount_inconsistent")
        elif verb == "raises" and len(parts) == 2 and allin_mark:
            act = "raise"  # older form: the number is the chips put in now
            text_amt = amount(parts[1])
            added = text_amt
        else:
            errors.add("unknown_grammar")
            continue
        if text_amt < 0 or added < 0:
            errors.add("unknown_grammar")
            continue
        if hidden:
            events.append((POST, p, "hidden", hidden, hidden, False))
            contrib[p] += hidden
            st[p] += hidden
            behind -= hidden
        auto = "unknown"
        if p in timed_out:
            timed_out.discard(p)
            if opt.timeout_marks_auto_action and act in ("fold", "check"):
                auto = "timeout"
        if added > behind:
            errors.add("stack_inconsistent")
        contrib[p] += added
        st[p] += added
        if act in ("bet", "raise") and st[p] > cur:
            cur = st[p]
        if act == "fold":
            folded[p] = True
        left = stacks[p] - contrib[p]
        if allin_mark and left != 0:
            errors.add("stack_inconsistent")
        elif left == 0 and added > 0 and not allin_mark:
            flags.add("allin_marker_missing")
        events.append((ACT, p, act, added, allin_mark, auto, text_amt, text_to))
        n_acts += 1
        if street == 0:
            pre_acts += 1
        last_actor = p

    # ---------------------------------------------------------------- gate
    variant = "standard"
    if sb == 0:
        return GateReject("unsupported_structure_no_sb", hid)
    if bb_player is None:
        # a blind seat whose ante took its whole stack posts no blind
        ante_put = [0] * n
        for e in events:
            if e[0] == POST and e[2] == "ante":
                ante_put[e[1]] += e[3]
        ante_only_all_in = [p for p in range(n) if 0 < ante_put[p] == stacks[p]]
        if pre_acts == 0 and boards[0] and not ante_only_all_in:
            if not opt.keep_bomb_pots:
                return GateReject("bomb_pot_no_blinds", hid)
            variant = "bomb_pot"
        elif ante_only_all_in:
            flags.add("blind_all_in_from_ante")
        else:
            return GateReject("no_big_blind_post", hid)
    if n_acts == 0 and variant == "standard" and not _closed_after_posts(events, stacks, n):
        return GateReject("no_actions", hid)
    if "unknown_grammar" in errors or bad_seat:
        return GateReject("unknown_grammar", hid)
    # cards: no card twice among the run-1 board and the known hole cards
    known = list(boards[0])
    for cs in dealt_cards.values():
        known.extend(cs)
    for p, cs in shown.items():
        if p not in dealt_cards:
            known.extend(cs)
        elif dealt_cards[p] != cs and set(dealt_cards[p]) != set(cs):
            errors.add("bad_card")
    if "bad_card" in errors or len(set(known)) != len(known):
        return GateReject("bad_card", hid)
    if "payee_not_dealt" in errors:
        return GateReject("payee_not_dealt", hid)
    if "stack_inconsistent" in errors:
        return GateReject("stack_inconsistent", hid)
    if "action_order_invalid" in errors:
        return GateReject("action_order_invalid", hid)
    if "amount_inconsistent" in errors:
        return GateReject("amount_inconsistent", hid)
    if total_pot is None:
        return GateReject("money_unreconciled", hid, "no total pot line")
    paid = sum(collected) + sum(returned)
    if abs(paid - (sum(contrib) - rake - fees)) > opt.money_tolerance:
        return GateReject("money_unreconciled", hid, f"paid {paid} vs put in {sum(contrib)} - {rake} - {fees}")
    hidden_flags = flags & _HIDDEN_FLAGS
    if hidden_flags:
        if opt.drop_hidden_post_hands:
            return GateReject("hidden_post_inferred", hid, ",".join(sorted(hidden_flags)))
        flags.add("hidden_post")
    flags -= _HIDDEN_FLAGS
    if total_pot != sum(contrib) - sum(returned_calc):
        flags.add("total_pot_differs")
    if summary_board is not None and summary_board != boards[0]:
        flags.add("board_text_differs")
    if int(tm["btn"]) not in seats:
        flags.add("dead_button")
    level = 0
    if variant == "standard":
        live_posts = [0] * n
        for e in events:
            if e[0] == POST and e[2] != "hidden":
                live_posts[e[1]] += e[4]
        level = max(bb, max(live_posts))
    return ParsedHand(
        hand_id=hid, started_at=started_at, time_zone=tz, sb=sb, bb=bb, ante=ante,
        table_name=tm["table"], max_seats=int(tm["max"]), button_seat=int(tm["btn"]),
        players=players, seats=seats, stacks=stacks, events=events,
        dealt_cards=dealt_cards, shown_cards=shown, shown_at_showdown=shown_sd, mucked=mucked,
        collected=collected, returned=returned, returned_calc=returned_calc, contributed=contrib,
        boards=boards, total_pot=total_pot, rake=rake, fees=fees, variant=variant,
        preflop_level=level, sb_player=sb_player, bb_player=bb_player, straddlers=straddlers,
        flags=flags,
    )


def _closed_after_posts(events: list[tuple], stacks: list[int], n: int) -> bool:
    """True when nobody had to act after the posts (every player all-in, or one covering the rest)."""
    put = [0] * n
    live = [0] * n
    for e in events:
        if e[0] == POST:
            put[e[1]] += e[3]
            live[e[1]] += e[4]
    can_act = [p for p in range(n) if stacks[p] - put[p] > 0]
    if not can_act:
        return True
    if len(can_act) == 1:
        p = can_act[0]
        return live[p] >= max(live[q] for q in range(n) if q != p)
    return False
