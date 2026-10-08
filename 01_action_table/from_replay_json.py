"""Operator replay rows (JSON) -> hand-history text in this repository's dialect, plus a decision-time file.

Input variant 2 of step 01. The per-suspect forensic package (``superuser/``) reads one operator's export: one row
per hand with a replay JSON and a game-record JSON, in a CSV, parquet or JSON-lines file
(``superuser.cli.read_hands``). This script parses every row with ``superuser.parser.parse_hand`` and writes the hand
as text in the dialect of ``sudetect/sim/hhtext.py`` (full information), which ``01_action_table/run.py`` loads into
the same three tables as any other hand history.

How the export maps to the text:

* Blinds: the big blind is ``TableSetup.big_blind``; the small blind is half of it unless the room record carries
  ``small_blind`` or ``sb``. The export has no post records, so the posts (and antes) are written from these values.
  Heads-up, the small blind is the first player to act preflop (the button); when the seat labels say otherwise the
  action order wins and the hand is counted under ``labels_overridden``.
* Actions: the export's bet, raise and all-in amounts are street totals and its call amounts are chips added. They
  become ``bets <total>``, ``raises <increment> to <total>``, ``calls <added>``; an all-in that does not exceed the
  current street total is a call. ``and is all-in`` is added when the player's stack is used up.
* ``Dealt to`` lines for every seat, the board as far as it was dealt, ``Uncalled bet (x) returned to``, a
  ``shows`` line for every player still in at the showdown.
* ``collected`` = the export's net result + the chips the player put in (after the uncalled bet). The net result of
  every player in ``hand_players`` therefore equals the export's settlement, and the pot left over is the rake.
  Hands whose settlement asks for a negative payout or more than the pot are skipped with a reason.

The text has no decision times. They go to ``decision_times.csv`` next to the text: ``hand_id, street, order,
seconds``, where ``order`` counts the actions of the street from 1, posts excluded. ``05_detection/run.py`` reads
it back.

Usage::

    python 01_action_table/from_replay_json.py --input data/sample_hands.csv --out <folder>
    python 01_action_table/run.py --dataset article_sample --input <folder>/hands.txt --source operator
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sudetect.evaluator import describe_hand  # noqa: E402
from superuser.cards import evaluate, to_str  # noqa: E402
from superuser.cli import read_hands  # noqa: E402
from superuser.parser import BOARD_LEN, STREETS, Hand, parse_hand  # noqa: E402

UNITS = 100                      # amounts are written with two decimals: one chip = 100 units
TIME_COLUMNS = ["hand_id", "street", "order", "seconds"]
_HEADERS = {"flop": "FLOP", "turn": "TURN", "river": "RIVER"}
_SUMMARY_STREET = {"flop": "Flop", "turn": "Turn", "river": "River"}


class ConvertError(ValueError):
    """A hand the text cannot represent; ``reason`` is counted in the funnel."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _units(chips: Any) -> int:
    return int(round(float(chips) * UNITS))


def _amt(units: int) -> str:
    if units < 0:
        raise ConvertError("negative_amount")
    return f"{units // UNITS}.{units % UNITS:02d}"


def _cards(ids) -> str:
    return " ".join(to_str(c) for c in ids)


def small_blind_of(row: Mapping[str, Any]) -> Optional[float]:
    """The small blind when the room record states it, else None (half the big blind is used)."""
    rp = row.get("hand_replay")
    try:
        room = (json.loads(rp) if isinstance(rp, str) else (rp or {})).get("TableSetup") or {}
    except (ValueError, AttributeError):
        return None
    for key in ("small_blind", "smallblind", "sb"):
        if room.get(key):
            return float(room[key])
    return None


def blind_seats(h: Hand) -> tuple[int, int, int, bool]:
    """(small blind seat, big blind seat, button seat, labels overridden)."""
    seats = sorted(h.seats)
    lab = {h.seats[s].label.upper(): s for s in seats if h.seats[s].label}
    if len(seats) == 2:
        pre = [a for a in h.actions["preflop"] if a.type != 7]
        sb = pre[0].seat_no if pre else lab.get("SB", seats[0])
        bb = seats[1] if sb == seats[0] else seats[0]
        return sb, bb, sb, "SB" in lab and lab["SB"] != sb
    sb, bb = lab.get("SB"), lab.get("BB")
    if sb is None or bb is None:
        raise ConvertError("no_blind_labels")
    btn = next((lab[k] for k in ("BTN", "BU", "D", "DEALER", "BUTTON") if k in lab), None)
    if btn is None:
        btn = seats[seats.index(sb) - 1]
    return sb, bb, btn, False


def hand_text(h: Hand, sb_chips: Optional[float] = None) -> tuple[str, list[tuple], Counter]:
    """One hand as text, its decision times and remark counts. Raises :class:`ConvertError`."""
    seats = sorted(h.seats)
    if len(seats) < 2:
        raise ConvertError("fewer_than_two_seats")
    off = 1 if min(seats) == 0 else 0
    tseat = {s: s + off for s in seats}
    name = {s: str(h.seats[s].uid) for s in seats}
    if len(set(name.values())) < len(seats):
        raise ConvertError("player_twice")
    sb_s, bb_s, btn_s, overridden = blind_seats(h)
    bb = _units(h.blind)
    if bb <= 0:
        raise ConvertError("no_big_blind")
    sb = _units(sb_chips) if sb_chips else bb // 2
    ante = _units(h.ante)
    stack = {s: _units(h.seats[s].stake) for s in seats}
    put = {s: 0 for s in seats}
    remarks: Counter = Counter()
    if overridden:
        remarks["labels_overridden"] += 1

    def put_in(s: int, x: int) -> str:
        if x < 0 or x > stack[s] - put[s]:
            raise ConvertError("amount_above_stack")
        put[s] += x
        return " and is all-in" if put[s] == stack[s] and x > 0 else ""

    t0 = datetime.fromtimestamp(int(h.start_ts), tz=timezone.utc)
    lines = [f"Hand #{h.hand_uid}: Hold'em No Limit ({_amt(sb)}/{_amt(bb)}) - {t0:%Y/%m/%d %H:%M:%S} UTC",
             f"Table '{h.table_no}' {max(len(seats), max(tseat.values()))}-max Seat #{tseat[btn_s]} is the button"]
    lines += [f"Seat {tseat[s]}: {name[s]} ({_amt(stack[s])} in chips)" for s in seats]
    if ante:
        for s in seats:
            x = min(ante, stack[s])
            lines.append(f"{name[s]}: posts the ante {_amt(x)}{put_in(s, x)}")
    total = {s: 0 for s in seats}
    for s, kind, x in ((sb_s, "small blind", sb), (bb_s, "big blind", bb)):
        x = min(x, stack[s] - put[s])
        lines.append(f"{name[s]}: posts {kind} {_amt(x)}{put_in(s, x)}")
        total[s] = x
    lines.append("*** HOLE CARDS ***")
    lines += [f"Dealt to {name[s]} [{_cards(h.seats[s].hole)}]" for s in seats if h.seats[s].hole]

    folded_on: dict[int, str] = {}
    times: list[tuple] = []
    street_lines: dict[str, list[str]] = {}
    last_aggr: dict[str, int] = {}
    for i, st in enumerate(STREETS):
        if i:
            total = {s: 0 for s in seats}
        cur = max(total.values())
        out: list[str] = []
        order = 0
        for a in h.actions[st]:
            s = a.seat_no
            if s not in h.seats:
                raise ConvertError("unknown_seat")
            if a.type == 7:                      # walk: nobody acted, the big blind takes the blinds back
                continue
            if s in folded_on:
                raise ConvertError("action_after_fold")
            x, mine = _units(a.amount), total[s]
            if a.type == 2:
                text = "folds"
                folded_on[s] = st
            elif a.type == 1:
                if cur > mine:
                    raise ConvertError("check_facing_bet")
                text = "checks"
            elif a.type == 3 or (a.type == 6 and x <= cur):
                add = x if a.type == 3 else x - mine
                if a.type == 3 and mine + add != cur and add != stack[s] - put[s]:
                    if x == cur and x > mine:    # an export that writes the call as the street total
                        add = x - mine
                    else:
                        raise ConvertError("call_amount_mismatch")
                text = f"calls {_amt(add)}" + put_in(s, add)
                total[s] = mine + add
            elif a.type in (4, 5, 6):
                if x <= cur:
                    raise ConvertError("raise_not_above_level")
                text = (f"bets {_amt(x)}" if cur == 0 else f"raises {_amt(x - cur)} to {_amt(x)}") + put_in(s, x - mine)
                total[s] = cur = x
                last_aggr[st] = s
            else:
                raise ConvertError("unknown_action")
            out.append(f"{name[s]}: {text}")
            order += 1
            times.append((h.hand_uid, st, order, float(a.time)))
        street_lines[st] = out

    alive = [s for s in seats if s not in folded_on]
    acted = [i for i, st in enumerate(STREETS) if street_lines[st]]
    closed = max(acted) if acted else 0
    last = closed if len(alive) == 1 else 3
    if len(h.board) < BOARD_LEN[STREETS[last]]:
        raise ConvertError("board_incomplete")
    board = h.board[:BOARD_LEN[STREETS[last]]]

    top = max(seats, key=lambda s: put[s])
    second = max(put[s] for s in seats if s != top)
    uncalled = put[top] - second
    if uncalled > 0:
        put[top] -= uncalled
    for i, st in enumerate(STREETS[:last + 1]):
        if st == "flop":
            lines.append(f"*** FLOP *** [{_cards(board[:3])}]")
        elif st == "turn":
            lines.append(f"*** TURN *** [{_cards(board[:3])}] [{_cards(board[3:4])}]")
        elif st == "river":
            lines.append(f"*** RIVER *** [{_cards(board[:4])}] [{_cards(board[4:5])}]")
        lines += street_lines[st]
        if i == closed and uncalled > 0:
            lines.append(f"Uncalled bet ({_amt(uncalled)}) returned to {name[top]}")

    showdown = len(alive) > 1
    shown = {s: h.seats[s].hole for s in alive if h.seats[s].hole} if showdown else {}
    desc = {s: describe_hand(_cards(c + board)) for s, c in shown.items()}
    if showdown:
        lines.append("*** SHOW DOWN ***")
        start = seats.index(btn_s) + 1
        order_sd = [seats[(start + k) % len(seats)] for k in range(len(seats))]
        order_sd = [s for s in order_sd if s in alive]
        if "river" in last_aggr and last_aggr["river"] in order_sd:
            order_sd.remove(last_aggr["river"])
            order_sd.insert(0, last_aggr["river"])
        for s in order_sd:
            lines.append(f"{name[s]}: shows [{_cards(shown[s])}] ({desc[s]})" if s in shown
                         else f"{name[s]}: mucks hand")

    coll: dict[int, int] = {}
    for s in seats:
        uid = h.seats[s].uid
        c = _units(h.settlement[uid]) + put[s] if uid in h.settlement else 0
        if c < 0:
            raise ConvertError("settlement_below_contribution")
        coll[s] = c
    pot = sum(put.values())
    rake = pot - sum(coll.values())
    if rake < 0:
        raise ConvertError("settlement_above_pot")
    best = set()
    if shown:
        top_rank = max(evaluate(board + c) for c in shown.values())
        best = {s for s, c in shown.items() if evaluate(board + c) == top_rank}
    else:
        best = set(alive)
    if any(c > 0 and s not in best for s, c in coll.items()):
        remarks["settlement_not_by_cards"] += 1
    lines += [f"{name[s]} collected {_amt(coll[s])} from pot" for s in seats if coll[s] > 0]

    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {_amt(pot)} | Rake {_amt(rake)}")
    if board:
        lines.append(f"Board [{_cards(board)}]")
    for s in seats:
        text = f"Seat {tseat[s]}: {name[s]}"
        text += " (button)" if s == btn_s else ""
        text += " (small blind)" if s == sb_s else ""
        text += " (big blind)" if s == bb_s else ""
        if s in folded_on:
            st = folded_on[s]
            text += " folded before Flop" if st == "preflop" else f" folded on the {_SUMMARY_STREET[st]}"
        elif not showdown:
            text += f" collected ({_amt(coll[s])})"
        elif s not in shown:
            text += " mucked"
        elif coll[s] > 0:
            text += f" showed [{_cards(shown[s])}] and won ({_amt(coll[s])}) with {desc[s]}"
        else:
            text += f" showed [{_cards(shown[s])}] and lost with {desc[s]}"
        lines.append(text)
    return "\n".join(lines) + "\n", times, remarks


def convert(rows, out_dir: str | Path, log=print) -> Counter:
    """Write ``hands.txt`` and ``decision_times.csv`` into ``out_dir``; returns the funnel."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    funnel: Counter = Counter()
    texts: list[str] = []
    times: list[tuple] = []
    for row in rows:
        h = parse_hand(row)
        if h is None:
            funnel["unparseable"] += 1
            continue
        try:
            text, t, remarks = hand_text(h, small_blind_of(row))
        except ConvertError as exc:
            funnel[exc.reason] += 1
            continue
        texts.append(text)
        times += t
        funnel["written"] += 1
        funnel.update(remarks)
    (out / "hands.txt").write_text("\n".join(texts), encoding="utf-8", newline="\n")
    with open(out / "decision_times.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TIME_COLUMNS)
        w.writerows(times)
    log(f"{funnel['written']:,} hands written to {out / 'hands.txt'}; "
        + ", ".join(f"{k}: {v}" for k, v in sorted(funnel.items()) if k != "written"))
    return funnel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Operator replay rows (JSON) -> hand-history text + decision times")
    ap.add_argument("--input", default=str(ROOT / "data" / "sample_hands.csv"),
                    help="CSV, parquet or JSON(L) file of replay rows (default: data/sample_hands.csv)")
    ap.add_argument("--out", required=True, help="folder for hands.txt and decision_times.csv")
    a = ap.parse_args(argv)
    convert(read_hands(a.input).to_dict("records"), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
