"""Rows in, equity out: five decision rows from the warehouse and their equity references.

For each row the script prints the input fields (cards, board, pot, amount to
call), computes the references with the single-spot functions of
:mod:`sudetect.equity` and prints them next to the pot odds and, when step 02
has run, the values stored in ``action_equity``:

* equity vs all hands: ``equity_vs_random(hero, board)``, against ONE random hand;
* equity vs the opponent's hand: ``equity_vs_hand(hero, villain, board)``, or
  ``multiway_equity([hero, *villains], board)[0]`` with several live opponents;
* equity vs the predicted range: computed by step 04 (``action_range_equity``).

Usage::

    python 02_equity/examples.py                       # article_sample
    python 02_equity/examples.py --dataset <name> --pick 3
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sudetect import warehouse  # noqa: E402
from sudetect.cards import format_cards, parse_cards  # noqa: E402
from sudetect.equity import equity_vs_hand, equity_vs_random, multiway_equity  # noqa: E402
from sudetect.evaluator import describe_hand  # noqa: E402

SQL_FILE = HERE / "sql" / "examples.sql"
KINDS = {1: "flop call, heads-up", 2: "turn fold, heads-up", 3: "river call, heads-up",
         4: "river bet, heads-up", 5: "facing a bet with two or more opponents",
         6: "turn decision, heads-up (no turn fold with known cards in this dataset)"}
LOCK_WAIT_S = 120.0


def _cards(text: Optional[str]) -> str:
    return format_cards(parse_cards(text)) if isinstance(text, str) and text else "-"


def _fmt(x: Any, digits: int = 4) -> str:
    return "NULL" if x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA else f"{x:.{digits}f}"


@dataclass
class Example:
    """One example row, the live opponents at that row and the stored step-02 values (None before step 02)."""

    row: Any
    opps: pd.DataFrame
    stored: Optional[pd.Series]


def _step_run() -> Any:
    """The step script ``run.py``, loaded under its own module name (its folder is not a package)."""
    name = "step02_equity_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def fetch(dataset: str, pick: int) -> list[Example]:
    """Read the example rows of one dataset in one short read-only session."""
    rows_sql, opps_sql = warehouse.split_statements(SQL_FILE.read_text(encoding="utf-8"))
    out: list[Example] = []
    with _step_run().session(dataset, True, LOCK_WAIT_S) as con:   # waits out a locked file
        rows = warehouse.query_df(con, rows_sql, pick=pick)
        stored_ok = warehouse.table_exists(con, "action_equity")
        for r in rows.itertuples(index=False):
            key = {"hand_id": r.hand_id, "action_index": int(r.action_index)}
            opps = warehouse.query_df(con, opps_sql, **key)
            stored = None
            if stored_ok:
                e = warehouse.query_df(con, "SELECT * FROM action_equity "
                                            "WHERE hand_id = :hand_id AND action_index = :action_index", **key)
                stored = e.iloc[0] if len(e) else None
            out.append(Example(r, opps, stored))
    return out


def show_dataset(dataset: str, pick: int, log: Callable[[str], None] = print) -> int:
    """Print the examples of one dataset; returns the number of rows shown. Equities are computed here."""
    examples = fetch(dataset, pick)
    log(f"=== {dataset}: {len(examples)} example rows ===\n")
    for ex in examples:
        r, opps, s = ex.row, ex.opps, ex.stored
        hero = parse_cards(r.hole_cards)
        board = parse_cards(r.board_at_action)
        log(f"Example {r.example}: {KINDS[int(r.example)]} (hand {r.hand_id}, row {r.action_index})")
        log(f"  input row: {r.player_id} ({r.position}) holds {_cards(r.hole_cards)}, {r.street} board "
            f"{_cards(r.board_at_action)}: {describe_hand(hero + board)}")
        log(f"  pot before the row {r.pot_before_bb:.2f} bb, to call {r.to_call_eff_bb:.2f} bb, "
            f"action {r.action} {r.amount_eff_bb:.2f} bb")
        for o in opps.itertuples(index=False):
            log(f"  live opponent {o.player_id} ({o.position}) holds {_cards(o.hole_cards)}")
        v_random = equity_vs_random(hero, board)
        log(f"  equity vs all hands (one random hand):  {v_random:.4f}   equity_vs_random(hero, board), exact")
        known = [c for c in opps["hole_cards"] if isinstance(c, str)]
        if len(known) == len(opps) and len(opps):
            villains = [parse_cards(c) for c in known]
            if len(villains) == 1:
                v_real = equity_vs_hand(hero, villains[0], board)
                call = "equity_vs_hand(hero, villain, board)"
            else:
                v_real = multiway_equity([hero] + villains, board)[0]
                call = "multiway_equity([hero, *villains], board)[0]"
            log(f"  equity vs the opponent's hand:          {v_real:.4f}   {call}, exact")
        else:
            log("  equity vs the opponent's hand:          NULL     (an opponent's cards are not known)")
        log("  equity vs the predicted range:          step 04  (action_range_equity)")
        if r.pot_odds is None or (isinstance(r.pot_odds, float) and math.isnan(r.pot_odds)):
            log("  pot odds:                               none     (no bet to call)")
        else:
            log(f"  pot odds:                               {r.pot_odds:.4f}   "
                f"(a call needs {100 * r.pot_odds:.1f} % of the final pot)")
        if s is not None:
            log(f"  stored in action_equity: eq_vs_random {_fmt(s.eq_vs_random)} ({s.eq_method}, se "
                f"{_fmt(s.eq_se)}), eq_vs_real {_fmt(s.eq_vs_real)}, ahead_now "
                f"{'NULL' if pd.isna(s.ahead_now) else str(bool(s.ahead_now)).lower()}")
        log("")
    return len(examples)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Print five decision rows and their equity references.")
    ap.add_argument("--dataset", default="article_sample")
    ap.add_argument("--pick", type=int, default=1, help="show the n-th row of each kind (default 1)")
    args = ap.parse_args(argv)
    rows_sql = warehouse.split_statements(SQL_FILE.read_text(encoding="utf-8"))[0]
    print("SQL that selects the rows (sql/examples.sql):\n")
    print(rows_sql.replace(":pick", str(args.pick)) + ";\n")
    for dataset in [args.dataset]:
        if not warehouse.db_path(dataset).exists():
            print(f"error: no warehouse for {dataset!r}; run 01_action_table/run.py first", file=sys.stderr)
            return 2
        show_dataset(dataset, args.pick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
