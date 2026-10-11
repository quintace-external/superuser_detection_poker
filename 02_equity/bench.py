"""Speed of step 02 on this machine: decision rows per second, per street and per backend.

Rows come from a loaded warehouse (default ``article_sample``): the first ``--rows``
decision rows of each street and the first ``--rows`` heads-up hands. Every
backend scores them in this process with :func:`sudetect.equity_backends.score_decisions`
(card parsing, equity vs a random hand, equity vs the real cards and
``ahead_now``) or :func:`sudetect.equity_backends.score_hu_hands`. Rates are
computed from the CPU time of this process, so other work on the machine
changes them less than the wall time, which is printed as well. With
``--procs`` the script also times the numpy backend on all the selected rows
through a process pool, the way ``run.py`` uses it (pool start-up excluded).
Finally it runs :func:`sudetect.equity_backends.cross_check` on ``--cross-check``
random spots for each pair of backends and prints the largest differences.

Usage::

    python 02_equity/bench.py
    python 02_equity/bench.py --dataset <name> --rows 5000 --procs 16
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sudetect import equity_backends as eb  # noqa: E402
from sudetect import warehouse  # noqa: E402

SQL_DIR = HERE / "sql"
STREETS = ("flop", "turn", "river")


def load_rows(dataset: str, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The first n decision rows of each street and the first n heads-up hands of a dataset."""
    con = warehouse.connect(dataset, read_only=True)
    try:
        plan = warehouse.query_df(con, (SQL_DIR / "plan_decisions.sql").read_text(encoding="utf-8"),
                                  include_preflop=False, preflop_max_opp=0)
        cum = plan["n_rows"].cumsum().to_numpy()
        last = plan["hand_id"].iloc[min(len(plan) - 1, int((cum < 8 * n).sum()))]
        rows = warehouse.query_df(con, (SQL_DIR / "inputs_decisions.sql").read_text(encoding="utf-8"),
                                  lo=plan["hand_id"].iloc[0], hi=last, include_preflop=False, preflop_max_opp=0)
        hplan = warehouse.query_df(con, (SQL_DIR / "plan_hands.sql").read_text(encoding="utf-8"))
        hands = warehouse.query_df(con, (SQL_DIR / "inputs_hands.sql").read_text(encoding="utf-8"),
                                   lo=hplan["hand_id"].iloc[0], hi=hplan["hand_id"].iloc[min(n, len(hplan)) - 1])
    finally:
        con.close()
    rows = pd.concat([rows[rows["street"] == s].head(n) for s in STREETS], ignore_index=True)
    return rows, hands.head(n)


def _cols(frame: pd.DataFrame, names: Sequence[str]) -> dict:
    return {c: frame[c].to_numpy() if frame[c].dtype.kind in "iuf" else frame[c].tolist() for c in names}


def backends(seed: int) -> dict[str, tuple[eb.Backend, eb.Backend]]:
    """name -> (backend for decision rows, backend for heads-up hands)."""
    out: dict[str, tuple[eb.Backend, eb.Backend]] = {
        "numpy": (eb.NumpyBackend(seed=seed), eb.NumpyBackend(seed=seed, mc_samples=8192)),
        "numpy, exact flop": (eb.NumpyBackend(seed=seed, flop_random_exact=True), None),
    }
    if eb.pyrust_available():
        out["pyrust"] = (eb.PyrustBackend(seed=seed), eb.PyrustBackend(seed=seed))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Decision rows per second, per street and per backend.")
    ap.add_argument("--dataset", default="article_sample")
    ap.add_argument("--rows", type=int, default=3000, help="rows per street (and heads-up hands)")
    ap.add_argument("--procs", type=int, default=0, help="also time a process pool of this size (0 = skip)")
    ap.add_argument("--seed", type=int, default=20011)
    ap.add_argument("--cross-check", type=int, default=200, help="random spots per backend pair (0 = skip)")
    args = ap.parse_args(argv)
    if not warehouse.db_path(args.dataset).exists():
        print(f"error: no warehouse for {args.dataset!r}", file=sys.stderr)
        return 2
    rows, hands = load_rows(args.dataset, args.rows)
    print(f"machine: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"Python {platform.python_version()}; dataset {args.dataset}")
    print("\n| backend | street | rows | CPU s | rows/s | ms/row | wall s |\n|---|---|---|---|---|---|---|")
    eb.score_decisions(_cols(rows.head(50), eb.DECISION_COLUMNS), eb.NumpyBackend())      # warm the tables
    for name, (dec_backend, hand_backend) in backends(args.seed).items():
        jobs = [(street, rows[rows["street"] == street], eb.DECISION_COLUMNS, eb.score_decisions, dec_backend)
                for street in STREETS]
        if hand_backend is not None and len(hands):
            jobs.append(("preflop (heads-up hand)", hands, eb.HAND_COLUMNS, eb.score_hu_hands, hand_backend))
        for label, part, names, score, backend in jobs:
            cols = _cols(part, names)
            c0, t0 = time.process_time(), time.perf_counter()
            score(cols, backend)
            cpu, wall = time.process_time() - c0, time.perf_counter() - t0
            print(f"| {name} | {label} | {len(part):,} | {cpu:.2f} | {len(part) / cpu:,.0f} | "
                  f"{1000 * cpu / len(part):.3f} | {wall:.2f} |")
    if args.procs > 1:
        sys.path.insert(0, str(HERE))
        from run import Runner, _chunks, _merge_results  # noqa: E402  (the step script, imported for its pool)

        runner = Runner(args.procs)
        try:
            warm = [("decisions", "numpy", {"seed": args.seed, "mc_samples": 4096, "flop_random_exact": False},
                     c) for c in _chunks(rows.head(20 * args.procs), "decisions", 20)]
            runner.collect(runner.submit(warm))
            tasks = [("decisions", "numpy", {"seed": args.seed, "mc_samples": 4096, "flop_random_exact": False},
                      c) for c in _chunks(rows, "decisions", max(100, len(rows) // (4 * args.procs)))]
            t0 = time.perf_counter()
            _merge_results(runner.collect(runner.submit(tasks)))
            dt = time.perf_counter() - t0
        finally:
            runner.close()
        print(f"\nnumpy backend, {args.procs} processes, {len(rows):,} rows ({', '.join(STREETS)} mixed "
              f"{args.rows:,} each): {dt:.2f} s, {len(rows) / dt:,.0f} rows/s")
    if args.cross_check > 0:
        print_cross_checks(args.cross_check, args.seed)
    return 0


def print_cross_checks(n_spots: int, seed: int) -> None:
    """Largest differences between backends on random flop, turn and river spots with one to three opponents."""
    # (label, a, b, tolerance for two exact values); pyrust splits pots with integer arithmetic, hence 1e-7
    pairs: list[tuple[str, eb.Backend, eb.Backend, float]] = [
        ("numpy, exact flop vs reference", eb.NumpyBackend(flop_random_exact=True), eb.ReferenceBackend(), 1e-9),
        ("numpy vs reference", eb.NumpyBackend(seed=seed), eb.ReferenceBackend(), 1e-9),
    ]
    if eb.pyrust_available():
        pairs.append(("pyrust vs numpy, exact flop", eb.PyrustBackend(), eb.NumpyBackend(flop_random_exact=True),
                      1e-7))
    print(f"\ncross-checks on {n_spots} random spots (flop, turn, river; 1 to 3 opponents; both equities per spot)")
    for label, a, b, tol in pairs:
        c = eb.cross_check(a, b, n_spots, seed=seed, exact_tol=tol)
        one = max(v for k, v in c.by_kind.items() if k.endswith("vs random") or k.endswith("vs 1 hand"))
        multi = max((v for k, v in c.by_kind.items() if k.endswith("hands")), default=0.0)
        print(f"  {label}: {c.n_values} values, max |diff| {c.max_abs_diff:.2e} (vs a random hand or one known "
              f"hand {one:.2e}, vs 2-3 known hands {multi:.2e}), max z {c.max_z:.2f}, "
              + ("pass" if c.passed else f"{len(c.failures)} outside the tolerance"))


if __name__ == "__main__":
    sys.exit(main())
