"""Step 04, preflop: the hero's equity against the opponent's predicted preflop range.

For every preflop decision with exactly one live opponent (everyone else has folded; a heads-up table, or the
button against the big blind after the rest of a six-max table folded), the opponent's range is the field's range
for his preflop line so far, read from ``rng_prior_pre`` (step 04's ``fit``): the hands the field held when it
opened, 3-bet, 4-bet or limped from his position (``sudetect.rangemodel.FitTables.prior_for``, levels P4 ... P0).
An opponent who has not acted yet holds every hand, so the equity is the one against a random hand.

The equity is a Monte Carlo estimate (``[enrich] mc_samples`` deals, the hero's cards removed from the range); one
estimate per distinct (hole cards, opponent line) is computed and shared by every decision with that key.

Writes ``action_range_equity_pre`` (one row per decision; ``05_detection/screen.py`` reads it as ``r`` of the
preflop rows). Run it after ``04_predicted_range/run.py --steps fit``::

    python 04_predicted_range/preflop.py --dataset sim_small
    python 04_predicted_range/preflop.py --dataset mined_pool --fit-from full_info_pool

Caveat: the prior is fitted on the hands that saw a flop, keyed by the last preflop line, so a 4-bet range is read
from 4-bets that were called; with few such hands the lookup falls back to coarser levels (the ``prior_level``
column says which).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import equity, rangemodel, warehouse  # noqa: E402
from sudetect.cards import CLASS_LABELS  # noqa: E402
from sudetect.equity_backends import cards_from_text  # noqa: E402

SQL = HERE / "sql" / "stage_preflop.sql"
OUT_TABLE = "action_range_equity_pre"
CHUNK = 2000


def _step04() -> Any:
    spec = importlib.util.spec_from_file_location("step04_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def range_equity(stage: pd.DataFrame, tables: rangemodel.FitTables, rcfg: rangemodel.RangeConfig,
                 n_samples: int, seed: int) -> pd.DataFrame:
    """``stage`` (rows of ``sql/stage_preflop.sql``) with ``prior_level``, ``n_combos`` and ``eq_vs_range``."""
    out = stage.copy()
    out["prior_level"] = None
    out["n_combos"] = np.nan
    out["eq_vs_range"] = np.nan
    if out.empty:
        return out
    random_row = out["opp_situation"].isna()
    cls_eq = dict(zip(CLASS_LABELS, equity.PREFLOP_VS_RANDOM))
    out.loc[random_row, "prior_level"] = "random"
    out.loc[random_row, "n_combos"] = 1225.0
    out.loc[random_row, "eq_vs_range"] = out.loc[random_row, "hand_class"].map(cls_eq).astype(float)
    todo = out[~random_row]
    if todo.empty:
        return out
    keys = ["hole_cards", "size_class", "opp_situation", "opp_position_class", "opp_action"]
    uniq = todo[keys].drop_duplicates().reset_index(drop=True)
    hero = cards_from_text(uniq["hole_cards"].tolist(), 2)
    levels: list[str] = []
    n_combos = np.zeros(len(uniq))
    weights = np.zeros((len(uniq), 1326))
    for i, row in enumerate(uniq.itertuples(index=False)):
        prior, level = tables.prior_for(row.size_class, row.opp_situation, row.opp_position_class or "",
                                        row.opp_action or "check", rcfg)
        levels.append(level)
        if prior is None:
            continue
        ids, w = rangemodel.expand_prior(prior, hero[i].tolist(), rcfg)
        weights[i, ids] = w
        n_combos[i] = len(ids)
    eq = np.full(len(uniq), np.nan)
    board = np.full((len(uniq), 5), -1, dtype=np.int64)
    ok = np.flatnonzero(weights.sum(axis=1) > 0)
    for lo in range(0, len(ok), CHUNK):
        rows = ok[lo:lo + CHUNK]
        eq[rows] = equity.equity_vs_range_batch(hero[rows], board[rows], weights[rows], n_samples=n_samples,
                                                seed=seed)
    uniq["prior_level"] = levels
    uniq["n_combos"] = n_combos
    uniq["eq_vs_range"] = eq
    merged = todo.drop(columns=["prior_level", "n_combos", "eq_vs_range"]).merge(uniq, on=keys, how="left")
    merged.index = todo.index
    out.loc[todo.index, ["prior_level", "n_combos", "eq_vs_range"]] = merged[
        ["prior_level", "n_combos", "eq_vs_range"]]
    return out


def run(dataset: str, cfg: dict, fit_from: Optional[str] = None, log=print) -> pd.DataFrame:
    s4 = _step04()
    t0 = time.perf_counter()
    con = s4.connect(fit_from or dataset, read_only=True)
    try:
        if not warehouse.table_exists(con, "rng_prior_pre"):
            raise SystemExit(f"{fit_from or dataset} has no rng_prior_pre: run 04_predicted_range/run.py --steps fit")
        tables, _ = s4.load_tables(con)
    finally:
        con.close()
    con = s4.connect(dataset, read_only=True)
    try:
        stage = warehouse.run_sql_file(con, SQL)
    finally:
        con.close()
    rcfg = rangemodel.RangeConfig.from_toml(cfg)
    en = cfg["enrich"]
    out = range_equity(stage, tables, rcfg, int(en["mc_samples"]), int(en["seed"]))
    cols = ["hand_id", "action_index", "player_id", "opp_id", "size_class", "opp_situation", "opp_position_class",
            "opp_action", "prior_level", "n_combos", "eq_vs_range"]
    out = out[cols].astype({"action_index": "int16", "n_combos": "float64", "eq_vs_range": "float64"})
    con = s4.connect(dataset)
    try:
        warehouse.insert_df(con, OUT_TABLE, out, mode="replace")
    finally:
        con.close()
    lv = out["prior_level"].value_counts().to_dict()
    log(f"[preflop] {dataset}: {len(out):,} decisions, {out['eq_vs_range'].notna().sum():,} with an equity; "
        f"prior levels {lv}; {time.perf_counter() - t0:.1f} s")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fit-from", default=None, help="read rng_prior_pre from this full-information dataset")
    ap.add_argument("--config", default=None, help="TOML file overriding config/step04_range.toml")
    a = ap.parse_args(argv)
    run(a.dataset, _step04().load_config(a.config), a.fit_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
