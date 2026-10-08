"""Step 03: one row per player with the 16 classic statistics and the 9 hidden-information signals.

The statistics and signals are those of the per-suspect forensic package (``superuser.stats.player_stats`` and
``superuser.signals.player_signals``), fed from the step 01 and step 02 tables through ``05_detection/adapter.py``.
Writes the table ``player_stats`` into the dataset's warehouse (replaced on every run). A signal is stored as its
value; its sample size goes to ``<signal>_n``.

Usage::

    python 03_player_stats/run.py --dataset article_sample [--heads-up-only] [--min-hands 1]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sudetect import warehouse  # noqa: E402
from superuser.signals import player_signals  # noqa: E402
from superuser.stats import player_stats  # noqa: E402

STATS = ["hands", "net_win", "bb_100", "vpip", "pfr", "three_bet", "fold_to_3bet", "flop_cbet", "fold_to_cbet",
         "aggression_factor", "aggression_frequency", "wtsd", "w_sd", "fold_to_river_bet", "river_call_win_rate",
         "median_think_time"]
SIGNALS = ["perfect_fold_ratio_postflop", "perfect_fold_ratio_turn", "perfect_fold_ratio_river",
           "pair_plus_folds_behind", "bluff_index", "fold_vs_value_bet", "fold_vs_bluff", "discrimination_gap",
           "river_decision_accuracy"]


def _adapter() -> Any:
    name = "step05_adapter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "05_detection" / "adapter.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def build_table(con: Any, heads_up_only: bool = False, times: Optional[dict] = None, min_hands: int = 1) -> pd.DataFrame:
    sql = "SELECT hand_id FROM hands WHERE variant = 'standard' AND NOT error_flag"
    if heads_up_only:
        sql += " AND n_dealt = 2"
    ids = warehouse.query_df(con, sql).hand_id.tolist()
    ad = _adapter()
    dec, hands, uid_of = ad.build(con, ids, times)
    by_uid: dict[int, list] = defaultdict(list)
    for h in hands.values():
        for u in h.uids:
            by_uid[u].append(h)
    rows = []
    for pid, uid in sorted(uid_of.items()):
        if len(by_uid[uid]) < min_hands:
            continue
        row: dict[str, Any] = {"player_id": pid}
        row.update({k: player_stats(by_uid[uid], uid)[k] for k in STATS})
        sig = player_signals(dec, uid) if not dec.empty else {}
        for k in SIGNALS:
            s = sig.get(k) or {}
            row[k] = s.get("value")
            row[f"{k}_n"] = s.get("n")
        rows.append(row)
    return pd.DataFrame(rows, columns=["player_id"] + STATS + [c for k in SIGNALS for c in (k, f"{k}_n")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 03: classic statistics and hidden-information signals per player")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heads-up-only", action="store_true")
    ap.add_argument("--min-hands", type=int, default=1)
    ap.add_argument("--times", help="decision_times.csv (default: $SU_DATA_DIR/<dataset>/input/decision_times.csv)")
    a = ap.parse_args(argv)
    times_file = Path(a.times) if a.times else warehouse.data_dir() / a.dataset / "input" / "decision_times.csv"
    con = warehouse.connect(a.dataset)
    try:
        df = build_table(con, a.heads_up_only, _adapter().read_times(times_file), a.min_hands)
        warehouse.insert_df(con, "player_stats", df, mode="replace")
    finally:
        con.close()
    print(f"player_stats: {len(df):,} players, {len(df.columns) - 1} columns")
    print(df[["player_id"] + STATS[:5] + SIGNALS[:3]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
