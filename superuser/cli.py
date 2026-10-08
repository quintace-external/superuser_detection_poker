"""Command line.

    python -m superuser                                   # sample file, sample player
    python -m superuser --input my_hands.csv --player 123 [--opponent 456] [--out output/]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_PLAYER
from .pipeline import analyse_player
from .report import write_report

COLUMNS = ["hand_uid", "start_ts", "end_ts", "table_no", "stake_level", "chip_type", "hand_replay", "hand_result"]


def read_hands(path) -> pd.DataFrame:
    """Read a CSV, parquet or JSON(L) file of hand records."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".json", ".jsonl"):
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    else:
        df = pd.read_csv(p, usecols=lambda c: c in COLUMNS)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Superuser forensics for hand-replay records")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="CSV, parquet or JSON(L) file of hand records (default: data/sample_hands.csv)")
    ap.add_argument("--player", type=int, default=DEFAULT_PLAYER, help=f"player id (default: {DEFAULT_PLAYER})")
    ap.add_argument("--opponent", type=int, help="only analyse hands where this player also sat")
    ap.add_argument("--min-blind", type=float)
    ap.add_argument("--heads-up-only", action="store_true")
    ap.add_argument("--chip-type")
    ap.add_argument("--flop-samples", type=int, default=200)
    ap.add_argument("--top", type=int, default=25, help="suspicious hands listed in the Markdown report")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT))
    a = ap.parse_args(argv)
    result = analyse_player(read_hands(a.input), a.player, a.opponent, a.min_blind, a.heads_up_only,
                            a.chip_type, a.flop_samples)
    out = write_report(result, a.out, a.top)
    print(f"{result['meta']['hands_analysed']:,} hands analysed, "
          f"{len(result['suspicious_hands']):,} suspicious; report written to {out}")


if __name__ == "__main__":
    main()
