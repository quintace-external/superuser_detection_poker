"""Generate a simulated dataset of hand histories (full information, or the mined view with --mined).

    python 00_simulator/simulate.py --dataset sim_small
    python 00_simulator/simulate.py --dataset sim_demo --procs 16
    python 00_simulator/simulate.py --dataset sim_demo --calibrate --hands 50000 --no-write
    python 00_simulator/simulate.py --dataset sim_small --mined --out $SU_DATA_DIR/sim_mined/hands

``--mined`` writes the same hands as an observer at the tables sees them: no
``Dealt to`` lines, hole cards only where a player shows them at the showdown.

Writes ``<out>/<date>_gNN.txt`` (default ``$SU_DATA_DIR/<dataset>/hands``)
and, next to that folder, ``ground_truth.json``, ``population.json`` and
``run_summary.json``. ``--calibrate`` counts the honest field's HUD
statistics while playing, prints them next to the target ranges of
``[sim.calibration]`` and writes ``calibration.json`` next to the other
JSON files (a run without ``--calibrate`` removes an older one, which
described the hands it replaced).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sudetect.sim.calibrate import (  # noqa: E402
    by_players_dealt,
    check_targets,
    format_calibration,
    player_table,
    summarize,
)
from sudetect.sim.population import load_sim_config  # noqa: E402
from sudetect.sim.run import data_dir, simulate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="sim_small",
                    help="sim_small | sim_demo | sim_pf | custom (sizes in [sim.datasets])")
    ap.add_argument("--out", type=Path, default=None, help="hands folder (default $SU_DATA_DIR/<dataset>/hands)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="worker processes")
    ap.add_argument("--seed", type=int, default=None, help="overrides [sim] seed")
    ap.add_argument("--config", type=Path, default=None, help="TOML file with a [sim] section")
    ap.add_argument("--calibrate", action="store_true", help="print the honest field's HUD stats vs the targets")
    ap.add_argument("--hands", type=int, default=None, help="override the dataset's target number of hands")
    ap.add_argument("--days", type=int, default=None, help="override the dataset's number of days")
    ap.add_argument("--players", type=int, default=None, help="override the dataset's number of players")
    ap.add_argument("--no-write", action="store_true", help="play without writing files (calibration runs)")
    ap.add_argument("--preflop-info", action="store_true",
                    help="the planted superusers use the information preflop too ([sim.superuser] preflop_info)")
    ap.add_argument("--mined", action="store_true",
                    help="write the hands as an observer at the tables sees them: no 'Dealt to' lines, hole cards "
                         "only where a player shows them at the showdown")
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.players is not None:
        overrides["datasets"] = {args.dataset: {"n_players": args.players}}
    if args.preflop_info:
        overrides["superuser"] = {"preflop_info": True}
    cfg = load_sim_config(args.config, overrides)
    out = args.out
    if out is None and not args.no_write:
        out = data_dir() / args.dataset / "hands"
    result = simulate(cfg, args.dataset, out_dir=out, procs=args.procs, seed=args.seed, calibrate=args.calibrate,
                      write=not args.no_write, target_hands=args.hands, n_days=args.days, log=print,
                      full_information=not args.mined)
    if result.out_dir is not None:
        print(f"hands in {result.out_dir}; ground truth in {result.out_dir.parent / 'ground_truth.json'}")
        print(f"{result.n_hands / max(result.seconds, 1e-9):,.0f} hands/s")
        stale = result.out_dir.parent / "calibration.json"
        if not args.calibrate and stale.exists():
            stale.unlink()  # it described the hands this run has just replaced
    if args.calibrate and result.counter is not None:
        groups = {}
        for p in result.population.players:
            groups[p.name] = p.plant_id if p.plant_id else p.archetype
        cal = cfg.raw["calibration"]
        summary = summarize(result.counter, groups, cfg.bb_units)
        players = player_table(result.counter, groups, cfg.bb_units)
        dealt = by_players_dealt(result.counter, [p.name for p in result.population.honest()])
        print()
        print(format_calibration(summary, cal["targets"], players, dealt, int(cal["player_min_hands"])))
        misses = [r for r in check_targets(summary, cal["targets"]) if not r[5]]
        print(f"\n{len(misses)} target(s) missed" if misses else "\nall targets met")
        if result.out_dir is not None:
            path = result.out_dir.parent / "calibration.json"
            rows = [dict(zip(("kind", "stat", "low", "high", "achieved", "inside"), r)) for r in
                    check_targets(summary, cal["targets"])]
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"targets": rows,
                           "groups": json.loads(summary.reset_index().to_json(orient="records"))}, f, indent=2)
                f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
