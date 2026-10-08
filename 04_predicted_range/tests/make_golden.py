"""Write ``golden/river_raise.json``: one decision of the narrowing example, the slices of the fitted tables the
range model reads for it, and the range the model gave when the file was made.

The decision is the hero's last one in the example hand of ``config/step04_range.toml`` (``[examples]``): he
faces the opponent's river raise after a flop bet and a turn bet. The file was generated once from ``sim_demo``,
checked by hand (after the river raise the range holds mostly sevens and full houses), and frozen;
``test_golden.py`` rebuilds the range from the file alone. ``river_raise.json`` holds the default settings,
``river_raise_cap240.json`` the narrow-setting constants of ``variants/cap240.toml``. Regenerate only on purpose,
after a change to the model that is meant to move the numbers::

    python 04_predicted_range/tests/make_golden.py --dataset <full-information dataset>
    python 04_predicted_range/tests/make_golden.py --dataset <full-information dataset> \
        --config 04_predicted_range/variants/cap240.toml --out river_raise_cap240.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step04_testlib import REPO, STEP_DIR, load_script  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import rangemodel as rm  # noqa: E402
from sudetect import warehouse  # noqa: E402
from sudetect.cards import CLASS_LABELS, format_cards  # noqa: E402
from sudetect.equity import equity_vs_range  # noqa: E402

GOLDEN = HERE / "golden" / "river_raise.json"


def _columns(df: pd.DataFrame) -> dict[str, list[Any]]:
    """A frame as {column: values}, NaN as None (compact JSON)."""
    out = {}
    for col in df.columns:
        vals = df[col].tolist()
        out[col] = [None if isinstance(v, float) and np.isnan(v) else (int(v) if isinstance(v, np.integer) else v)
                    for v in vals]
    return out


def expected(tables: rm.FitTables, cfg: rm.RangeConfig, line: rm.OpponentLine, hero: Sequence[int],
             board: Sequence[int]) -> dict[str, Any]:
    """The numbers the golden test checks."""
    dead = tuple(hero) + tuple(board)
    res = rm.predict_range(tables, cfg, line, dead)
    before = rm.predict_range(tables, cfg, dataclasses.replace(line, nodes=line.nodes[:-1]), dead)
    mass = rm.range_classes(res.ids, res.w)
    order = np.argsort(-mass, kind="stable")[:10]
    return {
        "prior_level": res.prior_level,
        "n_combos": int(res.ids.size),
        "top_classes": [[CLASS_LABELS[i], round(float(mass[i]), 6)] for i in order],
        "groups": {g: round(v, 6) for g, v in rm.group_shares(res.ids, res.w, board).items()},
        "air_before_last_action": round(rm.group_shares(before.ids, before.w, board)["air"], 6),
        "eq_vs_range": round(float(equity_vs_range(list(hero), list(board), rm.range_dense(res.ids, res.w))), 6),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="sim_demo")
    ap.add_argument("--config", default=None, help="TOML file overriding config/step04_range.toml")
    ap.add_argument("--out", default=GOLDEN.name, help="file name under golden/")
    args = ap.parse_args(argv)
    run04 = load_script("step04_run", STEP_DIR / "run.py")
    examples = load_script("step04_examples", STEP_DIR / "examples.py")
    cfg = run04.load_config(args.config)
    rcfg = rm.RangeConfig.from_toml(cfg)
    con = run04.connect(args.dataset, read_only=True)
    try:
        full, _ = run04.load_tables(con)
        hand_id = str(cfg["examples"]["hand_id"])
        hit = [c for c in examples.find_candidates(con, cfg["examples"]["pattern"]) if c[0] == hand_id]
        if not hit:
            raise SystemExit(f"example hand {hand_id} not found in {args.dataset}")
        _, opp_id, hero_id = hit[0]
        h = examples.load_hand(con, cfg, hand_id, opp_id, hero_id)
        acts = h["acts"]
        last = acts[(acts.player_id == hero_id) & ~acts.is_post].iloc[-1]
        board = rm.cards_of(last.board_at_action)
        nodes = tuple(n for ai, n in h["nodes"] if ai < int(last.action_index))
        line = dataclasses.replace(h["line"], nodes=nodes)
        lik = warehouse.query_df(con, "SELECT * FROM rng_lik_post ORDER BY level, street, situation, pose, "
                                      "pot_type, made_hand, draw")
        need = np.zeros(len(lik), dtype=bool)
        for n in nodes:                      # the rows a lookup at this spot can reach, level by level
            at = (lik.street == n.street) & (lik.situation == n.situation)
            need |= at & (lik.level == "K6") & (lik.pose == n.pose) & (lik.pot_type == n.pot_type)
            need |= at & (lik.level == "K5") & (lik.pose == n.pose)
            need |= at & lik.level.isin(["K4", "K3", "K1"])
            need |= at & (lik.level == "K2") & (lik.pose == n.pose) & (lik.pot_type == n.pot_type)
        lik = lik[need]
        pri = warehouse.query_df(con, "SELECT * FROM rng_prior_pre ORDER BY level, size_class, preflop_situation, "
                                      "position_class, action, hand_class")
        keep = (
            ((pri.level == "P4") & (pri.size_class == line.size_cls) & (pri.preflop_situation == line.pf_situation)
             & (pri.position_class == line.position_class) & (pri.action == line.pf_action))
            | ((pri.level == "P3") & (pri.preflop_situation == line.pf_situation)
               & (pri.position_class == line.position_class) & (pri.action == line.pf_action))
            | ((pri.level == "P2") & (pri.preflop_situation == line.pf_situation)
               & (pri.position_class == line.position_class))
            | ((pri.level == "P1") & (pri.position_class == line.position_class))
            | (pri.level == "P0")
        )
        pri = pri[keep]
        fld = warehouse.query_df(con, "SELECT * FROM rng_field ORDER BY size_class")
    finally:
        con.close()
    sliced = rm.FitTables.from_frames(lik, pri, fld)
    want = expected(full, rcfg, line, h["hero"], board)
    got = expected(sliced, rcfg, line, h["hero"], board)
    if want != got:
        raise SystemExit(f"the table slices do not reproduce the full tables:\n{want}\n{got}")
    doc = {
        "about": "One decision of the step-04 narrowing example, the slices of the fitted tables it reads and the "
                 "expected range (generated by make_golden.py, reviewed, frozen).",
        "dataset": args.dataset,
        "hand_id": hand_id,
        "range_config": dataclasses.asdict(rcfg),
        "hero": format_cards(h["hero"], sep=""),
        "board": format_cards(board, sep=""),
        "line": {k: v for k, v in dataclasses.asdict(line).items() if k != "nodes"},
        "nodes": [{"street": n.street, "situation": n.situation, "pose": n.pose, "pot_type": n.pot_type,
                   "action": n.action, "board": format_cards(n.board, sep="")} for n in nodes],
        "tables": {"rng_lik_post": _columns(lik), "rng_prior_pre": _columns(pri), "rng_field": _columns(fld)},
        "expected": want,
    }
    dest = GOLDEN.parent / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(want, indent=1))
    print(f"written: {dest} ({dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
