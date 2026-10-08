"""Evaluation of step 05 on a simulated dataset against its answer key.

This script reads the simulator's answer key (``ground_truth.json`` and ``population.json`` next to the hand
files); the pipeline never does. It prints the honest field's z_b distribution, the planted players' values, the
single-victim pair test and the onset changepoints.

    python 05_detection/eval_quick.py --dataset sim_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import warehouse  # noqa: E402

PLANT_COLS = ["time_window", "hands", "bb100", "z_b", "z_b_pct", "x_dead", "x_win", "nd", "nw", "e_disc", "cmh_z",
              "ofr_pair", "ofr_tr", "disc_gap", "bluff_index", "s1_flag", "s2_flag"]


def answer_key(dataset: str) -> tuple[dict, dict]:
    folder = warehouse.data_dir() / dataset
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    pop = json.loads((folder / "population.json").read_text(encoding="utf-8"))
    return truth, pop


def field_quantiles(x: pd.Series) -> dict:
    x = x.dropna().to_numpy(float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": round(float(np.median(x)), 3), "sd": round(float(x.std(ddof=1)), 3)
            if x.size > 1 else None, "p99": round(float(np.quantile(x, 0.99)), 3),
            "p99.9": round(float(np.quantile(x, 0.999)), 3), "max": round(float(x.max()), 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim_small")
    a = ap.parse_args(argv)
    truth, pop = answer_key(a.dataset)
    plants = {p["player"]: p for p in truth["plants"]}
    victims = {p["victim"] for p in truth["plants"] if p.get("victim")}
    strong = {p["name"] for p in pop["players"] if p.get("archetype") == "strong" and not p.get("planted")}
    con = warehouse.connect(a.dataset, read_only=True)
    ps = warehouse.query_df(con, "SELECT * FROM player_scores")
    pairs = warehouse.query_df(con, "SELECT * FROM pair_scores")
    esc = warehouse.query_df(con, "SELECT * FROM escalations")
    meta = warehouse.query_df(con, "SELECT * FROM report_meta")
    con.close()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    honest = ps[~ps["player_id"].isin(plants)]
    print(f"== {a.dataset}: honest field, z_b of players with >= 20 dead and >= 20 winning decisions")
    for w, g in honest[honest["rb_eligible"].fillna(False).astype(bool)].groupby("time_window"):
        print(f"  {w:7s} z_b {field_quantiles(g['z_b'])}")
        print(f"  {w:7s} E   {field_quantiles(g['e_disc'])}")
    print("\n== planted players")
    rows = ps[ps["player_id"].isin(plants)].copy()
    rows.insert(0, "plant", rows["player_id"].map(lambda p: plants[p]["plant_id"]))
    print(rows[["plant"] + PLANT_COLS].sort_values(["plant", "time_window"]).round(3).to_string(index=False))
    allw = ps[ps["time_window"] == "all"]
    print("\n== win-rate runs percentile (all window): plants and honest strong regulars")
    wr = allw[allw["player_id"].isin(set(plants) | strong) & allw["wr_pct_median"].notna()].copy()
    wr["who"] = wr["player_id"].map(lambda p: plants[p]["plant_id"] if p in plants else "strong_honest")
    print(wr[["who", "player_id", "wr_hands", "wr_run_bb100", "wr_pct_median", "wr_pct_lo", "wr_pct_hi",
              "wr_pct_pooled", "wr_pct_best_run", "z_b", "s1_flag", "s2_flag"]].round(2).to_string(index=False))
    flags = allw.assign(planted=allw["player_id"].isin(plants))
    print("\n== Stage 1 / z_b screen in each window (honest players flagged)")
    for w, g in ps.groupby("time_window"):
        h = g[~g["player_id"].isin(plants)]
        print(f"  {w:7s} s1 {int(h['s1_flag'].sum())}/{int(h['s1_eligible'].sum())}  "
              f"s2 {int(h['s2_flag'].sum())}  z_b>=4 {int(h['rb_screen'].fillna(False).sum())}/"
              f"{int(h['rb_eligible'].fillna(False).sum())}")
    print("\n== Stage 5 escalations: plants / honest")
    for (rs, rule), g in esc.groupby(["rule_set", "rule"]):
        pl = g[g["player_id"].isin(plants)]
        print(f"  {rs:18s} {rule:22s} plants {int(pl['escalate'].sum())}/{len(pl)}  honest "
              f"{int(g['escalate'].sum() - pl['escalate'].sum())}/{len(g) - len(pl)}  "
              f"honest median share {g[~g['player_id'].isin(plants)]['ratio'].median():.3f}  "
              f"plants' shares {sorted(pl['ratio'].round(3).tolist())}")
    print("\n== single victim: pair tests")
    print(meta[meta["item"].str.startswith("pairs")].to_string(index=False))
    for pid, p in plants.items():
        if p.get("victim"):
            r = pairs[(pairs["player_id"] == pid) & (pairs["opponent_id"] == p["victim"])]
            print(f"  {p['plant_id']} -> victim:",
                  r[["pair_hands", "pair_share", "nd", "nw", "s", "s_rest", "z_split", "a_i", "b_j", "eps",
                     "z_eps", "eligible", "test_z", "q_value", "bh_pass"]].round(3).to_string(index=False))
    honest_pairs = pairs[pairs["eligible"].fillna(False).astype(bool) & ~pairs["player_id"].isin(plants)]
    print(f"  honest eligible pairs: {len(honest_pairs)}, BH pass: {int(honest_pairs['bh_pass'].sum())}")
    print("\n== onset: changepoint (all window)")
    on = allw[allw["player_id"].isin(plants)][["player_id", "cp_z", "cp_week_start", "cp_splits"]].copy()
    on["plant"] = on["player_id"].map(lambda p: plants[p]["plant_id"])
    on["planted_from"] = on["player_id"].map(lambda p: plants[p]["from_date"])
    print(on.to_string(index=False))
    print(f"  honest cp_z: {field_quantiles(allw[~allw['player_id'].isin(plants)]['cp_z'])}")
    print(f"\nvictims (honest): {sorted(victims)}")


if __name__ == "__main__":
    main()
