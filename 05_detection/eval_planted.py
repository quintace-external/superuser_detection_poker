"""Evaluation of step 05 on a simulated dataset against its answer key (the planted players).

This is the only code of the step that reads the simulator's answer key (``ground_truth.json`` and
``population.json`` next to the hand files); the pipeline never does. For every planted player it reports each
method's verdict and values, then the honest field's distributions of the same statistics, the strong honest
regulars as a control and the acceptance checks, and writes

* ``05_detection/results/eval_<dataset>.json``,
* ``05_detection/results/eval_<dataset>.md`` (a short summary).

The bars are the configured ones (``[step05.eval]`` in ``config/step05_screening.toml``); none of them was tuned
on the planted players. Recall statements hold for the simulator only.

    python 05_detection/eval_planted.py --dataset sim_demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import warehouse  # noqa: E402

RESULTS = HERE / "results"
WINDOWS = ("all", "last30", "last15")
TABLES = ("player_scores", "pair_scores", "window_scores", "escalations", "report_meta")


def step05_module():
    """The screening script ``screen.py`` loaded under the name ``step05_screen`` (shared with the tests)."""
    if "step05_screen" not in sys.modules:
        spec = importlib.util.spec_from_file_location("step05_screen", HERE / "screen.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["step05_screen"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["step05_screen"]


def answer_key(dataset: str) -> tuple[dict, dict]:
    """``ground_truth.json`` and ``population.json`` of a simulated dataset (evaluation code only)."""
    folder = warehouse.data_dir() / dataset
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    pop = json.loads((folder / "population.json").read_text(encoding="utf-8"))
    return truth, pop


def read_outputs(dataset: str, cfg: Mapping) -> dict[str, pd.DataFrame]:
    """The step 05 output tables (read-only, waiting for another process's write lock)."""
    run = step05_module()
    con = run._connect(dataset, True, float(cfg.get("retry_seconds", 600)))
    try:
        return {t: warehouse.query_df(con, f"SELECT * FROM {t}") for t in TABLES}
    finally:
        con.close()


# --------------------------------------------------------------------------------------------- helpers


def clean(x: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN and NaT to None, timestamps to ISO dates."""
    if isinstance(x, Mapping):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(x) else str(pd.Timestamp(x).date())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    return x


def quantiles(x: pd.Series | np.ndarray) -> dict:
    """n, median, p90, p99, p99.9, sd and max of the finite values (empty dict values when none)."""
    v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.5, 0.9, 0.99, 0.999])
    return {"n": int(v.size), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "p99_9": float(q[3]), "sd": float(v.std(ddof=1)) if v.size > 1 else None, "max": float(v.max())}


def _date(v: Any) -> Optional[str]:
    return None if v is None or pd.isna(v) else str(pd.Timestamp(v).date())


def _bool(v: Any) -> bool:
    return bool(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) and not pd.isna(v) else False


def _row(ps: pd.DataFrame, player: str, window: str) -> Optional[pd.Series]:
    r = ps[(ps["player_id"] == player) & (ps["time_window"] == window)]
    return None if r.empty else r.iloc[0]


def _get(r: Optional[pd.Series], col: str) -> Any:
    return None if r is None or col not in r.index else r[col]


def rank_above(field: pd.Series, value: Any) -> Optional[int]:
    """How many values of the field are at least ``value`` (the player's rank among them, 1 = above all)."""
    if value is None or pd.isna(value):
        return None
    f = pd.to_numeric(field, errors="coerce").dropna()
    return int((f >= float(value)).sum()) + 1


# --------------------------------------------------------------------------------------------- evaluation


def eval_settings(cfg: Mapping) -> dict:
    ev = dict(cfg.get("eval", {}))
    return {"z_bar": float(ev.get("z_bar", cfg["method_b"]["z_b_screen"])),
            "runs_pct_flag": float(ev.get("runs_pct_flag", 95.0)),
            "centre_tol": float(ev.get("centre_tol", 0.25)),
            "sd_range": [float(v) for v in ev.get("sd_range", [0.8, 1.25])],
            "bh_q": float(cfg["pairs"]["bh_q"]), "z_b_screen": float(cfg["method_b"]["z_b_screen"]),
            "min_dead": int(cfg["method_b"]["min_dead"]), "min_win": int(cfg["method_b"]["min_win"])}


def player_methods(ps: pd.DataFrame, player: str, st: Mapping) -> dict:
    """Each method's verdict and values for one player, per window."""
    bar = st["z_bar"]
    out: dict[str, Any] = {"win_rate": {}, "scorecard": {}, "method_b": {}, "method_b_e": {}}
    for w in WINDOWS:
        r = _row(ps, player, w)
        if r is None:
            continue
        el = _bool(r.get("rb_eligible"))
        out["win_rate"][w] = {"flag": _bool(r.get("s1_flag")), "eligible": _bool(r.get("s1_eligible")),
                              "hands": r.get("hands"), "bb100": r.get("bb100"),
                              "composite_pct": r.get("s1_composite_pct")}
        out["scorecard"][w] = {"flag": _bool(r.get("s2_flag")), "ofr_tr": r.get("ofr_tr"),
                               "ofr_tr_n": r.get("ofr_tr__n"), "bluff_index": r.get("bluff_index"),
                               "bluff_index_n": r.get("bluff_index__n"), "disc_gap": r.get("disc_gap"),
                               "pct2_ofr_tr": r.get("pct2_ofr_tr"), "pct2_bluff_index": r.get("pct2_bluff_index")}
        out["method_b"][w] = {"flag": _bool(r.get("rb_screen")), "eligible": el, "z_b": r.get("z_b"),
                              "nd": r.get("nd"), "nw": r.get("nw"), "x_dead": r.get("x_dead"),
                              "x_win": r.get("x_win"), "z_b_pct": r.get("z_b_pct")}
        cmh = r.get("cmh_z")
        out["method_b_e"][w] = {"flag": bool(el and cmh is not None and not pd.isna(cmh) and cmh >= bar),
                                "eligible": el, "e_disc": r.get("e_disc"), "cmh_z": cmh}
    r = _row(ps, player, "all")
    out["runs"] = {"flag": bool(_get(r, "wr_pct_median") is not None and not pd.isna(_get(r, "wr_pct_median"))
                                and _get(r, "wr_pct_median") >= st["runs_pct_flag"]),
                   **{k: _get(r, c) for k, c in (("pct_median", "wr_pct_median"), ("pct_lo", "wr_pct_lo"),
                                                 ("pct_hi", "wr_pct_hi"), ("pct_pooled", "wr_pct_pooled"),
                                                 ("pct_best_run", "wr_pct_best_run"), ("hands", "wr_hands"),
                                                 ("run_bb100", "wr_run_bb100"))}}
    cp = _get(r, "cp_z")
    out["changepoint"] = {"flag": bool(cp is not None and not pd.isna(cp) and cp >= bar), "cp_z": cp,
                          "week_start": _date(_get(r, "cp_week_start")), "splits": _get(r, "cp_splits")}
    return out


def pair_result(pairs: pd.DataFrame, plant: Mapping) -> dict:
    """The pair test for one plant: its pair against the victim (single victim) and its passing pairs."""
    pa = pairs[(pairs["time_window"] == "all")]
    el = pa[pa["eligible"].fillna(False).astype(bool)].sort_values("test_z", ascending=False)
    mine = pa[pa["player_id"] == plant["player"]]
    out = {"pairs_bh_pass": int(mine["bh_pass"].fillna(False).astype(bool).sum()),
           "pairs_eligible": int(mine["eligible"].fillna(False).astype(bool).sum())}
    if plant.get("victim"):
        r = mine[mine["opponent_id"] == plant["victim"]]
        if r.empty:
            out["victim_pair"] = {"met": False}
        else:
            r = r.iloc[0]
            rank = None
            if _bool(r["eligible"]):
                rank = int((el["test_z"] >= r["test_z"]).sum())
            out["victim_pair"] = {"met": True, "pair_hands": r["pair_hands"], "pair_share": r["pair_share"],
                                  "eligible": _bool(r["eligible"]), "test_z": r["test_z"], "z_split": r["z_split"],
                                  "eps": r["eps"], "q_value": r["q_value"], "flag": _bool(r["bh_pass"]),
                                  "rank": rank, "of": int(len(el))}
    return out


def stage5(esc: pd.DataFrame, player: str) -> dict:
    out: dict[str, dict] = {}
    for r in esc[esc["player_id"] == player].itertuples(index=False):
        out.setdefault(r.rule_set, {})[r.rule] = {"flag": _bool(r.escalate), "flagged": r.flagged,
                                                  "hands": r.hands, "ratio": r.ratio}
    return out


def honest_field(ps: pd.DataFrame, pairs: pd.DataFrame, honest: set[str], st: Mapping) -> dict:
    """Distributions and false-flag counts of the same statistics over the honest players."""
    h = ps[ps["player_id"].isin(honest)]
    out: dict[str, Any] = {"players": len(honest), "windows": {}}
    bar = st["z_bar"]
    for w in WINDOWS:
        g = h[h["time_window"] == w]
        el = g[g["rb_eligible"].fillna(False).astype(bool)]
        out["windows"][w] = {
            "players": int(len(g)), "rb_eligible": int(len(el)),
            "z_b": quantiles(el["z_b"]), "e_disc": quantiles(el["e_disc"]), "cmh_z": quantiles(el["cmh_z"]),
            "x_dead": quantiles(el["x_dead"]), "x_win": quantiles(el["x_win"]),
            "flags": {"win_rate": int(g["s1_flag"].fillna(False).astype(bool).sum()),
                      "win_rate_eligible": int(g["s1_eligible"].fillna(False).astype(bool).sum()),
                      "scorecard": int(g["s2_flag"].fillna(False).astype(bool).sum()),
                      "method_b": int(el["rb_screen"].fillna(False).astype(bool).sum()),
                      "method_b_e": int((el["cmh_z"] >= bar).sum())},
        }
    allw = h[h["time_window"] == "all"]
    out["runs_pct_median"] = quantiles(allw["wr_pct_median"])
    out["runs_flag"] = int((allw["wr_pct_median"] >= st["runs_pct_flag"]).sum())
    out["cp_z"] = quantiles(allw["cp_z"])
    out["cp_flag"] = int((allw["cp_z"] >= bar).sum())
    pa = pairs[(pairs["time_window"] == "all") & pairs["player_id"].isin(honest)]
    el = pa[pa["eligible"].fillna(False).astype(bool)]
    out["pairs"] = {"eligible": int(len(el)), "bh_pass": int(el["bh_pass"].fillna(False).astype(bool).sum()),
                    "test_z": quantiles(el["test_z"])}
    return out


def strong_regulars(ps: pd.DataFrame, strong: list[str], st: Mapping) -> dict:
    rows = []
    for p in strong:
        m = player_methods(ps, p, st)
        a = m["method_b"].get("all", {})
        any_b = any(v.get("flag") for v in m["method_b"].values())
        wr_flag = bool(m["win_rate"].get("all", {}).get("flag")) or bool(m["runs"]["flag"])
        rows.append({"player": p, "hands": m["win_rate"].get("all", {}).get("hands"),
                     "bb100": m["win_rate"].get("all", {}).get("bb100"),
                     "s1_flag": bool(m["win_rate"].get("all", {}).get("flag")),
                     "runs_pct_median": m["runs"]["pct_median"], "runs_flag": m["runs"]["flag"],
                     "win_rate_flag": wr_flag, "s2_flag_any": any(v.get("flag") for v in m["scorecard"].values()),
                     "z_b": a.get("z_b"), "rb_eligible": a.get("eligible"), "method_b_flag_any": any_b,
                     "cmh_z": m["method_b_e"].get("all", {}).get("cmh_z")})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"players": [], "n": 0}
    wr = df[df["win_rate_flag"]]
    return {"players": rows, "n": int(len(df)), "win_rate_flagged": int(len(wr)),
            "s1_flagged": int(df["s1_flag"].sum()), "runs_flagged": int(df["runs_flag"].sum()),
            "win_rate_flagged_and_method_b": int(wr["method_b_flag_any"].sum()),
            "win_rate_flagged_and_scorecard": int(wr["s2_flag_any"].sum())}


ACCEPTANCE_DATASET = "sim_demo"   # the acceptance checks apply to this dataset; elsewhere they are informational


def acceptance(dataset: str, plants: Mapping[str, dict], field: Mapping, strong: Mapping, st: Mapping) -> list:
    """The acceptance checks with PASS / FAIL / N/A and the numbers behind each."""
    checks = []
    binding = dataset == ACCEPTANCE_DATASET

    def add(cid: str, text: str, status: Optional[bool], numbers: str) -> None:
        checks.append({"id": cid, "check": text, "status": "N/A" if status is None else
                       ("PASS" if status else "FAIL"), "numbers": numbers, "binding": binding})

    for pid in ("SU_FULL_100", "SU_VICTIM_92"):
        p = plants.get(pid)
        if p is None:
            add(f"{pid}_screen", f"{pid} flagged by the player-level screen", None, "plant not in this dataset")
            continue
        b = p["methods"]["method_b"].get("all", {})
        add(f"{pid}_screen", f"{pid} flagged by the player-level screen (z_b >= {st['z_b_screen']:g}, all window)",
            bool(b.get("flag")), f"z_b {_f(b.get('z_b'))}, dead {_f(b.get('nd'), 0)} / winning "
                                 f"{_f(b.get('nw'), 0)} decisions")
    p = plants.get("SU_VICTIM_20")
    if p is None:
        add("SU_VICTIM_20_pair", "SU_VICTIM_20 not flagged at player level, its pair flagged after BH", None,
            "plant not in this dataset")
    else:
        flags = {w: v.get("flag") for w, v in p["methods"]["method_b"].items()}
        zb = {w: v.get("z_b") for w, v in p["methods"]["method_b"].items()}
        vp = p["pair"].get("victim_pair", {})
        ok = (not any(flags.values())) and bool(vp.get("flag"))
        add("SU_VICTIM_20_pair", f"SU_VICTIM_20 not flagged at player level, its pair flagged after BH "
                                 f"(q = {st['bh_q']:g})", ok,
            f"player z_b all {_f(zb.get('all'))}, last30 {_f(zb.get('last30'))}, last15 {_f(zb.get('last15'))} "
            f"(screen flags: {sum(bool(v) for v in flags.values())}); pair test z {_f(vp.get('test_z'))}, "
            f"q {_f(vp.get('q_value'), 5)}, rank {vp.get('rank')} of {vp.get('of')}")
    p = plants.get("SU_ONSET")
    if p is None:
        add("SU_ONSET_window", "SU_ONSET flagged in last15 and not (or weaker) in all", None,
            "plant not in this dataset")
    else:
        b15, ball = p["methods"]["method_b"].get("last15", {}), p["methods"]["method_b"].get("all", {})
        weaker = (not ball.get("flag")) or ((b15.get("z_b") or -math.inf) > (ball.get("z_b") or math.inf))
        cp = p["methods"]["changepoint"]
        add("SU_ONSET_window", "SU_ONSET flagged in last15 and not (or weaker) in all",
            bool(b15.get("flag")) and weaker,
            f"z_b last15 {_f(b15.get('z_b'))} on {_f(b15.get('nd'), 0)} dead / {_f(b15.get('nw'), 0)} winning "
            f"decisions (eligible: {bool(b15.get('eligible'))}, needs {st['min_dead']}/{st['min_win']}); "
            f"z_b all {_f(ball.get('z_b'))}; changepoint z {_f(cp.get('cp_z'))} at week {cp.get('week_start')} "
            f"(planted from {p['from_date']})")
    if strong.get("n"):
        ok = strong["win_rate_flagged"] > 0 and strong["win_rate_flagged_and_method_b"] == 0
        add("strong_regulars", "strong honest regulars flagged by win rate but not by the decision signals", ok,
            f"{strong['n']} strong honest regulars; flagged by win rate (Stage 1 or runs percentile >= "
            f"{st['runs_pct_flag']:g}): {strong['win_rate_flagged']}; of these flagged by the z_b screen in any "
            f"window: {strong['win_rate_flagged_and_method_b']}, by the article's Stage 2: "
            f"{strong['win_rate_flagged_and_scorecard']}")
    else:
        add("strong_regulars", "strong honest regulars flagged by win rate but not by the decision signals", None,
            "no strong honest regulars")
    z = field["windows"]["all"]["z_b"]
    if z.get("n", 0) > 1:
        lo, hi = st["sd_range"]
        ok = abs(z["median"]) <= st["centre_tol"] and lo <= z["sd"] <= hi
        add("honest_centre", "honest field z_b centred near 0 with sd near 1 (all window)", ok,
            f"n {z['n']}, median {_f(z['median'])} (|median| <= {st['centre_tol']:g}), sd {_f(z['sd'])} "
            f"(inside {lo:g} to {hi:g}), p99.9 {_f(z['p99_9'])}, max {_f(z['max'])}")
    else:
        add("honest_centre", "honest field z_b centred near 0 with sd near 1", None, "too few eligible players")
    return checks


def _f(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return "n/a"
    return f"{float(v):.{nd}f}"


def evaluate(frames: Mapping[str, pd.DataFrame], truth: Mapping, pop: Mapping, cfg: Mapping) -> dict:
    """The whole evaluation from the step 05 tables and the answer key (pure, no I/O)."""
    st = eval_settings(cfg)
    ps, pairs, esc = frames["player_scores"], frames["pair_scores"], frames["escalations"]
    plant_rows = truth.get("plants", [])
    planted = {p["player"] for p in plant_rows}
    people = pop.get("players", [])
    honest = set(ps["player_id"]) - planted
    strong = sorted(p["name"] for p in people if p.get("archetype") == "strong" and not p.get("planted")
                    and p["name"] not in planted and p["name"] in honest)
    plants: dict[str, dict] = {}
    for p in plant_rows:
        m = player_methods(ps, p["player"], st)
        rb = ps[(ps["time_window"] == "all") & ~ps["player_id"].isin(planted)
                & ps["rb_eligible"].fillna(False).astype(bool)]
        plants[p["plant_id"]] = {
            "player": p["player"], "kind": p["kind"], "u": p.get("u"), "victim": p.get("victim"),
            "victim_share_target": p.get("victim_share_target"), "from_date": p.get("from_date"),
            "hands": p.get("hands"), "hu_hands": p.get("hu_hands"),
            "hu_hands_vs_victim": p.get("hu_hands_vs_victim"), "methods": m, "pair": pair_result(pairs, p),
            "stage5": stage5(esc, p["player"]),
            "honest_rank": {"z_b": rank_above(rb["z_b"], m["method_b"].get("all", {}).get("z_b")),
                            "cmh_z": rank_above(rb["cmh_z"], m["method_b_e"].get("all", {}).get("cmh_z"))},
        }
    field = honest_field(ps, pairs, honest, st)
    strong_res = strong_regulars(ps, strong, st)
    meta = frames["report_meta"]
    last = meta.loc[meta["item"] == "dataset_last_date", "text"]
    return clean({
        "dataset": truth.get("dataset"), "dataset_last_date": None if last.empty else last.iloc[0],
        "config_hash": next(iter(meta.loc[meta["item"] == "config_hash", "text"]), None),
        "settings": st, "answer_key": {"plants": [{k: p.get(k) for k in ("plant_id", "player", "kind", "u",
                                                                          "victim", "victim_share_target",
                                                                          "from_date")} for p in plant_rows],
                                       "strong_regulars": strong},
        "plants": plants, "honest_field": field, "strong_regulars": strong_res,
        "acceptance": acceptance(str(truth.get("dataset")), plants, field, strong_res, st),
        "note": "Recall and false-flag counts hold on the simulator with planted players only; they are not "
                "detection rates for real data.",
    })


# --------------------------------------------------------------------------------------------- markdown


def _yn(v: Any) -> str:
    return "yes" if v else "no"


def markdown(res: Mapping) -> str:
    st = res["settings"]
    lines = [f"# Planted-player evaluation: {res['dataset']}", "",
             f"Step 05 tables of the last run (config hash `{res['config_hash']}`, data up to "
             f"{res['dataset_last_date']}). Bars: z_b screen {st['z_b_screen']:g} with >= {st['min_dead']} dead and "
             f">= {st['min_win']} winning decisions; the same bar {st['z_bar']:g} for E's CMH z and "
             f"the changepoint z; runs percentile >= {st['runs_pct_flag']:g}; BH at q = {st['bh_q']:g}. "
             f"{res['note']}", "", "## Acceptance checks", ""]
    if res["acceptance"] and not res["acceptance"][0]["binding"]:
        lines += [f"The acceptance is defined on {ACCEPTANCE_DATASET}; on {res['dataset']} the checks are "
                  "informational (fewer players and plants).", ""]
    lines += ["| check | status | numbers |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['status']} | {c['numbers']} |" for c in res["acceptance"]]
    lines += ["", "## Planted players, all window (flagged? value)", "",
              "| plant | win rate (Stage 1) | runs pct | article (Stage 2) | method (b) z_b | "
              "E (CMH z) | pair test (BH) | changepoint z | last15 z_b |", "|---|---|---|---|---|---|---|---|---|"]
    for pid, p in res["plants"].items():
        m = p["methods"]
        a = lambda k: m[k].get("all", {})  # noqa: E731
        vp = p["pair"].get("victim_pair")
        pair = (f"{_yn(vp.get('flag'))} z {_f(vp.get('test_z'))}, q {_f(vp.get('q_value'), 4)}" if vp
                else f"{p['pair']['pairs_bh_pass']} pairs pass")
        l15 = m["method_b"].get("last15", {})
        lines.append(
            f"| {pid} | {_yn(a('win_rate').get('flag'))} | {_f(m['runs'].get('pct_median'), 1)} | "
            f"{_yn(a('scorecard').get('flag'))} | "
            f"{_yn(a('method_b').get('flag'))} {_f(a('method_b').get('z_b'))} | "
            f"{_yn(a('method_b_e').get('flag'))} {_f(a('method_b_e').get('cmh_z'))} | {pair} | "
            f"{_yn(m['changepoint'].get('flag'))} {_f(m['changepoint'].get('cp_z'))} | "
            f"{_yn(l15.get('flag'))} {_f(l15.get('z_b'))}{'' if l15.get('eligible') else ' (not eligible)'} |")
    lines += ["", "## Honest field", "", "| window | eligible | z_b median | sd | p90 | p99 | p99.9 | max | "
              "flagged: Stage 1 / Stage 2 / z_b / E |", "|---|---|---|---|---|---|---|---|---|"]
    for w, f in res["honest_field"]["windows"].items():
        z, fl = f["z_b"], f["flags"]
        lines.append(f"| {w} | {f['rb_eligible']} | {_f(z.get('median'))} | {_f(z.get('sd'))} | {_f(z.get('p90'))} | "
                     f"{_f(z.get('p99'))} | {_f(z.get('p99_9'))} | {_f(z.get('max'))} | {fl['win_rate']} / "
                     f"{fl['scorecard']} / {fl['method_b']} / {fl['method_b_e']} |")
    hf = res["honest_field"]
    lines += ["", f"Runs percentile >= {st['runs_pct_flag']:g}: {hf['runs_flag']} honest players. Changepoint "
                  f"z: median {_f(hf['cp_z'].get('median'))}, p99.9 {_f(hf['cp_z'].get('p99_9'))}, "
                  f"{hf['cp_flag']} at or above the bar. Pair test: {hf['pairs']['eligible']} honest eligible "
                  f"pairs, {hf['pairs']['bh_pass']} pass BH."]
    s = res["strong_regulars"]
    if s.get("n"):
        lines += ["", f"Strong honest regulars (control): {s['n']}; flagged by win rate {s['win_rate_flagged']} "
                      f"(Stage 1 {s['s1_flagged']}, runs {s['runs_flagged']}); of those, z_b screen "
                      f"{s['win_rate_flagged_and_method_b']}, Stage 2 {s['win_rate_flagged_and_scorecard']}."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", action="append", help="simulated dataset (repeatable); default from the config")
    ap.add_argument("--config", type=Path, help="TOML file whose [step05] keys override the defaults")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    cfg = step05_module().load_config(a.config)
    datasets = a.dataset or list(cfg.get("eval", {}).get("datasets", ["sim_demo", "sim_small"]))
    a.out.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        truth, pop = answer_key(ds)
        res = evaluate(read_outputs(ds, cfg), truth, pop, cfg)
        (a.out / f"eval_{ds}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / f"eval_{ds}.md").write_text(markdown(res), encoding="utf-8")
        print(f"== {ds}")
        for c in res["acceptance"]:
            print(f"  {c['status']:4s} {c['check']}: {c['numbers']}")


if __name__ == "__main__":
    main()
