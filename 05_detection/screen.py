"""Step 05, population screening: the article's scorecard, the range-adjusted test (method b), the screening
stages 1-5 and the three scenarios (full table, single victim, onset), over every player of a dataset.

Reads the tables of steps 01-04 through two queries in ``sql/`` and writes:

* ``player_scores``: one row per player x time window (``all``, ``last30``, ``last15``) x scope (``hu_tables``):
  volume and style, the scorecard, method (b), the Stage 1-2 flags, the win-rate runs percentile and the onset
  changepoint (``all`` rows), the Stage 5 verdicts;
* ``pair_scores``: ordered pairs (i -> j): the scorecard and the range-adjusted counters of i's decisions against
  j, the split test z_ij, the opponent-controlled residual eps_ij, the robust z, the BH q-value;
* ``window_scores``: weekly series per player (X_dead, X_win, S, z_b);
* ``flagged_hands`` (Stages 3-4), ``escalations`` (Stage 5), ``report_meta`` (config hash, populations, counts
  tested) and ``scored_decisions`` (the decision rows the metrics were computed on).

Usage::

    python 05_detection/screen.py --dataset sim_small
    python 05_detection/screen.py --dataset sim_demo --config my_overrides.toml

The warehouse is opened read-only to read and read-write only to write the outputs; a lock held by another
process is waited for (``retry_seconds``). ``05_detection/run.py`` then reports one suspect in depth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import metrics as M  # noqa: E402
from sudetect import screening as S  # noqa: E402
from sudetect import warehouse  # noqa: E402

SQL_DIR = HERE / "sql"
CONFIG = ROOT / "config" / "step05_screening.toml"

# Stand-ins for upstream tables a dataset may lack (no step 02 or 04 run): same columns, no rows.
EMPTY_INPUTS = {
    "action_equity": "SELECT CAST(NULL AS VARCHAR) AS hand_id, CAST(NULL AS SMALLINT) AS action_index, "
                     "CAST(NULL AS DOUBLE) AS eq_vs_real, CAST(NULL AS BOOLEAN) AS eq_real_known, "
                     "CAST(NULL AS DOUBLE) AS eq_vs_random WHERE FALSE",
    "action_range_equity": "SELECT CAST(NULL AS VARCHAR) AS hand_id, CAST(NULL AS SMALLINT) AS action_index, "
                           "CAST(NULL AS DOUBLE) AS eq_vs_range WHERE FALSE"
}
DECISION_COLUMNS = ["hand_id", "action_index", "player_id", "opponent_id", "hand_date", "street", "action",
                    "facing", "bettor_q", "q", "r", "req", "rbin", "p_all", "y", "bluff_success", "pot_bb",
                    "faced_size_pot", "made_hand"]


# --------------------------------------------------------------------------------------------- config and I/O


def _merge(base: dict, extra: Mapping) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
    return out


def load_config(extra: Optional[Path] = None) -> dict:
    """The ``[step05]`` table of config/step05_screening.toml, with the keys of ``extra`` (a TOML file) on top."""
    with open(CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)["step05"]
    if extra is not None:
        with open(extra, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh).get("step05", {}))
    return cfg


def config_hash(cfg: Mapping) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _connect(dataset: str, read_only: bool, wait_s: float):
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return warehouse.connect(dataset, read_only=read_only)
        except Exception as exc:  # duckdb raises IOException while another process holds the write lock
            text = str(exc).lower()
            if not any(w in text for w in ("lock", "being used", "conflicting")) or time.monotonic() > deadline:
                raise
            time.sleep(5)


def read_inputs(dataset: str, cfg: Mapping) -> dict[str, Any]:
    """The two input queries as frames, plus the dataset's last hand date and the missing upstream tables."""
    con = _connect(dataset, True, float(cfg.get("retry_seconds", 600)))
    try:
        missing = []
        for name, sql in EMPTY_INPUTS.items():
            if not warehouse.table_exists(con, name):
                warehouse.execute(con, f"CREATE TEMP VIEW {name} AS {sql}")
                missing.append(name)
        dec = warehouse.run_sql_file(con, SQL_DIR / "scored_decisions.sql")
        hands = warehouse.run_sql_file(con, SQL_DIR / "player_hands.sql")
        max_date = warehouse.query_df(con, "SELECT max(hand_date) AS d FROM hands")["d"].iloc[0]
    finally:
        con.close()
    return {"dec": dec, "hands": hands, "max_date": pd.Timestamp(max_date), "missing": missing}


def write_outputs(dataset: str, frames: Mapping[str, pd.DataFrame], cfg: Mapping) -> None:
    con = _connect(dataset, False, float(cfg.get("retry_seconds", 600)))
    try:
        for name, df in frames.items():
            warehouse.insert_df(con, name, _dates_to_date(df), mode="replace")
    finally:
        con.close()


def _dates_to_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("hand_date", "week_start", "week_end", "cp_week_start", "window_from", "window_to"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.date
    return out


# --------------------------------------------------------------------------------------------- preparation


def prepare_decisions(dec: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    """Typed flags, y, the pair-or-better flags and the range decile of every decision row."""
    d = dec.copy()
    d["hand_date"] = pd.to_datetime(d["hand_date"])
    d["started_at"] = pd.to_datetime(d["started_at"])
    for c in ("facing", "bluff_success", "won_at_showdown", "went_to_showdown", "is_allin"):
        d[c] = M.flag(d[c])
    for c in ("q", "q_random", "r", "req", "bettor_q", "pot_bb", "decision_s", "faced_size_pot"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    d["is_fold"] = d["action"] == "fold"
    d["is_cont"] = d["action"].isin(["call", "raise"])
    d["is_aggr"] = d["action"].isin(["bet", "raise"])
    d["y"] = d["is_fold"].astype(np.int64)
    d["pair_own"] = d["made_hand"].notna() & (d["made_hand"] != "nothing")
    d["pair_full"] = False
    m = (d["is_fold"] & d["q"].notna()).to_numpy()
    if m.any():
        full, _ = M.hand_tiers(d.loc[m, "hole_cards"].tolist(), d.loc[m, "board_at_action"].tolist())
        d.loc[m, "pair_full"] = full >= 1
    d["pair_full"] = d["pair_full"].astype(bool)
    d["rbin"] = M.range_bin(d["r"], int(cfg["method_b"]["range_bins"]))
    return d


def prepare_hands(hands: pd.DataFrame) -> pd.DataFrame:
    h = hands.copy()
    h["hand_date"] = pd.to_datetime(h["hand_date"])
    h["started_at"] = pd.to_datetime(h["started_at"])
    for c in ("vpip", "pfr", "walk", "saw_flop", "went_to_showdown", "won_at_showdown"):
        h[c] = M.flag(h[c])
    h["net_bb"] = h["net_bb"].astype(float)
    return h


def window_start(max_date: pd.Timestamp, name: str, cfg: Mapping) -> Optional[pd.Timestamp]:
    """First day of a window: the ``days`` days ending at the dataset's last hand date; None for ``all``."""
    days = {"all": 0, "last30": int(cfg["last30_days"]), "last15": int(cfg["last15_days"])}[name]
    return None if days == 0 else max_date - pd.Timedelta(days=days - 1)


def _in_window(df: pd.DataFrame, lo: Optional[pd.Timestamp]) -> pd.DataFrame:
    return df if lo is None else df[df["hand_date"] >= lo]


def _responses(dec: pd.DataFrame) -> pd.DataFrame:
    """Responses to a bet with q and r known (any street after the flop)."""
    return dec[dec["facing"] & dec["q"].notna() & dec["r"].notna() & dec["rbin"].notna()]


# --------------------------------------------------------------------------------------------- the steps


def scorecard_frame(dec: pd.DataFrame, keys: list[str], cfg: Mapping) -> pd.DataFrame:
    sc, dc = cfg["scorecard"], cfg["decisions"]
    d = dec[dec["street"].isin(dc["streets"])]
    return M.scorecard(d, keys, pair_plus_own_cards=bool(sc["pair_plus_own_cards"]),
                       bluff_includes_raises=bool(sc["bluff_includes_raises"]), behind_q=float(dc["behind_q"]),
                       value_bettor_q=float(dc["value_bettor_q"]), bluff_q=float(dc["bluff_q"]),
                       gap_min_n=int(sc["gap_min_n"]))


def method_b(resp: pd.DataFrame, keys: list[str], cfg: Mapping) -> tuple[pd.DataFrame, float, float, int]:
    """Method (b) per group on response rows that carry p: counters, scores, E / CMH, oracle events."""
    mb = cfg["method_b"]
    dq, wq = float(mb["dead_q"]), float(mb["win_q"])
    cnt = M.range_counters(resp, keys, dead_q=dq, win_q=wq)
    dd, dw, n_el = M.centring(cnt, min_dead=int(mb["min_dead"]), min_win=int(mb["min_win"]))
    out = M.range_scores(cnt, dd, dw)
    out["rb_eligible"] = (out["nd"] >= int(mb["min_dead"])) & (out["nw"] >= int(mb["min_win"]))
    out["rb_screen"] = out["rb_eligible"] & (out["z_b"] >= float(mb["z_b_screen"]))
    out = out.merge(M.excess_discrimination(resp, keys, dead_q=dq, win_q=wq), on=keys, how="left")
    by_street = M.excess_discrimination(resp, keys + ["street"], dead_q=dq, win_q=wq)
    if not by_street.empty:
        wide = by_street.pivot_table(index=keys, columns="street", values="e_disc", aggfunc="first")
        wide.columns = [f"e_disc_{c}" for c in wide.columns]
        out = out.merge(wide.reset_index(), on=keys, how="left")
    for st in mb["streets"]:
        if f"e_disc_{st}" not in out.columns:
            out[f"e_disc_{st}"] = np.nan
    ev = M.oracle_events(resp, keys, dead_q=dq, win_q=wq, fold_margin=float(mb["oracle_fold_margin"]),
                         call_margin=float(mb["oracle_call_margin"]), line=float(mb["line"]))
    return out.merge(ev, on=keys, how="left"), dd, dw, n_el


def player_windows(dec: pd.DataFrame, hands: pd.DataFrame, max_date: pd.Timestamp,
                   cfg: Mapping, meta: list) -> tuple[pd.DataFrame, dict]:
    """One row per player x window: volume, style, scorecard, method (b)."""
    rows, tables = [], {}
    mb = cfg["method_b"]
    for w in cfg["time_windows"]:
        lo = window_start(max_date, w, cfg)
        dw, hw = _in_window(dec, lo), _in_window(hands, lo)
        base = M.hand_stats(hw, ["player_id"], vpip_excludes_walks=bool(cfg["scorecard"]["vpip_excludes_walks"]))
        card = scorecard_frame(dw, ["player_id"], cfg)
        ptab = M.fit_expected_fold(_responses(dw))
        tables[w] = ptab
        resp = M.attach_expected(_responses(dw)[lambda x: x["street"].isin(mb["streets"])], ptab)
        rb, dd, dlw, n_el = method_b(resp, ["player_id"], cfg)
        e = rb["rb_eligible"]
        rb["e_disc_pct"] = M.midrank_pct(rb["e_disc"].where(e))
        rb["z_b_pct"] = M.midrank_pct(rb["z_b"].where(e))
        p = base.merge(card, on="player_id", how="left").merge(rb, on="player_id", how="left")
        p.insert(1, "time_window", w)
        p.insert(2, "scope", cfg["scope"])
        p["window_from"] = lo if lo is not None else hw["hand_date"].min()
        p["window_to"] = max_date
        rows.append(p)
        meta += [("delta_dead", w, dd, ""), ("delta_win", w, dlw, ""), ("players_rb_eligible", w, n_el, ""),
                 ("players_with_hands", w, len(base), ""), ("responses_method_b", w, len(resp), ""),
                 ("decisions_scorecard", w, len(dw), "")]
    return pd.concat(rows, ignore_index=True), tables


def weekly(dec: pd.DataFrame, hands: pd.DataFrame, ptab: pd.DataFrame, dd: float, dw: float,
           max_date: pd.Timestamp, cfg: Mapping) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly series (weeks counted back from the dataset's last hand date) and the changepoint per player.

    The weekly values use the expected fold p and the centring of the ``all`` window, so weeks compare."""
    oc, mb = cfg["onset"], cfg["method_b"]
    period = int(oc["period_days"])

    def week_cols(df: pd.DataFrame) -> pd.DataFrame:
        k = ((max_date - df["hand_date"]).dt.days // period).astype(int)
        return df.assign(week_index=k, week_start=max_date - pd.to_timedelta(k * period + period - 1, unit="D"))

    keys = ["player_id", "week_start"]
    resp = week_cols(M.attach_expected(_responses(dec)[lambda x: x["street"].isin(mb["streets"])], ptab))
    cnt = M.range_counters(resp, keys, dead_q=float(mb["dead_q"]), win_q=float(mb["win_q"]))
    wk = M.range_scores(cnt, dd, dw)
    hw = week_cols(hands).groupby(keys).agg(hands=("hand_id", "size"), week_index=("week_index", "first"))
    out = hw.reset_index().merge(wk, on=keys, how="left")
    out["week_end"] = out["week_start"] + pd.Timedelta(days=period - 1)
    out["scope"] = cfg["scope"]
    cp = M.changepoint(cnt[keys + list(M.COUNTERS)], min_dead=int(oc["min_dead"]), min_win=int(oc["min_win"]))
    return out.sort_values(keys).reset_index(drop=True), cp


def pair_scores(dec: pd.DataFrame, hands: pd.DataFrame, max_date: pd.Timestamp, tables: Mapping[str, pd.DataFrame],
                deltas: Mapping[str, tuple[float, float]], cfg: Mapping, meta: list) -> pd.DataFrame:
    """Single victim: every ordered pair (i -> j) that met heads-up, the pair test on the eligible ones."""
    pc, mb = cfg["pairs"], cfg["method_b"]
    keys = ["player_id", "opponent_id"]
    out = []
    for w in pc["time_windows"]:
        lo = window_start(max_date, w, cfg)
        dw, hw = _in_window(dec, lo), _in_window(hands, lo)
        nh = hw.groupby(keys).size().rename("pair_hands").reset_index()
        nh["pair_share"] = nh["pair_hands"] / nh.groupby("player_id")["pair_hands"].transform("sum")
        resp = M.attach_expected(_responses(dw)[lambda x: x["street"].isin(mb["streets"])], tables[w])
        dq, wq = float(mb["dead_q"]), float(mb["win_q"])
        pair = M.range_counters(resp, keys, dead_q=dq, win_q=wq)
        tot = M.range_counters(resp, ["player_id"], dead_q=dq, win_q=wq)
        m = M.range_scores(M.split_z(pair, tot, ["player_id"]), *deltas[w])
        fit = ((m["nd"] >= int(pc["min_dead"])) & (m["nw"] >= int(pc["min_win"])) & (m["var_s"] > 0)).to_numpy()
        for c in ("mu", "a_i", "b_j", "eps", "z_eps"):
            m[c] = np.nan
        if fit.any():
            f = m[fit]
            mu, a, b, eps = M.two_way_fit(f["s"].to_numpy(), 1.0 / f["var_s"].to_numpy(), f["player_id"].to_numpy(),
                                          f["opponent_id"].to_numpy(), passes=int(pc["fit_passes"]),
                                          prior_weight=float(pc["prior_weight"]))
            m.loc[fit, "mu"], m.loc[fit, "a_i"], m.loc[fit, "b_j"], m.loc[fit, "eps"] = mu, a, b, eps
            m.loc[fit, "z_eps"] = eps / np.sqrt(f["var_s"].to_numpy())
        m = nh.merge(m, on=keys, how="left")
        card = scorecard_frame(dw, keys, cfg)
        m = m.merge(card, on=keys, how="left")
        m["fit_pair"] = (m["nd"] >= int(pc["min_dead"])) & (m["nw"] >= int(pc["min_win"])) & m["eps"].notna()
        m["eligible"] = m["fit_pair"] & (m["pair_hands"] >= int(pc["min_hands"]))
        stat = m["z_eps"] if pc.get("test", "eps") == "eps" else m["z_split"]
        el = m["eligible"].to_numpy()
        n_pairs = int(el.sum())
        robust = n_pairs >= int(pc["robust_min_pairs"])
        m["test_z"] = np.nan
        m.loc[el, "test_z"] = M.robust_z(stat[el].to_numpy()) if robust else stat[el].to_numpy()
        m["p_value"] = M.upper_tail_p(m["test_z"].to_numpy())
        m["q_value"] = np.nan
        m.loc[el, "q_value"] = M.benjamini_hochberg(m.loc[el, "p_value"].to_numpy())
        m["bh_pass"] = m["q_value"] <= float(pc["bh_q"])
        m.insert(2, "time_window", w)
        m.insert(3, "scope", cfg["scope"])
        out.append(m)
        meta += [("pairs_met", w, len(m), ""), ("pairs_fit", w, int(m["fit_pair"].sum()), ""),
                 ("pairs_tested_M", w, n_pairs, f"test={pc.get('test', 'eps')}, robust={robust}"),
                 ("pairs_bh_pass", w, int(m["bh_pass"].sum()), f"q={pc['bh_q']}")]
    return pd.concat(out, ignore_index=True)


def compute(inp: Mapping[str, Any], cfg: Mapping) -> dict[str, pd.DataFrame]:
    """Every output table from the two input frames."""
    impl = importlib.import_module(cfg["impl"]["module"])
    dec = prepare_decisions(inp["dec"], cfg)
    hands = prepare_hands(inp["hands"])
    max_date = inp["max_date"]
    meta: list = [("dataset_last_date", "", math.nan, str(max_date.date())),
                  ("missing_inputs", "", float(len(inp["missing"])), ",".join(inp["missing"]))]

    players, tables = player_windows(dec, hands, max_date, cfg, meta)
    players = impl.screen(players, cfg)
    deltas = {w: (_meta_value(meta, "delta_dead", w), _meta_value(meta, "delta_win", w))
              for w in cfg["time_windows"]}

    runs = S.win_rate_runs(hands, cfg["runs"], int(cfg["seed"]))
    weeks, cp = weekly(dec, hands, tables["all"], *deltas["all"], max_date, cfg)
    extra = runs.astype({"player_id": object}).merge(cp, on="player_id", how="outer")
    extra = extra.assign(time_window="all")
    players = players.merge(extra, on=["player_id", "time_window"], how="left")

    dec_p = M.attach_expected(dec, tables["all"]).rename(columns={"p": "p_all", "p_n": "p_all_n"})
    flagged = impl.flag_hands(dec_p, cfg)
    esc = impl.escalate(flagged, hands, cfg)
    if not esc.empty:
        wide = esc.assign(col="esc_" + esc["rule_set"] + "_" + esc["rule"]).pivot_table(
            index="player_id", columns="col", values="escalate", aggfunc="first")
        wide = wide.astype(bool).reset_index().assign(time_window="all")
        players = players.merge(wide, on=["player_id", "time_window"], how="left")

    pairs = pair_scores(dec, hands, max_date, tables, deltas, cfg, meta)

    for w in cfg["time_windows"]:
        p = players[players["time_window"] == w]
        meta += [("players_stage1_eligible", w, int(p["s1_eligible"].sum()), ""),
                 ("players_stage1_flag", w, int(p["s1_flag"].sum()), cfg["screening"]["rule"]),
                 ("players_stage1_top10", w, int(p["s1_top10"].sum()), ""),
                 ("players_stage1_p95_3of5", w, int(p["s1_p95"].sum()), ""),
                 ("players_stage2_flag", w, int(p["s2_flag"].sum()), cfg["screening"]["stage2_basis"]),
                 ("players_z_b_screen", w, int(p["rb_screen"].fillna(False).sum()),
                  f"z_b >= {cfg['method_b']['z_b_screen']}")]
    for rs in cfg["hand_tests"]["rule_sets"]:
        f = flagged[flagged["rule_set"] == rs]
        meta.append(("flagged_hands", rs, len(f), ",".join(f"{c}={n}" for c, n in
                                                          f["category"].value_counts().sort_index().items())))
        for rule in ("article_fixed", "population_calibrated"):
            e = esc[(esc["rule_set"] == rs) & (esc["rule"] == rule)]
            meta.append((f"escalations_{rule}", rs, int(e["escalate"].sum()), f"of {len(e)} players"))
    meta.append(("players_runs_field", "all", float(len(runs)), f"run_length={cfg['runs']['run_length']}"))
    meta.append(("config_hash", "", math.nan, config_hash(cfg)))
    report_meta = pd.DataFrame(meta, columns=["item", "scope", "value", "text"])
    report_meta["value"] = report_meta["value"].astype(float)

    decisions = dec_p.copy()
    for c in DECISION_COLUMNS:
        if c not in decisions.columns:
            decisions[c] = np.nan
    return {"player_scores": players, "pair_scores": pairs, "window_scores": weeks, "flagged_hands": flagged,
            "escalations": esc, "report_meta": report_meta, "scored_decisions": decisions[DECISION_COLUMNS]}


def _meta_value(meta: list, item: str, scope: str) -> float:
    return next(v for i, s, v, _ in meta if i == item and s == scope)


def summary(frames: Mapping[str, pd.DataFrame]) -> str:
    m = frames["report_meta"]
    lines = [f"{r.item:28s} {r.scope:18s} {'' if pd.isna(r.value) else round(r.value, 4)!s:>10s}  {r.text}"
             for r in m.itertuples(index=False)]
    lines += [f"{k}: {len(v)} rows" for k, v in frames.items() if k != "report_meta"]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", type=Path, help="TOML file whose [step05] keys override the defaults")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    t0 = time.perf_counter()
    inp = read_inputs(a.dataset, cfg)
    frames = compute(inp, cfg)
    if not a.dry_run:
        write_outputs(a.dataset, frames, cfg)
    print(summary(frames))
    print(f"step 05 on {a.dataset}: {time.perf_counter() - t0:.1f} s{' (dry run)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()
