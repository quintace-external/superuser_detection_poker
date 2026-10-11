"""Step 04: the opponent's predicted range and the hero's equity against it (detection method (b) inputs).

Stages, in order (``--steps`` picks a subset):

``fit``
    Field tables from a full-information dataset (``sql/fit_*.sql``): ``rng_lik_post`` (postflop action counts
    per spot and hand class, levels K6..K1), ``rng_prior_pre`` (169-class counts of the hands that saw a flop,
    per preflop key, levels P4..P0), ``rng_field`` (field VPIP / PFR / aggression); then ``rng_player_tilt``
    (every player's tilt inputs) and ``rng_meta``. A dataset with showdown-only hands is never fitted on: pass
    ``--fit-from <dataset>`` and its tables are copied over (the tilt inputs are always computed on the dataset).
``enrich``
    Stages the decisions to score as parquet chunks under ``$SU_DATA_DIR/<dataset>_step04/``, runs
    :func:`sudetect.rangemodel.score_chunk` over them in a process pool (finished chunks are kept, so an
    interrupted run resumes) and writes ``action_range_equity``.
``leakage``
    The leakage split on the heads-up responses to a bet: how much of the field's fold behaviour the range bins
    explain and how much the real-card class adds. Writes ``rng_leak_matrix`` and the summary JSON.
``examples``
    The narrowing example of ``examples.py`` (figures and ``action_range_opp``), for simulator datasets.

Usage::

    python 04_predicted_range/run.py --dataset mined --steps fit --config 04_predicted_range/variants/observer_fit.toml
    python 04_predicted_range/run.py --dataset article_sample --fit-from mined --config 04_predicted_range/variants/observer_fit.toml
    python 04_predicted_range/run.py --dataset <full-information dataset> --procs 16

The summary is written to ``$SU_DATA_DIR/<dataset>_step04/summary.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import rangemodel, warehouse  # noqa: E402

SQL_DIR = HERE / "sql"
DEFAULT_CONFIG = ROOT / "config" / "step04_range.toml"
PUBLIC_DATASETS: tuple[str, ...] = ()
STEPS = ("fit", "enrich", "leakage", "examples")
FIT_TABLES = ("rng_lik_post", "rng_lik_size", "rng_prior_pre", "rng_field")
STAGE_VERSION = 1


# ------------------------------------------------------------------ configuration


def _merge(base: dict, extra: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(extra: Optional[str | os.PathLike[str]] = None) -> dict:
    """``config/step04_range.toml`` with the keys of ``extra`` (a TOML file) merged over it."""
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if extra:
        with open(extra, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    streets = list(cfg["enrich"]["streets"])
    if not streets or any(s not in ("flop", "turn", "river") for s in streets):
        raise ValueError(f"enrich.streets must be a subset of flop, turn, river: {streets}")
    if cfg["enrich"]["tables"] not in ("hu", "all"):
        raise ValueError("enrich.tables must be 'hu' or 'all'")
    if not 1 <= int(cfg["enrich"]["max_live_opp"]) <= 3:
        raise ValueError("enrich.max_live_opp must be 1, 2 or 3")
    return cfg


def connect(dataset: str, read_only: bool = False, wait_s: float = 900.0) -> Any:
    """Open the warehouse, waiting while another process holds its write lock (DuckDB allows one writer)."""
    t0 = time.perf_counter()
    while True:
        try:
            return warehouse.connect(dataset, read_only=read_only)
        except Exception as exc:  # duckdb.IOException; this module does not import duckdb
            text = str(exc).lower()
            busy = any(s in text for s in ("lock", "being used by another process", "already open"))
            if not busy or time.perf_counter() - t0 > wait_s:
                raise
            time.sleep(5.0)


def stage_dir(dataset: str, tag: str = "") -> Path:
    """Folder of the staged chunks and the summary of one dataset (of one variant when ``tag`` is set)."""
    return warehouse.data_dir() / (f"{dataset}_step04_{tag}" if tag else f"{dataset}_step04")


def _sql_statement(path: Path, table: str) -> str:
    """The one statement of a schema file that creates ``table``."""
    for stmt in warehouse.split_statements(path.read_text(encoding="utf-8")):
        if f"TABLE {table} " in stmt or f"TABLE {table}\n" in stmt or f"TABLE {table}(" in stmt:
            return stmt
    raise KeyError(f"no CREATE TABLE {table} in {path}")


def create_output_table(con: Any, table: str) -> None:
    """(Re)create one of the tables of ``sql/schema.sql``."""
    warehouse.execute(con, _sql_statement(SQL_DIR / "schema.sql", table))


def _frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


# ------------------------------------------------------------------ fit


def full_share(con: Any) -> float:
    """Share of the dataset's hands with every dealt player's cards in the text."""
    df = warehouse.query_df(
        con, "SELECT avg(CASE WHEN cards_visibility = 'full' THEN 1.0 ELSE 0.0 END) AS s, count(*) AS n FROM hands"
    )
    return float(df.s.iat[0]) if int(df.n.iat[0]) else 0.0


def step_fit(con: Any, dataset: str, fit_from: Optional[str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fit (or copy) the field tables, then the dataset's own tilt inputs; returns a summary."""
    t0 = time.perf_counter()
    source = fit_from or dataset
    tl = cfg["tilt"]
    if source == dataset:
        share = full_share(con)
        if share < float(cfg["fit"]["min_full_share"]):
            raise SystemExit(
                f"{dataset}: only {share:.1%} of the hands show every player's cards. The field tables need "
                "full-information hands; run with --fit-from <a full-information dataset> to apply tables fitted there, "
                "or, for mined hands without one, with --config 04_predicted_range/variants/observer_fit.toml "
                "(docs/HAND_HISTORY_FORMAT.md, 'The example')."
            )
        warehouse.run_sql_file(con, SQL_DIR / "fit_lik_post.sql")
        warehouse.run_sql_file(con, SQL_DIR / "fit_lik_size.sql", **size_edges())
        warehouse.run_sql_file(con, SQL_DIR / "fit_prior_pre.sql")
        warehouse.run_sql_file(con, SQL_DIR / "tilt_raw.sql")
        warehouse.run_sql_file(con, SQL_DIR / "fit_field.sql", min_hands=int(tl["clean_min_hands"]),
                               sigma=float(tl["clean_sigma"]))
    else:
        src = connect(source, read_only=True)
        try:
            missing = [t for t in FIT_TABLES if not warehouse.table_exists(src, t)]
            if missing:
                raise SystemExit(f"{source} has no fitted tables {missing}: run step 04 on {source} first")
            frames = {t: warehouse.query_df(src, f"SELECT * FROM {t}") for t in FIT_TABLES}
        finally:
            src.close()
        for t, df in frames.items():
            warehouse.insert_df(con, t, df, mode="replace")
        warehouse.run_sql_file(con, SQL_DIR / "tilt_raw.sql")
    warehouse.run_sql_file(con, SQL_DIR / "player_tilt.sql", agg_min_n=int(tl["agg_min_n"]),
                           agg_full_n=int(tl["agg_full_n"]))
    lik = warehouse.query_df(con, """
        SELECT level, count(*) AS cells, sum(CASE WHEN n >= :m THEN 1 ELSE 0 END) AS cells_supported,
               median(n) AS median_n, sum(n) AS actions
        FROM rng_lik_post GROUP BY level ORDER BY level DESC""", m=int(cfg["likelihood"]["min_support"]))
    lik_size = warehouse.query_df(con, """
        SELECT level, count(*) AS cells, sum(CASE WHEN n >= :m THEN 1 ELSE 0 END) AS cells_supported,
               median(n) AS median_n, sum(n) AS actions
        FROM rng_lik_size GROUP BY level ORDER BY level DESC""", m=int(cfg["likelihood"]["min_support"]))
    pri = warehouse.query_df(con, """
        SELECT level, count(*) AS keys, sum(CASE WHEN n >= :m THEN 1 ELSE 0 END) AS keys_supported,
               sum(n) AS hands
        FROM (SELECT level, size_class, preflop_situation, position_class, action, sum(n) AS n
              FROM rng_prior_pre GROUP BY level, size_class, preflop_situation, position_class, action) AS k
        GROUP BY level ORDER BY level DESC""", m=int(cfg["prior"]["min_support"]))
    fld = warehouse.query_df(con, "SELECT * FROM rng_field ORDER BY size_class")
    tilt = warehouse.query_df(con, """
        SELECT size_class, count(*) AS players, sum(CASE WHEN n >= :lo THEN 1 ELSE 0 END) AS players_loose_tilt,
               sum(CASE WHEN n >= :ag THEN 1 ELSE 0 END) AS players_agg_tilt
        FROM rng_player_tilt GROUP BY size_class ORDER BY size_class""",
                              lo=int(tl["loose_min_n"]), ag=int(tl["agg_min_n"]))
    meta = pd.DataFrame({
        "key": ["dataset", "fit_dataset", "table_size_classes", "lik_rows", "prior_rows"],
        "value": [dataset, source, str(bool(cfg["fit"]["table_size_classes"])).lower(),
                  str(int(lik.cells.sum())), str(int(pri["keys"].sum()))],
    })
    warehouse.insert_df(con, "rng_meta", meta, mode="replace")
    return {
        "fit_dataset": source,
        "seconds": time.perf_counter() - t0,
        "rng_lik_post": lik.to_dict(orient="records"),
        "rng_lik_size": lik_size.to_dict(orient="records"),
        "rng_prior_pre": pri.to_dict(orient="records"),
        "rng_field": fld.to_dict(orient="records"),
        "rng_player_tilt": tilt.to_dict(orient="records"),
    }


def size_edges() -> dict[str, float]:
    """The tier edges of ``fit_lik_size.sql`` (parameters e1 ... e4), from :data:`sudetect.rangemodel.SIZE_EDGES`."""
    return {f"e{i + 1}": float(e) for i, e in enumerate(rangemodel.SIZE_EDGES)}


def load_tables(con: Any) -> tuple[rangemodel.FitTables, str]:
    """The fitted tables as a :class:`sudetect.rangemodel.FitTables` and a hash of their contents
    (``rng_lik_size`` is read when it exists: tables fitted before it was added have none)."""
    lik = warehouse.query_df(con, "SELECT * FROM rng_lik_post ORDER BY level, street, situation, pose, pot_type, "
                                  "made_hand, draw")
    pri = warehouse.query_df(con, "SELECT * FROM rng_prior_pre ORDER BY level, size_class, preflop_situation, "
                                  "position_class, action, hand_class")
    fld = warehouse.query_df(con, "SELECT * FROM rng_field ORDER BY size_class")
    frames = [lik, pri, fld]
    size = None
    if warehouse.table_exists(con, "rng_lik_size"):
        size = warehouse.query_df(con, "SELECT * FROM rng_lik_size ORDER BY level, street, situation, pose, "
                                       "pot_type, made_hand, draw, faced_tier")
        frames.append(size)
    tables = rangemodel.FitTables.from_frames(lik, pri, fld, size)
    digest = hashlib.sha256("".join(_frame_hash(d) for d in frames).encode()).hexdigest()
    return tables, digest


# ------------------------------------------------------------------ enrich


def model_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a worker needs besides the tables (picklable)."""
    rc = rangemodel.RangeConfig.from_toml(cfg)
    en = cfg["enrich"]
    return {
        "range": dataclasses.asdict(rc),
        "exact": bool(en["exact"]),
        "mc_samples": int(en["mc_samples"]),
        "seed": int(en["seed"]),
    }


def stage(con: Any, dataset: str, cfg: Mapping[str, Any], tables_hash: str, restart: bool,
          limit_hands: Optional[int], tag: str = "") -> tuple[Path, list[dict[str, str]], dict[str, Any]]:
    """Write the decisions to score, their live opponents and the opponents' postflop actions as parquet chunks.

    The chunks and their outputs are reused when the settings, the fitted tables and the staged rows are the
    same as in the previous run (an interrupted run resumes); otherwise the folder is rebuilt.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    en = cfg["enrich"]
    t0 = time.perf_counter()
    warehouse.run_sql_file(con, SQL_DIR / "stage_decisions.sql", max_live_opp=int(en["max_live_opp"]),
                           hu_only=en["tables"] == "hu")
    if limit_hands:
        warehouse.execute(con, """
            DELETE FROM rng_stage_dec WHERE hand_id NOT IN (
                SELECT hand_id FROM (SELECT DISTINCT hand_id FROM rng_stage_dec ORDER BY hand_id LIMIT :n) AS k)""",
                          n=int(limit_hands))
    dec = warehouse.query_df(con, "SELECT * FROM rng_stage_dec ORDER BY hand_id, action_index")
    dec = dec[dec.street.isin(list(en["streets"]))].reset_index(drop=True)
    opp = warehouse.run_sql_file(con, SQL_DIR / "stage_opponents.sql",
                                 split_sizes=bool(cfg["fit"]["table_size_classes"]))
    nodes = warehouse.run_sql_file(con, SQL_DIR / "stage_nodes.sql")
    code = hashlib.sha256(Path(rangemodel.__file__).read_bytes()).hexdigest()   # outputs are only reused unchanged
    key = hashlib.sha256(json.dumps({
        "v": STAGE_VERSION, "code": code, "dataset": dataset, "tables": tables_hash, "settings": model_settings(cfg),
        "enrich": {k: en[k] for k in sorted(en) if k != "procs"}, "limit": limit_hands,
        "dec": _frame_hash(dec), "opp": _frame_hash(opp), "nodes": _frame_hash(nodes),
    }, sort_keys=True, default=str).encode()).hexdigest()
    sdir = stage_dir(dataset, tag)
    manifest_path = sdir / "manifest.json"
    if manifest_path.exists() and not restart:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("key") == key and all(Path(j["dec"]).exists() for j in old["jobs"]):
            return sdir, old["jobs"], {"reused": True, "decisions": len(dec), "seconds": time.perf_counter() - t0}
    for sub in ("in", "out"):
        shutil.rmtree(sdir / sub, ignore_errors=True)
        (sdir / sub).mkdir(parents=True, exist_ok=True)

    per_hand = dec.groupby("hand_id", sort=True).size()
    chunk_of_hand = (per_hand.cumsum().shift(fill_value=0) // int(en["chunk_decisions"])).astype(int)
    hand_chunk = chunk_of_hand.to_dict()
    jobs: list[dict[str, str]] = []
    parts = {
        "dec": dec.assign(_c=dec.hand_id.map(hand_chunk)),
        "opp": opp.assign(_c=opp.hand_id.map(hand_chunk)),
        "nodes": nodes.assign(_c=nodes.hand_id.map(hand_chunk)),
    }
    groups = {name: dict(tuple(df.dropna(subset=["_c"]).groupby("_c", sort=True))) for name, df in parts.items()}
    for c in sorted(set(hand_chunk.values())):
        job = {"chunk": f"c{c:05d}"}
        for name in ("dec", "opp", "nodes"):
            df = groups[name].get(c, parts[name].iloc[0:0]).drop(columns="_c")
            path = sdir / "in" / f"c{c:05d}_{name}.parquet"
            pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), path)
            job[name] = str(path)
        job["out"] = str(sdir / "out" / f"c{c:05d}.parquet")
        jobs.append(job)
    manifest_path.write_text(json.dumps({"key": key, "jobs": jobs}, indent=1), encoding="utf-8")
    return sdir, jobs, {"reused": False, "decisions": len(dec), "chunks": len(jobs),
                        "seconds": time.perf_counter() - t0}


def run_pool(jobs: Sequence[dict[str, str]], fit_path: Path, settings: Mapping[str, Any],
             procs: int) -> tuple[list[dict[str, Any]], float]:
    """Score every chunk that has no output yet; returns the workers' counters and the wall time."""
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if not todo:
        return results, 0.0
    procs = max(1, min(procs, len(todo)))
    print(f"  scoring {len(todo)} of {len(jobs)} chunks on {procs} processes", flush=True)
    if procs == 1:
        rangemodel.init_worker(str(fit_path), settings)
        for k, job in enumerate(todo, 1):
            results.append(_keep_stats(rangemodel.score_chunk(job)))
            _progress(k, len(todo), t0)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs, initializer=rangemodel.init_worker, initargs=(str(fit_path), dict(settings))) as pool:
            for k, res in enumerate(pool.imap_unordered(rangemodel.score_chunk, todo), 1):
                results.append(_keep_stats(res))
                _progress(k, len(todo), t0)
    return results, time.perf_counter() - t0


def _keep_stats(res: dict[str, Any]) -> dict[str, Any]:
    """Store a chunk's counters next to its output, so a resumed run still reports every chunk."""
    Path(res["chunk"]).with_suffix(".json").write_text(json.dumps(res), encoding="utf-8")
    return res


def chunk_stats(jobs: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """The stored counters of every scored chunk."""
    out = []
    for j in jobs:
        p = Path(j["out"]).with_suffix(".json")
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _progress(k: int, n: int, t0: float) -> None:
    if k == n or k % max(1, n // 10) == 0:
        print(f"    {k}/{n} chunks, {time.perf_counter() - t0:.0f} s", flush=True)


def read_outputs(jobs: Sequence[dict[str, str]]) -> pd.DataFrame:
    """All chunk outputs, in hand and row order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(j["out"]) for j in jobs if Path(j["out"]).exists()]
    if not tables:
        return pd.DataFrame()
    df = pa.concat_tables(tables).to_pandas()
    return df.sort_values(["hand_id", "action_index"], kind="stable").reset_index(drop=True)


def step_enrich(dataset: str, cfg: Mapping[str, Any], procs: int, restart: bool,
                limit_hands: Optional[int], tag: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage, score in the pool, write ``action_range_equity``; returns the scored rows and a summary.

    A tagged run (a variant of the settings) keeps its chunks in its own folder and leaves the warehouse alone.
    """
    con = connect(dataset, read_only=True)      # staging only reads (and fills a temporary table)
    try:
        tables, tables_hash = load_tables(con)
        if bool(cfg["range"].get("sizing", False)) and not any(tables.lik_size.values()):
            raise SystemExit(f"{dataset}: [range] sizing = true needs the table rng_lik_size; run --steps fit again "
                             "(with --fit-from, fit the source dataset again first) or set sizing = false")
        sdir, jobs, stage_info = stage(con, dataset, cfg, tables_hash, restart, limit_hands, tag)
    finally:
        con.close()                      # the warehouse stays free for other readers while the pool runs
    fit_path = sdir / "fit.pkl"
    with open(fit_path, "wb") as fh:
        pickle.dump(tables, fh, protocol=pickle.HIGHEST_PROTOCOL)
    settings = model_settings(cfg)
    results, wall = run_pool(jobs, fit_path, settings, procs)
    out = read_outputs(jobs)
    if len(out) != stage_info["decisions"]:
        raise RuntimeError(f"{len(out)} scored rows for {stage_info['decisions']} staged decisions")
    scored = out[out.eq_vs_range.notna()]
    con = connect(dataset, read_only=bool(tag))
    try:
        if not tag:
            create_output_table(con, "action_range_equity")
            cols = ["hand_id", "action_index", "n_live_opp", "eq_vs_range", "eq_range_method", "eq_range_se",
                    "n_combos_opp", "range_top_class", "range_strong_share", "range_air_share"]
            warehouse.insert_df(con, "action_range_equity", scored[cols], mode="append")
        step02 = _step02_check(con, scored)
    finally:
        con.close()
    lik_mass = np.zeros(len(rangemodel.LIK_LEVELS) + 1)
    sized_mass = 0.0
    prior_levels: dict[str, int] = {}
    cpu = cpu_range = cpu_eq = 0.0
    stats = chunk_stats(jobs)                       # this run's chunks and those kept from an earlier one
    for r in stats:
        lik_mass += np.asarray(r["lik_mass"])
        sized_mass += float(r.get("sized_mass", 0.0))
        for k, v in r["prior_levels"].items():
            prior_levels[k] = prior_levels.get(k, 0) + int(v)
        cpu += r["seconds"]
        cpu_range += r["seconds_range"]
        cpu_eq += r["seconds_equity"]
    summary = {
        "stage": stage_info,
        "decisions_staged": int(len(out)),
        "decisions_scored": int(len(scored)),
        "by_street": scored.groupby("street").size().to_dict(),
        "by_n_live_opp": scored.groupby("n_live_opp").size().to_dict(),
        "procs": procs,
        "pool_seconds": wall,
        "decisions_per_second": (sum(r["rows"] for r in results) / wall) if wall > 0 else None,
        "worker_cpu_seconds": cpu,
        "worker_cpu_ms_per_decision": {
            "range": 1000 * cpu_range / max(1, sum(r["rows"] for r in stats)),
            "equity": 1000 * cpu_eq / max(1, sum(r["rows"] for r in stats)),
        },
        "chunks_scored_this_run": len(results),
        "n_combos_opp": {
            "mean": float(scored.n_combos_opp.mean()) if len(scored) else None,
            "median": float(scored.n_combos_opp.median()) if len(scored) else None,
            "share_at_cap": float((scored.n_combos_opp >= int(cfg["range"]["cap"])).mean()) if len(scored) else None,
            "mean_by_street": scored.groupby("street").n_combos_opp.mean().to_dict(),
        },
        "effective_combos_opp": {          # 1 / sum(w^2): the number of equally weighted combos with the same spread
            "mean": float(scored.n_eff_opp.mean()) if len(scored) else None,
            "mean_by_street": scored.groupby("street").n_eff_opp.mean().to_dict(),
        },
        "real_hand_in_range": {            # hindsight check on full-information data: the opponent's real combo kept
            "share": float(scored.real_in_range.mean()) if scored.real_in_range.notna().any() else None,
            "share_by_street": scored.groupby("street").real_in_range.mean().to_dict(),
        },
        "mean_eq_vs_range_minus_eq_vs_real": float((scored.eq_vs_range - scored.eq_vs_real).mean())
        if scored.eq_vs_real.notna().any() else None,
        "prior_levels": prior_levels,
        "likelihood_level_mass": {lv: float(m) for lv, m in zip(list(rangemodel.LIK_LEVELS) + ["uniform"],
                                                                lik_mass / max(lik_mass.sum(), 1e-12))},
        # sizing = true: share of the update weight read from a populated rng_lik_size cell (the rest: the plain
        # probability times a coarser tier mix, or the plain probability for a bet or raise of unknown size)
        "likelihood_sized_cell_share": float(sized_mass / lik_mass.sum()) if lik_mass.sum() > 0 else None,
        "step02_check": step02,
    }
    return out, summary


def _step02_check(con: Any, scored: pd.DataFrame) -> dict[str, Any]:
    """Compare the real-card equity computed here with step 02's ``action_equity`` when that table exists."""
    if not warehouse.table_exists(con, "action_equity") or not len(scored):
        return {"action_equity": False}
    ref = warehouse.query_df(con, "SELECT hand_id, action_index, eq_vs_real FROM action_equity "
                                  "WHERE eq_vs_real IS NOT NULL")
    if not len(ref):
        return {"action_equity": True, "joined": 0}
    m = scored[["hand_id", "action_index", "eq_vs_real"]].merge(ref, on=["hand_id", "action_index"],
                                                                suffixes=("", "_step02"))
    m = m[m.eq_vs_real.notna()]
    diff = (m.eq_vs_real - m.eq_vs_real_step02).abs()
    return {"action_equity": True, "joined": int(len(m)),
            "max_abs_diff": float(diff.max()) if len(m) else None,
            "share_above_1e-6": float((diff > 1e-6).mean()) if len(m) else None}


# ------------------------------------------------------------------ leakage and sanity


def responses(out: pd.DataFrame, streets: Sequence[str]) -> pd.DataFrame:
    """Heads-up responses to a bet with both equities: street, r, q, y (1 = fold) and the decider."""
    d = out[(out.n_live_opp == 1) & out.pot_odds.notna() & out.action.isin(["fold", "call", "raise"])
            & out.eq_vs_range.notna() & out.eq_vs_real.notna() & out.street.isin(list(streets))]
    return pd.DataFrame({
        "street": d.street.to_numpy(),
        "player_id": d.player_id.to_numpy(),
        "r": d.eq_vs_range.to_numpy(),
        "q": d.eq_vs_real.to_numpy(),
        "req": d.pot_odds.to_numpy(),
        "y": (d.action == "fold").to_numpy().astype(np.int64),
    })


LEAK_TOP_PLAYERS = 20          # players with the most responses get their own leakage split


def leak_matrix(resp: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Counts per (street, range bin, real class)."""
    lk = cfg["leakage"]
    bins = int(lk["bins"])
    b = np.minimum(bins - 1, np.floor(resp.r.to_numpy() * bins)).astype(np.int64)
    cls = np.where(resp.q < float(lk["dead_below"]), "dead", np.where(resp.q >= float(lk["win_from"]), "win", "mid"))
    m = pd.DataFrame({"street": resp.street, "range_bin": b, "real_class": cls, "y": resp.y})
    g = m.groupby(["street", "range_bin", "real_class"], sort=True).y.agg(["size", "sum"]).reset_index()
    return g.rename(columns={"size": "n", "sum": "folds"})


def _loglik(n: np.ndarray, k: np.ndarray) -> float:
    n = n.astype(np.float64)
    k = k.astype(np.float64)
    p = np.where(n > 0, k / np.where(n > 0, n, 1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(k > 0, k * np.log(np.where(p > 0, p, 1)), 0.0)
        b = np.where(n - k > 0, (n - k) * np.log(np.where(p < 1, 1 - p, 1)), 0.0)
    return float(a.sum() + b.sum())


def leakage_split(mat: pd.DataFrame, streets: Sequence[str]) -> dict[str, Any]:
    """Binomial log-likelihoods of the fold indicator under street, street x bin, street x bin x real class.

    ``2 (l1 - l0)`` is what the range bins explain, ``2 (l2 - l1)`` what the real class adds, and the real share
    is ``(l2 - l1) / (l2 - l0)``.
    """
    m = mat[mat.street.isin(list(streets))]
    if not len(m):
        return {"n": 0}
    g0 = m.groupby("street")[["n", "folds"]].sum()
    g1 = m.groupby(["street", "range_bin"])[["n", "folds"]].sum()
    l0 = _loglik(g0.n.to_numpy(), g0.folds.to_numpy())
    l1 = _loglik(g1.n.to_numpy(), g1.folds.to_numpy())
    l2 = _loglik(m.n.to_numpy(), m.folds.to_numpy())
    total = l2 - l0
    return {
        "n": int(m.n.sum()),
        "fold_rate": float(m.folds.sum() / m.n.sum()),
        "deviance_range_bins": 2 * (l1 - l0),
        "deviance_real_class_added": 2 * (l2 - l1),
        "real_share": (l2 - l1) / total if total > 0 else None,
    }


def fold_curves(resp: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fold rate per street and range bin, overall and split dead / winning."""
    lk = cfg["leakage"]
    bins = int(lk["bins"])
    out: dict[str, Any] = {}
    for street, d in resp.groupby("street", sort=True):
        b = np.minimum(bins - 1, np.floor(d.r.to_numpy() * bins)).astype(np.int64)
        dead = d.q.to_numpy() < float(lk["dead_below"])
        win = d.q.to_numpy() >= float(lk["win_from"])
        rows = []
        for k in range(bins):
            sel = b == k
            n = int(sel.sum())
            nd, nw = int((sel & dead).sum()), int((sel & win).sum())
            rows.append({
                "bin": k, "n": n, "fold": float(d.y.to_numpy()[sel].mean()) if n else None,
                "n_dead": nd, "fold_dead": float(d.y.to_numpy()[sel & dead].mean()) if nd else None,
                "n_win": nw, "fold_win": float(d.y.to_numpy()[sel & win].mean()) if nw else None,
            })
        out[str(street)] = rows
    return out


def step_leakage(dataset: str, out: pd.DataFrame, cfg: Mapping[str, Any], tag: str = "") -> dict[str, Any]:
    """Leakage split, fold curves and the agreement of the two equities; writes ``rng_leak_matrix``."""
    resp = responses(out, ["flop", "turn", "river"])
    mat = leak_matrix(resp, cfg)
    if not tag:
        con = connect(dataset)
        try:
            create_output_table(con, "rng_leak_matrix")
            warehouse.insert_df(con, "rng_leak_matrix", mat, mode="append")
        finally:
            con.close()
    both = out[out.eq_vs_range.notna() & out.eq_vs_real.notna() & (out.n_live_opp == 1)]
    corr = {}
    for street, d in both.groupby("street"):
        corr[str(street)] = {"n": int(len(d)), "pearson": float(np.corrcoef(d.eq_vs_range, d.eq_vs_real)[0, 1]),
                             "mean_abs_diff": float((d.eq_vs_range - d.eq_vs_real).abs().mean())}
    if len(both) > 1:
        corr["all"] = {"n": int(len(both)), "pearson": float(np.corrcoef(both.eq_vs_range, both.eq_vs_real)[0, 1]),
                       "mean_abs_diff": float((both.eq_vs_range - both.eq_vs_real).abs().mean())}
    streets = list(cfg["leakage"]["streets"])
    # the same split for each of the players with the most responses: the pooled split mixes a suspect with his
    # opponents, so a per-suspect statement needs his own rows
    own = resp[resp.street.isin(streets)].player_id.value_counts()
    by_player = {str(pid): leakage_split(leak_matrix(resp[resp.player_id == pid], cfg), streets)
                 for pid in own.index[:LEAK_TOP_PLAYERS]}
    return {
        "decision_set": "heads-up responses to a bet (fold / call / raise) with both equities known",
        "split": leakage_split(mat, streets),
        "split_by_player": by_player,
        "split_streets": streets,
        "split_by_street": {s: leakage_split(mat, [s]) for s in ("flop", "turn", "river")},
        "split_all_postflop": leakage_split(mat, ["flop", "turn", "river"]),
        "fold_curves": fold_curves(resp, cfg),
        "eq_vs_range_vs_eq_vs_real": corr,
    }


# ------------------------------------------------------------------ main


def _load_examples_module() -> Any:
    spec = importlib.util.spec_from_file_location("step04_examples", HERE / "examples.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_summary(dataset: str, summary: Mapping[str, Any], tag: str = "") -> list[Path]:
    """Write the run summary to the data folder and, for the public datasets, to ``results/``."""
    text = json.dumps(_jsonable(summary), indent=1, sort_keys=False)
    paths = [stage_dir(dataset, tag) / "summary.json"]
    if dataset in PUBLIC_DATASETS:
        paths.append(HERE / "results" / (f"{dataset}_{tag}_summary.json" if tag else f"{dataset}_summary.json"))
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 04: predicted ranges and equity against them.")
    ap.add_argument("--dataset", required=True, help="warehouse name, e.g. article_sample")
    ap.add_argument("--fit-from", default=None,
                    help="take the field tables from this full-information dataset instead of fitting here")
    ap.add_argument("--config", default=None, help="TOML file overriding config/step04_range.toml")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {','.join(STEPS)}")
    ap.add_argument("--procs", type=int, default=None, help="worker processes (config enrich.procs)")
    ap.add_argument("--restart", action="store_true", help="drop the staged chunks and score everything again")
    ap.add_argument("--limit-hands", type=int, default=None, help="score only the first N hands (tests)")
    ap.add_argument("--stage-tag", default="",
                    help="score a variant of the settings (with --config) into <dataset>_step04_<tag>/ without "
                         "touching the warehouse tables; the fitted tables are reused")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        ap.error(f"unknown steps {bad}; choose from {STEPS}")
    tag = args.stage_tag.strip()
    if tag and not all(ch.isalnum() or ch in "_-" for ch in tag):
        ap.error("--stage-tag uses letters, digits, '_' and '-'")
    if tag:
        steps = [s for s in steps if s in ("enrich", "leakage")]
    procs = int(args.procs or cfg["enrich"]["procs"])
    summary: dict[str, Any] = {"dataset": args.dataset, "steps": steps}
    if tag:
        summary["variant"] = tag
    summary_path = stage_dir(args.dataset, tag) / "summary.json"
    if summary_path.exists():                      # keep the parts of a previous run that this run does not redo
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            for k in ("fit", "enrich", "leakage", "examples"):
                if k in old and k not in steps:
                    summary[k] = old[k]
        except (OSError, ValueError):
            pass
    t_all = time.perf_counter()

    if "fit" in steps:
        print(f"[fit] {args.dataset}" + (f" (tables from {args.fit_from})" if args.fit_from else ""), flush=True)
        con = connect(args.dataset)
        try:
            summary["fit"] = step_fit(con, args.dataset, args.fit_from, cfg)
        finally:
            con.close()
        for row in summary["fit"]["rng_lik_post"]:
            print(f"  {row['level']}: {row['cells']} cells, {row['cells_supported']} with >= "
                  f"{cfg['likelihood']['min_support']} actions", flush=True)

    out: Optional[pd.DataFrame] = None
    if "enrich" in steps:
        print(f"[enrich] {args.dataset}" + (f" (variant {tag})" if tag else ""), flush=True)
        out, summary["enrich"] = step_enrich(args.dataset, cfg, procs, args.restart, args.limit_hands, tag)
        e = summary["enrich"]
        print(f"  {e['decisions_scored']} of {e['decisions_staged']} decisions scored, "
              f"{e['pool_seconds']:.0f} s in the pool"
              + (f", {e['decisions_per_second']:.0f} decisions/s" if e["decisions_per_second"] else ""), flush=True)

    if "leakage" in steps:
        if out is None:
            sdir = stage_dir(args.dataset, tag)
            manifest = sdir / "manifest.json"
            if not manifest.exists():
                raise SystemExit("leakage needs the enrich step's outputs: run --steps enrich first")
            out = read_outputs(json.loads(manifest.read_text(encoding="utf-8"))["jobs"])
        print(f"[leakage] {args.dataset}", flush=True)
        summary["leakage"] = step_leakage(args.dataset, out, cfg, tag)
        sp = summary["leakage"]["split"]
        if sp.get("n"):
            share = "n/a" if sp["real_share"] is None else f"{sp['real_share']:.3f}"
            print(f"  {sp['n']} responses on {summary['leakage']['split_streets']}: range bins explain "
                  f"{sp['deviance_range_bins']:.0f}, the real class adds {sp['deviance_real_class_added']:.0f} "
                  f"(real share {share})", flush=True)

    if "examples" in steps:
        ex_dataset = str(cfg["examples"]["dataset"])
        if args.dataset == ex_dataset:
            print(f"[examples] {args.dataset}", flush=True)
            summary["examples"] = _load_examples_module().run_example(args.dataset, cfg)
        else:                       # the figures have one name for all datasets: only one dataset draws them
            print(f"[examples] skipped: the example is drawn from {ex_dataset} (config examples.dataset)", flush=True)

    summary["seconds_total"] = time.perf_counter() - t_all
    for p in write_summary(args.dataset, summary, tag):
        print(f"summary: {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
