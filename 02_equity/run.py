"""Step 02: equity references for the decision rows -> ``action_equity``, ``hand_equity`` and three views.

For every decision on the flop, turn and river whose actor's cards are known,
step 02 stores the actor's equity against one random hand (``eq_vs_random``),
against the real cards of every live opponent (``eq_vs_real``, NULL unless all
of them are known) and whether the actor's hand is ahead right now
(``ahead_now``). For every heads-up hand it stores each player's preflop
equity against the other's hand (``hand_equity``). The views ``actions_eq``,
``decisions_eq`` and ``hand_players_eq`` join the new columns to the step-01
tables.

Usage::

    python 02_equity/run.py --dataset article_sample
    python 02_equity/run.py --dataset <name> --procs 16
    python 02_equity/run.py --dataset <name> --backend pyrust --fresh

How it runs: the hands are cut into batches of about ``batch_rows`` decision
rows (never splitting a hand). Each batch is read with
``sql/inputs_decisions.sql``, scored in a process pool and written as a
parquet part under ``$SU_DATA_DIR/<dataset>/stage_equity/``. A part records a
fingerprint of its input rows and settings; a rerun reuses every part whose
fingerprint still matches, so an interrupted run resumes where it stopped.
The tables are then rebuilt from the parts (``CREATE OR REPLACE``), the views
are created and ``sql/checks.sql`` runs. The exit code is 1 when a check fails.

The warehouse file is open only while SQL runs: read-only for each read, and
read-write once, for the single transaction that rebuilds the tables. While
another process holds the file lock, a connection attempt is retried with a
growing wait for up to ``lock_wait_s`` seconds.

Options:
    --backend           numpy | pyrust (config step02.backend)
    --procs             worker processes, 0 = all logical CPUs, 1 = no pool (config step02.procs)
    --seed              base seed of the Monte Carlo estimates (config step02.seed)
    --config            TOML file whose keys override config/step02_equity.toml
    --fresh             delete the staged parts first
    --no-hand-equity    skip hand_equity (the table is left empty)
    --identity-sample   heads-up hands to recompute in both directions after the run (0 = skip)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional, Union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sudetect import equity_backends as eb  # noqa: E402
from sudetect import warehouse  # noqa: E402

SQL_DIR = HERE / "sql"
DEFAULT_CONFIG = ROOT / "config" / "step02_equity.toml"
REQUIRED_TABLES = ("hands", "hand_players", "actions", "decisions", "hu_decisions")
KINDS = ("decisions", "hands")
CODE_FILES = ("equity_backends.py", "equity.py", "evaluator.py", "cards.py")


def code_hash() -> str:
    """SHA-256 (first 16 hex digits) of the library modules that compute the equities.

    It is part of every staged part's fingerprint, so an edit of any of these
    files makes the next run recompute the parts instead of reusing them.
    """
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((ROOT / "sudetect" / name).read_bytes())
    return h.hexdigest()[:16]


CODE_HASH = code_hash()


# ------------------------------------------------------------------ configuration


def _merge(base: dict, extra: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(extra: Optional[Union[str, os.PathLike[str]]] = None) -> dict:
    """``config/step02_equity.toml`` with the keys of ``extra`` (a TOML file) merged over it."""
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if extra:
        with open(extra, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


@dataclass(frozen=True)
class Settings:
    """The ``[step02]`` table of the configuration (see ``config/step02_equity.toml``)."""

    backend: str = "numpy"
    include_preflop: bool = False
    seed: int = 20011
    mc_samples: int = 4096
    flop_random_exact: bool = False
    hand_equity: bool = True
    hand_equity_samples: int = 8192
    batch_rows: int = 50_000
    batch_hands: int = 50_000
    chunk_rows: int = 2000
    procs: int = 0
    stage_dir: str = "stage_equity"
    identity_sample: int = 1000
    lock_wait_s: float = 300.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], **overrides: Any) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in dict(cfg.get("step02", {})).items() if k in known}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def backend_settings(self, kind: str) -> dict[str, Any]:
        """Constructor arguments of the backend for one kind of rows (``decisions`` or ``hands``)."""
        if kind == "decisions":
            return {"seed": int(self.seed), "mc_samples": int(self.mc_samples),
                    "flop_random_exact": bool(self.flop_random_exact)}
        return {"seed": int(self.seed), "mc_samples": int(self.hand_equity_samples)}

    def n_procs(self) -> int:
        return int(self.procs) if self.procs and self.procs > 0 else (os.cpu_count() or 1)


# ------------------------------------------------------------------ staging


def plan_batches(plan: pd.DataFrame, batch_rows: int) -> list[tuple[str, str, int]]:
    """Cut hands (``hand_id``, ``n_rows`` in hand order) into ``(first hand, last hand, rows)`` batches.

    A batch closes as soon as it holds ``batch_rows`` rows or more; a hand is never split.
    """
    ids = plan["hand_id"].tolist()
    counts = plan["n_rows"].to_numpy(dtype=np.int64)
    out: list[tuple[str, str, int]] = []
    start, acc = 0, 0
    for i, c in enumerate(counts.tolist()):
        acc += c
        if acc >= batch_rows:
            out.append((ids[start], ids[i], acc))
            start, acc = i + 1, 0
    if start < len(ids):
        out.append((ids[start], ids[-1], acc))
    return out


def fingerprint(kind: str, frame: pd.DataFrame, backend: str, settings: Mapping[str, Any]) -> str:
    """Hash of a batch's input rows, the backend, its settings and the library code; reused only on a match."""
    h = hashlib.sha256()
    meta = {"kind": kind, "code": [eb.CODE_VERSION, CODE_HASH], "backend": backend, "settings": dict(settings),
            "columns": list(frame.columns)}
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


class Stage:
    """The folder of staged parquet parts: ``<kind>_<00000>.parquet``, each with a fingerprint in its metadata."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path(self, kind: str, index: int) -> Path:
        return self.folder / f"{kind}_{index:05d}.parquet"

    def fingerprint_of(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        try:
            meta = pq.read_schema(path).metadata or {}
        except (OSError, pa.ArrowInvalid):
            return None
        value = meta.get(b"fingerprint")
        return value.decode("ascii") if value else None

    def write(self, path: Path, table: pa.Table, fp: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        meta = dict(table.schema.metadata or {})
        meta[b"fingerprint"] = fp.encode("ascii")
        pq.write_table(table.replace_schema_metadata(meta), tmp)
        os.replace(tmp, path)

    def prune(self, kind: str, n_parts: int) -> int:
        """Delete parts of ``kind`` numbered ``n_parts`` or above, and unfinished temporary files."""
        removed = 0
        if not self.folder.exists():
            return 0
        for p in self.folder.glob(f"{kind}_*.parquet*"):
            stem = p.name.split(".", 1)[0]
            try:
                index = int(stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if p.name.endswith(".tmp") or index >= n_parts:
                p.unlink()
                removed += 1
        return removed

    def parts(self, kind: str, n_parts: int) -> list[str]:
        return [self.path(kind, i).as_posix() for i in range(n_parts)]


class Runner:
    """Runs scoring tasks in this process (``procs == 1``) or in a spawn-started process pool made on first use."""

    def __init__(self, procs: int) -> None:
        self.procs = procs
        self._pool: Optional[ProcessPoolExecutor] = None

    def submit(self, tasks: Sequence[tuple]) -> list[Any]:
        if self.procs <= 1:
            return [eb.run_task(t) for t in tasks]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.procs, mp_context=get_context("spawn"),
                                             initializer=eb.init_worker)
        return [self._pool.submit(eb.run_task, t) for t in tasks]

    @staticmethod
    def collect(handles: Sequence[Any]) -> list[dict[str, np.ndarray]]:
        return [h.result() if isinstance(h, Future) else h for h in handles]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def _chunks(frame: pd.DataFrame, kind: str, chunk_rows: int) -> list[dict[str, Any]]:
    columns = eb.DECISION_COLUMNS if kind == "decisions" else eb.HAND_COLUMNS
    out = []
    for lo in range(0, len(frame), max(1, chunk_rows)):
        part = frame.iloc[lo:lo + chunk_rows]
        cols: dict[str, Any] = {}
        for c in columns:
            s = part[c]
            cols[c] = s.to_numpy() if s.dtype.kind in "iuf" else s.tolist()
        out.append(cols)
    return out


def _merge_results(results: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([r[k] for r in results]) for k in results[0]}


def _method(exact: np.ndarray) -> np.ndarray:
    return np.where(exact, "exact", "mc").astype(object)


def decision_table(frame: pd.DataFrame, res: Mapping[str, np.ndarray]) -> pa.Table:
    """The ``action_equity`` rows of one batch as an arrow table."""
    rnd_null = np.isnan(res["eq_vs_random"])
    real_null = np.isnan(res["eq_vs_real"])
    ahead = res["ahead_now"]
    return pa.table({
        "hand_id": pa.array(frame["hand_id"].tolist(), pa.string()),
        "action_index": pa.array(frame["action_index"].to_numpy(dtype=np.int16), pa.int16()),
        "street": pa.array(frame["street"].tolist(), pa.string()),
        "n_live_opp": pa.array(res["n_live_opp"].astype(np.int16), pa.int16()),
        "eq_vs_random": pa.array(res["eq_vs_random"], pa.float64(), mask=rnd_null),
        "eq_method": pa.array(_method(res["eq_random_exact"]), pa.string(), mask=rnd_null),
        "eq_se": pa.array(res["eq_se"], pa.float64(), mask=rnd_null),
        "eq_vs_real": pa.array(res["eq_vs_real"], pa.float64(), mask=real_null),
        "eq_real_known": pa.array(res["eq_real_known"], pa.bool_()),
        "eq_real_method": pa.array(_method(res["eq_real_exact"]), pa.string(), mask=real_null),
        "eq_real_se": pa.array(res["eq_real_se"], pa.float64(), mask=real_null),
        "ahead_now": pa.array(ahead == 1, pa.bool_(), mask=ahead < 0),
    })


def hand_table(frame: pd.DataFrame, res: Mapping[str, np.ndarray]) -> pa.Table:
    """The ``hand_equity`` rows of one batch: player A then player B of every hand."""
    def pairs(a: Any, b: Any) -> np.ndarray:
        return np.column_stack([np.asarray(a, dtype=object), np.asarray(b, dtype=object)]).ravel()

    eq_a = res["eq_a"]
    eq = np.column_stack([eq_a, 1.0 - eq_a]).ravel()
    null = np.isnan(eq)
    return pa.table({
        "hand_id": pa.array(np.repeat(frame["hand_id"].to_numpy(dtype=object), 2).tolist(), pa.string()),
        "player_id": pa.array(pairs(frame["player_a"], frame["player_b"]).tolist(), pa.string()),
        "opp_player_id": pa.array(pairs(frame["player_b"], frame["player_a"]).tolist(), pa.string()),
        "preflop_eq_vs_opp": pa.array(eq, pa.float64(), mask=null),
        "eq_method": pa.array(np.repeat(_method(res["exact"]), 2), pa.string(), mask=null),
        "eq_se": pa.array(np.repeat(res["se"], 2), pa.float64(), mask=null),
    })


@dataclass
class KindStats:
    """What one kind of rows (``decisions`` or ``hands``) did in a run."""

    kind: str
    rows: int = 0
    parts: int = 0
    parts_reused: int = 0
    parts_computed: int = 0
    rows_computed: int = 0
    seconds: float = 0.0

    @property
    def rows_per_second(self) -> float:
        return self.rows_computed / self.seconds if self.seconds > 0 and self.rows_computed else float("nan")


def stage_kind(dataset: str, kind: str, settings: Settings, stage: Stage, runner: Runner,
               log: Callable[[str], None] = print, stop_after: Optional[int] = None) -> KindStats:
    """Score every batch of one kind into its staged part; returns the counts and the time spent.

    The plan and every batch are read in short read-only sessions, so the
    warehouse file is free while the workers compute.
    """
    t0 = time.perf_counter()
    if kind == "decisions":
        plan_sql, inputs_sql, batch_size = "plan_decisions.sql", "inputs_decisions.sql", settings.batch_rows
    else:
        plan_sql, inputs_sql, batch_size = "plan_hands.sql", "inputs_hands.sql", settings.batch_hands
    params = {"include_preflop": bool(settings.include_preflop)}
    wait_s = float(settings.lock_wait_s)
    plan = read_df(dataset, (SQL_DIR / plan_sql).read_text(encoding="utf-8"), wait_s, log, **params)
    inputs_text = (SQL_DIR / inputs_sql).read_text(encoding="utf-8")
    batches = plan_batches(plan, int(batch_size))
    stats = KindStats(kind, rows=int(plan["n_rows"].sum()) if len(plan) else 0, parts=len(batches))
    bsettings = settings.backend_settings(kind)
    make_table = decision_table if kind == "decisions" else hand_table
    pending: Optional[tuple[int, pd.DataFrame, str, list[Any]]] = None

    def finish(item: tuple[int, pd.DataFrame, str, list[Any]]) -> None:
        index, frame, fp, handles = item
        table = make_table(frame, _merge_results(runner.collect(handles)))
        stage.write(stage.path(kind, index), table, fp)
        stats.parts_computed += 1
        stats.rows_computed += len(frame)
        log(f"  {kind} part {index + 1}/{len(batches)}: {len(frame):,} rows")

    try:
        for index, (lo, hi, _) in enumerate(batches):
            if stop_after is not None and stats.parts_computed + (pending is not None) >= stop_after:
                break
            frame = read_df(dataset, inputs_text, wait_s, log, lo=lo, hi=hi, **params)
            fp = fingerprint(kind, frame, settings.backend, bsettings)
            if stage.fingerprint_of(stage.path(kind, index)) == fp:
                stats.parts_reused += 1
                continue
            tasks = [(kind, settings.backend, bsettings, cols) for cols in _chunks(frame, kind, settings.chunk_rows)]
            handles = runner.submit(tasks)   # the pool works on this batch while the next one is read
            if pending is not None:
                finish(pending)
            pending = (index, frame, fp, handles)
        if pending is not None:
            finish(pending)
    finally:
        stats.seconds = time.perf_counter() - t0
    if stop_after is None:
        stage.prune(kind, len(batches))
    return stats


# ------------------------------------------------------------------ the run


@dataclass
class RunResult:
    """What a run did: per-kind counts, the checks, the summary tables and timings."""

    dataset: str
    settings: Settings
    stats: dict[str, KindStats] = field(default_factory=dict)
    checks: Optional[pd.DataFrame] = None
    summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    identity: Optional[dict[str, float]] = None
    seconds_build: float = 0.0
    seconds_total: float = 0.0

    @property
    def violations(self) -> int:
        return 0 if self.checks is None else int(self.checks["violations"].sum())


def connect(dataset: str, read_only: bool, wait_s: float, log: Callable[[str], None] = print) -> Any:
    """Open the warehouse, retrying while another process holds its file lock.

    The wait between attempts starts at 0.25 s and doubles up to 8 s, with a
    random jitter so that several waiting processes do not retry in step; the
    lock error is raised once ``wait_s`` seconds have passed.
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    delay = 0.25
    warned = False
    while True:
        try:
            return warehouse.connect(dataset, read_only=read_only)
        except Exception as exc:  # the engine's lock error has no portable type
            text = str(exc).lower()
            busy = "lock" in text or "used by another process" in text or "already open" in text
            if not busy or time.monotonic() >= deadline:
                raise
            if not warned:
                log(f"waiting for another process to release {warehouse.db_path(dataset).name} ...")
                warned = True
            time.sleep(min(delay * random.uniform(0.75, 1.25), max(0.0, deadline - time.monotonic())))
            delay = min(2.0 * delay, 8.0)


@contextmanager
def session(dataset: str, read_only: bool, wait_s: float, log: Callable[[str], None] = print) -> Iterator[Any]:
    """A connection that is closed on exit; step 02 holds one only while SQL runs, never while it computes."""
    con = connect(dataset, read_only, wait_s, log)
    try:
        yield con
    finally:
        con.close()


def read_df(dataset: str, sql: str, wait_s: float, log: Callable[[str], None] = print, **params: Any) -> pd.DataFrame:
    """One query in its own read-only session."""
    with session(dataset, True, wait_s, log) as con:
        return warehouse.query_df(con, sql, **params)


def _require_tables(con: Any, dataset: str) -> None:
    missing = [t for t in REQUIRED_TABLES if not warehouse.table_exists(con, t)]
    if missing:
        raise RuntimeError(f"dataset {dataset!r} has no {', '.join(missing)}: run 01_action_table/run.py first")


def run(dataset: str, *, settings: Optional[Settings] = None, fresh: bool = False,
        log: Callable[[str], None] = print, stop_after: Optional[int] = None) -> RunResult:
    """Run step 02 on one dataset; see the module docstring.

    ``stop_after`` stops after computing that many parts of the first kind
    without building the tables (it simulates an interrupted run in the tests).
    """
    settings = settings or Settings.from_config(load_config())
    if settings.backend not in eb.BACKENDS:
        raise RuntimeError(f"unknown backend {settings.backend!r}")
    if settings.backend == "pyrust" and not eb.pyrust_available():
        raise RuntimeError("backend 'pyrust' needs the optional package pyrust_poker")
    result = RunResult(dataset, settings)
    t_start = time.perf_counter()
    if not warehouse.db_path(dataset).exists():
        raise RuntimeError(f"no warehouse at {warehouse.db_path(dataset)}: run 01_action_table/run.py first")
    stage = Stage(warehouse.data_dir() / dataset / settings.stage_dir)
    if fresh and stage.folder.exists():
        shutil.rmtree(stage.folder)
    kinds = KINDS if settings.hand_equity else ("decisions",)

    wait_s = float(settings.lock_wait_s)
    with session(dataset, True, wait_s, log) as con:
        _require_tables(con, dataset)

    runner = Runner(settings.n_procs())
    try:
        for kind in kinds:
            result.stats[kind] = stage_kind(dataset, kind, settings, stage, runner, log=log, stop_after=stop_after)
            s = result.stats[kind]
            log(f"{kind}: {s.rows:,} rows in {s.parts} parts ({s.parts_reused} reused, {s.parts_computed} computed) "
                f"in {s.seconds:.1f} s" + (f", {s.rows_per_second:,.0f} rows/s" if s.rows_computed else ""))
            if stop_after is not None:
                return result
    finally:
        runner.close()

    t0 = time.perf_counter()
    with session(dataset, False, wait_s, log) as con:   # the only read-write session: one transaction
        warehouse.execute(con, "BEGIN TRANSACTION")
        try:
            warehouse.run_sql_file(con, SQL_DIR / "schema.sql")
            for kind, sql in (("decisions", "load_action_equity.sql"), ("hands", "load_hand_equity.sql")):
                if kind in result.stats and result.stats[kind].parts:
                    warehouse.run_sql_file(con, SQL_DIR / sql, parts=stage.parts(kind, result.stats[kind].parts))
            settings_rows = {**asdict(settings), "code_version": eb.CODE_VERSION, "code_hash": CODE_HASH}
            warehouse.insert_df(con, "equity_settings", pd.DataFrame(
                {"setting": list(settings_rows), "value": [str(v) for v in settings_rows.values()]}))
            warehouse.run_sql_file(con, SQL_DIR / "views.sql")
            warehouse.execute(con, "COMMIT")
        except BaseException:
            warehouse.execute(con, "ROLLBACK")
            raise
    result.seconds_build = time.perf_counter() - t0
    with session(dataset, True, wait_s, log) as con:
        result.checks = warehouse.run_sql_file(con, SQL_DIR / "checks.sql",
                                               include_preflop=bool(settings.include_preflop),
                                               hand_equity=bool(settings.hand_equity))
        result.summary = summarize(con)
    if settings.hand_equity and settings.identity_sample > 0:
        result.identity = identity_check(dataset, settings, int(settings.identity_sample), log)
    result.seconds_total = time.perf_counter() - t_start
    return result


# ------------------------------------------------------------------ reporting


def summarize(con: Any) -> dict[str, pd.DataFrame]:
    """Run every query of ``sql/summary.sql``; keys are the titles on their first comment line."""
    out: dict[str, pd.DataFrame] = {}
    text = (SQL_DIR / "summary.sql").read_text(encoding="utf-8")
    for statement in warehouse.split_statements(text):
        lines = [ln.strip() for ln in statement.splitlines()]
        titles = [ln[2:].strip() for ln in lines if ln.startswith("--")]
        title = titles[-1] if titles else f"query {len(out) + 1}"
        out[title] = warehouse.query_df(con, statement)
    return out


def identity_check(dataset: str, settings: Settings, n: int,
                   log: Callable[[str], None] = print) -> dict[str, float]:
    """Recompute both directions of ``n`` heads-up hands independently: eq(A vs B) + eq(B vs A) should be 1.

    ``hand_equity`` stores B as 1 - A, so its rows sum to 1 by construction;
    this check computes B's side again with its own Monte Carlo deals.
    """
    with session(dataset, True, float(settings.lock_wait_s), log) as con:
        plan = warehouse.query_df(con, (SQL_DIR / "plan_hands.sql").read_text(encoding="utf-8"))
        if not len(plan):
            return {"hands": 0}
        ids = plan["hand_id"].tolist()[:n]
        frame = warehouse.query_df(con, (SQL_DIR / "inputs_hands.sql").read_text(encoding="utf-8"),
                                   lo=ids[0], hi=ids[-1])
    backend = eb.get_backend(settings.backend, **settings.backend_settings("hands"))
    ab = eb.score_hu_hands({"cards_a": frame["cards_a"].tolist(), "cards_b": frame["cards_b"].tolist()}, backend)
    ba = eb.score_hu_hands({"cards_a": frame["cards_b"].tolist(), "cards_b": frame["cards_a"].tolist()}, backend)
    dev = np.abs(ab["eq_a"] + ba["eq_a"] - 1.0)
    se = np.sqrt(ab["se"] ** 2 + ba["se"] ** 2)
    z = np.where(se > 0, dev / np.where(se > 0, se, 1.0), 0.0)
    return {"hands": float(len(frame)), "mean_abs_dev": float(dev.mean()), "max_abs_dev": float(dev.max()),
            "mean_z": float(z.mean()), "max_z": float(z.max()), "share_z_above_3": float((z > 3).mean())}


def report(result: RunResult, log: Callable[[str], None] = print) -> None:
    """Print the checks, the summary tables and the identity check."""
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        if result.checks is not None:
            bad = result.checks[result.checks["violations"] > 0]
            if len(bad):
                log("check violations:")
                for row in bad.itertuples(index=False):
                    log(f"  {row.check_name}: {row.violations}")
            else:
                log(f"checks: all {len(result.checks)} pass")
        for title, frame in result.summary.items():
            log(f"\n{title}\n{frame.to_string(index=False)}")
    if result.identity:
        i = result.identity
        if i.get("hands"):
            log(f"\nheads-up identity on {int(i['hands']):,} hands computed both ways: mean |eqA + eqB - 1| = "
                f"{i['mean_abs_dev']:.5f}, max {i['max_abs_dev']:.5f}; max z = {i['max_z']:.2f}, "
                f"share with z > 3: {i['share_z_above_3']:.4f}")
    log(f"\nbuild {result.seconds_build:.1f} s; total {result.seconds_total:.1f} s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Equity references for every decision row (step 02).")
    ap.add_argument("--dataset", required=True, help="warehouse name, e.g. article_sample")
    ap.add_argument("--backend", choices=eb.BACKENDS[:2], default=None)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None, help="TOML file overriding config/step02_equity.toml")
    ap.add_argument("--fresh", action="store_true", help="delete the staged parts first")
    ap.add_argument("--no-hand-equity", action="store_true", help="skip hand_equity")
    ap.add_argument("--identity-sample", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: cannot read the configuration: {exc}", file=sys.stderr)
        return 2
    settings = Settings.from_config(cfg, backend=args.backend, procs=args.procs, seed=args.seed,
                                    identity_sample=args.identity_sample)
    if args.no_hand_equity:
        settings = replace(settings, hand_equity=False)
    try:
        result = run(args.dataset, settings=settings, fresh=args.fresh)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report(result)
    return 1 if result.violations else 0


if __name__ == "__main__":
    sys.exit(main())
