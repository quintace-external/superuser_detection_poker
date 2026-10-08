"""Step 01: raw hand histories -> ``hands``, ``hand_players`` and ``actions`` in the warehouse.

Reads hand-history text files (and ``.txt`` members of ``.zip`` archives),
splits them into hands, parses each hand, drops the ones that fail the gate
(every reason is counted in ``parse_funnel``), replays the rest and inserts the
three tables in batches. Then it numbers the hands per table, creates the views
``decisions`` and ``hu_decisions`` and runs the invariant checks.

Usage::

    python 01_action_table/run.py --dataset article_sample --input $SU_DATA_DIR/article_sample/input/hands.txt --source operator
    python 01_action_table/run.py --dataset mined --input "D:/data/**/*.zip" --hash-ids

Options:
    --input        folder, glob or file; repeat for several (folders are read recursively)
    --hash-ids     replace player, table and hand ids by salted HMAC-SHA256 hashes;
                   the secret comes from the environment variable SU_ID_SALT
    --incremental  replace only the days present in the input (DELETE + INSERT per hand_date);
                   load every file of a day in the same run. Default: rebuild the tables.
    --batch-hands  hands per bulk insert (config step01.batch_hands)
    --config       TOML file whose keys override config/defaults.toml
    --source       label stored in hands.source: operator (full information) | mined (observer) | sim (tests)

The exit code is 1 when an invariant fails or a hand could not be replayed
(``--allow-violations`` reports them and exits 0).
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
import tomllib
import zipfile
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sudetect import warehouse  # noqa: E402
from sudetect.hhparse import GateReject, ParseOptions, parse_hand, split_hands  # noqa: E402
from sudetect.replay import Batch, IdMapper, ReplayContext, handclass_available, replay_hand  # noqa: E402

SQL_DIR = HERE / "sql"
DEFAULT_CONFIG = ROOT / "config" / "defaults.toml"
STEP_TABLES = ("hands", "hand_players", "actions", "parse_funnel")


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
    """``config/defaults.toml`` with the keys of ``extra`` (a TOML file) merged over it."""
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if extra:
        with open(extra, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def parse_options(step: Mapping[str, Any]) -> ParseOptions:
    """Parser switches from the ``[step01]`` table."""
    return ParseOptions(
        amount_decimals=int(step["amount_decimals"]),
        infer_hidden_posts=bool(step["infer_hidden_posts"]),
        drop_hidden_post_hands=bool(step["drop_hidden_post_hands"]),
        keep_bomb_pots=bool(step["keep_bomb_pots"]),
        timeout_marks_auto_action=bool(step["timeout_marks_auto_action"]),
        money_tolerance=int(step["money_tolerance_units"]),
    )


def default_source(dataset: str) -> str:
    if dataset == "fixture":
        return "fixture"
    if dataset.startswith("sim"):
        return "sim"
    return "mined"


# ------------------------------------------------------------------ inputs


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").lstrip("\ufeff")


def iter_sources(inputs: Sequence[str], extensions: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(label, text)`` for every input file and every ``.txt`` member of a ``.zip``.

    Labels are paths relative to the folder given on the command line (zip
    members as ``archive.zip:member``), so no absolute local path is stored.
    """
    exts = {e.lower() for e in extensions}
    for spec in inputs:
        path = Path(spec)
        if path.is_dir():
            base = path
            files = sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        elif any(ch in spec for ch in "*?["):
            files = sorted(Path(x) for x in glob.glob(spec, recursive=True) if Path(x).is_file())
            base = Path(os.path.commonpath([str(f.parent) for f in files])) if files else Path(".")
        elif path.is_file():
            base = path.parent
            files = [path]
        else:
            raise FileNotFoundError(f"no input at {spec}")
        for f in files:
            try:
                label = f.relative_to(base).as_posix()
            except ValueError:
                label = f.name
            if f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as archive:
                    for info in sorted(archive.infolist(), key=lambda i: i.filename):
                        if info.is_dir() or not info.filename.lower().endswith(".txt"):
                            continue
                        yield f"{label}:{info.filename}", _decode(archive.read(info))
            else:
                yield label, _decode(f.read_bytes())


# ------------------------------------------------------------------ load


@dataclass
class LoadResult:
    """What a load did: counts per (file, reason), invariant results, replay errors, timings."""

    funnel: Counter = field(default_factory=Counter)
    invariants: Optional[pd.DataFrame] = None
    replay_errors: list[tuple[str, str, str]] = field(default_factory=list)
    hands_loaded: int = 0
    seconds_parse_replay: float = 0.0
    seconds_insert: float = 0.0
    seconds_total: float = 0.0

    @property
    def violations(self) -> int:
        if self.invariants is None:
            return 0
        return int(self.invariants["violations"].sum())

    def reasons(self) -> Counter:
        out: Counter = Counter()
        for (_file, reason), k in self.funnel.items():
            out[reason] += k
        return out


def load(
    dataset: str,
    inputs: Sequence[str],
    *,
    config: Optional[Mapping[str, Any]] = None,
    hash_ids: bool = False,
    source: Optional[str] = None,
    batch_hands: Optional[int] = None,
    incremental: bool = False,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> LoadResult:
    """Load hand histories into the dataset's warehouse; see the module docstring."""
    cfg = dict(config or load_config())
    step = cfg["step01"]
    opts = parse_options(step)
    batch_size = int(batch_hands or step["batch_hands"])
    source = source or default_source(dataset)
    if hash_ids:
        salt = os.environ.get("SU_ID_SALT", "")
        if not salt:
            raise RuntimeError("--hash-ids needs the secret salt in the environment variable SU_ID_SALT")
        h = step["hash_ids"]
        ids = IdMapper(salt.encode("utf-8"), int(h["player_hex"]), int(h["table_hex"]), int(h["hand_hex"]))
    else:
        ids = IdMapper()
    want_classes = bool(step["hand_classes"])
    if want_classes and not handclass_available():
        log("note: sudetect.handclass is not importable; hand-class columns stay NULL")
    ctx = ReplayContext(source=source, ids=ids, hand_classes=want_classes)

    result = LoadResult()
    t_start = time.perf_counter()
    con = warehouse.connect(dataset)
    gc_was_enabled = gc.isenabled()
    gc.disable()  # rows are plain tuples without reference cycles; the cyclic collector only costs time
    try:
        if not incremental:
            warehouse.drop_tables(con, STEP_TABLES)
        warehouse.run_sql_file(con, SQL_DIR / "schema.sql")
        batch = Batch()
        seen: dict[int, int] = {}
        done_dates: set = set()
        files_seen: list[str] = []

        def flush() -> None:
            if not len(batch):
                return
            t0 = time.perf_counter()
            tables = batch.to_arrow()
            if incremental:
                new_dates = sorted(batch.dates() - done_dates)
                for d in new_dates:
                    for name in ("hands", "hand_players", "actions"):
                        warehouse.execute(con, f"DELETE FROM {name} WHERE hand_date = :d", d=d)
                done_dates.update(new_dates)
            for name in ("hands", "hand_players", "actions"):
                warehouse.insert_df(con, name, tables[name], mode="append")
            result.seconds_insert += time.perf_counter() - t0
            batch.clear()

        t_parse = 0.0
        stop = False
        for label, text in iter_sources(inputs, step["input_extensions"]):
            files_seen.append(label)
            t0 = time.perf_counter()
            for hand_text in split_hands(text):
                parsed = parse_hand(hand_text, opts)
                if isinstance(parsed, GateReject):
                    result.funnel[(label, parsed.reason)] += 1
                    continue
                key = hash(parsed.hand_id)
                crc = zlib.crc32(hand_text.encode("utf-8"))
                previous = seen.get(key)
                if previous is not None:
                    result.funnel[(label, "duplicate" if previous == crc else "duplicate_diff")] += 1
                    continue
                seen[key] = crc
                try:
                    rows = replay_hand(parsed, ctx)
                except Exception as exc:  # counted and reported; the exit code says so
                    result.funnel[(label, "replay_error")] += 1
                    result.replay_errors.append((label, parsed.hand_id, repr(exc)))
                    continue
                batch.add(rows)
                result.funnel[(label, "loaded")] += 1
                result.hands_loaded += 1
                if len(batch) >= batch_size:
                    t_parse += time.perf_counter() - t0
                    flush()
                    log(f"  {result.hands_loaded:,} hands loaded")
                    t0 = time.perf_counter()
                if limit is not None and result.hands_loaded >= limit:
                    stop = True
                    break
            t_parse += time.perf_counter() - t0
            if stop:
                break
        flush()
        result.seconds_parse_replay = t_parse

        funnel_df = pd.DataFrame(
            [(dataset, f, r, k) for (f, r), k in sorted(result.funnel.items())],
            columns=["dataset", "file", "reason", "n_hands"],
        )
        if incremental and files_seen:
            for f in files_seen:
                warehouse.execute(con, "DELETE FROM parse_funnel WHERE dataset = :ds AND file = :f", ds=dataset, f=f)
        if len(funnel_df):
            warehouse.insert_df(con, "parse_funnel", funnel_df, mode="append")
        warehouse.run_sql_file(con, SQL_DIR / "post_load.sql")
        warehouse.run_sql_file(con, SQL_DIR / "views.sql")
        result.invariants = warehouse.run_sql_file(con, SQL_DIR / "invariants.sql", tol=float(step["invariant_tol"]))
    finally:
        con.close()
        if gc_was_enabled:
            gc.enable()
    result.seconds_total = time.perf_counter() - t_start
    return result


def report(result: LoadResult, log: Callable[[str], None] = print) -> None:
    """Print the funnel, the speed and the invariant table."""
    reasons = result.reasons()
    total = sum(reasons.values())
    log(f"hands read: {total:,}; loaded: {reasons.get('loaded', 0):,}")
    for reason, k in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if reason != "loaded":
            log(f"  dropped {reason}: {k:,}")
    if result.seconds_parse_replay > 0:
        rate = result.hands_loaded / result.seconds_parse_replay
        log(f"parse + replay: {result.seconds_parse_replay:.1f} s ({rate:,.0f} hands/s); "
            f"insert: {result.seconds_insert:.1f} s; total: {result.seconds_total:.1f} s")
    for label, hand_id, err in result.replay_errors[:10]:
        log(f"  replay error in {label} hand {hand_id}: {err}")
    if result.invariants is not None:
        bad = result.invariants[result.invariants["violations"] > 0]
        if len(bad):
            log("invariant violations:")
            for row in bad.itertuples(index=False):
                log(f"  {row.check_name}: {row.violations}")
        else:
            log(f"invariants: all {len(result.invariants)} checks pass")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Load hand histories into hands / hand_players / actions.")
    ap.add_argument("--dataset", required=True, help="warehouse name, e.g. article_sample or mined")
    ap.add_argument("--input", required=True, action="append", help="folder, glob or file (repeatable)")
    ap.add_argument("--hash-ids", action="store_true", help="hash ids with the salt in env SU_ID_SALT")
    ap.add_argument("--batch-hands", type=int, default=None)
    ap.add_argument("--config", default=None, help="TOML file overriding config/defaults.toml")
    ap.add_argument("--source", choices=("operator", "mined", "sim", "fixture"), default=None)
    ap.add_argument("--incremental", action="store_true", help="replace only the days in the input")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many loaded hands")
    ap.add_argument("--allow-violations", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = load(args.dataset, args.input, config=load_config(args.config), hash_ids=args.hash_ids,
                      source=args.source, batch_hands=args.batch_hands, incremental=args.incremental,
                      limit=args.limit)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report(result)
    failed = result.violations > 0 or bool(result.replay_errors)
    return 1 if failed and not args.allow_violations else 0


if __name__ == "__main__":
    sys.exit(main())
