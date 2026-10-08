"""Play a scheduled dataset in parallel and write it as hand-history text.

:func:`simulate` builds the population and the schedule
(:mod:`sudetect.sim.population`), plays every work unit in a pool of
``spawn`` processes and writes one text file per unit (a day's share of the
tables) in full-information mode, so every seat's hole cards are in the
text. Next to the hands folder (never inside it) it writes:

* ``ground_truth.json``: the planted superusers, what they see and from
  when, their victims and how many hands they played. Detection code never
  reads it; the report uses it as the answer key.
* ``population.json``: every player's archetype and style parameters,
  activity and hand counts. Also never loaded into the warehouse.
* ``run_summary.json``: sizes (hands, files, tables, days) and the number
  of honest heads-up pairs with enough hands for the single-victim test.

Each hand's deck comes from a numpy Generator seeded with ``(seed,
hand_id)`` and each player's decisions from ``(seed, hand_id, player
index, stream tag)``, so every output file is byte-identical whatever the
number of processes; the HUD counts of the units are merged in file-name
order, so the calibration numbers are too. Stacks start at the player's
buy-in when he sits down and are topped up to it between hands when they
fall below ``top_up_below`` of it; a player who lost his whole stack buys
in again.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import re
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .calibrate import HudCounter
from .engine import HandRecord, SeatState, TableConfig, play_hand
from .hhtext import write_hand
from .policies import HonestPolicy, SuperuserPolicy
from .population import (
    Population,
    Schedule,
    SimConfig,
    TableSession,
    WorkUnit,
    build_population,
    build_schedule,
    load_sim_config,
)

__all__ = ["SimResult", "data_dir", "simulate", "play_unit", "make_agents", "table_config"]

_UNIT_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}_g\d{2,}\.txt$")  # g00 ... g99, g100 ...


def data_dir() -> Path:
    """``$SU_DATA_DIR``, or ``work/`` at the repository root.

    The same rule as :func:`sudetect.warehouse.data_dir`, without importing
    the database engine into the simulator.
    """
    env = os.environ.get("SU_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "work"


@dataclass
class SimResult:
    """What a run produced."""

    dataset: str
    out_dir: Optional[Path]
    n_hands: int
    n_files: int
    n_tables: int
    seconds: float
    procs: int
    counter: Optional[HudCounter]
    population: Population
    schedule: Schedule
    plant_stats: dict[str, dict[str, int]] = field(default_factory=dict)


def table_config(cfg: SimConfig, max_seats: int) -> TableConfig:
    """Stakes and house rules of a table with ``max_seats`` seats."""
    return TableConfig(
        max_seats=max_seats, sb=cfg.sb_units, bb=cfg.bb_units, ante=int(round(cfg.ante_bb * cfg.bb_units)),
        rake_pct=cfg.rake_pct, rake_cap=int(round(cfg.rake_cap_bb * cfg.bb_units)),
        no_flop_no_drop=cfg.no_flop_no_drop,
    )


def make_agents(cfg: SimConfig, pop: Population, seed: int) -> dict[int, HonestPolicy]:
    """One policy object per player index (plants get :class:`SuperuserPolicy`)."""
    reads = {p.name: p.archetype for p in pop.players}
    plant_u = {pl.player: pl.spec.u for pl in pop.plants.values()}
    agents: dict[int, HonestPolicy] = {}
    for p in pop.players:
        if p.index in plant_u:
            agents[p.index] = SuperuserPolicy(p.name, p.index, p.archetype, p.style, cfg.model, seed, reads,
                                              u=plant_u[p.index])
        else:
            agents[p.index] = HonestPolicy(p.name, p.index, p.archetype, p.style, cfg.model, seed, reads)
    return agents


# --------------------------------------------------------------------------- one unit


def play_unit(cfg: SimConfig, pop: Population, unit: WorkUnit, agents: Mapping[int, HonestPolicy], seed: int,
              *, out_path: Optional[Path] = None, counter: Optional[HudCounter] = None,
              keep_records: bool = False) -> tuple[int, list[HandRecord]]:
    """Play every session of a work unit; optionally write its text and count HUD stats.

    Returns the number of hands played and, with ``keep_records``, the records.
    """
    tables = {2: table_config(cfg, 2), cfg.ring_seats: table_config(cfg, cfg.ring_seats)}
    names = [p.name for p in pop.players]
    buy_in = [p.buy_in_bb * cfg.bb_units for p in pop.players]
    kept: list[HandRecord] = []
    handle = open(out_path, "w", encoding="utf-8", newline="\n") if out_path is not None else None
    n_written = 0
    try:
        for sess in unit.sessions:
            grants = pop.grants_for_day(sess.day)
            for rec in _play_session(cfg, sess, tables[sess.max_seats], names, buy_in, agents, grants, seed):
                if handle is not None:
                    if n_written:
                        handle.write("\n")
                    handle.write(write_hand(rec, full_information=True))
                if counter is not None:
                    counter.add(rec)
                if keep_records:
                    kept.append(rec)
                n_written += 1
    finally:
        if handle is not None:
            handle.close()
    return n_written, kept


def _play_session(cfg: SimConfig, sess: TableSession, table: TableConfig, names: list[str], buy_in: list[int],
                  agents: Mapping[int, HonestPolicy], grants: Mapping[int, Any], seed: int):
    """Yield the records of one table session in order."""
    sch = cfg.schedule
    gaps = sch.hu_hand_seconds if sess.max_seats == 2 else sch.ring_hand_seconds
    clock = cfg.epoch + timedelta(days=sess.day, seconds=sess.start_s)
    time_rng = np.random.default_rng([seed, sess.table_no, 3])
    stacks: dict[int, int] = {}
    stints = sorted(sess.stints, key=lambda x: x[2])
    button: Optional[int] = None
    top_up = cfg.top_up_below
    for k in range(sess.n_hands):
        seated = sorted((seat, p) for (seat, p, a, b) in stints if a <= k < b)
        here = {p for _, p in seated}
        for p in list(stacks):
            if p not in here:
                del stacks[p]  # left the table: a later stay starts with a fresh buy-in
        if len(seated) < 2:
            continue
        seats = []
        for seat, p in seated:
            st = stacks.get(p)
            if st is None or st <= 0 or st < top_up * buy_in[p]:  # a busted player rebuys even without top-ups
                st = buy_in[p] if st is None else max(st, buy_in[p])
            stacks[p] = st
            seats.append(SeatState(seat, names[p], st))
        occupied = [s for s, _ in seated]
        if button is None:
            button = occupied[int(time_rng.integers(len(occupied)))]
        else:
            later = [s for s in occupied if s > button]
            button = later[0] if later else occupied[0]
        info: dict[str, Any] = {}
        for _, p in seated:
            g = grants.get(p)
            if g is None:
                continue
            if g == "all":
                info[names[p]] = "all"
            elif g in here:
                info[names[p]] = {names[g]}
        hand_id = sess.hand_id0 + k
        rng = np.random.default_rng([seed, hand_id])
        rec = play_hand(table, seats, button, {names[p]: agents[p] for _, p in seated}, hand_id,
                        sess.table_name, clock, rng, info)
        for seat, p in seated:
            stacks[p] = rec.final_stacks[seat]
        n_dec = sum(1 for a in rec.actions if not a.kind.startswith("post"))
        clock = clock + timedelta(seconds=round(gaps[0] + gaps[1] * n_dec + time_rng.exponential(gaps[2])))
        yield rec


# --------------------------------------------------------------------------- workers

_WORKER: dict[str, Any] = {}


def _init_worker(payload: bytes) -> None:
    state = pickle.loads(payload)
    _WORKER.update(state)
    _WORKER["agents"] = make_agents(state["cfg"], state["pop"], state["seed"])


def _run_unit(unit: WorkUnit) -> dict[str, Any]:
    w = _WORKER
    t0 = time.perf_counter()
    counter = HudCounter() if w["calibrate"] else None
    out_path = (w["out_dir"] / unit.file_name) if w["out_dir"] is not None else None
    n, _ = play_unit(w["cfg"], w["pop"], unit, w["agents"], w["seed"], out_path=out_path, counter=counter)
    plants = {}
    for pl in w["pop"].plants.values():
        a = w["agents"][pl.player]
        plants[pl.spec.plant_id] = (a.eligible_decisions, a.informed_decisions)
        a.eligible_decisions = a.informed_decisions = 0
    return {"file": unit.file_name, "hands": n, "seconds": time.perf_counter() - t0,
            "counter": counter, "plants": plants}


# --------------------------------------------------------------------------- driver


def simulate(cfg: Optional[SimConfig] = None, dataset: str = "sim_small", *, out_dir: Optional[Path] = None,
             procs: int = 1, seed: Optional[int] = None, calibrate: bool = False, write: bool = True,
             target_hands: Optional[int] = None, n_days: Optional[int] = None,
             log: Optional[Any] = None) -> SimResult:
    """Simulate one dataset.

    Args:
        cfg: Simulator configuration (default: ``config/sim.toml``).
        dataset: Name of a ``[sim.datasets.*]`` entry.
        out_dir: Folder for the hand files (default ``$SU_DATA_DIR/<dataset>/hands``);
            the JSON files go to its parent (after resolving the path, so a
            relative ``.`` works too; a filesystem root is refused).
        procs: Worker processes (1 plays in this process).
        seed: Overrides the configured seed.
        calibrate: Count HUD statistics while playing.
        write: Write the hand files and JSON files (False for quick calibration runs).
        target_hands, n_days: Override the dataset's size.
        log: Callable taking a string, for progress lines.
    """
    cfg = cfg or load_sim_config()
    seed = cfg.seed if seed is None else int(seed)
    say = log or (lambda s: None)
    t0 = time.perf_counter()
    pop = build_population(cfg, dataset, seed, n_days=n_days)
    sched = build_schedule(cfg, pop, target_hands=target_hands, n_days=n_days)
    say(f"population {len(pop.players)} players ({len(pop.plants)} planted); schedule {sched.n_hands:,} hands "
        f"on {sched.n_tables:,} tables in {len(sched.units)} files ({time.perf_counter() - t0:.1f}s)")

    if write:
        out_dir = (Path(out_dir) if out_dir is not None else data_dir() / dataset / "hands").resolve()
        if out_dir.parent == out_dir:
            raise ValueError(f"{out_dir} has no parent folder for the JSON files (the answer key stays out of it)")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            if f.is_file() and _UNIT_FILE.match(f.name):
                f.unlink()
    else:
        out_dir = None

    counter = HudCounter() if calibrate else None
    plant_stats = {pid: {"eligible_decisions": 0, "informed_decisions": 0} for pid in pop.plants}
    units = sorted(sched.units, key=lambda u: -u.n_hands)
    done = 0
    results: dict[str, dict[str, Any]] = {}
    if procs <= 1:
        _init_worker(pickle.dumps({"cfg": cfg, "pop": pop, "seed": seed, "out_dir": out_dir, "calibrate": calibrate}))
        for u in units:
            res = _run_unit(u)
            results[res["file"]] = res
            done += int(res["hands"])
    else:
        payload = pickle.dumps({"cfg": cfg, "pop": pop, "seed": seed, "out_dir": out_dir, "calibrate": calibrate})
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=procs, mp_context=ctx, initializer=_init_worker,
                                 initargs=(payload,)) as ex:
            futures = [ex.submit(_run_unit, u) for u in units]
            last = time.perf_counter()
            for fut in as_completed(futures):
                res = fut.result()
                results[res["file"]] = res
                done += int(res["hands"])
                if time.perf_counter() - last > 15:
                    last = time.perf_counter()
                    say(f"  {done:,} / {sched.n_hands:,} hands, {time.perf_counter() - t0:.0f}s")
    for name in sorted(results):  # a fixed merge order: the float sums do not depend on the process count
        _collect(results[name], counter, plant_stats)
    seconds = time.perf_counter() - t0
    say(f"played {done:,} hands in {seconds:.1f}s with {procs} process(es)")

    result = SimResult(dataset=dataset, out_dir=out_dir, n_hands=done, n_files=len(units) if write else 0,
                       n_tables=sched.n_tables, seconds=seconds, procs=procs, counter=counter, population=pop,
                       schedule=sched, plant_stats=plant_stats)
    if write and out_dir is not None:
        write_metadata(cfg, result, seed)
    return result


def _collect(res: dict[str, Any], counter: Optional[HudCounter], plant_stats: dict[str, dict[str, int]]) -> int:
    if counter is not None and res["counter"] is not None:
        counter.merge(res["counter"])
    for pid, (eligible, informed) in res["plants"].items():
        plant_stats[pid]["eligible_decisions"] += eligible
        plant_stats[pid]["informed_decisions"] += informed
    return int(res["hands"])


# --------------------------------------------------------------------------- metadata


def player_hands(sched: Schedule, n_players: int) -> dict[str, np.ndarray]:
    """Hands dealt per player index at heads-up and 6-max tables (from the schedule)."""
    hu = np.zeros(n_players, dtype=np.int64)
    ring = np.zeros(n_players, dtype=np.int64)
    for s in sched.sessions():
        target = hu if s.max_seats == 2 else ring
        for (_, p, a, b) in s.stints:
            target[p] += b - a
    return {"hu": hu, "ring": ring}


def honest_pairs(sched: Schedule, pop: Population, min_hands: int) -> int:
    """Pairs of honest players with at least ``min_hands`` heads-up hands together."""
    plants = {pl.player for pl in pop.plants.values()}
    return sum(1 for (i, j), h in sched.hu_pair_hands.items() if h >= min_hands and i not in plants and j not in plants)


def write_metadata(cfg: SimConfig, result: SimResult, seed: int) -> None:
    """``ground_truth.json``, ``population.json`` and ``run_summary.json`` next to the hands folder."""
    assert result.out_dir is not None
    root = result.out_dir.parent
    pop, sched = result.population, result.schedule
    hands = player_hands(sched, len(pop.players))
    names = [p.name for p in pop.players]

    plants = []
    for pid, pl in pop.plants.items():
        victim_hu = 0
        if pl.victim is not None:
            key = (min(pl.player, pl.victim), max(pl.player, pl.victim))
            victim_hu = sched.hu_pair_hands.get(key, 0)
        plants.append({
            "plant_id": pid,
            "player": names[pl.player],
            "kind": pl.spec.kind,
            "sees": "every opponent" if pl.spec.kind == "full_table" else [names[pl.victim]],
            "u": pl.spec.u,
            "from_day": pl.from_day,
            "from_date": (cfg.epoch + timedelta(days=pl.from_day)).strftime("%Y-%m-%d"),
            "victim_role": pl.spec.victim,
            "victim": names[pl.victim] if pl.victim is not None else None,
            "victim_share_target": pl.spec.victim_share,
            "honest_style": pop.players[pl.player].archetype,
            "hands": int(hands["hu"][pl.player] + hands["ring"][pl.player]),
            "hu_hands": int(hands["hu"][pl.player]),
            "hu_hands_vs_victim": int(victim_hu),
            **result.plant_stats.get(pid, {}),
        })
    truth = {
        "about": "Answer key of the simulated dataset. Detection code never reads this file.",
        "dataset": result.dataset,
        "seed": seed,
        "epoch": cfg.epoch.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preflop_information": cfg.model.superuser.preflop_info,
        "plants": plants,
        "victims": {role: names[i] for role, i in pop.victims.items()},
    }
    _dump(root / "ground_truth.json", truth)

    players = []
    for p in pop.players:
        players.append({
            "name": p.name,
            "archetype": p.archetype,
            "planted": p.plant_id is not None,
            "activity": round(p.activity, 4),
            "hu_pref": round(p.hu_pref, 4),
            "buy_in_bb": p.buy_in_bb,
            "stickiness": round(p.stickiness, 4),
            "rivals": [names[j] for j in p.rivals],
            "hands_hu": int(hands["hu"][p.index]),
            "hands_ring": int(hands["ring"][p.index]),
            "style": {k: _rounded(v) for k, v in p.style.to_dict().items()},
        })
    _dump(root / "population.json", {
        "about": "Style parameters of every simulated player. Never loaded into the warehouse.",
        "dataset": result.dataset, "seed": seed, "archetype_mix": cfg.mix, "players": players,
    })

    min_pair = int(cfg.raw.get("calibration", {}).get("pair_min_hands", 500))
    summary = {
        "dataset": result.dataset,
        "seed": seed,
        "hands": result.n_hands,
        "files": result.n_files,
        "tables": result.n_tables,
        "hu_hands": int(sum(s.n_hands for s in sched.sessions() if s.max_seats == 2)),
        "ring_hands": int(sum(s.n_hands for s in sched.sessions() if s.max_seats != 2)),
        "days": max(u.day for u in sched.units) + 1 if sched.units else 0,
        f"honest_pairs_with_{min_pair}_hu_hands": honest_pairs(sched, pop, min_pair),
    }
    _dump(root / "run_summary.json", summary)


def _rounded(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, list):
        return [_rounded(x) for x in v]
    return v


def _dump(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
