"""Who plays, when and where: the simulated population and its table schedule.

A dataset is built in three steps, all deterministic for a given seed:

1. :func:`load_sim_config` reads the ``[sim]`` section of
   ``config/sim.toml``: dataset sizes, house rules, the style archetypes
   and their mix, the shared policy model and the planted superusers.
2. :func:`build_population` creates the players. Names are ``P_`` plus six
   hex digits of a hash of ``(seed, index)``. Honest players get an archetype
   from the configured mix and their own jittered copy of its parameters,
   an activity level (mean sessions per day), a preference for heads-up
   tables, a buy-in and two or so preferred heads-up opponents ("rivals").
   Planted players play a strong regular's style; their activity is a fixed
   quantile of the same activity distribution (a typical volume, so volume
   alone does not give them away) and their heads-up preference can be set.
   Single-victim plants get no rivals and their victims are ordinary TAG or
   LAG regulars.
3. :func:`build_schedule` lays out every table session of every day:
   heads-up matches (two players for the whole match) and 6-max tables where
   players come and go. Each player starts a Poisson number of heads-up
   matches per day; the opponent is one of the player's rivals with the
   player's stickiness, otherwise anyone (weighted by activity). A
   single-victim plant starts all of its own matches (nobody else picks it)
   and plays its victim in the configured share of its heads-up hands. The
   number of hands per day follows ``target_hands``, ``hu_hand_share`` of
   them at heads-up tables. Hand ids run consecutively over the whole
   dataset; table names are ``T0001``, ``T0002``, ... Sessions are grouped
   into work units of about ``unit_hands`` hands, one output file each,
   which the runner plays in parallel.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .policies import ARCHETYPES, PolicyModel, PostflopModel, PreflopModel, Style, SuperuserRules

__all__ = [
    "DEFAULT_CONFIG",
    "PlantSpec",
    "DatasetSpec",
    "ScheduleParams",
    "SimConfig",
    "Player",
    "Plant",
    "Population",
    "TableSession",
    "WorkUnit",
    "Schedule",
    "load_sim_config",
    "player_name",
    "build_population",
    "build_schedule",
]

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "sim.toml"
"""``config/sim.toml`` at the repository root."""

_DAY = 86_400.0


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class PlantSpec:
    """One planted superuser as configured.

    Attributes:
        plant_id: Label such as ``SU_FULL_100`` (ground truth only, never a name at a table).
        kind: ``full_table`` (sees every opponent) or ``single_victim``.
        u: Share of the eligible postflop decisions played with the information.
        from_day: First day with information (``from_day_frac`` of the dataset's days when given).
        victim: Victim role (``V1``, ``V2``, ...) for a single-victim plant.
        victim_share: Share of the plant's heads-up hands played against the victim.
        from_day_frac: When given, ``from_day`` is this share of the dataset's days.
        activity_quantile: The plant's activity is this quantile of the activity
            distribution every player is drawn from (a typical volume, so
            volume alone does not single the plant out).
        hu_pref: The plant's share of sessions started heads-up (None: drawn like everybody's).
    """

    plant_id: str
    kind: str
    u: float
    from_day: int
    victim: Optional[str] = None
    victim_share: Optional[float] = None
    from_day_frac: Optional[float] = None
    activity_quantile: Optional[float] = None
    hu_pref: Optional[float] = None


@dataclass(frozen=True)
class DatasetSpec:
    """Size of one named dataset."""

    name: str
    n_players: int
    n_days: int
    target_hands: int
    plants: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleParams:
    """How sessions are drawn (see ``[sim.schedule]`` in the config)."""

    activity_sigma: float
    hu_pref_mean: float
    hu_pref_concentration: float
    rivals_per_player: int
    stickiness_mean: float
    stickiness_concentration: float
    hu_match_hands_mean: float
    hu_match_hands_sigma: float
    hu_match_hands_min: int
    ring_table_hands: tuple[int, int]
    ring_stay_hands_mean: float
    ring_refill_hands_mean: float
    ring_min_players: int
    hu_hand_seconds: tuple[float, float, float]
    ring_hand_seconds: tuple[float, float, float]


@dataclass(frozen=True)
class SimConfig:
    """Everything the simulator reads from ``[sim]``."""

    seed: int
    epoch: datetime
    hand_id_base: int
    sb_units: int
    bb_units: int
    ante_bb: float
    rake_pct: float
    rake_cap_bb: float
    no_flop_no_drop: bool
    hu_hand_share: float
    ring_seats: int
    buy_in_mix: tuple[tuple[int, float], ...]
    top_up_below: float
    unit_hands: int
    schedule: ScheduleParams
    mix: dict[str, float]
    archetypes: dict[str, dict[str, Any]]
    jitter: dict[str, float]
    model: PolicyModel
    plants: dict[str, PlantSpec]
    datasets: dict[str, DatasetSpec]
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    def dataset(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            raise KeyError(f"unknown dataset {name!r}; configured: {sorted(self.datasets)}")
        return self.datasets[name]


def _tuple(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_tuple(v) for v in x)
    return x


def load_sim_config(path: str | Path | None = None, overrides: Optional[Mapping[str, Any]] = None) -> SimConfig:
    """Read the ``[sim]`` section of a TOML file (default: ``config/sim.toml``).

    ``overrides`` is merged into the section first (nested mappings are
    merged key by key), which is how tests and the CLI change single values.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    if "sim" not in doc:
        raise KeyError(f"{path} has no [sim] section")
    sim = _merge(doc["sim"], overrides or {})
    sch = sim["schedule"]
    schedule = ScheduleParams(**{k: _tuple(sch[k]) for k in ScheduleParams.__dataclass_fields__})
    model = PolicyModel(
        preflop=PreflopModel.from_mapping(sim["preflop"]),
        postflop=PostflopModel.from_mapping(sim["postflop"]),
        superuser=SuperuserRules.from_mapping(sim["superuser"]),
        exploit={k: tuple(v) for k, v in sim["exploit"].items()},
    )
    plants = {}
    for pid, p in sim["plants"].items():
        plants[pid] = PlantSpec(plant_id=pid, kind=p["kind"], u=float(p["u"]), from_day=int(p.get("from_day", 0)),
                                victim=p.get("victim"), victim_share=p.get("victim_share"),
                                from_day_frac=p.get("from_day_frac"), activity_quantile=p.get("activity_quantile"),
                                hu_pref=p.get("hu_pref"))
    datasets = {}
    for name, d in sim["datasets"].items():
        datasets[name] = DatasetSpec(name=name, n_players=int(d["n_players"]), n_days=int(d["n_days"]),
                                     target_hands=int(d["target_hands"]), plants=tuple(d["plants"]))
    archetypes = {k: dict(v) for k, v in sim["archetypes"].items()}
    for a in ARCHETYPES:
        if a not in archetypes:
            raise KeyError(f"[sim.archetypes.{a}] is missing")
        Style.from_mapping(archetypes[a])  # validates the field set
    epoch = datetime.fromisoformat(str(sim["epoch"]).replace("Z", "+00:00"))
    return SimConfig(
        seed=int(sim["seed"]), epoch=epoch, hand_id_base=int(sim["hand_id_base"]),
        sb_units=int(sim["sb_units"]), bb_units=int(sim["bb_units"]), ante_bb=float(sim["ante_bb"]),
        rake_pct=float(sim["rake_pct"]), rake_cap_bb=float(sim["rake_cap_bb"]),
        no_flop_no_drop=bool(sim["no_flop_no_drop"]), hu_hand_share=float(sim["hu_hand_share"]),
        ring_seats=int(sim["ring_seats"]), buy_in_mix=tuple((int(a), float(b)) for a, b in sim["buy_in_mix"]),
        top_up_below=float(sim["top_up_below"]), unit_hands=int(sim["unit_hands"]),
        schedule=schedule, mix={k: float(v) for k, v in sim["mix"].items()}, archetypes=archetypes,
        jitter={k: float(v) for k, v in sim["jitter"].items()}, model=model, plants=plants,
        datasets=datasets, raw=sim,
    )


def _merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- population


@dataclass(frozen=True)
class Player:
    """One simulated player.

    Attributes:
        index: Position in the population (seeds depend on it).
        name: ``P_`` + 6 hex digits.
        archetype: Style class; planted players use their honest style's class.
        style: The player's own (jittered) style parameters.
        activity: Mean number of sessions per day.
        hu_pref: Share of the player's sessions started at heads-up tables.
        buy_in_bb: Buy-in and top-up target in big blinds.
        rivals: Indices of the preferred heads-up opponents.
        stickiness: Probability that a heads-up match the player starts is against a rival.
        plant_id: Plant label for a planted superuser, else None.
    """

    index: int
    name: str
    archetype: str
    style: Style
    activity: float
    hu_pref: float
    buy_in_bb: int
    rivals: tuple[int, ...]
    stickiness: float
    plant_id: Optional[str] = None


@dataclass(frozen=True)
class Plant:
    """A planted superuser placed in the population."""

    spec: PlantSpec
    player: int
    from_day: int
    victim: Optional[int] = None


@dataclass
class Population:
    """The players of one dataset plus the ground truth of the plants.

    ``n_days`` is the number of days the plants' onset days were computed
    for (None: the dataset's configured days).
    """

    dataset: str
    seed: int
    players: list[Player]
    plants: dict[str, Plant]
    victims: dict[str, int]
    n_days: Optional[int] = None

    def by_name(self) -> dict[str, Player]:
        return {p.name: p for p in self.players}

    def honest(self) -> list[Player]:
        return [p for p in self.players if p.plant_id is None]

    def grants_for_day(self, day: int) -> dict[int, Any]:
        """Player index -> ``"all"`` or a victim's index, for plants informed on ``day``."""
        out: dict[int, Any] = {}
        for plant in self.plants.values():
            if day < plant.from_day:
                continue
            out[plant.player] = "all" if plant.spec.kind == "full_table" else plant.victim
        return out


def player_name(seed: int, index: int, salt: int = 0) -> str:
    """``P_`` + six hex digits of SHA-256 of ``"seed:index[:salt]"``."""
    text = f"{seed}:{index}" + (f":{salt}" if salt else "")
    return "P_" + hashlib.sha256(text.encode()).hexdigest()[:6]


def _jitter_style(base: Mapping[str, Any], jitter: Mapping[str, float], rng: np.random.Generator) -> Style:
    """A player's copy of an archetype: relative jitter on rates and widths,
    absolute jitter on margins and shifts (keys listed in ``jitter``)."""
    values = dict(base)
    rel = jitter.get("relative", 0.0)
    for key, v in base.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        abs_sd = jitter.get(key)
        if abs_sd is not None:
            values[key] = float(v) + rng.normal(0.0, abs_sd)
        elif rel > 0:
            values[key] = float(v) * math.exp(rng.normal(0.0, rel))
    for key in ("cbet", "barrel", "stab", "bluff", "semibluff", "raise_prob", "slowplay", "polar", "fold_3bet"):
        values[key] = min(0.98, max(0.0, values[key]))
    for key in ("hu_open", "hu_limp", "hu_defend", "hu_3bet", "hu_iso"):
        values[key] = min(1.0, max(0.0, values[key]))
    for key in ("noise", "pf_noise"):  # standard deviations: an absolute jitter must not make them negative
        values[key] = max(0.0, values[key])
    return Style.from_mapping(values)


def _normal_quantile(q: float) -> float:
    """Inverse of the standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if q < 0.02425:
        t = math.sqrt(-2 * math.log(q))
        num = ((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]
        return num / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    if q > 1 - 0.02425:
        return -_normal_quantile(1 - q)
    t = q - 0.5
    r = t * t
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t
    return num / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _counts(shares: Mapping[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of ``n`` by ``shares``."""
    total = sum(shares.values())
    raw = {k: n * v / total for k, v in shares.items()}
    out = {k: int(math.floor(x)) for k, x in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:rest]:
        out[k] += 1
    return out


def build_population(cfg: SimConfig, dataset: str, seed: Optional[int] = None,
                     n_days: Optional[int] = None) -> Population:
    """Create the players of ``dataset`` (plants included in ``n_players``).

    ``n_days`` overrides the dataset's number of days; a plant's
    ``from_day_frac`` is taken of it.
    """
    spec = cfg.dataset(dataset)
    seed = cfg.seed if seed is None else int(seed)
    days = spec.n_days if n_days is None else int(n_days)
    rng = np.random.default_rng([seed, 101])
    n = spec.n_players
    plant_ids = list(spec.plants)
    if len(plant_ids) + 10 > n:
        raise ValueError("too few players for the plants")

    names: list[str] = []
    seen: set[str] = set()
    for i in range(n):
        salt = 0
        name = player_name(seed, i)
        while name in seen:
            salt += 1
            name = player_name(seed, i, salt)
        seen.add(name)
        names.append(name)

    plant_idx = [int(x) for x in rng.choice(n, size=len(plant_ids), replace=False)]
    honest_idx = [i for i in range(n) if i not in set(plant_idx)]
    counts = _counts(cfg.mix, len(honest_idx))
    labels = [a for a in ARCHETYPES for _ in range(counts.get(a, 0))]
    rng.shuffle(labels)
    archetype = {i: a for i, a in zip(honest_idx, labels)}
    for i in plant_idx:
        archetype[i] = "strong"

    sch = cfg.schedule
    sigma = sch.activity_sigma
    activity = np.exp(rng.normal(-sigma * sigma / 2, sigma, size=n))
    a, b = sch.hu_pref_mean * sch.hu_pref_concentration, (1 - sch.hu_pref_mean) * sch.hu_pref_concentration
    hu_pref = rng.beta(a, b, size=n)
    stick_a = sch.stickiness_mean * sch.stickiness_concentration
    stick_b = (1 - sch.stickiness_mean) * sch.stickiness_concentration
    stickiness = rng.beta(stick_a, stick_b, size=n)
    for pid, i in zip(plant_ids, plant_idx):
        p = cfg.plants[pid]
        if p.activity_quantile is not None:  # lognormal quantile of the same distribution
            z = _normal_quantile(float(p.activity_quantile))
            activity[i] = math.exp(-sigma * sigma / 2 + sigma * z)
        if p.hu_pref is not None:
            hu_pref[i] = float(p.hu_pref)
    buy_ins, shares = zip(*cfg.buy_in_mix)
    buy_in = rng.choice(np.asarray(buy_ins), size=n, p=np.asarray(shares) / sum(shares))
    styles = {i: _jitter_style(cfg.archetypes[archetype[i]], cfg.jitter, rng) for i in range(n)}

    # victims: ordinary regulars (TAG or LAG) who are not plants
    plants_cfg = [cfg.plants[p] for p in plant_ids]
    roles = sorted({p.victim for p in plants_cfg if p.kind == "single_victim" and p.victim})
    pool = [i for i in honest_idx if archetype[i] == "tag"] + [i for i in honest_idx if archetype[i] == "lag"]
    victims: dict[str, int] = {}
    for role in roles:
        choice = pool[int(rng.integers(len(pool)))]
        pool.remove(choice)
        victims[role] = choice

    # rival graph among everyone but single-victim plants
    single = {plant_idx[k] for k, p in enumerate(plants_cfg) if p.kind == "single_victim"}
    eligible = [i for i in range(n) if i not in single]
    rivals: dict[int, list[int]] = {i: [] for i in range(n)}
    k = sch.rivals_per_player
    order = list(eligible)
    rng.shuffle(order)
    for i in order:
        tries = 0
        while len(rivals[i]) < k and tries < 50:
            tries += 1
            j = eligible[int(rng.integers(len(eligible)))]
            if j == i or j in rivals[i] or len(rivals[j]) >= k + 1:
                continue
            rivals[i].append(j)
            rivals[j].append(i)

    plants: dict[str, Plant] = {}
    plant_of = {}
    for pid, i in zip(plant_ids, plant_idx):
        p = cfg.plants[pid]
        from_day = p.from_day
        if p.from_day_frac is not None:
            from_day = int(round(p.from_day_frac * days))
        plants[pid] = Plant(spec=p, player=i, from_day=from_day,
                            victim=victims.get(p.victim) if p.kind == "single_victim" else None)
        plant_of[i] = pid

    players = [
        Player(index=i, name=names[i], archetype=archetype[i], style=styles[i], activity=float(activity[i]),
               hu_pref=float(hu_pref[i]), buy_in_bb=int(buy_in[i]), rivals=tuple(rivals[i]),
               stickiness=float(stickiness[i]), plant_id=plant_of.get(i))
        for i in range(n)
    ]
    return Population(dataset=dataset, seed=seed, players=players, plants=plants, victims=victims,
                      n_days=None if n_days is None else days)


# --------------------------------------------------------------------------- schedule


@dataclass(frozen=True)
class TableSession:
    """One heads-up match or one 6-max table session.

    ``stints`` lists ``(seat, player index, first hand, end hand)`` with hand
    offsets relative to the session (end exclusive); a seat may hold several
    players one after another.
    """

    table_no: int
    table_name: str
    day: int
    max_seats: int
    start_s: float
    n_hands: int
    hand_id0: int
    stints: tuple[tuple[int, int, int, int], ...]


@dataclass
class WorkUnit:
    """Sessions written to one output file by one worker."""

    day: int
    group: int
    file_name: str
    sessions: list[TableSession]

    @property
    def n_hands(self) -> int:
        return sum(s.n_hands for s in self.sessions)


@dataclass
class Schedule:
    units: list[WorkUnit]
    n_hands: int
    n_tables: int
    hu_pair_hands: dict[tuple[int, int], int]

    def sessions(self) -> list[TableSession]:
        return [s for u in self.units for s in u.sessions]


def build_schedule(cfg: SimConfig, pop: Population, *, target_hands: Optional[int] = None,
                   n_days: Optional[int] = None) -> Schedule:
    """Lay out every table session of the dataset (deterministic for the population's seed).

    The number of days is ``n_days``, else the population's, else the dataset's.
    """
    spec = cfg.dataset(pop.dataset)
    days = n_days if n_days is not None else (pop.n_days if pop.n_days is not None else spec.n_days)
    target = target_hands if target_hands is not None else spec.target_hands
    sch = cfg.schedule
    rng = np.random.default_rng([pop.seed, 202])
    players = pop.players
    n = len(players)
    activity = np.array([p.activity for p in players])
    hu_pref = np.array([p.hu_pref for p in players])

    single = {pl.player: pl for pl in pop.plants.values() if pl.spec.kind == "single_victim"}
    choosable = np.array([i not in single for i in range(n)])

    # heads-up: per-player Poisson starts scaled to the target number of hands
    hu_hands_day = target * cfg.hu_hand_share / days
    start_rate = activity * hu_pref
    start_rate = np.where(choosable, start_rate, 2.0 * start_rate)  # never picked by others
    matches_day = hu_hands_day / sch.hu_match_hands_mean
    start_rate = start_rate * matches_day / start_rate.sum()
    pick_w = np.where(choosable, activity * hu_pref, 0.0)

    ring_hands_day = target * (1.0 - cfg.hu_hand_share) / days
    lo, hi = sch.ring_table_hands
    ring_tables_day = ring_hands_day / ((lo + hi) / 2.0)
    ring_rate = activity * (1.0 - hu_pref)

    raw: list[tuple[int, float, int, tuple[tuple[int, int, int, int], ...]]] = []  # day, start, seats, stints
    tally = {i: [0, 0] for i in single}  # single-victim plant -> [hands vs the victim, heads-up hands]
    pair_hands: dict[tuple[int, int], int] = {}
    mu = math.log(sch.hu_match_hands_mean) - sch.hu_match_hands_sigma ** 2 / 2

    carry = 0.5
    for day in range(days):
        day_raw: list[tuple[float, int, int, tuple]] = []
        starts = rng.poisson(start_rate)
        for i in np.flatnonzero(starts):
            for _ in range(int(starts[i])):
                length = max(sch.hu_match_hands_min, int(round(math.exp(rng.normal(mu, sch.hu_match_hands_sigma)))))
                j = _pick_opponent(int(i), length, players, single, pick_w, rng, tally)
                # rough duration (four decisions a hand, 30% slack) keeps the match inside its day
                dur = length * (sch.hu_hand_seconds[0] + 4 * sch.hu_hand_seconds[1] + sch.hu_hand_seconds[2]) * 1.3
                start = float(rng.uniform(0.0, max(1.0, _DAY - dur)))
                seats = (1, 2) if rng.random() < 0.5 else (2, 1)
                stints = ((seats[0], int(i), 0, length), (seats[1], j, 0, length))
                day_raw.append((start, 2, length, stints))
                key = (min(int(i), j), max(int(i), j))
                pair_hands[key] = pair_hands.get(key, 0) + length
        # 6-max tables with players coming and going; today's weights are Poisson session counts
        stays_day = ring_tables_day * cfg.ring_seats * ((lo + hi) / 2.0) / sch.ring_stay_hands_mean
        ring_w = rng.poisson(ring_rate * stays_day / ring_rate.sum()).astype(float)
        if np.count_nonzero(ring_w) < cfg.ring_seats + 2:
            ring_w = ring_rate.copy()
        carry += ring_tables_day
        n_ring = int(carry)
        carry -= n_ring
        for _ in range(n_ring):
            length = int(rng.integers(lo, hi + 1))
            stints = _ring_stints(length, cfg.ring_seats, ring_w, sch, rng)
            if not stints:
                continue
            dur = length * (sch.ring_hand_seconds[0] + 8 * sch.ring_hand_seconds[1] + sch.ring_hand_seconds[2]) * 1.3
            start = float(rng.uniform(0.0, max(1.0, _DAY - dur)))
            day_raw.append((start, cfg.ring_seats, length, stints))
        day_raw.sort(key=lambda r: r[0])
        raw.extend((day, *r) for r in day_raw)

    width = max(4, len(str(len(raw))))
    sessions: list[TableSession] = []
    hand_id = cfg.hand_id_base + 1
    for k, (day, start, seats, length, stints) in enumerate(raw):
        sessions.append(TableSession(table_no=k + 1, table_name=f"T{k + 1:0{width}d}", day=day, max_seats=seats,
                                     start_s=start, n_hands=length, hand_id0=hand_id, stints=stints))
        hand_id += length

    units: list[WorkUnit] = []
    for day in range(days):
        day_sessions = [s for s in sessions if s.day == day]
        if not day_sessions:
            continue
        hands = sum(s.n_hands for s in day_sessions)
        groups = max(1, int(round(hands / cfg.unit_hands)))
        loads = [0] * groups
        members: list[list[TableSession]] = [[] for _ in range(groups)]
        for s in sorted(day_sessions, key=lambda s: (-s.n_hands * s.max_seats, s.table_no)):
            g = loads.index(min(loads))
            members[g].append(s)
            loads[g] += s.n_hands * (1 if s.max_seats == 2 else 3)
        date = (cfg.epoch + timedelta(days=day)).strftime("%Y-%m-%d")
        for g, ms in enumerate(members):
            if ms:
                ms.sort(key=lambda s: s.table_no)
                units.append(WorkUnit(day=day, group=g, file_name=f"{date}_g{g:02d}.txt", sessions=ms))
    total = sum(s.n_hands for s in sessions)
    return Schedule(units=units, n_hands=total, n_tables=len(sessions), hu_pair_hands=pair_hands)


def _pick_opponent(i: int, length: int, players: Sequence[Player], single: Mapping[int, Plant],
                   pick_w: np.ndarray, rng: np.random.Generator, tally: dict[int, list[int]]) -> int:
    """Opponent of a ``length``-hand heads-up match started by player ``i``.

    A single-victim plant meets its victim whenever that keeps its share of
    heads-up hands against the victim closest to the configured share
    (error diffusion, so the realised share is on target); its other
    matches go to players drawn by activity. Everybody else picks a rival
    with probability ``stickiness`` and otherwise anyone, by activity.
    """
    if i in single:
        plant = single[i]
        v, t = tally[i]
        share = float(plant.spec.victim_share or 0.0)
        if plant.victim is not None and abs((v + length) / (t + length) - share) <= abs(v / (t + length) - share):
            tally[i] = [v + length, t + length]
            return plant.victim
        tally[i] = [v, t + length]
        w = pick_w.copy()
        w[i] = 0.0
        if plant.victim is not None:
            w[plant.victim] = 0.0
        return int(rng.choice(len(w), p=w / w.sum()))
    p = players[i]
    if p.rivals and rng.random() < p.stickiness:
        return int(p.rivals[int(rng.integers(len(p.rivals)))])
    w = pick_w.copy()
    w[i] = 0.0
    return int(rng.choice(len(w), p=w / w.sum()))


def _ring_stints(length: int, seats: int, weights: np.ndarray, sch: ScheduleParams,
                 rng: np.random.Generator) -> tuple[tuple[int, int, int, int], ...]:
    """Seat timeline of one 6-max table.

    Every seat is taken at the first hand. A player stays for a geometric
    number of hands (mean ``ring_stay_hands_mean``); the seat then stays
    empty for a geometric wait (mean ``ring_refill_hands_mean``) unless the
    table would drop below ``ring_min_players``. New players are drawn in
    proportion to ``weights`` among those not already at the table.
    """
    w = weights / weights.sum()
    n = len(w)
    player = [-1] * seats
    since = [0] * seats
    until = [0] * seats
    wait = [0] * seats
    stints: list[tuple[int, int, int, int]] = []
    p_stay = 1.0 / sch.ring_stay_hands_mean
    p_refill = 1.0 / sch.ring_refill_hands_mean

    def sit(seat: int, t: int) -> None:
        seated = set(player)
        for _ in range(50):
            cand = int(rng.choice(n, p=w))
            if cand not in seated:
                player[seat] = cand
                since[seat] = t
                until[seat] = t + int(rng.geometric(p_stay))
                return

    for t in range(length):
        for seat in range(seats):
            if player[seat] >= 0 and t >= until[seat]:
                stints.append((seat + 1, player[seat], since[seat], t))
                player[seat] = -1
                wait[seat] = t + int(rng.geometric(p_refill)) - 1
            if player[seat] < 0 and t >= wait[seat]:
                sit(seat, t)
        empty = [seat for seat in range(seats) if player[seat] < 0]
        while seats - len(empty) < sch.ring_min_players and empty:
            sit(empty.pop(int(rng.integers(len(empty)))), t)
    for seat in range(seats):
        if player[seat] >= 0:
            stints.append((seat + 1, player[seat], since[seat], length))
    return tuple(sorted(stints, key=lambda x: (x[2], x[0])))
