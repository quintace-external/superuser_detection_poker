"""Adversarial tests for the simulator's policies, population, runner and calibration.

These try to break what ``test_sim_policies.py`` takes for granted:

* random streams: no player's decision stream may replay the deck's stream,
  and per-player jitter may never produce a negative noise level;
* information: honest players never touch ``revealed_cards`` (a spy view
  patched into the engine, and a differential check that their decisions do
  not change when the cards are in the view), a single-victim plant sees
  only its victim, the onset plant sees nothing before its day, full-table
  plants see every opponent, ``u`` sets the informed share;
* the informed rule uses the action table's pot odds (the call capped by the
  stack, side-pot aware);
* reproducibility: serial and parallel runs give the same text, the same
  per-hand text whatever the work split, and the same calibration numbers;
* outputs: the answer key never lands in the hands folder or in the text,
  stale unit files and a stale calibration of an earlier run are always
  removed, the onset day follows a ``--days`` override;
* schedule: plants' volume and active days look like the honest field's,
  sticky heads-up pairs and the victim pairs reach the single-victim test's
  500 hands in the demo dataset on several seeds;
* robustness: 2- to 9-seat tables, tiny stacks, antes and no top-up never
  crash, and the honest field meets the calibration targets on a fresh seed.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sudetect.cards import parse_cards
from sudetect.equity import equity_vs_hand
from sudetect.sim import engine
from sudetect.sim.calibrate import check_targets, summarize
from sudetect.sim.engine import ActionRecord, Decision, DecisionView, SeatState, TableConfig, play_hand
from sudetect.sim.policies import HonestPolicy, SuperuserPolicy
from sudetect.sim.population import TableSession, WorkUnit, build_population, build_schedule, load_sim_config
from sudetect.sim.run import make_agents, play_unit, player_hands, simulate

T0 = datetime(2001, 1, 1)
PROCS = max(2, min(8, (os.cpu_count() or 2)))


def cards(text: str) -> tuple[int, ...]:
    return tuple(parse_cards(text))


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(overrides={
        "unit_hands": 300,
        "datasets": {
            "tiny": {"n_players": 24, "n_days": 3, "target_hands": 1200,
                     "plants": ["SU_FULL_100", "SU_VICTIM_92", "SU_ONSET"]},
            "spy": {"n_players": 30, "n_days": 8, "target_hands": 4000,
                    "plants": ["SU_FULL_100", "SU_FULL_050", "SU_VICTIM_92", "SU_VICTIM_20", "SU_ONSET"]},
        },
    })


# ----------------------------------------------------------------------------------- random streams


def test_no_decision_stream_replays_the_deck_stream(cfg):
    """numpy pads a short seed key with zeros, so a key like [seed, hand, 0]
    is the deck's [seed, hand]: player 0's noise would be the shuffle's bits."""
    pop = build_population(cfg, "tiny")
    agents = make_agents(cfg, pop, cfg.seed)
    for hand_id in (1_000_001, 1_000_777, 10_000_000):
        deck = np.random.default_rng([cfg.seed, hand_id]).random(32)
        streams = []
        for p in pop.players:
            agent = agents[p.index]
            agent._hand_id = None  # force a fresh stream for this hand
            view = _river_view(hand_id=hand_id)
            streams.append(agent._begin(view).random(32))
            assert not np.array_equal(streams[-1], deck), f"player {p.index} decides with the deck's random stream"
            if isinstance(agent, SuperuserPolicy):
                assert not np.array_equal(agent._u_rng.random(32), streams[-1])
        assert len({s.tobytes() for s in streams}) == len(streams)


def test_jitter_never_gives_a_negative_noise_level(cfg):
    # seed 97 of the demo population used to give a strong regular pf_noise < 0 (numpy: "scale < 0")
    pop = build_population(cfg, "sim_demo", 97)
    assert min(p.style.pf_noise for p in pop.players) >= 0 and min(p.style.noise for p in pop.players) >= 0
    wild = load_sim_config(overrides={"jitter": {"noise": 0.2, "pf_noise": 0.2}})
    pop = build_population(wild, "sim_small", 3)
    assert min(p.style.pf_noise for p in pop.players) >= 0 and min(p.style.noise for p in pop.players) >= 0
    agents = make_agents(wild, pop, 3)
    group = pop.players[:6]
    amap = {p.name: agents[p.index] for p in group}
    for h in range(120):
        seats = [SeatState(i + 1, p.name, 10_000) for i, p in enumerate(group)]
        play_hand(TableConfig(max_seats=6), seats, 1 + h % 6, amap, h, "T", T0, np.random.default_rng([4, h]))


# ----------------------------------------------------------------------------------- hidden information

_SLOT = DecisionView.__dict__["revealed_cards"]


class _SpyView(DecisionView):
    """The engine's view with a logging accessor on ``revealed_cards``."""

    __slots__ = ()
    readers: list[str] = []

    @property
    def revealed_cards(self):  # type: ignore[override]
        _SpyView.readers.append(self.player)
        return _SLOT.__get__(self, DecisionView)

    @revealed_cards.setter
    def revealed_cards(self, value):
        _SLOT.__set__(self, value)


def test_honest_players_never_touch_revealed_cards_in_a_full_run(cfg, monkeypatch):
    monkeypatch.setattr(engine, "DecisionView", _SpyView)
    _SpyView.readers = []
    r = simulate(cfg, "tiny", procs=1, write=False)
    assert r.n_hands > 800
    plants = {r.population.players[pl.player].name for pl in r.population.plants.values()}
    honest_reads = [name for name in _SpyView.readers if name not in plants]
    assert honest_reads == []
    assert len(_SpyView.readers) > 100  # the plants read the view, so the spy works


class _Differential:
    """Decides on a copy with every opponent's cards in the view, then for real
    with an empty dict; the two decisions must be equal."""

    def __init__(self, inner: HonestPolicy) -> None:
        self.inner = inner
        self.checked = 0

    def act(self, view: DecisionView) -> Decision:
        assert len(view.revealed_cards) == len(view.seat_order) - 1  # the engine offered every opponent
        seen = copy.deepcopy(self.inner).act(view)
        view.revealed_cards = {}
        blind = self.inner.act(view)
        assert seen == blind, f"{self.inner.archetype} changed its decision with the hidden cards in view"
        self.checked += 1
        return blind


def test_honest_decisions_do_not_depend_on_hidden_cards_any_style(cfg):
    pop = build_population(cfg, "sim_demo")
    agents = make_agents(cfg, pop, 21)
    by_style: dict[str, list] = {}
    for p in pop.players:
        if p.plant_id is None:
            by_style.setdefault(p.archetype, []).append(p)
    assert len(by_style) == 6
    group = [ps[k] for k in range(2) for ps in by_style.values()]  # 12 players, every style twice
    checked = 0
    for max_seats, hands in ((2, 60), (6, 60), (9, 40)):
        for h in range(hands):
            sel = [group[(h + k) % len(group)] for k in range(max_seats)]
            wrapped = {p.name: _Differential(agents[p.index]) for p in sel}
            seats = [SeatState(i + 1, p.name, [3_000, 10_000, 25_000][(h + i) % 3]) for i, p in enumerate(sel)]
            grants = {p.name: "all" for p in sel}
            play_hand(TableConfig(max_seats=max_seats), seats, 1 + h % max_seats, wrapped, 50_000 + h, "T", T0,
                      np.random.default_rng([21, max_seats, h]), grants)
            checked += sum(w.checked for w in wrapped.values())
    assert checked > 1000


class _Logger:
    def __init__(self, inner, log: list) -> None:
        self.inner, self.log = inner, log

    def act(self, view: DecisionView) -> Decision:
        self.log.append((view.hand_id, view.player, view.seat, frozenset(view.revealed_cards)))
        return self.inner.act(view)


def test_information_grants_follow_the_ground_truth(cfg):
    pop = build_population(cfg, "spy")
    sched = build_schedule(cfg, pop)
    agents = make_agents(cfg, pop, cfg.seed)
    log: list = []
    spied = {i: _Logger(a, log) for i, a in agents.items()}
    recs = {}
    for unit in sched.units:
        for rec in play_unit(cfg, pop, unit, spied, cfg.seed, keep_records=True)[1]:
            recs[rec.hand_id] = rec
    assert len(recs) == sched.n_hands
    day_of = {s.hand_id0 + k: s.day for s in sched.sessions() for k in range(s.n_hands)}
    names = [p.name for p in pop.players]
    plant_of = {names[pl.player]: pl for pl in pop.plants.values()}
    onset = pop.plants["SU_ONSET"]
    assert 0 < onset.from_day < 8
    seen = {pid: 0 for pid in pop.plants}
    onset_days = set()
    for hand_id, player, seat, revealed in log:
        rec = recs[hand_id]
        seat_of = {s.player: s.seat for s in rec.seats}
        others = frozenset(s.seat for s in rec.seats if s.seat != seat)
        pl = plant_of.get(player)
        if pl is None:
            assert revealed == frozenset(), f"honest {player} was offered cards"
            continue
        day = day_of[hand_id]
        if day < pl.from_day:
            assert revealed == frozenset(), f"{pl.spec.plant_id} saw cards on day {day}"
            continue
        if pl.spec.kind == "full_table":
            assert revealed == others
        else:
            victim_seat = seat_of.get(names[pl.victim])
            assert revealed == (frozenset({victim_seat}) if victim_seat is not None else frozenset())
        if revealed:
            seen[pl.spec.plant_id] += 1
            if pl is onset:
                onset_days.add(day)
    assert all(v > 0 for v in seen.values()), seen
    assert onset_days and min(onset_days) >= onset.from_day
    assert any(day_of[h] < onset.from_day for h, player, *_ in log if player == names[onset.player])
    half = agents[pop.plants["SU_FULL_050"].player]
    assert half.eligible_decisions > 100
    assert abs(half.informed_decisions / half.eligible_decisions - 0.5) < 0.12
    full = agents[pop.plants["SU_FULL_100"].player]
    assert full.informed_decisions == full.eligible_decisions > 100


def test_single_victim_plant_at_a_six_max_table_sees_only_the_victim(cfg):
    pop = build_population(cfg, "spy")
    pl = pop.plants["SU_VICTIM_92"]
    others = [p.index for p in pop.players if p.plant_id is None and p.index != pl.victim][:4]
    seated = [pl.player, pl.victim, *others]
    session = TableSession(table_no=9_999, table_name="T9999", day=0, max_seats=6, start_s=0.0, n_hands=300,
                           hand_id0=5_000_001, stints=tuple((k + 1, p, 0, 300) for k, p in enumerate(seated)))
    agents = make_agents(cfg, pop, cfg.seed)
    log: list = []
    spied = {i: _Logger(a, log) for i, a in agents.items()}
    play_unit(cfg, pop, WorkUnit(day=0, group=0, file_name="x.txt", sessions=[session]), spied, cfg.seed)
    me = pop.players[pl.player].name
    mine = [revealed for _, player, _, revealed in log if player == me]
    assert mine and all(revealed == frozenset({2}) for revealed in mine)  # the victim sits in seat 2
    assert all(revealed == frozenset() for _, player, _, revealed in log if player != me)
    plant = agents[pl.player]
    assert 0 < plant.eligible_decisions < len(mine)  # informed only once the victim is the last opponent


# ----------------------------------------------------------------------------------- informed rule


def _river_view(**kw) -> DecisionView:
    base = dict(hand_id=7, street="river", seat=1, player="A", hole_cards=cards("Qh Qd"),
                board=cards("2c 7d 9s Jh 3c"), pot=1000, to_call=500, current_bet=500, min_raise_to=1000,
                max_raise_to=9500, can_raise=True, stacks={1: 9500, 2: 9000}, street_contrib={1: 0, 2: 500},
                players_in_hand=(1, 2), players_all_in=(), button_seat=2, seat_order=(1, 2),
                action_history=[], raises_this_street=1, last_aggressor=2, revealed_cards={}, bb=100)
    base.update(kw)
    return DecisionView(**base)


def _act(order, street, seat, player, kind, amount, to=None):
    return ActionRecord(order, street, seat, player, kind, amount, to, None, False, 0, 0, 0)


def test_informed_call_uses_pot_odds_capped_by_the_stack(cfg):
    """A short plant facing an overbet shove only risks what it has: the
    action table's pot odds (call capped by the stack, side-pot aware) decide."""
    hero, villain, board = cards("9h 8h"), cards("Ks Tc"), cards("Th 7c 2h Kd")
    eq = equity_vs_hand(hero, villain, board)
    history = [
        _act(0, "preflop", 2, "B", "post_sb", 50), _act(1, "preflop", 1, "A", "post_bb", 100),
        _act(2, "preflop", 2, "B", "raise", 250, 300), _act(3, "preflop", 1, "A", "call", 200),
        _act(4, "flop", 1, "A", "check", 0), _act(5, "flop", 2, "B", "bet", 700, 700),
        _act(6, "flop", 1, "A", "call", 700), _act(7, "turn", 1, "A", "check", 0),
        _act(8, "turn", 2, "B", "bet", 19_000, 19_000),
    ]
    view = _river_view(street="turn", hole_cards=hero, board=board, pot=21_000, to_call=19_000,
                       current_bet=19_000, min_raise_to=0, max_raise_to=0, can_raise=False,
                       stacks={1: 900, 2: 0}, street_contrib={1: 0, 2: 19_000}, players_all_in=(2,),
                       action_history=history, revealed_cards={2: villain})
    capped = 900 / (1_900 + 1_900)  # risk 900 to win the 1,900 the villain can lose
    uncapped = 19_000 / (21_000 + 19_000)
    margin = cfg.model.superuser.call_margin
    assert capped + margin <= eq < uncapped + margin
    p = build_population(cfg, "tiny").players[0]
    plant = SuperuserPolicy("A", p.index, "strong", p.style, cfg.model, 1, u=1.0)
    assert plant.act(view).kind == "call"
    # covered stacks: the same rule as plain pot odds
    deep = _river_view(revealed_cards={2: cards("Js Jc")})
    assert plant.act(deep).kind == "fold"


# ----------------------------------------------------------------------------------- reproducibility


def _hands_by_id(folder: Path) -> dict[int, str]:
    out = {}
    for f in sorted(folder.glob("*.txt")):
        for block in re.split(r"\n\n(?=Hand #)", f.read_text(encoding="utf-8")):
            m = re.match(r"Hand #(\d+):", block)
            assert m, block[:80]
            assert int(m.group(1)) not in out
            out[int(m.group(1))] = block.rstrip("\n")
    return out


def test_serial_parallel_and_work_split_give_the_same_hands_and_numbers(cfg, tmp_path):
    fine = load_sim_config(overrides={"datasets": cfg.raw["datasets"], "unit_hands": 50})
    coarse = load_sim_config(overrides={"datasets": cfg.raw["datasets"], "unit_hands": 100_000})
    r1 = simulate(fine, "tiny", out_dir=tmp_path / "s" / "hands", procs=1, calibrate=True)
    r3 = simulate(fine, "tiny", out_dir=tmp_path / "p" / "hands", procs=3, calibrate=True)
    rs = simulate(coarse, "tiny", out_dir=tmp_path / "w" / "hands", procs=1, calibrate=True)
    assert len(list(r1.out_dir.glob("*.txt"))) > len(list(rs.out_dir.glob("*.txt")))  # really a different split
    h1, h3, hs = _hands_by_id(r1.out_dir), _hands_by_id(r3.out_dir), _hands_by_id(rs.out_dir)
    assert h1 == h3 == hs and len(h1) == r1.n_hands
    groups = {p.name: (p.plant_id or p.archetype) for p in r1.population.players}
    f1, f3, fs = (r.counter.frame() for r in (r1, r3, rs))
    assert f1.equals(f3)  # same rows in the same order: float sums do not depend on the process count
    s1, s3 = summarize(r1.counter, groups), summarize(r3.counter, groups)
    assert s1.equals(s3)
    key = ["player", "max_seats", "n_dealt"]
    assert f1.sort_values(key).reset_index(drop=True).equals(fs.sort_values(key).reset_index(drop=True))
    assert r1.plant_stats == r3.plant_stats == rs.plant_stats


# ----------------------------------------------------------------------------------- outputs


def test_answer_key_never_in_the_hands_folder_or_the_text(cfg, tmp_path, monkeypatch):
    hands = tmp_path / "ds" / "hands"
    hands.mkdir(parents=True)
    monkeypatch.chdir(hands)
    r = simulate(cfg, "tiny", out_dir=Path("."), procs=1)  # a relative folder with no parent in its name
    assert sorted(p.name for p in hands.iterdir() if not p.name.endswith(".txt")) == []
    for name in ("ground_truth.json", "population.json", "run_summary.json"):
        assert (tmp_path / "ds" / name).exists()
    text = "".join(f.read_text(encoding="utf-8") for f in hands.glob("*.txt"))
    for word in ("SU_", "planted", "victim", "superuser", "truth", *r.population.plants):
        assert word.lower() not in text.lower()


def test_stale_unit_files_are_always_removed(tmp_path):
    many = load_sim_config(overrides={
        "unit_hands": 1, "hu_hand_share": 1.0,
        "schedule": {"hu_match_hands_mean": 10.0, "hu_match_hands_sigma": 0.01, "hu_match_hands_min": 10},
        "datasets": {"one": {"n_players": 30, "n_days": 1, "target_hands": 1500,
                             "plants": ["SU_FULL_100", "SU_VICTIM_92", "SU_ONSET"]}},
    })
    out = tmp_path / "one" / "hands"
    first = simulate(many, "one", out_dir=out, procs=1, seed=5)
    assert any(re.search(r"_g\d{3,}\.txt$", f.name) for f in out.iterdir())  # more than 100 groups in a day
    few = load_sim_config(overrides={**{k: v for k, v in many.raw.items() if k in ("datasets", "schedule")},
                                     "hu_hand_share": 1.0, "unit_hands": 100_000})
    second = simulate(few, "one", out_dir=out, procs=1, seed=6)
    assert first.n_files > 100 and second.n_files == 1
    assert sorted(f.name for f in out.iterdir()) == ["2001-01-01_g00.txt"]
    assert len(_hands_by_id(out)) == second.n_hands


def test_cli_never_leaves_a_calibration_of_older_hands(tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "00_simulator" / "simulate.py"
    spec = importlib.util.spec_from_file_location("simulate_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "ds" / "hands"
    args = ["--dataset", "sim_small", "--hands", "1500", "--days", "2", "--procs", "1", "--out", str(out)]
    assert cli.main(args + ["--calibrate"]) == 0
    assert (out.parent / "calibration.json").exists()
    assert cli.main(args + ["--seed", "7"]) == 0  # new hands, no calibration: the old file must go
    assert not (out.parent / "calibration.json").exists()
    assert sorted(p.name for p in out.parent.iterdir()) == ["ground_truth.json", "hands", "population.json",
                                                             "run_summary.json"]
    capsys.readouterr()


def test_onset_day_follows_a_days_override(cfg, tmp_path):
    r = simulate(cfg, "tiny", out_dir=tmp_path / "d" / "hands", procs=1, n_days=8, target_hands=2400)
    onset = r.population.plants["SU_ONSET"]
    assert onset.from_day == round(0.75 * 8)
    assert max(u.day for u in r.schedule.units) == 7
    assert r.plant_stats["SU_ONSET"]["informed_decisions"] > 0


# ----------------------------------------------------------------------------------- schedule


@pytest.mark.parametrize("seed", [None, 5, 6])
def test_demo_plants_have_ordinary_volume_and_active_days(cfg, seed):
    pop = build_population(cfg, "sim_demo", seed)
    sched = build_schedule(cfg, pop)
    n, days = len(pop.players), cfg.dataset("sim_demo").n_days
    per_day = np.zeros((n, days))
    for s in sched.sessions():
        for (_, p, a, b) in s.stints:
            per_day[p, s.day] += b - a
    honest = np.array([p.plant_id is None for p in pop.players])
    total, active = per_day.sum(1), (per_day > 0).sum(1)
    lo_t, hi_t = np.percentile(total[honest], [5, 95])
    lo_d, hi_d = np.percentile(active[honest], [2.5, 97.5])
    for pid, pl in pop.plants.items():
        assert lo_t <= total[pl.player] <= hi_t, (pid, total[pl.player], lo_t, hi_t)
        assert lo_d <= active[pl.player] <= hi_d, (pid, active[pl.player], lo_d, hi_d)


@pytest.mark.parametrize("seed", [None, 1, 2])
def test_demo_schedule_has_sticky_pairs_and_testable_victim_pairs(cfg, seed):
    pop = build_population(cfg, "sim_demo", seed)
    sched = build_schedule(cfg, pop)
    cal = cfg.raw["calibration"]
    plants = {pl.player for pl in pop.plants.values()}
    pairs = sum(1 for (i, j), h in sched.hu_pair_hands.items()
                if h >= cal["pair_min_hands"] and i not in plants and j not in plants)
    assert pairs >= cal["min_pairs"]
    hu = player_hands(sched, len(pop.players))["hu"]
    for pid, pl in pop.plants.items():
        if pl.victim is None:
            continue
        together = sched.hu_pair_hands.get((min(pl.player, pl.victim), max(pl.player, pl.victim)), 0)
        assert together >= cal["pair_min_hands"], (pid, together)
        assert abs(together / hu[pl.player] - pl.spec.victim_share) < 0.03, (pid, together / hu[pl.player])
    assert abs(sum(s.n_hands for s in sched.sessions() if s.max_seats == 2) / sched.n_hands
               - cfg.hu_hand_share) < 0.02


# ----------------------------------------------------------------------------------- robustness


def test_two_to_nine_seats_tiny_stacks_and_antes_never_crash(cfg):
    pop = build_population(cfg, "sim_small", 8)
    agents = make_agents(cfg, pop, 8)
    rng = np.random.default_rng(8)
    stacks = [1, 10, 30, 49, 50, 51, 99, 100, 101, 150, 199, 250, 1_000, 10_000, 40_000]
    sizes = {}
    for h in range(1_500):
        n = int(rng.integers(2, 10))
        ante = int(rng.choice([0, 0, 10, 25, 100]))
        who = rng.choice(len(pop.players), size=n, replace=False)
        seat_nos = sorted(int(x) for x in rng.choice(np.arange(1, 10), size=n, replace=False))
        seats = [SeatState(s, pop.players[p].name, int(rng.choice(stacks))) for s, p in zip(seat_nos, who)]
        amap = {pop.players[p].name: agents[int(p)] for p in who}
        grants = {pop.players[p].name: "all" for p in who if pop.players[p].plant_id}
        rec = play_hand(TableConfig(max_seats=9, ante=ante), seats, int(rng.choice(seat_nos)), amap, 70_000 + h, "T",
                        T0, np.random.default_rng([8, h]), grants)
        assert sum(rec.net.values()) + rec.rake == 0
        sizes[n] = sizes.get(n, 0) + 1
    assert set(sizes) == set(range(2, 10))


@pytest.mark.parametrize("over", [
    {"ring_seats": 9, "buy_in_mix": [[1, 0.3], [2, 0.3], [5, 0.2], [200, 0.2]], "ante_bb": 0.25},
    {"ring_seats": 9, "buy_in_mix": [[1, 0.5], [3, 0.5]], "top_up_below": 0.0},  # no top-up: busted players rebuy
    {"hu_hand_share": 1.0, "buy_in_mix": [[1, 0.5], [2, 0.5]], "top_up_below": 0.0},
])
def test_runner_survives_short_stacks_big_tables_and_no_top_up(over):
    small = load_sim_config(overrides=over)
    r = simulate(small, "sim_small", procs=1, write=False, target_hands=2_000)
    assert r.n_hands == r.schedule.n_hands > 1_000


def test_calibration_targets_hold_on_a_fresh_seed():
    cfg = load_sim_config()
    r = simulate(cfg, "sim_demo", seed=97, procs=PROCS, calibrate=True, write=False, target_hands=60_000)
    groups = {p.name: (p.plant_id or p.archetype) for p in r.population.players}
    rows = check_targets(summarize(r.counter, groups, cfg.bb_units), cfg.raw["calibration"]["targets"])
    misses = [(kind, stat, round(value, 2), (lo, hi)) for kind, stat, lo, hi, value, ok in rows if not ok]
    assert misses == []
