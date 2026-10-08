"""Tests for the simulator's policies, population, runner and HUD counters.

``sudetect.sim.policies``, ``population``, ``run`` and ``calibrate``: honest
players never look at hidden cards, planted superusers follow their informed
rules, runs are reproducible and independent of the number of processes, the
metadata files land next to the hands folder, and the HUD counter matches
hand-checked numbers on scripted hands.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sudetect.cards import class_index, parse_cards
from sudetect.sim.calibrate import FIELDS, HudCounter, duel, summarize
from sudetect.sim.engine import Decision, DecisionView, SeatState, TableConfig, play_hand
from sudetect.sim.policies import (
    HonestPolicy,
    SuperuserPolicy,
    draw_kind,
    position_class,
    preflop_percentile,
    preflop_percentile_table,
)
from sudetect.sim.population import build_population, build_schedule, load_sim_config, player_name
from sudetect.sim.run import make_agents, play_unit, simulate

T0 = datetime(2001, 1, 1)


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config(overrides={"datasets": {"tiny": {"n_players": 24, "n_days": 3, "target_hands": 1200,
                                                            "plants": ["SU_FULL_100", "SU_VICTIM_92", "SU_ONSET"]}},
                                      "unit_hands": 300})


@pytest.fixture(scope="module")
def pop(cfg):
    return build_population(cfg, "tiny")


def cards(text: str) -> tuple[int, ...]:
    return tuple(parse_cards(text))


# ----------------------------------------------------------------------------------- hand helpers


def test_preflop_ranking_puts_aces_first_and_spreads_over_the_unit_interval():
    pct = preflop_percentile_table()
    assert pct.shape == (169,)
    assert np.all((pct > 0) & (pct < 1))
    assert np.argmin(pct) == class_index("AA")
    assert pct[class_index("KK")] < pct[class_index("AKs")] < pct[class_index("72o")]
    assert pct[class_index("72o")] > 0.95
    assert preflop_percentile(cards("Ah As")) == pct[class_index("AA")]
    assert preflop_percentile(cards("Kd Ah")) == preflop_percentile(cards("Ah Kd")) == pct[class_index("AKo")]


@pytest.mark.parametrize("n, index, label", [
    (2, 1, "BTN"), (2, 0, "BB"),
    (6, 0, "SB"), (6, 1, "BB"), (6, 2, "EP"), (6, 3, "HJ"), (6, 4, "CO"), (6, 5, "BTN"),
    (3, 2, "BTN"), (4, 2, "CO"), (5, 2, "HJ"),
])
def test_position_classes(n, index, label):
    assert position_class(n, index) == label


@pytest.mark.parametrize("hole, board, kind", [
    ("Ah Kh", "2h 7h Tc", 2),        # nut flush draw
    ("9c 8d", "7h 6s 2c", 2),        # open-ended straight draw
    ("9c 8d", "Jh 7s 2c", 1),        # gutshot
    ("9h 8h", "7h 6h 2c", 3),        # flush draw + open-ender
    ("Ah Kd", "Qc Jc 2d", 1),        # gutshot to the broadway
    ("5c 4d", "Ah 3s 9c", 1),        # wheel gutshot (needs a 2)
    ("Ac Kc", "Qc Jc Tc", 0),        # made straight flush: no draw
    ("9c 8d", "7h 6s 5c", 0),        # made straight
    ("2c 3d", "7h 8h 9h", 0),        # the board's draw, not ours
    ("Ah Kh", "2h 7h Tc 3s 4d", 0),  # river: never a draw
])
def test_draw_kind(hole, board, kind):
    assert draw_kind(cards(hole), cards(board)) == kind


# ----------------------------------------------------------------------------------- honest players


class _Trap(dict):
    """Any access to the hidden cards fails the test."""

    def _fail(self, *a, **k):
        raise AssertionError("an honest policy looked at revealed_cards")

    __getitem__ = __contains__ = __iter__ = __len__ = __bool__ = get = items = keys = values = _fail


class _Guard:
    """Wraps an honest policy and replaces the hidden cards by a trap before every decision."""

    def __init__(self, inner: HonestPolicy) -> None:
        self.inner = inner
        self.decisions = 0

    def act(self, view: DecisionView) -> Decision:
        view.revealed_cards = _Trap()
        self.decisions += 1
        return self.inner.act(view)


def test_honest_policies_never_read_revealed_cards(cfg, pop):
    honest = [p for p in pop.players if p.plant_id is None][:6]
    agents = make_agents(cfg, pop, 5)
    for max_seats in (2, 6):
        table = TableConfig(max_seats=max_seats)
        group = honest[:max_seats]
        guards = {p.name: _Guard(agents[p.index]) for p in group}
        grants = {p.name: "all" for p in group}  # the engine offers the cards; nobody may look
        for h in range(150):
            seats = [SeatState(i + 1, p.name, 10_000) for i, p in enumerate(group)]
            play_hand(table, seats, 1 + h % max_seats, guards, h, "T", T0, np.random.default_rng([5, h]), grants)
        assert sum(g.decisions for g in guards.values()) > 300


def _view(**kw) -> DecisionView:
    base = dict(hand_id=7, street="river", seat=1, player="A", hole_cards=cards("Qh Qd"),
                board=cards("2c 7d 9s Jh 3c"), pot=1000, to_call=500, current_bet=500, min_raise_to=1000,
                max_raise_to=9500, can_raise=True, stacks={1: 9500, 2: 9000}, street_contrib={1: 0, 2: 500},
                players_in_hand=(1, 2), players_all_in=(), button_seat=2, seat_order=(1, 2),
                action_history=[], raises_this_street=1, last_aggressor=2, revealed_cards={}, bb=100)
    base.update(kw)
    return DecisionView(**base)


def test_honest_decision_is_the_same_whatever_the_opponent_holds(cfg, pop):
    p = next(p for p in pop.players if p.plant_id is None)
    decisions = set()
    for opp in ("Ah Ad", "4s 5s", "Kc Kh"):
        agent = HonestPolicy(p.name, p.index, p.archetype, p.style, cfg.model, 11)
        decisions.add(agent.act(_view(revealed_cards={2: cards(opp)})))
    assert len(decisions) == 1


def test_policies_return_legal_decisions_on_many_random_hands(cfg, pop):
    agents = make_agents(cfg, pop, 3)
    group = [p for p in pop.players][:6]
    table = TableConfig(max_seats=6)
    amap = {p.name: agents[p.index] for p in group}
    grants = {p.name: "all" for p in group if p.plant_id is not None}
    kinds = set()
    for h in range(300):  # play_hand raises on any illegal decision
        seats = [SeatState(i + 1, p.name, [4000, 10_000, 20_000][(h + i) % 3]) for i, p in enumerate(group)]
        rec = play_hand(table, seats, 1 + h % 6, amap, h, "T", T0, np.random.default_rng([3, h]), grants)
        kinds.update(a.kind for a in rec.actions)
    assert {"fold", "check", "call", "bet", "raise"} <= kinds


# ----------------------------------------------------------------------------------- planted players


def _plant(cfg, pop, u: float = 1.0, seed: int = 1) -> SuperuserPolicy:
    p = pop.players[pop.plants["SU_FULL_100"].player]
    return SuperuserPolicy(p.name, p.index, p.archetype, p.style, cfg.model, seed, u=u)


def test_superuser_folds_when_behind_and_continues_when_ahead(cfg, pop):
    rules = cfg.model.superuser
    # river: hero QQ, opponent holds a set of jacks -> 0% equity, must fold to a pot-sized bet
    behind = _plant(cfg, pop).act(_view(revealed_cards={2: cards("Js Jc")}))
    assert behind.kind == "fold"
    # opponent holds 8-6: hero wins for sure -> never folds; raises for value about half the time
    kinds = [_plant(cfg, pop, seed=s).act(_view(hand_id=s, revealed_cards={2: cards("8s 6h")})).kind
             for s in range(200)]
    assert "fold" not in kinds
    share = kinds.count("raise") / len(kinds)
    assert abs(share - rules.raise_prob) < 0.12


def test_superuser_value_bets_bigger_against_a_hand_that_will_call(cfg, pop):
    open_view = dict(to_call=0, current_bet=0, min_raise_to=100, max_raise_to=9500, raises_this_street=0,
                     last_aggressor=None, street_contrib={1: 0, 2: 0})
    rules = cfg.model.superuser
    strong_opp = _plant(cfg, pop).act(_view(revealed_cards={2: cards("Jc Ts")}, **open_view))  # top pair
    weak_opp = _plant(cfg, pop).act(_view(revealed_cards={2: cards("5h 4h")}, **open_view))  # nothing
    assert strong_opp.kind == weak_opp.kind == "bet"
    assert strong_opp.to == round(rules.value_size_strong * 1000)
    assert weak_opp.to == round(rules.value_size_weak * 1000)


def test_superuser_bluffs_weak_opponents_at_the_configured_rate(cfg, pop):
    open_view = dict(to_call=0, current_bet=0, min_raise_to=100, max_raise_to=9500, raises_this_street=0,
                     last_aggressor=None, street_contrib={1: 0, 2: 0}, hole_cards=cards("5c 4c"),
                     board=cards("Ac Kd Qs 8h 2d"))
    # hero 5-4 loses to 7-6 (ace high, seven kicker) but 7-6 is too weak to call: bluff at bluff_prob;
    # against J-T (a straight) the opponent is strong: never bluff
    bets = [_plant(cfg, pop, seed=s).act(_view(hand_id=s, revealed_cards={2: cards("7s 6d")}, **open_view)).kind
            for s in range(300)]
    assert abs(bets.count("bet") / 300 - cfg.model.superuser.bluff_prob) < 0.1
    never = {_plant(cfg, pop, seed=s).act(_view(hand_id=s, revealed_cards={2: cards("Jh Td")}, **open_view)).kind
             for s in range(50)}
    assert never == {"check"}


def test_superuser_with_u_zero_plays_like_its_honest_style(cfg, pop):
    p = pop.players[pop.plants["SU_FULL_100"].player]
    for s in range(30):
        view = _view(hand_id=s, revealed_cards={2: cards("Js Jc")})
        honest = HonestPolicy(p.name, p.index, p.archetype, p.style, cfg.model, 9).act(view)
        plant = SuperuserPolicy(p.name, p.index, p.archetype, p.style, cfg.model, 9, u=0.0)
        assert plant.act(_view(hand_id=s, revealed_cards={2: cards("Js Jc")})) == honest
        assert plant.informed_decisions == 0 and plant.eligible_decisions == 1


def test_superuser_needs_every_live_opponent_visible(cfg, pop):
    plant = _plant(cfg, pop)
    three = _view(players_in_hand=(1, 2, 3), seat_order=(1, 2, 3), stacks={1: 9500, 2: 9000, 3: 9000},
                  street_contrib={1: 0, 2: 500, 3: 0}, revealed_cards={2: cards("Js Jc")})
    plant.act(three)
    assert plant.eligible_decisions == 0  # seat 3's cards are unknown: honest policy


def test_superuser_beats_the_strong_regular_by_far(cfg, pop):
    p = next(p for p in pop.players if p.archetype == "strong" and p.plant_id is None)
    reads = {q.name: q.archetype for q in pop.players}
    reg = HonestPolicy("HERO_1", 9001, "strong", p.style, cfg.model, cfg.seed, reads)
    su = SuperuserPolicy("HERO_1", 9001, "strong", p.style, cfg.model, cfg.seed, reads, u=1.0)
    a = duel(cfg, pop, reg, n_hands=1500, seed=4)
    b = duel(cfg, pop, su, n_hands=1500, seed=4, info="all")
    diff = b.nets - a.nets
    assert diff.mean() * 100 > 60
    assert diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) > 3


# ----------------------------------------------------------------------------------- population and schedule


def test_names_are_deterministic_hex_and_unique(cfg, pop):
    names = [p.name for p in pop.players]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"P_[0-9a-f]{6}", n) for n in names)
    assert names[0] == player_name(pop.seed, 0)
    again = build_population(cfg, "tiny")
    assert [p.name for p in again.players] == names
    assert [p.style for p in again.players] == [p.style for p in pop.players]


def test_population_mix_plants_and_victims(cfg, pop):
    plants = {pid: pl for pid, pl in pop.plants.items()}
    assert set(plants) == {"SU_FULL_100", "SU_VICTIM_92", "SU_ONSET"}
    assert plants["SU_ONSET"].from_day == round(0.75 * 3)
    victim = plants["SU_VICTIM_92"].victim
    assert victim == pop.victims["V1"] and pop.players[victim].plant_id is None
    assert pop.players[victim].archetype in ("tag", "lag")
    honest = [p for p in pop.players if p.plant_id is None]
    assert len(honest) == 21
    assert {p.archetype for p in honest} <= {"tag", "lag", "nit", "loose_passive", "maniac", "strong"}


def test_schedule_ids_tables_and_seating(cfg, pop):
    sched = build_schedule(cfg, pop)
    sessions = sorted(sched.sessions(), key=lambda s: s.hand_id0)
    assert sessions[0].hand_id0 == cfg.hand_id_base + 1
    for a, b in zip(sessions, sessions[1:]):
        assert b.hand_id0 == a.hand_id0 + a.n_hands
    names = [s.table_name for s in sessions]
    assert len(set(names)) == len(names) and all(re.fullmatch(r"T\d{4,}", n) for n in names)
    for s in sessions:
        for k in range(0, s.n_hands, 7):
            seated = [(seat, p) for (seat, p, a, b) in s.stints if a <= k < b]
            players = [p for _, p in seated]
            assert len(players) == len(set(players)) and len({seat for seat, _ in seated}) == len(seated)
            if s.max_seats == 2:
                assert len(seated) == 2
            else:
                assert cfg.schedule.ring_min_players <= len(seated) <= cfg.ring_seats
    plant = pop.plants["SU_VICTIM_92"]
    total = sum(h for (i, j), h in sched.hu_pair_hands.items() if plant.player in (i, j))
    vs_victim = sched.hu_pair_hands.get(tuple(sorted((plant.player, plant.victim))), 0)
    assert total > 0 and vs_victim / total > 0.6


# ----------------------------------------------------------------------------------- runner


def _files(folder: Path) -> dict[str, bytes]:
    return {f.name: f.read_bytes() for f in sorted(folder.iterdir())}


def test_run_writes_hands_and_metadata_next_to_the_folder_and_is_reproducible(cfg, tmp_path):
    out1, out2 = tmp_path / "a" / "hands", tmp_path / "b" / "hands"
    r1 = simulate(cfg, "tiny", out_dir=out1, procs=1)
    r2 = simulate(cfg, "tiny", out_dir=out2, procs=2)
    assert r1.n_hands == r2.n_hands == r1.schedule.n_hands > 800
    files = _files(out1)
    assert files == _files(out2)  # byte-identical whatever the number of processes
    assert all(re.fullmatch(r"2001-01-0\d_g\d\d\.txt", n) for n in files)
    assert not any(n.endswith(".json") for n in files)
    for name in ("ground_truth.json", "population.json", "run_summary.json"):
        assert (out1.parent / name).read_bytes() == (out2.parent / name).read_bytes()
    truth = json.loads((out1.parent / "ground_truth.json").read_text(encoding="utf-8"))
    by_id = {p["plant_id"]: p for p in truth["plants"]}
    assert by_id["SU_FULL_100"]["sees"] == "every opponent" and by_id["SU_ONSET"]["from_day"] == 2
    assert by_id["SU_VICTIM_92"]["sees"] == [truth["victims"]["V1"]]
    text = files[sorted(files)[0]].decode("utf-8")
    first = text.split("\n\n")[0]
    assert re.match(r"Hand #\d+: Hold'em No Limit \(0\.50/1\.00\) - 2001/01/01 \d\d:\d\d:\d\d UTC\n", first)
    everything = b"".join(files.values()).decode("utf-8")
    ids = sorted(int(x) for x in re.findall(r"^Hand #(\d+):", everything, flags=re.M))
    assert ids == list(range(cfg.hand_id_base + 1, cfg.hand_id_base + 1 + r1.n_hands))
    n_seats = len(re.findall(r"^Seat \d+: P_[0-9a-f]{6} \(\d+\.\d\d in chips\)$", first, flags=re.M))
    assert n_seats >= 2
    assert len(re.findall(r"^Dealt to P_[0-9a-f]{6} \[", first, flags=re.M)) == n_seats  # full information
    population = json.loads((out1.parent / "population.json").read_text(encoding="utf-8"))
    assert len(population["players"]) == 24 and "style" in population["players"][0]


def test_rerun_with_another_seed_changes_the_cards(cfg, tmp_path):
    a = simulate(cfg, "tiny", out_dir=tmp_path / "a" / "hands", procs=1, seed=1)
    b = simulate(cfg, "tiny", out_dir=tmp_path / "b" / "hands", procs=1, seed=2)
    assert _files(a.out_dir) != _files(b.out_dir)


def test_play_unit_keeps_records_and_plants_get_information(cfg, pop):
    sched = build_schedule(cfg, pop)
    agents = make_agents(cfg, pop, cfg.seed)
    recs = []
    for unit in sched.units:
        recs += play_unit(cfg, pop, unit, agents, cfg.seed, keep_records=True)[1]
    assert len(recs) == sched.n_hands
    plant = agents[pop.plants["SU_FULL_100"].player]
    assert plant.informed_decisions > 0
    onset = agents[pop.plants["SU_ONSET"].player]
    day0 = [r for r in recs if r.start_time.day == 1]
    assert day0  # the onset plant has no information on day 0; its counters only grow from day 2
    assert all(sum(r.net.values()) + r.rake == 0 for r in recs)
    assert all(r.rake <= 300 for r in recs)
    assert onset.eligible_decisions >= onset.informed_decisions


# ----------------------------------------------------------------------------------- HUD counter


class _Script:
    def __init__(self, moves):
        self.moves = list(moves)

    def act(self, view):
        return self.moves.pop(0)


def test_hud_counter_on_a_scripted_three_handed_hand():
    # seats: 1 = SB, 2 = BB, 3 = button (first to act preflop, three-handed)
    F, X, C = Decision.fold(), Decision.check(), Decision.call()
    agents = {
        "btn": _Script([Decision.raise_to(250), C, Decision.bet(400), X, X]),     # opens, calls the 3-bet, c-bets
        "sb": _Script([F]),
        "bb": _Script([Decision.raise_to(900), X, C, X, X]),                      # 3-bets, checks, calls the bet
    }
    seats = [SeatState(1, "sb", 10_000), SeatState(2, "bb", 10_000), SeatState(3, "btn", 10_000)]
    deck = parse_cards("2c 3d Ah Kd 7s 7h Qs 8c 4d 9h 2s 5c 6d Ts")
    rec = play_hand(TableConfig(max_seats=6), seats, 3, agents, 1, "T", T0, None, deck=deck)
    c = HudCounter()
    c.add(rec)
    row = {k[0]: dict(zip(FIELDS, v))
           for k, v in c.rows.items()}
    btn, bb, sb = row["btn"], row["bb"], row["sb"]
    assert (btn["vpip"], btn["pfr"], btn["rfi_opp"], btn["rfi"]) == (1, 1, 1, 1)
    assert (bb["tb_opp"], bb["tb"], sb["tb_opp"], sb["tb"]) == (1, 1, 1, 0)
    assert (btn["f3b_opp"], btn["f3b"]) == (1, 0)
    assert (sb["vpip"], sb["pf_dec"], sb["saw_flop"]) == (0, 1, 0)
    # the 3-bettor is the preflop aggressor: bb has the c-bet chance and checks; btn bets into him
    assert (bb["cbet_opp"], bb["cbet"], btn["cbet_opp"]) == (1, 0, 0)
    assert btn["fcb_opp"] == 0 and bb["fcb_opp"] == 0  # a bet after a check is not a c-bet
    assert (btn["saw_flop"], btn["wtsd"], bb["wtsd"]) == (1, 1, 1)
    assert btn["wsd"] + bb["wsd"] >= 1
    assert btn["p_bet"] == 1 and bb["p_call"] == 1 and bb["p_check"] == 3
    assert btn["net"] + bb["net"] + sb["net"] + rec.rake == 0


def test_hud_counter_c_bet_and_fold_to_c_bet():
    F, X, C = Decision.fold(), Decision.check(), Decision.call()
    agents = {"a": _Script([Decision.raise_to(250), Decision.bet(300)]), "b": _Script([C, X, F])}
    seats = [SeatState(1, "a", 10_000), SeatState(2, "b", 10_000)]
    deck = parse_cards("2c 3d Ah Kd 7s 7h Qs 8c 4d")
    rec = play_hand(TableConfig(max_seats=2), seats, 1, agents, 2, "T", T0, None, deck=deck)
    c = HudCounter()
    c.add(rec)
    fields = FIELDS
    a, b = (dict(zip(fields, c.rows[(n, 2, 2)])) for n in ("a", "b"))
    assert (a["cbet_opp"], a["cbet"], b["fcb_opp"], b["fcb"]) == (1, 1, 1, 1)
    assert (a["wtsd"], b["wtsd"], a["saw_flop"], b["saw_flop"]) == (0, 0, 1, 1)
    summary = summarize(c, {"a": "tag", "b": "nit"})
    assert summary.loc[("hu", "honest"), "flop_cbet"] == 100.0
    assert summary.loc[("hu", "honest"), "fold_to_flop_cbet"] == 100.0
