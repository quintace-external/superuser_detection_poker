"""The range model on small hand-made field tables: invariants, the two tilts, back-off, pruning, narrowing,
Monte Carlo equity and one worker chunk end to end."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from sudetect import rangemodel as rm
from sudetect.cards import CLASS_LABELS, COMBO_INDEX, COMBO_TO_CLASS, COMBOS, N_CLASSES, class_index, parse_cards
from sudetect.equity import equity_vs_hand, equity_vs_range, multiway_equity
from sudetect.handclass import CHEN_PERCENTILE, MADE_CLASSES, classes_for_all_combos

# The narrow-setting constants (variants/cap240.toml): the tests of the cap, the class floor and the relative prune
# need them switched on, whatever the public default is.
CFG = rm.RangeConfig(cap=240, prune_frac=2e-4, class_floor=0.0008, prune_each_step=True, combo_weight="live_combos")
CFG_FULL = dataclasses.replace(CFG, cap=1326)          # no cap: tests of the narrowing itself
CFG_POSTERIOR = rm.RangeConfig(cap=1326, prune_frac=0.0, class_floor=0.0, combo_weight="class_size")
STREETS = ("flop", "turn", "river")
SITUATIONS = ("cbet_opp", "donk_opp", "probe_opp", "lead_opp", "vs_cbet", "vs_donk", "vs_probe", "vs_bet",
              "vs_raise")


def uniform_prior() -> np.ndarray:
    """169 class counts proportional to the class sizes: every combo equally likely."""
    return np.array([6.0 if len(c) == 2 else (4.0 if c.endswith("s") else 12.0) for c in CLASS_LABELS]) * 100


def make_tables(lik_k3=None, uniform_actions: bool = True, prior=None) -> rm.FitTables:
    """Field tables with a K1 cell per (street, situation) and optional K3 cells per made class."""
    lik = {lv: {} for lv in rm.LIK_LEVELS}
    for st in STREETS:
        for sit in SITUATIONS:
            lik["K1"][(st, sit)] = np.full(5, 200.0) if uniform_actions else np.array([10, 40, 50, 60, 40.0])
    for key, counts in (lik_k3 or {}).items():
        lik["K3"][key] = np.asarray(counts, dtype=float)
    p = uniform_prior() if prior is None else prior
    pri = {"P4": {}, "P3": {}, "P2": {}, "P1": {}, "P0": {(): p.copy()}}
    return rm.FitTables(lik=lik, prior=pri, field={"2": (0.6, 0.4, 0.4), "all": (0.5, 0.3, 0.4)})


def line(nodes=(), **kw) -> rm.OpponentLine:
    base = dict(size_cls="2", pf_situation="unopened", position_class="BTN", pf_action="raise")
    base.update(kw)
    return rm.OpponentLine(nodes=tuple(nodes), **base)


def node(street, situation, action, board, pose="ip", pot="single_raised") -> rm.Node:
    return rm.Node(street=street, situation=situation, pose=pose, pot_type=pot, action=action,
                   board=tuple(parse_cards(board)))


# ------------------------------------------------------------------ invariants


@pytest.mark.parametrize("each", [True, False])
def test_invariants_on_random_spots(each):
    cfg = dataclasses.replace(CFG, prune_each_step=each)
    rng = np.random.default_rng(7)
    k3 = {("flop", "vs_cbet", m): [10 + i, 50, 30 + 3 * i, 5, 2 + 4 * i] for i, m in enumerate(MADE_CLASSES)}
    tables = make_tables(k3, uniform_actions=False)
    for _ in range(40):
        deck = rng.permutation(52)
        hero, board = deck[:2].tolist(), deck[2:7].tolist()
        n_board = int(rng.choice([3, 4, 5]))
        nodes = [node("flop", "vs_cbet", "call", board[:3]), node("turn", "lead_opp", "bet", board[:4])]
        res = rm.predict_range(tables, cfg, line(nodes[: n_board - 2]), hero + board[:n_board])
        assert res is not None
        assert math.isclose(res.w.sum(), 1.0, abs_tol=1e-12)
        assert (res.w > 0).all()
        assert 0 < res.ids.size <= cfg.cap
        assert np.unique(res.ids).size == res.ids.size
        dead = set(hero + board[:n_board])
        assert not any(int(c) in dead for c in COMBOS[res.ids].ravel())
        again = rm.predict_range(tables, cfg, line(nodes[: n_board - 2]), hero + board[:n_board])
        assert np.array_equal(res.ids, again.ids) and np.array_equal(res.w, again.w)


@pytest.mark.parametrize("each", [True, False])
def test_uniform_likelihood_leaves_the_prior_unchanged(each):
    cfg = dataclasses.replace(CFG, prune_each_step=each)
    tables = make_tables()
    hero, board = parse_cards("Ah Kd"), parse_cards("7c 8d 2s Qh 4c")
    before = rm.predict_range(tables, cfg, line(), hero + board)
    nodes = [node("flop", "vs_cbet", "call", board[:3]), node("turn", "lead_opp", "bet", board[:4]),
             node("river", "vs_raise", "raise", board)]
    after = rm.predict_range(tables, cfg, line(nodes), hero + board)
    assert np.array_equal(before.ids, after.ids)
    assert np.allclose(before.w, after.w, rtol=1e-12, atol=0)


def test_no_postflop_action_gives_the_tilted_prior_cut_to_the_cap():
    prior = uniform_prior()
    prior[class_index("AA")] *= 50                       # one heavy class must survive the cap
    tables = make_tables(prior=prior)
    res = rm.predict_range(tables, CFG, line(), parse_cards("2c 3d 9h Th Js"))
    assert res.ids.size == CFG.cap
    assert rm.top_class(res.ids, res.w) == "AA"


# ------------------------------------------------------------------ narrowing


def test_a_river_raise_removes_air():
    strong = {"two_pair", "trips_set", "straight", "flush", "full_house_plus"}
    k3 = {("river", "vs_bet", m): ([300, 0, 600, 0, 100] if m in strong
                                   else [600, 0, 400, 0, 0] if m == "nothing" else [500, 0, 499, 0, 1])
          for m in MADE_CLASSES}
    tables = make_tables(k3)
    hero, board = parse_cards("Ac Qd"), parse_cards("Kh 9s 4d 7c 2h")
    before = rm.predict_range(tables, CFG_FULL, line(), hero + board)
    after = rm.predict_range(tables, CFG_FULL, line([node("river", "vs_bet", "raise", board)]), hero + board)
    air_before = rm.group_shares(before.ids, before.w, board)["air"]
    air_after = rm.group_shares(after.ids, after.w, board)["air"]
    assert air_before > 0.3
    assert air_after < 0.03 and air_after < air_before / 10
    eq_before = equity_vs_range(hero, board, rm.range_dense(before.ids, before.w))
    eq_after = equity_vs_range(hero, board, rm.range_dense(after.ids, after.w))
    assert eq_after < eq_before - 0.3


def test_update_uses_the_classes_of_the_board_of_that_action():
    # a flop bet with made hands only: on a 7-8-2 flop the combos that pair the flop gain weight even if the
    # decision is on a later street where other combos pair the turn card
    k3 = {("flop", "lead_opp", m): ([0, 900, 0, 100, 0] if m == "nothing" else [0, 100, 0, 900, 0])
          for m in MADE_CLASSES}
    tables = make_tables(k3)
    hero, board = parse_cards("As Ks"), parse_cards("7c 8d 2s Qh")
    res = rm.predict_range(tables, CFG_FULL, line([node("flop", "lead_opp", "bet", board[:3])]), hero + board)
    made_flop, _ = classes_for_all_combos(board[:3])
    share_nothing_on_flop = res.w[made_flop[res.ids] == 0].sum()
    assert share_nothing_on_flop < 0.15


def test_aggression_tilt_keeps_more_air_for_an_aggressive_player():
    k3 = {("flop", "lead_opp", m): ([0, 500, 0, 500, 0] if m == "nothing" else [0, 300, 0, 700, 0])
          for m in MADE_CLASSES}
    tables = make_tables(k3)
    hero, board = parse_cards("As Ks"), parse_cards("7c 8d 2s")
    nodes = [node("flop", "lead_opp", "bet", board)]
    calm = rm.predict_range(tables, CFG_FULL, line(nodes, n=200, agg=0.4), hero + board)        # agg = field
    wild = rm.predict_range(tables, CFG_FULL, line(nodes, n=200, agg=0.8), hero + board)        # twice the field
    air = lambda r: rm.group_shares(r.ids, r.w, board)["air"]
    assert air(wild) > air(calm) + 0.02


def test_aggression_tilt_weights():
    assert rm.aggression_tilt(7, 0.8, 0.4, CFG) == (1.0, 0.0)
    assert rm.aggression_tilt(8, 0.8, 0.4, CFG) == (2.0, 0.0)
    af, w = rm.aggression_tilt(29, 0.1, 0.4, CFG)
    assert af == 0.5 and math.isclose(w, 0.5)
    af, w = rm.aggression_tilt(500, 0.6, 0.4, CFG)
    assert math.isclose(af, 1.5) and w == 1.0
    assert rm.aggression_tilt(500, None, 0.4, CFG) == (1.0, 0.0)


# ------------------------------------------------------------------ prior and tilts


def test_looseness_tilt_moves_weight_between_weak_and_strong_classes():
    prior = uniform_prior() / uniform_prior().sum()
    loose = rm.looseness_tilt(prior, 0.9, 0.45, 100, CFG)                 # ratio 2
    tight = rm.looseness_tilt(prior, 0.15, 0.45, 100, CFG)                # ratio 1/3, clipped to 0.5
    strength = np.asarray(CHEN_PERCENTILE)
    assert math.isclose(loose.sum(), 1.0) and math.isclose(tight.sum(), 1.0)
    assert (loose * strength).sum() < (prior * strength).sum() < (tight * strength).sum()
    aa, worst = class_index("AA"), int(np.argmin(strength))
    assert math.isclose(loose[aa] / prior[aa] / (loose[worst] / prior[worst]), 2 ** (-1.4), rel_tol=1e-9)
    assert rm.looseness_tilt(prior, 0.9, 0.45, 14, CFG) is prior          # too few actions
    assert rm.looseness_tilt(prior, 0.4502, 0.45, 100, CFG) is prior      # |r - 1| < 0.001
    assert rm.looseness_tilt(prior, None, 0.45, 100, CFG) is prior


def test_preflop_tilt_picks_pfr_after_a_raise():
    tables = make_tables()
    hero = parse_cards("2c 3d")
    base = rm.predict_range(tables, CFG_FULL, line(n=100, vpip=0.6, pfr=0.4), hero)        # vpip = field: no tilt
    raiser = rm.predict_range(tables, CFG_FULL, line(raised_preflop=True, n=100, vpip=0.6, pfr=0.8), hero)
    strength = np.asarray(CHEN_PERCENTILE)[COMBO_TO_CLASS]
    assert (raiser.w * strength[raiser.ids]).sum() < (base.w * strength[base.ids]).sum()


def test_prior_backoff():
    tables = make_tables()
    p4 = np.zeros(N_CLASSES)
    p4[class_index("AA")] = 29                                             # below min_support = 30
    p3 = np.zeros(N_CLASSES)
    p3[class_index("KK")] = 40
    tables.prior["P4"][("2", "vs_open", "BB", "call")] = p4
    tables.prior["P3"][("vs_open", "BB", "call")] = p3
    prior, level = tables.prior_for("2", "vs_open", "BB", "call", CFG)
    assert level == "P3" and prior[class_index("KK")] == 1.0
    p4[class_index("AA")] = 30
    tables._prior_cache.clear()
    assert tables.prior_for("2", "vs_open", "BB", "call", CFG)[1] == "P4"
    assert tables.prior_for("2", "vs_open", "BB", "call", dataclasses.replace(CFG, use_size_class=False))[1] == "P3"
    # a player who never acted skips the situation levels; call and raise merge at the position
    c = np.zeros(N_CLASSES)
    c[class_index("QQ")] = 20
    r = np.zeros(N_CLASSES)
    r[class_index("JJ")] = 20
    tables.prior["P1"][("SB", "call")] = c
    tables.prior["P1"][("SB", "raise")] = r
    prior, level = tables.prior_for("3-6", "none", "SB", "call", CFG)
    assert level == "P1m" and math.isclose(prior[class_index("QQ")], 0.5)
    assert tables.prior_for("3-6", "none", "CO", "check", CFG)[1] == "P0"
    empty = rm.FitTables(lik={lv: {} for lv in rm.LIK_LEVELS}, prior={k: {} for k in ("P4", "P3", "P2", "P1", "P0")},
                         field={})
    assert empty.prior_for("2", "unopened", "BTN", "raise", CFG) == (None, "")


def test_expansion_removes_dead_cards_and_light_classes():
    prior = np.zeros(N_CLASSES)
    prior[class_index("AKs")] = 0.6
    prior[class_index("72o")] = 0.3995
    prior[class_index("32o")] = 0.0005                                    # under the class floor
    dead = parse_cards("Ah Kd 2c")
    for mode in ("class_size", "live_combos"):
        ids, w = rm.expand_prior(prior, dead, dataclasses.replace(CFG, combo_weight=mode))
        labels = np.array([CLASS_LABELS[COMBO_TO_CLASS[i]] for i in ids])
        assert set(labels) == {"AKs", "72o"}
        # AcKc and AsKs; 72o: 3 deuces x 4 sevens - 3 blocked combos
        assert (labels == "AKs").sum() == 2 and (labels == "72o").sum() == 9
        aks, sev = w[labels == "AKs"], w[labels == "72o"]
        assert np.allclose(aks, aks[0]) and np.allclose(sev, sev[0])
        if mode == "live_combos":       # each class keeps its prior weight whatever the board blocks
            assert math.isclose(aks.sum(), 0.6 / 0.9995)
        else:                           # each combo keeps its own probability: 0.6 / 4 against 0.3995 / 12
            assert math.isclose(aks[0] / sev[0], (0.6 / 4) / (0.3995 / 12))
        assert math.isclose(w.sum(), 1.0)

def test_whole_posterior_keeps_every_live_combo_with_a_prior():
    prior = uniform_prior()
    prior[class_index("72o")] = 0.0                                   # never seen in this spot
    tables = make_tables(prior=prior)
    hero, board = parse_cards("Ah Kd"), parse_cards("7c 8d 2s Qh")
    nodes = [node("flop", "vs_cbet", "call", board[:3]), node("turn", "lead_opp", "bet", board)]
    res = rm.predict_range(tables, CFG_POSTERIOR, line(nodes), hero + board)
    dead = set(hero + board)
    live = [i for i, (a, b) in enumerate(COMBOS) if a not in dead and b not in dead
            and CLASS_LABELS[COMBO_TO_CLASS[i]] != "72o"]
    assert np.array_equal(res.ids, np.asarray(live)) and (res.w > 0).all()
    assert math.isclose(res.w.sum(), 1.0, abs_tol=1e-12)


def test_prune_cap_and_ties():
    ids = np.arange(10)
    w = np.array([5, 1, 3, 3, 3, 0.0001, 2, 3, 1, 4.0])
    cfg = dataclasses.replace(CFG, cap=4, prune_frac=1e-4)
    out_ids, out_w = rm.prune(ids, w, cfg)
    assert out_ids.tolist() == [0, 2, 3, 9]                               # ties at 3 keep the lowest ids
    cfg = dataclasses.replace(CFG, cap=100, prune_frac=0.1)
    assert 5 not in rm.prune(ids, w, cfg)[0].tolist()


def test_node_table_backoff_levels():
    tables = make_tables({("flop", "vs_cbet", "nothing"): [80, 0, 15, 0, 5]})
    tables.lik["K6"][("flop", "vs_cbet", "ip", "single_raised", "overpair", "none")] = np.array([0, 0, 20, 0, 5.0])
    probs, levels = tables.node_table("flop", "vs_cbet", "ip", "single_raised", CFG)
    nothing, overpair = MADE_CLASSES.index("nothing"), MADE_CLASSES.index("overpair")
    assert levels[nothing, 0] == rm.LIK_LEVELS.index("K3")
    assert math.isclose(probs[nothing, 0, 0], (80 + 1e-4) / (100 + 5e-4))
    assert levels[overpair, 0] == rm.LIK_LEVELS.index("K6")              # 25 actions: supported
    assert levels[overpair, 1] == rm.LIK_LEVELS.index("K1")
    assert np.allclose(probs.sum(axis=2), 1.0)
    probs, levels = tables.node_table("flop", "no_such_spot", "ip", "limped", CFG)
    assert (levels == len(rm.LIK_LEVELS)).all() and np.allclose(probs, 0.2)


# ------------------------------------------------------------------ views and equity


def test_class_grid_layout():
    w = np.zeros(N_CLASSES)
    for i, label in enumerate(CLASS_LABELS):
        w[i] = i
    grid, labels = rm.class_grid(w)
    assert labels[0][0] == "AA" and labels[0][1] == "AKs" and labels[1][0] == "AKo" and labels[12][12] == "22"
    assert labels[11][12] == "32s" and labels[12][11] == "32o"
    assert grid[0][1] == class_index("AKs")


def test_monte_carlo_multiway_matches_exact_for_known_hands():
    hero = parse_cards("Ah Kh")
    board = parse_cards("Qh 7s 2h")
    opps = [parse_cards("Qs Qd"), parse_cards("Jc Tc")]
    exact = multiway_equity([hero] + opps, board)[0]
    ranges = [(np.array([int(COMBO_INDEX[o[0], o[1]])]), np.array([1.0])) for o in opps]
    eq, se = rm.equity_vs_ranges_mc(hero, board, ranges, n_samples=20000, seed=3)
    assert abs(eq - exact) < 4 * se + 1e-3
    assert rm.equity_vs_ranges_mc(hero, board, ranges, n_samples=2000, seed=3) == \
        rm.equity_vs_ranges_mc(hero, board, ranges, n_samples=2000, seed=3)
    assert rm.row_seed(1, "h1", 5) == rm.row_seed(1, "h1", 5) != rm.row_seed(1, "h1", 6)


def test_score_chunk_end_to_end(tmp_path):
    """One staged chunk with two heads-up decisions through the worker, against direct computations."""
    import pickle

    k3 = {("flop", "vs_cbet", m): [30, 0, 60, 0, 10] for m in MADE_CLASSES}
    tables = make_tables(k3)
    with open(tmp_path / "fit.pkl", "wb") as fh:
        pickle.dump(tables, fh)
    dec = pd.DataFrame({
        "hand_id": ["h1", "h1"], "action_index": [9, 12], "player_id": ["A", "A"], "street": ["flop", "turn"],
        "hole_cards": ["AhKd", "AhKd"], "board": ["7c8d2s", "7c8d2sQh"], "n_live_opp": [1, 1],
        "action": ["call", "check"], "pot_odds": [0.25, None], "size_class": ["2", "2"],
    })
    opp = pd.DataFrame({
        "hand_id": ["h1", "h1"], "action_index": [9, 12], "opp_id": ["B", "B"], "opp_cards": ["QsJs", "QsJs"],
        "pf_sit": ["unopened", "unopened"], "pf_act": ["raise", "raise"], "position_class": ["BTN", "BTN"],
        "raised_pf": [True, True], "t_n": [100.0, 100.0], "t_vpip": [0.6, 0.6], "t_pfr": [0.4, 0.4],
        "t_agg": [0.4, 0.4],
    })
    nodes = pd.DataFrame({
        "hand_id": ["h1", "h1"], "action_index": [8, 10], "player_id": ["B", "B"], "street": ["flop", "turn"],
        "action": ["bet", "bet"], "situation": ["cbet_opp", "cbet_opp"], "pose": ["ip", "ip"],
        "pot_type": ["single_raised", "single_raised"], "board": ["7c8d2s", "7c8d2sQh"],
    })
    job = {}
    for name, df in (("dec", dec), ("opp", opp), ("nodes", nodes)):
        job[name] = str(tmp_path / f"{name}.parquet")
        df.to_parquet(job[name], index=False)
    job["out"] = str(tmp_path / "out.parquet")
    rm.init_worker(str(tmp_path / "fit.pkl"), {"range": dataclasses.asdict(CFG), "exact": True,
                                                "mc_samples": 1024, "seed": 1})
    stats = rm.score_chunk(job)
    out = pd.read_parquet(job["out"])
    assert stats["rows"] == 2 and stats["scored"] == 2
    assert out.eq_range_method.tolist() == ["exact", "exact"]
    assert (out.eq_range_se == 0).all()
    hero = parse_cards("Ah Kd")
    for k, board_text in enumerate(["7c 8d 2s", "7c 8d 2s Qh"]):
        board = parse_cards(board_text)
        n_used = 1 if k == 0 else 2
        ln = line([node("flop", "cbet_opp", "bet", board[:3]),
                   node("turn", "cbet_opp", "bet", parse_cards("7c 8d 2s Qh"))][:n_used],
                  raised_preflop=True, n=100.0, vpip=0.6, pfr=0.4, agg=0.4)
        res = rm.predict_range(tables, CFG, ln, hero + board)
        assert out.n_combos_opp.iat[k] == res.ids.size
        assert math.isclose(out.eq_vs_range.iat[k], equity_vs_range(hero, board, rm.range_dense(res.ids, res.w)),
                            abs_tol=1e-12)
        assert math.isclose(out.eq_vs_real.iat[k], equity_vs_hand(hero, parse_cards("Qs Js"), board), abs_tol=1e-12)
    # the first decision (index 9) sees only the flop bet (index 8), not the turn bet (index 10)
    assert out.range_top_class.notna().all()
