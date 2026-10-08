"""Step 02 end to end on a tiny warehouse: values against direct library calls, NULL rules, reruns, resuming."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import DATASET, STEP_DIR, load_module, step02

from sudetect import warehouse
from sudetect.cards import parse_cards
from sudetect.equity import (
    equity_vs_hand,
    equity_vs_hand_batch,
    equity_vs_random,
    equity_vs_random_batch,
    multiway_equity,
)
from sudetect.evaluator import eval_hand


def settings(**overrides):
    s = step02().Settings(procs=1, identity_sample=0, lock_wait_s=0)
    return step02().replace(s, **overrides) if overrides else s


def run(**overrides):
    return step02().run(DATASET, settings=settings(**overrides), log=lambda _msg: None)


def table(name: str, order: str) -> pd.DataFrame:
    con = warehouse.connect(DATASET, read_only=True)
    try:
        return warehouse.query_df(con, f"SELECT * FROM {name} ORDER BY {order}")
    finally:
        con.close()


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return table("action_equity", "hand_id, action_index"), table("hand_equity", "hand_id, player_id")


def decisions_with_cards() -> pd.DataFrame:
    """Every decision with the actor's cards and each live opponent's cards, read without step-02 SQL."""
    con = warehouse.connect(DATASET, read_only=True)
    try:
        d = warehouse.query_df(con, "SELECT hand_id, action_index, street, hole_cards, board_at_action, n_live, "
                                    "live_opponents FROM decisions ORDER BY hand_id, action_index")
        hp = warehouse.query_df(con, "SELECT hand_id, player_id, hole_cards FROM hand_players")
    finally:
        con.close()
    cards = {(r.hand_id, r.player_id): r.hole_cards for r in hp.itertuples(index=False)}
    d["opp_cards"] = [[c if isinstance(c := cards.get((h, o["player_id"])), str) else None for o in opps]
                      for h, opps in zip(d["hand_id"], d["live_opponents"])]
    return d


# ---------------------------------------------------------------------------------------------- the run


def test_run_builds_tables_views_and_passes_checks(tiny):
    result = run()
    assert result.violations == 0, result.checks[result.checks["violations"] > 0]
    d = decisions_with_cards()
    scored = d[d["hole_cards"].notna() & (d["street"] != "preflop")]
    eq, he = frames()
    assert len(eq) == len(scored) > 500
    assert set(eq["street"]) == {"flop", "turn", "river"}
    con = warehouse.connect(DATASET, read_only=True)
    try:
        for view in ("actions_eq", "decisions_eq", "hu_decisions_eq", "hand_players_eq"):
            assert warehouse.table_exists(con, view)
        n = warehouse.query_df(con, "SELECT count(*) AS n, count(eq_vs_random) AS k FROM actions_eq").iloc[0]
        assert n["k"] == len(eq) and n["n"] > n["k"]
        hp = warehouse.query_df(con, "SELECT count(preflop_eq_vs_opp) AS k FROM hand_players_eq").iloc[0]
        assert hp["k"] == len(he) > 50
    finally:
        con.close()


def test_values_match_direct_library_calls(tiny):
    run()
    s = settings()
    eq, he = frames()
    d = decisions_with_cards().merge(eq, on=["hand_id", "action_index", "street"])
    worst_exact, worst_z = 0.0, 0.0
    for r in d.itertuples(index=False):
        hero, board = parse_cards(r.hole_cards), parse_cards(r.board_at_action)
        # equity vs one random hand: the exact single-spot value, and the same seeded batch call
        exact_random = equity_vs_random(hero, board)
        if r.eq_method == "exact":
            worst_exact = max(worst_exact, abs(r.eq_vs_random - exact_random))
        else:
            assert r.street == "flop" and r.eq_se > 0 or r.eq_vs_random in (0.0, 1.0)
            worst_z = max(worst_z, abs(r.eq_vs_random - exact_random) / max(r.eq_se, 1e-9))
            pad = np.array([board + [-1, -1]])
            again = equity_vs_random_batch(np.array([hero]), pad, n_samples=s.mc_samples, seed=s.seed)[0]
            assert again == r.eq_vs_random
        # equity vs the real cards of every live opponent
        known = all(c is not None for c in r.opp_cards)
        assert bool(r.eq_real_known) == known
        assert r.n_live_opp == r.n_live - 1 == len(r.opp_cards)
        if not known:
            assert pd.isna(r.eq_vs_real) and pd.isna(r.ahead_now)
            continue
        opps = [parse_cards(c) for c in r.opp_cards]
        want = equity_vs_hand(hero, opps[0], board) if len(opps) == 1 else multiway_equity([hero] + opps, board)[0]
        worst_exact = max(worst_exact, abs(r.eq_vs_real - want))
        mine = eval_hand(hero + board)
        assert bool(r.ahead_now) == all(mine > eval_hand(o + board) for o in opps)
    assert worst_exact <= 1e-9
    assert worst_z < 4.5
    # hand_equity: the seeded batch call gives the same numbers; player B holds 1 - A
    con = warehouse.connect(DATASET, read_only=True)
    try:
        hands = warehouse.query_df(con, (step02().SQL_DIR / "inputs_hands.sql").read_text(),
                                   lo="0", hi="999999999")
    finally:
        con.close()
    a = np.array([parse_cards(c) for c in hands["cards_a"]])
    b = np.array([parse_cards(c) for c in hands["cards_b"]])
    want = equity_vs_hand_batch(a, b, np.full((len(a), 5), -1), n_samples=s.hand_equity_samples, seed=s.seed)
    got = he.set_index(["hand_id", "player_id"])["preflop_eq_vs_opp"]
    assert np.array_equal(got.loc[list(zip(hands["hand_id"], hands["player_a"]))].to_numpy(), want)
    assert np.allclose(got.loc[list(zip(hands["hand_id"], hands["player_b"]))].to_numpy(), 1 - want, atol=1e-15)
    for i in range(3):     # and the Monte Carlo value sits near the exact preflop equity
        se = math.sqrt(want[i] * (1 - want[i]) / s.hand_equity_samples)
        assert abs(want[i] - equity_vs_hand(a[i].tolist(), b[i].tolist())) < 4.5 * se + 1e-9


def test_null_rules_on_observed_hands(tiny):
    run()
    eq, _ = frames()
    d = decisions_with_cards()
    post = d[d["street"] != "preflop"]
    assert post["hole_cards"].isna().sum() > 0                      # observed hands hide most hole cards
    merged = post.merge(eq, on=["hand_id", "action_index"], how="left", indicator=True)
    assert (merged["_merge"] == "both").tolist() == post["hole_cards"].notna().tolist()
    rows = merged[merged["_merge"] == "both"]
    unknown = rows[~rows["eq_real_known"].astype(bool)]
    assert len(unknown) > 10                                        # an opponent's cards were never shown
    assert unknown["eq_vs_real"].isna().all() and unknown["ahead_now"].isna().all()
    assert unknown["eq_real_method"].isna().all() and unknown["eq_vs_random"].notna().all()
    known = rows[rows["eq_real_known"].astype(bool)]
    assert known["eq_vs_real"].notna().all() and known["ahead_now"].notna().all()
    assert (known["eq_real_method"] == "exact").all() and (known["eq_real_se"] == 0).all()


def test_preflop_rows_on_request(tiny):
    result = run(include_preflop=True)
    assert result.violations == 0
    eq, _ = frames()
    pre = eq[eq["street"] == "preflop"]
    assert len(pre) > 300
    assert (pre["eq_method"] == "exact").all() and (pre["eq_se"] == 0).all()
    known = pre[pre["eq_real_known"]]
    assert (known["eq_real_method"] == "mc").all() and known["eq_real_se"].between(0, 0.5 / 64).all()
    assert (known["n_live_opp"] >= 2).any() and pre["ahead_now"].isna().all()


# ---------------------------------------------------------------------------------------------- reruns


def test_rerun_is_idempotent_and_reuses_parts(tiny):
    first = run(batch_rows=300)
    tables = frames()
    second = run(batch_rows=300)
    assert second.stats["decisions"].parts_computed == 0
    assert second.stats["decisions"].parts_reused == first.stats["decisions"].parts > 1
    assert second.stats["hands"].parts_computed == 0
    for old, new in zip(tables, frames()):
        pd.testing.assert_frame_equal(old, new)
    step02().run(DATASET, settings=settings(batch_rows=300), fresh=True, log=lambda _msg: None)
    for old, new in zip(tables, frames()):
        pd.testing.assert_frame_equal(old, new)


def test_interrupted_run_resumes_and_batching_does_not_change_values(tiny):
    reference = run()                                   # one part per kind
    ref_tables = frames()
    stage = tiny / DATASET / settings().stage_dir
    for p in stage.glob("*.parquet"):
        p.unlink()
    with_stop = step02().run(DATASET, settings=settings(batch_rows=150), stop_after=3, log=lambda _msg: None)
    assert with_stop.stats["decisions"].parts_computed == 3
    assert len(list(stage.glob("decisions_*.parquet"))) == 3
    resumed = run(batch_rows=150)
    s = resumed.stats["decisions"]
    assert s.parts_reused == 3 and s.parts_computed == s.parts - 3 > 0
    for old, new in zip(ref_tables, frames()):
        pd.testing.assert_frame_equal(old, new)
    # a lost part is recomputed alone
    (stage / "decisions_00001.parquet").unlink()
    again = run(batch_rows=150)
    assert again.stats["decisions"].parts_computed == 1
    assert reference.violations == resumed.violations == again.violations == 0


def test_new_seed_recomputes_and_changes_only_monte_carlo_values(tiny):
    run()
    old, _ = frames()
    result = run(seed=12345)
    assert result.stats["decisions"].parts_reused == 0
    new, _ = frames()
    mc = old["eq_method"] == "mc"
    assert (old.loc[mc, "eq_vs_random"] != new.loc[mc, "eq_vs_random"]).mean() > 0.9
    pd.testing.assert_series_equal(old.loc[~mc, "eq_vs_random"], new.loc[~mc, "eq_vs_random"])
    pd.testing.assert_series_equal(old["eq_vs_real"], new["eq_vs_real"])


def test_process_pool_gives_the_same_tables(tiny):
    run()
    single = frames()
    stage = tiny / DATASET / settings().stage_dir
    for p in stage.glob("*.parquet"):
        p.unlink()
    pooled = run(procs=2, chunk_rows=200)
    assert pooled.violations == 0
    for old, new in zip(single, frames()):
        pd.testing.assert_frame_equal(old, new)


def test_pyrust_backend_run(tiny):
    pytest.importorskip("pyrust_poker")
    run()
    numpy_eq, numpy_he = frames()
    result = run(backend="pyrust")
    assert result.violations == 0
    eq, he = frames()
    assert (eq["eq_method"] == "exact").all() and (he["eq_method"] == "exact").all()
    assert np.allclose(eq["eq_vs_real"].fillna(-1), numpy_eq["eq_vs_real"].fillna(-1), atol=1e-7)
    exact = numpy_eq["eq_method"] == "exact"
    assert np.allclose(eq.loc[exact, "eq_vs_random"], numpy_eq.loc[exact, "eq_vs_random"], atol=1e-9)
    z = (eq["eq_vs_random"] - numpy_eq["eq_vs_random"]).abs() / numpy_eq["eq_se"].clip(lower=1e-9)
    assert z[~exact].max() < 4.5
    z = (he["preflop_eq_vs_opp"] - numpy_he["preflop_eq_vs_opp"]).abs() / numpy_he["eq_se"].clip(lower=1e-9)
    assert z.max() < 4.5


def test_cli(tiny, capsys):
    assert step02().main(["--dataset", DATASET, "--procs", "1", "--identity-sample", "40"]) == 0
    out = capsys.readouterr().out
    assert "checks: all" in out and "heads-up identity on 40 hands" in out


def test_missing_warehouse_is_an_error(tiny):
    with pytest.raises(RuntimeError, match="run 01_action_table"):
        step02().run("no_such_dataset", settings=settings(), log=lambda _msg: None)


def test_warehouse_is_free_while_rows_are_scored(tiny, monkeypatch):
    """No connection is open while rows are scored: a writer can open the file from inside every task."""
    eb = step02().eb
    real_task = eb.run_task
    kinds = []

    def probe(task):
        con = warehouse.connect(DATASET, read_only=False)   # fails while this process holds a read-only connection
        con.close()
        kinds.append(task[0])
        return real_task(task)

    monkeypatch.setattr(eb, "run_task", probe)
    assert run().violations == 0
    assert {"decisions", "hands"} <= set(kinds)


def test_connect_retries_a_locked_file_and_gives_up_after_lock_wait_s(tiny, monkeypatch):
    mod = step02()
    real_connect = mod.warehouse.connect
    calls, sleeps = [], []

    def locked_twice(dataset, read_only=False):
        calls.append(read_only)
        if len(calls) <= 2:
            raise OSError("IO Error: Could not set lock on file: Conflicting lock is held")
        return real_connect(dataset, read_only=read_only)

    monkeypatch.setattr(mod.warehouse, "connect", locked_twice)
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    mod.connect(DATASET, True, 60.0, log=lambda _msg: None).close()
    assert len(calls) == 3 and len(sleeps) == 2 and 0 < sleeps[0] < sleeps[1]

    calls.clear()
    with pytest.raises(OSError, match="lock"):
        mod.connect(DATASET, True, 0.0, log=lambda _msg: None)
    assert len(calls) == 1


def test_examples_script_prints_stored_values_next_to_direct_calls(tiny):

    run()
    examples = load_module("step02_examples", STEP_DIR / "examples.py")
    lines = []
    shown = examples.show_dataset(DATASET, 1, log=lines.append)
    text = "\n".join(lines)
    assert shown >= 4 and text.count("stored in action_equity") == shown
    for ex in examples.fetch(DATASET, 1):
        if ex.stored is not None and ex.stored.eq_method == "exact":
            hero, board = parse_cards(ex.row.hole_cards), parse_cards(ex.row.board_at_action)
            assert abs(ex.stored.eq_vs_random - equity_vs_random(hero, board)) < 1e-9


def test_library_code_change_recomputes_every_part(tiny, monkeypatch):
    first = run()
    monkeypatch.setattr(step02(), "CODE_HASH", "0" * 16)
    again = run()
    for kind in ("decisions", "hands"):
        assert again.stats[kind].parts_reused == 0
        assert again.stats[kind].parts_computed == first.stats[kind].parts > 0
