"""Step 04 end to end on a small warehouse: simulated hands (random legal agents) loaded by step 01, then
``run.py`` fit -> enrich -> leakage. Checks the GROUPING SETS levels against plain counts, the output tables'
invariants, the refusal to fit on showdown-only data, ``--fit-from``, and that the scores do not depend on the
number of worker processes or on a rerun."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest
from step04_testlib import REPO, STEP_DIR, load_script

from sudetect import warehouse

step01 = load_script("step01_run", REPO / "01_action_table" / "run.py")
testlib01 = load_script("step01_testlib", REPO / "01_action_table" / "tests" / "step01_testlib.py")
run04 = load_script("step04_run", STEP_DIR / "run.py")
QUIET = {"log": lambda message: None}
N_HANDS = 700


@pytest.fixture(scope="module")
def warehouses(tmp_path_factory):
    """Two warehouses of the same hands: full information ('t04') and showdown-only text ('t04sd')."""
    root = tmp_path_factory.mktemp("step04")
    mp = pytest.MonkeyPatch()
    mp.setenv("SU_DATA_DIR", str(root / "work"))
    for name, full in (("t04", True), ("t04sd", False)):
        texts = testlib01.simulated_hands(41, N_HANDS, 400001, "T41", full_information=full,
                                          start=datetime(2001, 1, 1, 10))
        folder = root / f"hh_{name}"
        folder.mkdir()
        (folder / "day.txt").write_text("\n".join(texts), encoding="utf-8")
        step01.load(name, [str(folder)], source="sim", batch_hands=500, **QUIET)
    assert run04.main(["--dataset", "t04", "--steps", "fit,enrich,leakage", "--procs", "1"]) == 0
    yield root
    mp.undo()


def _q(dataset: str, sql: str, **params) -> pd.DataFrame:
    con = warehouse.connect(dataset, read_only=True)
    try:
        return warehouse.query_df(con, sql, **params)
    finally:
        con.close()


def test_likelihood_levels_add_up(warehouses):
    lik = _q("t04", "SELECT * FROM rng_lik_post")
    total = _q("t04", """SELECT count(*) AS n FROM decisions WHERE street IN ('flop', 'turn', 'river')
                         AND made_hand IS NOT NULL AND action IN ('fold', 'check', 'call', 'bet', 'raise')""").n.iat[0]
    assert set(lik.level) == {"K6", "K5", "K4", "K3", "K2", "K1"}
    for level, g in lik.groupby("level"):
        assert g.n.sum() == total, level
    assert (lik[["c_fold", "c_check", "c_call", "c_bet", "c_raise"]].sum(axis=1) == lik.n).all()
    k1 = lik[lik.level == "K1"].set_index(["street", "situation"]).n
    k6 = lik[lik.level == "K6"].groupby(["street", "situation"]).n.sum()
    assert k6.sort_index().equals(k1.sort_index())
    k3 = lik[lik.level == "K3"]
    assert k3.pose.isna().all() and k3.pot_type.isna().all() and k3.draw.isna().all() and k3.made_hand.notna().all()
    k2 = lik[lik.level == "K2"]
    assert k2.made_hand.isna().all() and k2.pose.notna().all() and k2.pot_type.notna().all()


def test_prior_levels_add_up(warehouses):
    pri = _q("t04", "SELECT * FROM rng_prior_pre")
    seen = _q("t04", "SELECT count(*) AS n FROM hand_players WHERE saw_flop AND hand_class IS NOT NULL").n.iat[0]
    for level, g in pri.groupby("level"):
        assert g.n.sum() == seen, level
    assert set(pri[pri.level == "P4"].size_class) <= {"2", "3-6", "7+"}
    assert pri[pri.level == "P0"].preflop_situation.isna().all()
    assert set(pri.action.dropna()) <= {"check", "call", "raise"}


def test_tilt_tables(warehouses):
    field = _q("t04", "SELECT * FROM rng_field")
    tilt = _q("t04", "SELECT * FROM rng_player_tilt")
    # random seat names: nobody reaches the 300 hands a field baseline needs, so the tilts stay off
    assert list(field.columns) == ["size_class", "n_eligible", "n_clean", "field_vpip", "field_pfr", "field_agg"]
    assert (field.n_clean <= field.n_eligible).all()
    assert tilt.groupby("player_id").size().min() >= 2          # one row per size class played, plus 'all'
    assert tilt["agg"].dropna().between(0, 1).all() and tilt["vpip"].between(0, 1).all()
    assert (tilt.pfr <= tilt.vpip + 1e-12).all()
    assert set(tilt.size_class) <= {"2", "3-6", "7+", "all"}


def test_range_equity_rows(warehouses):
    out = _q("t04", "SELECT * FROM action_range_equity")
    staged = _q("t04", """SELECT count(*) AS n FROM decisions WHERE n_live = 2 AND
                          street IN ('flop', 'turn', 'river') AND hole_cards IS NOT NULL""").n.iat[0]
    assert len(out) == staged > 50
    assert out.eq_vs_range.between(0, 1).all()
    assert (out.n_combos_opp > 0).all() and (out.n_combos_opp <= int(run04.load_config()["range"]["cap"])).all()
    assert (out.eq_range_method == "exact").all() and (out.eq_range_se == 0).all()
    assert (out.n_live_opp == 1).all()
    assert out.range_top_class.notna().all()
    assert ((out.range_strong_share + out.range_air_share) <= 1 + 1e-9).all()
    assert not out.duplicated(["hand_id", "action_index"]).any()


def test_leak_matrix_and_summary(warehouses):
    mat = _q("t04", "SELECT * FROM rng_leak_matrix")
    assert set(mat.real_class) <= {"dead", "mid", "win"} and mat.range_bin.between(0, 9).all()
    assert (mat.folds <= mat.n).all()
    summary = json.loads((warehouses / "work" / "t04_step04" / "summary.json").read_text(encoding="utf-8"))
    assert summary["enrich"]["decisions_scored"] == len(_q("t04", "SELECT * FROM action_range_equity"))
    assert summary["leakage"]["split_all_postflop"]["n"] == int(mat.n.sum())


def test_scores_do_not_depend_on_processes_or_reruns(warehouses):
    before = _q("t04", "SELECT * FROM action_range_equity ORDER BY hand_id, action_index")
    assert run04.main(["--dataset", "t04", "--steps", "enrich", "--procs", "2", "--restart"]) == 0
    after = _q("t04", "SELECT * FROM action_range_equity ORDER BY hand_id, action_index")
    pd.testing.assert_frame_equal(before, after)


def test_showdown_only_data_needs_fit_from(warehouses):
    with pytest.raises(SystemExit):
        run04.main(["--dataset", "t04sd", "--steps", "fit", "--procs", "1"])
    assert run04.main(["--dataset", "t04sd", "--fit-from", "t04", "--steps", "fit,enrich", "--procs", "1"]) == 0
    a = _q("t04", "SELECT * FROM rng_lik_post ORDER BY level, street, situation, pose, pot_type, made_hand, draw")
    b = _q("t04sd", "SELECT * FROM rng_lik_post ORDER BY level, street, situation, pose, pot_type, made_hand, draw")
    pd.testing.assert_frame_equal(a, b)
    meta = _q("t04sd", "SELECT * FROM rng_meta").set_index("key").value
    assert meta["fit_dataset"] == "t04"
    rows = _q("t04sd", "SELECT count(*) AS n FROM action_range_equity").n.iat[0]
    known = _q("t04sd", """SELECT count(*) AS n FROM decisions WHERE n_live = 2
                           AND street IN ('flop', 'turn', 'river')
                           AND hole_cards IS NOT NULL""").n.iat[0]
    assert rows == known


def test_variant_files_build_and_state_every_range_setting():
    from sudetect.rangemodel import RangeConfig

    files = sorted((STEP_DIR / "variants").glob("*.toml"))
    assert {f.stem for f in files} >= {"cap240", "observer_fit"}
    for f in files:
        cfg = run04.load_config(f)
        RangeConfig.from_toml(cfg)
        text = f.read_text(encoding="utf-8")
        for key in ("class_floor", "cap", "prune_frac", "prune_each_step", "combo_weight", "sizing"):
            assert f"\n{key} = " in text, (f.name, key)


def test_library_defaults_are_the_config_file():
    from sudetect.rangemodel import RangeConfig

    assert RangeConfig() == RangeConfig.from_toml(run04.load_config())



def test_preflop_range_equity(warehouses):
    pre04 = load_script("step04_preflop", STEP_DIR / "preflop.py")
    out = pre04.run("t04", run04.load_config(), log=lambda _m: None)
    n = _q("t04", """SELECT count(*) AS n FROM decisions WHERE street = 'preflop' AND n_live = 2
                     AND hole_cards IS NOT NULL""").n.iat[0]
    assert len(out) == n > 20
    assert out.eq_vs_range.between(0, 1).all()
    acted = out.opp_situation.notna()
    assert (out.prior_level[~acted] == "random").all()
    assert out.prior_level[acted].isin(["P4", "P3", "P2", "P1", "P1m", "P0"]).all()
    assert _q("t04", "SELECT count(*) AS n FROM action_range_equity_pre").n.iat[0] == len(out)
