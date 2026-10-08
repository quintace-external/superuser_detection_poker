"""End to end through run.py: folders, zip members and globs -> the warehouse.

Checks the funnel (every hand counted once, by reason), duplicates, the views,
the table order, the invariants (and that they do catch a corrupted row),
incremental loads, salted ids and the schema contract between the SQL file and
the replay's columns.
"""

from __future__ import annotations

import re
import zipfile
from datetime import date, datetime

import pytest
from step01_testlib import hand, load_run_module, simulated_hands

from sudetect import warehouse
from sudetect.replay import ACTION_COLUMNS, HAND_COLUMNS, PLAYER_COLUMNS

run = load_run_module()
QUIET = {"log": lambda message: None}

BAD = hand("""
    Hand #999999: Hold'em No Limit (0.50/1.00) - 2001/01/01 23:00:00 UTC
    Table 'T09' 2-max Seat #1 is the button
    Seat 1: X (100.00 in chips)
    Seat 2: Y (100.00 in chips)
    X: posts small blind 0.50
    Y: posts big blind 1.00
    *** HOLE CARDS ***
    X: raises 2.00 to 3.00
    Y: folds
    Uncalled bet (2.00) returned to X
    X collected 1.99 from pot
    *** SUMMARY ***
    Total pot 2.00 | Rake 0.00
""")


@pytest.fixture
def inputs(tmp_path):
    day1 = simulated_hands(1, 150, 100001, "T01", start=datetime(2001, 1, 1, 10))
    day2 = simulated_hands(2, 120, 200001, "T02", start=datetime(2001, 1, 2, 10))
    folder = tmp_path / "hh"
    folder.mkdir()
    (folder / "day1.txt").write_text("\n".join(day1 + [BAD]), encoding="utf-8")
    same = day1[0]
    different = day1[1].replace("Table 'T01'", "Table 'T01-copy'")
    with zipfile.ZipFile(folder / "day2.zip", "w") as archive:
        archive.writestr("a/day2a.txt", "\n".join(day2[:60]))
        archive.writestr("a/day2b.txt", "\n".join(day2[60:] + [same, different]))
        archive.writestr("notes.md", "not hand histories")
    return folder


def test_folder_with_a_zip_loads_every_hand_once(data_dir, inputs):
    result = run.load("t_load", [str(inputs)], source="sim", batch_hands=100, **QUIET)
    assert result.reasons() == {"loaded": 270, "money_unreconciled": 1, "duplicate": 1, "duplicate_diff": 1}
    assert result.violations == 0 and not result.replay_errors
    con = warehouse.connect("t_load", read_only=True)
    try:
        q = lambda sql: warehouse.query_df(con, sql)  # noqa: E731
        assert q("SELECT count(*) AS n FROM hands")["n"][0] == 270
        funnel = q("SELECT file, reason, n_hands FROM parse_funnel ORDER BY file, reason")
        assert set(funnel["file"]) == {"day1.txt", "day2.zip:a/day2a.txt", "day2.zip:a/day2b.txt"}
        assert funnel["n_hands"].sum() == 273
        seq = q("SELECT table_id, min(table_seq) AS lo, max(table_seq) AS hi, count(*) AS n "
                "FROM hands GROUP BY table_id")
        assert (seq["lo"] == 1).all() and (seq["hi"] == seq["n"]).all()
        dates = set(q("SELECT DISTINCT hand_date FROM hands")["hand_date"].dt.date)
        assert dates == {date(2001, 1, 1), date(2001, 1, 2)}
        n_dec = q("SELECT count(*) AS n FROM actions WHERE is_decision AND NOT free_fold AND NOT error_flag")["n"][0]
        assert q("SELECT count(*) AS n FROM decisions")["n"][0] == n_dec > 0
        hu = q("SELECT count(*) AS n, count(*) FILTER (WHERE NOT is_hu_table OR n_live <> 2) AS bad "
               "FROM hu_decisions")
        assert hu["n"][0] > 0 and hu["bad"][0] == 0
        assert set(q("SELECT DISTINCT source FROM hands")["source"]) == {"sim"}
        assert (q("SELECT count(*) AS n FROM hand_players")["n"][0]
                == q("SELECT sum(n_dealt) AS n FROM hands")["n"][0])
    finally:
        con.close()


def test_invariants_catch_a_corrupted_row(data_dir, inputs):
    run.load("t_corrupt", [str(inputs)], **QUIET)
    con = warehouse.connect("t_corrupt")
    try:
        warehouse.execute(con, "UPDATE actions SET pot_before_bb = pot_before_bb + 1 "
                               "WHERE action_index = 3 AND hand_id = (SELECT min(hand_id) FROM hands)")
        warehouse.execute(con, "UPDATE hand_players SET net_bb = net_bb + 0.5 "
                               "WHERE hand_id = (SELECT max(hand_id) FROM hands)")
        inv = warehouse.run_sql_file(con, run.SQL_DIR / "invariants.sql", tol=1e-6).set_index("check_name")
    finally:
        con.close()
    assert inv.loc["pot_before_accumulates", "violations"] >= 1
    assert inv.loc["money_identity", "violations"] == 1
    assert inv["violations"].sum() >= 2


def test_incremental_load_replaces_whole_days(data_dir, inputs):
    full = run.load("t_inc", [str(inputs)], **QUIET)
    again = run.load("t_inc", [str(inputs)], incremental=True, **QUIET)
    assert again.violations == 0 and full.reasons() == again.reasons()
    only_day1 = run.load("t_inc", [str(inputs / "day1.txt")], incremental=True, **QUIET)
    assert only_day1.violations == 0
    con = warehouse.connect("t_inc", read_only=True)
    try:
        per_day = warehouse.query_df(con, "SELECT hand_date, count(*) AS n FROM hands GROUP BY 1 ORDER BY 1")
        funnel = warehouse.query_df(con, "SELECT sum(n_hands) AS n FROM parse_funnel")
    finally:
        con.close()
    assert per_day["n"].tolist() == [150, 120]
    assert funnel["n"][0] == 273


def test_glob_and_limit(data_dir, inputs):
    result = run.load("t_glob", [str(inputs / "*.txt")], **QUIET)
    assert result.reasons()["loaded"] == 150
    result = run.load("t_glob", [str(inputs)], limit=40, **QUIET)
    assert result.hands_loaded == 40 and result.violations == 0


def test_hash_ids_need_the_salt_and_hide_every_name(data_dir, inputs, monkeypatch, capsys):
    monkeypatch.delenv("SU_ID_SALT", raising=False)
    assert run.main(["--dataset", "t_hash", "--input", str(inputs), "--hash-ids"]) == 2
    monkeypatch.setenv("SU_ID_SALT", "a-test-salt")
    assert run.main(["--dataset", "t_hash", "--input", str(inputs), "--hash-ids"]) == 0
    assert "all 20 checks pass" in capsys.readouterr().out

    def ids() -> tuple[set, set, set]:
        con = warehouse.connect("t_hash", read_only=True)
        try:
            players = set(warehouse.query_df(con, "SELECT DISTINCT player_id FROM actions")["player_id"])
            hands = set(warehouse.query_df(con, "SELECT hand_id FROM hands")["hand_id"])
            tables = set(warehouse.query_df(con, "SELECT DISTINCT table_id FROM hands")["table_id"])
        finally:
            con.close()
        return players, hands, tables

    players, hands, tables = ids()
    assert all(re.fullmatch(r"P_[0-9a-f]{12}", p) for p in players)
    assert all(re.fullmatch(r"H_[0-9a-f]{16}", h) for h in hands)
    assert all(re.fullmatch(r"T_[0-9a-f]{12}", t) for t in tables) and len(tables) == 2
    assert run.main(["--dataset", "t_hash", "--input", str(inputs), "--hash-ids"]) == 0
    assert ids() == (players, hands, tables)
    monkeypatch.setenv("SU_ID_SALT", "another-salt")
    assert run.main(["--dataset", "t_hash", "--input", str(inputs), "--hash-ids"]) == 0
    assert not (ids()[0] & players)


def test_missing_input_is_an_error(data_dir, tmp_path):
    assert run.main(["--dataset", "t_missing", "--input", str(tmp_path / "nothing-here")]) == 2


def test_schema_matches_the_replay_columns():
    con = warehouse.connect(":memory:")
    try:
        warehouse.run_sql_file(con, run.SQL_DIR / "schema.sql")
        for table, columns in (("hands", HAND_COLUMNS), ("hand_players", PLAYER_COLUMNS),
                               ("actions", ACTION_COLUMNS)):
            assert warehouse.table_columns(con, table) == [c[0] for c in columns], table
        assert warehouse.table_columns(con, "parse_funnel") == ["dataset", "file", "reason", "n_hands"]
    finally:
        con.close()


def test_default_source_from_the_dataset_name():
    assert run.default_source("fixture") == "fixture"
    assert run.default_source("sim_small") == "sim" and run.default_source("sim_demo") == "sim"
    assert run.default_source("mined") == "mined"


def test_kept_bomb_pots_stay_out_of_the_decision_views(data_dir, tmp_path):
    from test_step01_cases_synthetic import CASE_A, CASE_K

    folder = tmp_path / "bomb"
    folder.mkdir()
    (folder / "hands.txt").write_text(hand(CASE_A) + "\n" + hand(CASE_K), encoding="utf-8")
    cfg = run.load_config()
    cfg["step01"] = dict(cfg["step01"], keep_bomb_pots=True)
    result = run.load("t_bomb", [str(folder)], config=cfg, **QUIET)
    assert result.reasons() == {"loaded": 2} and result.violations == 0
    con = warehouse.connect("t_bomb", read_only=True)
    try:
        per_hand = warehouse.query_df(con, "SELECT hand_id, count(*) AS n FROM decisions GROUP BY 1")
    finally:
        con.close()
    assert per_hand["hand_id"].tolist() == ["1001"]
