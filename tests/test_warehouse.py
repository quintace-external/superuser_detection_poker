"""Tests for ``sudetect.warehouse``: SQL scripts with named parameters, inserts, the data folder."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from sudetect import warehouse


def test_split_statements_respects_quotes_and_comments():
    sql = """
        -- a comment; with a semicolon
        CREATE TABLE t (x VARCHAR);   /* block ; comment */
        INSERT INTO t VALUES ('a;b'), ("x"";y");
        ;
        SELECT * FROM t
    """
    parts = warehouse.split_statements(sql)
    assert len(parts) == 3
    assert parts[0].endswith("CREATE TABLE t (x VARCHAR)")
    assert "INSERT INTO t" in parts[1] and "'a;b'" in parts[1] and parts[2] == "SELECT * FROM t"


def test_bind_named_leaves_casts_strings_and_comments_alone():
    text, names = warehouse.bind_named(
        "SELECT x::VARCHAR, ':not' AS s, -:tol AS t -- :ignored\nFROM y WHERE d = :day AND z IN (:day)")
    assert names == ["tol", "day", "day"]
    assert "x::VARCHAR" in text and "':not'" in text and "-$tol" in text and "-- :ignored" in text
    assert text.endswith("WHERE d = $day AND z IN ($day)")


def test_run_sql_returns_the_last_result_and_checks_parameters():
    con = warehouse.connect(":memory:")
    try:
        out = warehouse.run_sql(con, "CREATE TABLE t AS SELECT 1 AS a; SELECT a + :k AS b FROM t", k=41)
        assert out["b"].tolist() == [42]
        assert warehouse.run_sql(con, "INSERT INTO t VALUES (2)") is None
        with pytest.raises(KeyError):
            warehouse.run_sql(con, "SELECT :missing")
        assert warehouse.query_df(con, "SELECT count(*) AS n FROM t WHERE a >= :lo", lo=1)["n"][0] == 2
    finally:
        con.close()


def test_insert_append_casts_to_the_declared_types_and_replace_recreates():
    con = warehouse.connect(":memory:")
    try:
        warehouse.execute(con, "CREATE TABLE t (a SMALLINT, b VARCHAR)")
        n = warehouse.insert_df(con, "t", pd.DataFrame({"b": ["x", "y"], "a": [1, 2]}), mode="append")
        assert n == 2 and warehouse.table_columns(con, "t") == ["a", "b"]
        warehouse.insert_df(con, "t", pa.table({"a": pa.array([3], pa.int64()), "b": ["z"]}), mode="append")
        types = warehouse.query_df(con, "SELECT data_type FROM information_schema.columns "
                                        "WHERE table_name = 't' ORDER BY ordinal_position")
        assert types["data_type"].tolist() == ["SMALLINT", "VARCHAR"]
        assert warehouse.query_df(con, "SELECT sum(a) AS s FROM t")["s"][0] == 6
        warehouse.insert_df(con, "t", pd.DataFrame({"c": [1.5]}))  # replace is the default
        assert warehouse.table_columns(con, "t") == ["c"]
        assert warehouse.table_exists(con, "t") and not warehouse.table_exists(con, "nope")
        warehouse.insert_df(con, "fresh", pd.DataFrame({"q": [1]}))
        assert warehouse.table_exists(con, "fresh")
        warehouse.drop_tables(con, ["fresh", "never_was"])
        assert not warehouse.table_exists(con, "fresh")
        with pytest.raises(ValueError):
            warehouse.insert_df(con, "t; DROP TABLE t", pd.DataFrame({"c": [1.0]}))
    finally:
        con.close()


def test_data_dir_and_dataset_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SU_DATA_DIR", str(tmp_path / "w"))
    assert warehouse.data_dir() == tmp_path / "w"
    assert warehouse.db_path("sim_small") == tmp_path / "w" / "sim_small.duckdb"
    con = warehouse.connect("sim_small")
    warehouse.execute(con, "CREATE TABLE k AS SELECT 7 AS v")
    con.close()
    ro = warehouse.connect("sim_small", read_only=True)
    try:
        assert warehouse.query_df(ro, "SELECT v FROM k")["v"][0] == 7
    finally:
        ro.close()
    with pytest.raises(ValueError):
        warehouse.db_path("../escape")
    monkeypatch.delenv("SU_DATA_DIR")
    assert warehouse.data_dir() == warehouse.REPO_ROOT / "work"
