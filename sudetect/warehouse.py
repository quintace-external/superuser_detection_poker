"""The warehouse: a thin adapter around one DuckDB file per dataset.

Every pipeline step reads tables and writes tables through this module, and
it is the only module that imports :mod:`duckdb`; pointing the same SQL files
at another engine means replacing these few functions.

* Data lives in ``$SU_DATA_DIR/<dataset>.duckdb`` (default: ``<repo>/work``).
* SQL files use named parameters written ``:name``; :func:`run_sql_file`
  turns them into DuckDB's ``$name`` and passes only the parameters a
  statement uses. ``::`` casts and text inside quotes or comments are left alone.

Example::

    con = connect("article_sample")
    run_sql_file(con, "01_action_table/sql/schema.sql")
    insert_df(con, "hands", frame, mode="append")   # into the table schema.sql created
    df = query_df(con, "SELECT count(*) AS n FROM hands WHERE hand_date = :d", d="2001-01-01")
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import pandas as pd

__all__ = [
    "REPO_ROOT",
    "data_dir",
    "db_path",
    "connect",
    "split_statements",
    "bind_named",
    "run_sql",
    "run_sql_file",
    "query_df",
    "execute",
    "insert_df",
    "table_exists",
    "table_columns",
    "drop_tables",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
"""The repository root (the folder that holds ``sudetect/``)."""

Connection = duckdb.DuckDBPyConnection


def data_dir() -> Path:
    """Folder of the warehouse files: ``$SU_DATA_DIR``, or ``<repo>/work``."""
    env = os.environ.get("SU_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "work"


def db_path(dataset: str) -> Path:
    """Path of a dataset's DuckDB file."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset):
        raise ValueError(f"dataset names use letters, digits, '_', '.' and '-': {dataset!r}")
    return data_dir() / f"{dataset}.duckdb"


def connect(dataset: str, read_only: bool = False) -> Connection:
    """Open (and create when needed) the warehouse of one dataset; ``":memory:"`` opens a scratch one."""
    if dataset == ":memory:":
        return duckdb.connect(":memory:")
    path = db_path(dataset)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


# ------------------------------------------------------------------ SQL text

_TOKEN_RE = re.compile(
    r"""(?P<sq>'(?:[^']|'')*')"""        # 'string'
    r"""|(?P<dq>"(?:[^"]|"")*")"""       # "identifier"
    r"""|(?P<lc>--[^\n]*)"""             # -- comment
    r"""|(?P<bc>/\*.*?\*/)"""            # /* comment */
    r"""|(?P<semi>;)"""
    r"""|(?P<param>(?<![:\w]):(?P<name>[A-Za-z_]\w*))""",
    re.S,
)


def split_statements(sql: str) -> list[str]:
    """Split a script into statements at semicolons outside quotes and comments; empty ones are dropped."""
    out: list[str] = []
    start = 0
    for m in _TOKEN_RE.finditer(sql):
        if m.lastgroup == "semi":
            out.append(sql[start:m.start()])
            start = m.end()
    out.append(sql[start:])
    return [s.strip() for s in out if _has_code(s)]


def _has_code(statement: str) -> bool:
    stripped = _TOKEN_RE.sub(lambda m: "" if m.lastgroup in ("lc", "bc") else m.group(0), statement)
    return bool(stripped.strip())


def bind_named(sql: str) -> tuple[str, list[str]]:
    """``:name`` -> ``$name`` outside quotes and comments; returns the SQL and the names used."""
    names: list[str] = []

    def sub(m: re.Match) -> str:
        if m.lastgroup == "param":
            names.append(m.group("name"))
            return "$" + m.group("name")
        return m.group(0)

    return _TOKEN_RE.sub(sub, sql), names


def _execute(con: Connection, statement: str, params: dict[str, Any]) -> Any:
    text, names = bind_named(statement)
    missing = [x for x in names if x not in params]
    if missing:
        raise KeyError(f"SQL parameters not given: {sorted(set(missing))}")
    used = {x: params[x] for x in dict.fromkeys(names)}
    return con.execute(text, used) if used else con.execute(text)


_QUERY_WORDS = frozenset({"SELECT", "WITH", "VALUES", "FROM", "TABLE", "DESCRIBE", "SHOW", "PRAGMA",
                          "EXPLAIN", "SUMMARIZE"})


def _first_keyword(statement: str) -> str:
    text = _TOKEN_RE.sub(lambda m: " " if m.lastgroup in ("lc", "bc") else m.group(0), statement).strip()
    return text.lstrip("(").split(None, 1)[0].upper() if text else ""


def run_sql(con: Connection, sql: str, **params: Any) -> Optional[pd.DataFrame]:
    """Run a script of one or more statements.

    Returns the rows of the last statement when it is a query (``SELECT``,
    ``WITH``, ...), else None.
    """
    result: Optional[pd.DataFrame] = None
    for statement in split_statements(sql):
        cur = _execute(con, statement, params)
        result = cur.fetchdf() if _first_keyword(statement) in _QUERY_WORDS else None
    return result


def run_sql_file(con: Connection, path: Union[str, os.PathLike[str]], **params: Any) -> Optional[pd.DataFrame]:
    """Run a ``.sql`` file (see :func:`run_sql`); relative paths are taken from the repository root."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    return run_sql(con, p.read_text(encoding="utf-8"), **params)


def query_df(con: Connection, sql: str, **params: Any) -> pd.DataFrame:
    """Run one query and return its rows as a DataFrame."""
    cur = _execute(con, sql, params)
    return cur.fetchdf()


def execute(con: Connection, sql: str, **params: Any) -> None:
    """Run one statement that returns no rows."""
    _execute(con, sql, params)


# ------------------------------------------------------------------ tables


def insert_df(con: Connection, table: str, df: Any, mode: str = "replace") -> int:
    """Write a pandas DataFrame or a pyarrow Table into ``table``; returns the number of rows.

    ``mode="replace"`` (the default) recreates the table from the frame's own
    columns and types. ``mode="append"`` inserts by column name into an
    existing table (created by a schema file), so values are cast to its
    declared types; a missing table is created from the frame.
    """
    if mode not in ("append", "replace"):
        raise ValueError("mode must be 'append' or 'replace'")
    _check_identifier(table)
    view = "_insert_source"
    con.register(view, df)
    try:
        if mode == "replace" or not table_exists(con, table):
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {view}")
        else:
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM {view}")
    finally:
        con.unregister(view)
    return len(df)


def table_exists(con: Connection, name: str) -> bool:
    """True if a table or view of that name exists in the current schema."""
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = $name", {"name": name}
    ).fetchone()
    return bool(row and row[0])


def table_columns(con: Connection, name: str) -> list[str]:
    """Column names of a table, in order."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = $name ORDER BY ordinal_position",
        {"name": name},
    ).fetchall()
    return [r[0] for r in rows]


def drop_tables(con: Connection, names: Iterable[str]) -> None:
    """Drop tables if they exist."""
    for name in names:
        _check_identifier(name)
        con.execute(f"DROP TABLE IF EXISTS {name}")


def _check_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"not a plain table name: {name!r}")
