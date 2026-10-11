"""docs/HAND_HISTORY_FORMAT.md against the loader: its example hands parse and load in both views, its drop table
names every reason, input that cannot be loaded stops before the warehouse is touched, and the bundled sample the page
points to loads in full."""

from __future__ import annotations

import re
from pathlib import Path

from step01_testlib import load_run_module

from sudetect import warehouse
from sudetect.hhparse import GATE_REASONS, GateReject, parse_hand

run = load_run_module()
REPO = Path(__file__).resolve().parents[2]
QUIET = {"log": lambda message: None}
DOC = REPO / "docs" / "HAND_HISTORY_FORMAT.md"
SAMPLE = REPO / "data" / "sample_hands.txt"


def doc_hands() -> list[str]:
    blocks = re.findall(r"```text\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    return [b.strip("\n") for b in blocks if re.match(r".*Hand #\S+: ", b)]


def without_dealt_lines(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith("Dealt to "))


def hand_count(dataset: str) -> int:
    con = warehouse.connect(dataset, read_only=True)
    try:
        return int(warehouse.query_df(con, "SELECT count(*) AS n FROM hands")["n"].iloc[0])
    finally:
        con.close()


def visibility(dataset: str) -> list[str]:
    con = warehouse.connect(dataset, read_only=True)
    try:
        return list(warehouse.query_df(con, "SELECT cards_visibility FROM hands ORDER BY hand_id")["cards_visibility"])
    finally:
        con.close()


def folder(tmp_path: Path, name: str, **files: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    for file_name, text in files.items():
        (path / f"{file_name}.txt").write_text(text + "\n", encoding="utf-8")
    return path


def test_every_example_hand_of_the_doc_parses_and_the_first_is_the_bundled_one():
    hands = doc_hands()
    assert len(hands) == 2
    for text in hands:
        assert not isinstance(parse_hand(text), GateReject), text.split("\n")[0]
    assert hands[0].count("Dealt to ") == 2 and "Dealt to" not in hands[1]
    assert hands[0] in SAMPLE.read_text(encoding="utf-8")


def test_the_drop_table_of_the_doc_names_every_reason():
    text = DOC.read_text(encoding="utf-8")
    table = text[text.index("## What is dropped, and why"):text.index("## Check your files")]
    for reason in GATE_REASONS + ("duplicate", "duplicate_diff", "replay_error", "no_hand_header"):
        assert f"`{reason}`" in table, reason


def test_the_doc_hand_loads_as_full_information_and_without_its_dealt_to_lines_as_mined(data_dir, tmp_path):
    full = doc_hands()[0]
    for name, text, expected in (("doc_full", full, "full"), ("doc_mined", without_dealt_lines(full), "showdown")):
        result = run.load(name, [str(folder(tmp_path, name, hands=text))], source="mined", **QUIET)
        assert result.hands_loaded == 1 and result.violations == 0
        assert visibility(name) == [expected]


def test_input_without_a_hand_header_stops_before_the_warehouse(data_dir, tmp_path, capsys):
    good = folder(tmp_path, "good", hands="\n\n".join(doc_hands()))
    assert run.main(["--dataset", "t_keep", "--input", str(good)]) == 0
    other = folder(tmp_path, "other", notes="Game Hand 1 - another layout\nnothing the grammar knows")
    assert run.main(["--dataset", "t_keep", "--input", str(other)]) == 2
    assert "no hand history found" in capsys.readouterr().err
    assert hand_count("t_keep") == 2  # the earlier load is still there


def test_input_whose_every_hand_is_dropped_stops_before_the_warehouse(data_dir, tmp_path, capsys):
    good = folder(tmp_path, "good", hands="\n\n".join(doc_hands()))
    assert run.main(["--dataset", "t_keep", "--input", str(good)]) == 0
    bad = folder(tmp_path, "bad", hands=doc_hands()[0].replace("collected 650.00 from pot", "collected 649.00 from pot"))
    assert run.main(["--dataset", "t_keep", "--input", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "no hand in" in err and "money_unreconciled 1" in err
    assert hand_count("t_keep") == 2


def test_a_missing_second_input_stops_before_the_warehouse(data_dir, tmp_path):
    good = folder(tmp_path, "good", hands="\n\n".join(doc_hands()))
    assert run.main(["--dataset", "t_keep", "--input", str(good)]) == 0
    assert run.main(["--dataset", "t_keep", "--input", str(good), "--input", str(tmp_path / "missing")]) == 2
    assert hand_count("t_keep") == 2


def test_a_file_without_a_hand_header_next_to_good_ones_is_named(data_dir, tmp_path):
    mixed = folder(tmp_path, "mixed", good="\n\n".join(doc_hands()), other="Game Hand 1 - another layout")
    lines: list[str] = []
    result = run.load("t_mixed", [str(mixed)], log=lines.append)
    run.report(result, log=lines.append)
    assert result.hands_loaded == 2 and result.no_header_files == ["other.txt"]
    assert any("files without a hand header" in line and "other.txt" in line for line in lines)
    con = warehouse.connect("t_mixed", read_only=True)
    try:
        funnel = warehouse.query_df(con, "SELECT file, reason, n_hands FROM parse_funnel ORDER BY file")
    finally:
        con.close()
    assert funnel.values.tolist() == [["good.txt", "loaded", 2], ["other.txt", "no_hand_header", 0]]


def test_the_bundled_sample_loads_in_full_as_full_information(data_dir):
    result = run.load("sample", [str(SAMPLE)], source="operator", **QUIET)
    assert result.reasons() == {"loaded": 718}
    assert result.violations == 0 and not result.replay_errors and not result.no_header_files
    assert set(visibility("sample")) == {"full"}
