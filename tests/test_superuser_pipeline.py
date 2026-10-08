"""The per-suspect package on the sample case: data/sample_hands.csv, the article's case with ids and dates replaced."""
from superuser import DEFAULT_INPUT, analyse_player, parse_hand, read_hands
from superuser.report import to_markdown

ROWS = read_hands(DEFAULT_INPUT).to_dict("records")
SUSPECT, VICTIM = 1001, 1002


def test_sample_file_loads():
    assert len(ROWS) == 718
    assert {"hand_uid", "start_ts", "hand_replay", "hand_result"} <= set(ROWS[0])


def test_parse_hands():
    parsed = [parse_hand(r) for r in ROWS]
    assert sum(h is not None for h in parsed) == 718


def test_suspect_is_flagged_and_escalated():
    res = analyse_player(ROWS, SUSPECT)
    sus = res["suspicious_hands"]
    assert res["meta"]["hands_analysed"] == 718
    assert len(sus) > 0 and set(sus["category"]) <= {"A", "B", "C", "D"}
    assert res["escalation"]["escalate"]


def test_opponent_scope_and_report():
    res = analyse_player(ROWS, SUSPECT, opponent=VICTIM)
    assert res["meta"]["hands_analysed"] > 700
    md = to_markdown(res)
    assert "Hidden-information signals" in md and "Suspicious hands" in md


def test_player_not_present():
    res = analyse_player(ROWS, 9999)
    assert res["meta"]["hands_analysed"] == 0
    assert res["suspicious_hands"].empty


def test_write_report_overwrites_output(tmp_path):
    from superuser.report import OUTPUT_FILES, write_report

    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "old_file.txt").write_text("left over from a previous run")
    res = analyse_player(ROWS, SUSPECT)
    write_report(res, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == set(OUTPUT_FILES) | {".gitkeep"}
    page = (tmp_path / "suspicious_hands.html").read_text(encoding="utf-8")
    assert page.count('class="hand"') == len(res["suspicious_hands"])


def test_package_numbers_on_the_sample():
    """The package on its own, all 718 records: the numbers it reports for the suspect."""
    res = analyse_player(ROWS, SUSPECT)
    sus = res["suspicious_hands"]
    assert len(sus) == 76 and sus["category"].value_counts().to_dict() == {"A": 31, "D": 23, "B": 15, "C": 7}
    assert sus["decision_label"].value_counts().to_dict() == {
        "Perfect fold": 37, "Good fold": 10, "Average fold": 7, "Average call": 6, "Good call": 5,
        "Perfect call": 4, "Perfect bluff": 3, "Good bluff": 2, "Average bluff": 2}
    assert int(sus["snap_decision"].sum()) == 3
    esc = res["escalation"]
    assert (esc["suspicious"], esc["hands"], esc["escalate"]) == (29, 250, True)
    sig = res["signals"]
    assert (sig["pair_plus_folds_behind"]["k"], sig["pair_plus_folds_behind"]["n"]) == (56, 62)
    assert (sig["fold_vs_value_bet"]["k"], sig["fold_vs_value_bet"]["n"]) == (81, 94)
    assert (sig["fold_vs_bluff"]["k"], sig["fold_vs_bluff"]["n"]) == (14, 86)
    assert (sig["river_decision_accuracy"]["k"], sig["river_decision_accuracy"]["n"]) == (46, 54)
