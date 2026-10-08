"""Input variant 2: operator replay rows -> text -> the three tables, on the bundled article-case sample."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from step01_testlib import load_run_module

from sudetect import warehouse
from superuser import DEFAULT_INPUT, parse_hand, read_hands

STEP_DIR = Path(__file__).resolve().parents[1]


def test_replay_rows_round_trip_to_tables(data_dir):
    spec = importlib.util.spec_from_file_location("step01_from_json", STEP_DIR / "from_replay_json.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    rows = read_hands(DEFAULT_INPUT).to_dict("records")
    out = data_dir / "article_sample" / "input"
    funnel = conv.convert(rows, out, log=lambda _m: None)
    assert len(rows) == 718 and funnel["written"] == 718      # the real case, heads-up only; every hand passes

    run = load_run_module()
    res = run.load("article_sample", [str(out / "hands.txt")], source="operator", log=lambda _m: None)
    assert res.hands_loaded == funnel["written"] and set(res.reasons()) == {"loaded"}
    assert all(int(v) == 0 for v in res.invariants["violations"])

    con = warehouse.connect("article_sample", read_only=True)
    try:
        hp = warehouse.query_df(con, "SELECT hand_id, player_id, net_bb FROM hand_players")
        vis = warehouse.query_df(con, "SELECT DISTINCT cards_visibility FROM hands").cards_visibility.tolist()
    finally:
        con.close()
    assert vis == ["full"]
    loaded = set(hp.hand_id)
    for r in rows:
        h = parse_hand(r)
        if h is None or h.hand_uid not in loaded:
            continue
        got = hp[hp.hand_id == h.hand_uid].set_index("player_id").net_bb * h.blind
        assert {int(k): round(v, 6) for k, v in got.items()} == {k: round(v, 6) for k, v in h.settlement.items()}

    with open(out / "decision_times.csv", encoding="utf-8") as f:
        times = list(csv.DictReader(f))
    hands = [h for h in map(parse_hand, rows) if h is not None and h.hand_uid in loaded]
    n_actions = sum(len(h.actions[st]) for h in hands for st in ("preflop", "flop", "turn", "river"))
    assert len(times) == n_actions
