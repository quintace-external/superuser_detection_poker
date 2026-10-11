"""Parity of the screening's article rule set with the per-suspect package (``superuser/`` through ``adapter.py``).

With the switches set to the module's definitions (board pairs count, raises are bluffs, walks count, the gap
needs one response per side) the warehouse scorecard, the flagged hands with their labels and the Stage 5 verdict
must equal what the module computes from the same tables through ``05_detection/adapter.py``, for the planted
players of sim_small and five honest players. Evaluation code: it reads the simulator's answer key to pick them.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import step05_screen as step05
from sudetect import metrics as M
from sudetect import screening as S
from sudetect import warehouse

DATASET = "sim_small"
pytestmark = pytest.mark.skipif(not warehouse.db_path(DATASET).exists(), reason="needs the sim_small warehouse")

SIGNALS = {  # module signal -> step 05 column
    "perfect_fold_ratio_postflop": "ofr_post", "perfect_fold_ratio_turn": "ofr_turn",
    "perfect_fold_ratio_river": "ofr_river", "pair_plus_folds_behind": "ofr_pair", "bluff_index": "bluff_index",
    "fold_vs_value_bet": "fold_to_value", "fold_vs_bluff": "fold_to_bluff", "discrimination_gap": "disc_gap",
    "river_decision_accuracy": "river_accuracy",
}


def module_config() -> dict:
    cfg = step05.load_config()
    cfg["scope"] = "hu_tables"          # the module reads two-player tables only
    cfg["scorecard"].update(pair_plus_own_cards=False, bluff_includes_raises=True, vpip_excludes_walks=False,
                            gap_min_n=1)
    cfg["hand_tests"]["rule_sets"] = ["article"]
    cfg["escalation"]["anchor"] = "player_last_hand"
    return cfg


@pytest.fixture(scope="module")
def ours():
    cfg = module_config()
    inp = step05.read_inputs(DATASET, cfg)
    dec = step05.prepare_decisions(inp["dec"][lambda x: x["street"] != "preflop"], cfg)
    hands = step05.prepare_hands(inp["hands"])
    card = step05.scorecard_frame(dec, ["player_id"], cfg).set_index("player_id")
    ptab = M.fit_expected_fold(step05._responses(dec))
    flagged = S.flag_hands(M.attach_expected(dec, ptab).rename(columns={"p": "p_all"}), cfg)
    esc = S.escalate(flagged, hands, cfg)
    esc = esc[esc["rule"] == "article_fixed"].set_index("player_id")
    return cfg, card, flagged, esc, hands


def players(hands: pd.DataFrame) -> list[str]:
    truth = json.loads((warehouse.data_dir() / DATASET / "ground_truth.json").read_text(encoding="utf-8"))
    planted = [p["player"] for p in truth["plants"]]
    # five honest players with at least 300 heads-up hands, the smallest such volumes first (the module replays
    # every hand in Python, so the test stays well under two minutes)
    n = hands.groupby("player_id").size().drop(planted, errors="ignore")
    n = n[n >= 300].sort_values(kind="mergesort")
    return planted + n.index[:5].tolist()


def test_module_parity(ours):
    from adapter import analyse_player   # 05_detection/adapter.py

    cfg, card, flagged, esc, hands = ours
    con = step05._connect(DATASET, True, 600)
    try:
        for pid in players(hands):
            res = analyse_player(con, pid, scope="hu_tables")   # the screen's scope above
            sig = res["signals"]
            for name, col in SIGNALS.items():
                theirs, mine = sig[name]["value"], card.loc[pid, col]
                if theirs is None:
                    assert math.isnan(mine), (pid, name, mine)
                else:
                    assert abs(theirs - mine) <= 1e-9, (pid, name, theirs, mine)
                if sig[name]["n"] is not None:
                    assert sig[name]["n"] == card.loc[pid, f"{col}__n"], (pid, name)
                    assert sig[name]["k"] == card.loc[pid, f"{col}__k"], (pid, name)
            sus = res["suspicious_hands"]
            mine = flagged[flagged["player_id"] == pid].set_index("hand_id")
            theirs = sus.set_index("hand_uid") if not sus.empty else pd.DataFrame(columns=["category"])
            assert set(mine.index) == set(theirs.index), pid
            for hid, row in theirs.iterrows():
                assert mine.loc[hid, "category"] == row["category"], (pid, hid)
                assert mine.loc[hid, "decision_label"] == row["decision_label"], (pid, hid)
                assert abs(mine.loc[hid, "pot_bb"] - row["pot_bb"]) <= 0.05 + 1e-9, (pid, hid)
            order = mine.sort_values("review_rank").index.tolist()
            labels_in_order = [mine.loc[h, "label"] for h in order]
            assert labels_in_order == sorted(labels_in_order, key=S.LABEL_ORDER.get), pid
            e = res["escalation"]
            assert e["hands"] == esc.loc[pid, "hands"] and e["suspicious"] == esc.loc[pid, "flagged"], pid
            assert e["escalate"] == bool(esc.loc[pid, "escalate"]), pid
    finally:
        con.close()
