"""Hidden-information signals: oracle (perfect) fold ratios, bluff index, response discrimination."""
from __future__ import annotations

import pandas as pd


def _r(k: int, n: int) -> dict:
    return {"value": k / n if n else None, "k": int(k), "n": int(n)}


def player_signals(dec: pd.DataFrame, uid: int) -> dict:
    """`dec` = decision rows (see decisions.decision_rows) for the hands in scope."""
    g = dec[dec["uid"] == uid]
    post = g["street"].isin(["flop", "turn", "river"])
    has_eq = g["hero_eq"].notna()
    folds = g[post & has_eq & g["is_fold"]]
    behind = folds["hero_eq"] < 0.5
    pair = folds[folds["hero_pair_plus"] == True]          # noqa: E712
    bluffs = g[post & has_eq & g["is_aggr"] & (g["hero_eq"] < 0.5)]
    faced = g[post & g["facing_bet"] & g["bettor_eq"].notna()]
    vs_value, vs_bluff = faced[faced["bettor_eq"] >= 0.5], faced[faced["bettor_eq"] < 0.5]
    river = g[(g["street"] == "river") & g["facing_bet"] & has_eq & (g["is_fold"] | g["is_continue"])]
    correct = ((river["is_fold"] & (river["hero_eq"] < 0.5)) | (river["is_continue"] & (river["hero_eq"] >= 0.5))).sum()
    fv, fb = _r(vs_value["is_fold"].sum(), len(vs_value)), _r(vs_bluff["is_fold"].sum(), len(vs_bluff))
    gap = (fv["value"] - fb["value"]) if fv["value"] is not None and fb["value"] is not None else None
    return {
        "perfect_fold_ratio_postflop": _r(behind.sum(), len(folds)),
        "perfect_fold_ratio_turn": _r((folds[folds["street"] == "turn"]["hero_eq"] < 0.5).sum(), (folds["street"] == "turn").sum()),
        "perfect_fold_ratio_river": _r((folds[folds["street"] == "river"]["hero_eq"] < 0.5).sum(), (folds["street"] == "river").sum()),
        "pair_plus_folds_behind": _r((pair["hero_eq"] < 0.5).sum(), len(pair)),
        "bluff_index": _r((bluffs["bluff_success"] == True).sum(), len(bluffs)),   # noqa: E712
        "fold_vs_value_bet": fv,
        "fold_vs_bluff": fb,
        "discrimination_gap": {"value": gap, "k": None, "n": None},
        "river_decision_accuracy": _r(correct, len(river)),
    }
