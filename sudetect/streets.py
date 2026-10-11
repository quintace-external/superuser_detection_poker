"""Preflop and flop tests of step 05: the decisions the article's core tests (turn and river) leave out.

Both work on the rows of ``05_detection/sql/scored_decisions.sql``, so both see heads-up confrontations at tables of
any size: preflop, every decision with exactly one live opponent; on the flop, every decision of a hand that exactly
two players saw the flop of.

Player level, per time window:

* preflop fold test: responses to a raise, q and r known. A player who sees the opponent's cards folds more when he
  is behind the real hand (q < ``behind_q``) and less when he is ahead (q >= ``ahead_q``) than the field folds with
  the same equity against the opponent's line (r, the field's range for the line, ``04_predicted_range/preflop.py``)
  in the same situation (vs_open, vs_3bet, ...). Counted like method (b): ``pf_fold_z_b`` (centred on the
  population) and the within-player CMH z ``pf_fold_cmh_z`` (behind vs ahead inside each situation x r stratum);
* preflop aggression test: every decision, raise or not. He raises more when ahead of the real hand than when
  behind with the same r: ``pf_aggr_z`` (the CMH z, sign turned so that more aggression when ahead is positive);
* the raw gaps behind both tests, not adjusted for r: ``pf_fold_gap`` = ``pf_fold_behind`` - ``pf_fold_ahead`` (fold
  rate facing a raise when behind minus when ahead) and ``pf_aggr_gap`` = ``pf_aggr_ahead`` - ``pf_aggr_behind``
  (raise rate when ahead minus when behind). Honest players have positive gaps too (their own hand strength follows
  q), so the z values above, not the gaps, carry the test;
* ``pf_z`` = (``pf_fold_cmh_z`` + ``pf_aggr_z``) / sqrt 2. Honest players do not sit at 0 (on the simulator their
  median is about +1, sd 1.2: a strata cell of r still mixes stronger and weaker own hands), so the screen uses
  ``pf_zr``: ``pf_z`` minus the population median, over 1.4826 x MAD, once ``robust_min_players`` players are
  eligible (otherwise ``pf_z`` itself); ``pf_screen`` = eligible and ``pf_zr >= z_screen``;
* re-raise success ("4-bet bluff success"): 3-bets, 4-bets and 5-bets and the share the opponent folded to at once
  (``pf_3bet_n`` / ``pf_3bet_succ`` ...), and the light ones (r < ``light_r``: weak against the line): how many, how
  often they worked and how often the real hand behind them was ahead (``pf_rr_light_*``);
* flop test: method (b) on flop responses with a flop-sized "behind" class (q < ``flop.dead_q``; with two cards to
  come few hands are dead): ``z_b_flop``, ``fl_cmh_z``, robust-standardised ``z_b_flop_r`` like ``pf_zr``,
  ``fl_screen``, and the flop's excess discrimination ``fl_e_disc`` (E of :func:`sudetect.metrics.excess_discrimination`
  with the same behind and winning classes, counted on ``fl_nd`` and ``fl_nw`` decisions).

``pf_zr`` and the flop pair ``z_b_flop`` + ``fl_e_disc`` feed the escalation rule set ``street_tails``
(:func:`sudetect.screening.street_tails`).

Hand level (``street_flags``, one row per player and decision): the preflop rules PF1-PF4 and the flop rules FL1-FL2
(see :data:`RULES`). Each needs the real cards to have made the decision right and the predicted range or the price
to have said otherwise; ``gap`` = |q - r| ranks them. They are reported, not counted in Stage 5: their false-positive
rate is not calibrated yet (see ``05_detection/README.md``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M

RULES: dict[str, str] = {
    "PF1": "Preflop fold, beaten: the range said continue at that price, the real hand was ahead",
    "PF2": "Preflop call, ahead: the range said fold at that price, the real hand was behind",
    "PF3": "Preflop re-raise, light and ahead: a hand weak against the line, the real hand weaker still",
    "PF4": "Preflop flat with a strong hand, beaten: strong against the line, the real hand stronger",
    "FL1": "Flop fold, beaten: the range said continue at that price, the real hand left little equity",
    "FL2": "Flop continue, ahead: the range said fold at that price, the real hand was behind",
}
FLAG_COLUMNS = ["player_id", "hand_id", "action_index", "street", "rule", "rule_name", "opponent_id", "table_class",
                "position", "situation", "opp_situation", "opp_action", "action", "hole_cards", "opp_hole_cards",
                "board_at_action", "q", "r", "req", "q_random", "gap", "tier", "pot_bb", "faced_size_pot",
                "bluff_success", "hand_date"]
PLAYER_COLUMNS = ["pf_resp", "pf_nd", "pf_nw", "pf_fold_behind", "pf_fold_ahead", "pf_fold_gap", "pf_fold_z_b",
                  "pf_fold_cmh_z", "pf_aggr_n", "pf_aggr_behind", "pf_aggr_ahead", "pf_aggr_gap", "pf_aggr_z", "pf_z",
                  "pf_zr", "pf_eligible", "pf_screen", "pf_3bet_n", "pf_3bet_succ", "pf_4bet_n", "pf_4bet_succ",
                  "pf_5bet_n", "pf_5bet_succ", "pf_rr_light_n", "pf_rr_light_succ", "pf_rr_light_ahead", "fl_nd",
                  "fl_nw", "z_b_flop", "fl_e_disc", "fl_cmh_z", "z_b_flop_r", "fl_eligible", "fl_screen"]
RERAISE = {"3bet": "vs_open", "4bet": "vs_3bet", "5bet": "vs_4bet_plus"}   # a raise in this situation


def _strata(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Copy of ``df`` with ``street`` = 'pf:' + situation (the stratum key method (b) groups on) and ``rbin``."""
    out = df.copy()
    out["street"] = "pf:" + out["situation"].fillna("none").astype(str)
    out["rbin"] = M.range_bin(out["r"], bins)
    return out


def _cmh(rows: pd.DataFrame, keys: list[str], lo: float, hi: float) -> pd.DataFrame:
    return M.excess_discrimination(rows, keys, dead_q=lo, win_q=hi)


def robust_standardise(z: pd.Series, eligible: pd.Series, min_players: int) -> pd.Series:
    """(z - median) / (1.4826 MAD) over the eligible rows once there are ``min_players`` of them; else z."""
    ref = z[eligible & z.notna()]
    if len(ref) < min_players:
        return z.astype(float)
    med = float(ref.median())
    mad = 1.4826 * float((ref - med).abs().median())
    return (z - med) / mad if mad > 0 else z - med


def reraise_success(pre: pd.DataFrame, keys: list[str], light_r: float) -> pd.DataFrame:
    """3-bets, 4-bets, 5-bets and the share the opponent folded to at once; light re-raises (r < light_r)."""
    rr = pre[pre["is_aggr"] & pre["situation"].isin(list(RERAISE.values()))]
    succ = rr["bluff_success"].astype(float)
    cols: dict[str, Any] = {k: rr[k] for k in keys}
    for name, sit in RERAISE.items():
        m = (rr["situation"] == sit).astype(float)
        cols[f"n_{name}"], cols[f"s_{name}"] = m, m * succ
    light = (rr["r"] < light_r).astype(float)
    cols["n_l"], cols["s_l"] = light, light * succ
    cols["k_l"] = light * rr["q"].notna()
    cols["a_l"] = light * (rr["q"] >= 0.5)
    g = pd.DataFrame(cols).groupby(keys).sum().reset_index()
    out = g[keys].copy()
    for name in RERAISE:
        out[f"pf_{name}_n"] = g[f"n_{name}"]
        out[f"pf_{name}_succ"] = M.rate(g[f"s_{name}"], g[f"n_{name}"])
    out["pf_rr_light_n"] = g["n_l"]
    out["pf_rr_light_succ"] = M.rate(g["s_l"], g["n_l"])
    out["pf_rr_light_ahead"] = M.rate(g["a_l"], g["k_l"])
    return out


def preflop_players(pre: pd.DataFrame, keys: list[str], pc: Mapping[str, Any]) -> tuple[pd.DataFrame, dict]:
    """The preflop fold and aggression tests per group (``keys``) on the preflop rows of one window."""
    bins, lo, hi = int(pc["range_bins"]), float(pc["behind_q"]), float(pc["ahead_q"])
    base = pre[pre["q"].notna() & pre["r"].notna()]
    resp = _strata(base[base["facing"]], bins)
    resp = resp.assign(y=resp["is_fold"].astype(np.int64))
    table = M.fit_expected_fold(resp)
    resp = M.attach_expected(resp, table)
    cnt = M.range_counters(resp, keys, dead_q=lo, win_q=hi)
    dd, dw, n_el = M.centring(cnt, min_dead=int(pc["min_behind"]), min_win=int(pc["min_ahead"]))
    fold = M.range_scores(cnt, 0.0 if math.isnan(dd) else dd, 0.0 if math.isnan(dw) else dw)
    fold = fold.rename(columns={"nd": "pf_nd", "nw": "pf_nw", "z_b": "pf_fold_z_b"})
    fold = fold.merge(resp.groupby(keys).size().rename("pf_resp").reset_index(), on=keys, how="left")
    fold["pf_fold_behind"] = M.rate(fold["fd"], fold["pf_nd"])
    fold["pf_fold_ahead"] = M.rate(fold["fw"], fold["pf_nw"])
    fold["pf_fold_gap"] = fold["pf_fold_behind"] - fold["pf_fold_ahead"]
    fold = fold.merge(_cmh(resp, keys, lo, hi).rename(columns={"cmh_z": "pf_fold_cmh_z"})[keys + ["pf_fold_cmh_z"]],
                      on=keys, how="left")
    agg = _strata(base, bins).assign(y=lambda x: x["is_aggr"].astype(np.int64))
    a = _cmh(agg, keys, lo, hi).rename(columns={"cmh_z": "aggr_cmh"})
    behind, ahead = agg["q"] < lo, agg["q"] >= hi
    rates = agg.assign(b=behind.astype(float), a=ahead.astype(float), yb=behind * agg["y"], ya=ahead * agg["y"])
    rates = rates.groupby(keys)[["b", "a", "yb", "ya"]].sum().reset_index()
    rates["pf_aggr_n"] = rates["b"] + rates["a"]
    rates["pf_aggr_behind"] = M.rate(rates["yb"], rates["b"])
    rates["pf_aggr_ahead"] = M.rate(rates["ya"], rates["a"])
    rates["pf_aggr_gap"] = rates["pf_aggr_ahead"] - rates["pf_aggr_behind"]
    out = fold[keys + ["pf_resp", "pf_nd", "pf_nw", "pf_fold_behind", "pf_fold_ahead", "pf_fold_gap", "pf_fold_z_b",
                       "pf_fold_cmh_z"]].merge(rates[keys + ["pf_aggr_n", "pf_aggr_behind", "pf_aggr_ahead",
                                                             "pf_aggr_gap"]], on=keys, how="outer")
    out = out.merge(a[keys + ["aggr_cmh"]], on=keys, how="left")
    out["pf_aggr_z"] = -out.pop("aggr_cmh")
    out["pf_z"] = (out["pf_fold_cmh_z"].fillna(0) + out["pf_aggr_z"].fillna(0)) / math.sqrt(2)
    out.loc[out["pf_fold_cmh_z"].isna() & out["pf_aggr_z"].isna(), "pf_z"] = np.nan
    out["pf_eligible"] = (out["pf_nd"].fillna(0) >= int(pc["min_behind"])) & (
        out["pf_nw"].fillna(0) >= int(pc["min_ahead"]))
    out["pf_zr"] = robust_standardise(out["pf_z"], out["pf_eligible"], int(pc["robust_min_players"]))
    out["pf_screen"] = out["pf_eligible"] & (out["pf_zr"] >= float(pc["z_screen"]))
    out = out.merge(reraise_success(pre, keys, float(pc["light_r"])), on=keys, how="outer")
    return out, {"pf_delta_behind": dd, "pf_delta_ahead": dw, "pf_players_eligible": n_el,
                 "pf_responses": len(resp), "pf_decisions": len(base)}


def flop_players(post: pd.DataFrame, keys: list[str], fc: Mapping[str, Any], range_bins: int) -> tuple[
        pd.DataFrame, dict]:
    """Method (b) on the flop responses of one window, with the flop's own "behind" class; ``fl_e_disc`` is E on the
    same responses and classes (``z_b_flop`` and ``fl_e_disc`` together feed the ``street_tails`` rule set)."""
    dq, wq = float(fc["dead_q"]), float(fc["win_q"])
    resp = post[(post["street"] == "flop") & post["facing"] & post["q"].notna() & post["r"].notna()].copy()
    resp["rbin"] = M.range_bin(resp["r"], range_bins)
    resp = M.attach_expected(resp, M.fit_expected_fold(resp))
    cnt = M.range_counters(resp, keys, dead_q=dq, win_q=wq)
    dd, dw, n_el = M.centring(cnt, min_dead=int(fc["min_dead"]), min_win=int(fc["min_win"]))
    sc = M.range_scores(cnt, 0.0 if math.isnan(dd) else dd, 0.0 if math.isnan(dw) else dw)
    sc = sc.rename(columns={"nd": "fl_nd", "nw": "fl_nw", "z_b": "z_b_flop"})
    ed = _cmh(resp, keys, dq, wq).rename(columns={"cmh_z": "fl_cmh_z", "e_disc": "fl_e_disc"})
    sc = sc.merge(ed[keys + ["fl_e_disc", "fl_cmh_z"]], on=keys, how="left")
    sc["fl_e_disc"] = sc["fl_e_disc"].astype(float)          # a float column even when no stratum has both classes
    sc["fl_eligible"] = (sc["fl_nd"] >= int(fc["min_dead"])) & (sc["fl_nw"] >= int(fc["min_win"]))
    sc["z_b_flop_r"] = robust_standardise(sc["z_b_flop"], sc["fl_eligible"], int(fc["robust_min_players"]))
    sc["fl_screen"] = sc["fl_eligible"] & (sc["z_b_flop_r"] >= float(fc["z_screen"]))
    return sc[keys + ["fl_nd", "fl_nw", "z_b_flop", "fl_e_disc", "fl_cmh_z", "z_b_flop_r", "fl_eligible",
                      "fl_screen"]], {"fl_delta_dead": dd, "fl_delta_win": dw, "fl_players_eligible": n_el,
                                      "fl_responses": len(resp)}


def player_scores(pre: pd.DataFrame, post: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Both player tests for one window: one row per player with :data:`PLAYER_COLUMNS`."""
    keys = ["player_id"]
    pf, m1 = preflop_players(pre, keys, cfg["preflop"])
    fl, m2 = flop_players(post, keys, cfg["flop"], int(cfg["method_b"]["range_bins"]))
    out = pf.merge(fl, on=keys, how="outer")
    for c in PLAYER_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    for c in ("pf_eligible", "pf_screen", "fl_eligible", "fl_screen"):
        out[c] = out[c].fillna(False).astype(bool)
    return out[keys + PLAYER_COLUMNS], {**m1, **m2}


def _tier(gap: pd.Series) -> np.ndarray:
    return np.select([gap >= 0.30, gap >= 0.15], ["BLATANT", "STRONG"], default="THIN")


def flag_streets(pre: pd.DataFrame, post: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The hand rules PF1-PF4 (preflop) and FL1-FL2 (flop): one row per flagged decision, ``gap`` = |q - r|."""
    pc, fc = cfg["preflop"], cfg["flop"]
    lo, hi = float(pc["behind_q"]), float(pc["ahead_q"])
    p = pre[pre["facing"] & pre["q"].notna() & pre["r"].notna() & pre["req"].notna()]
    q, r, req = p["q"], p["r"], p["req"]
    call = p["action"] == "call"
    raise_ = p["action"] == "raise"
    rules = {
        "PF1": p["is_fold"] & ((r - req) >= float(pc["fold_margin"])) & (q < req) & (q < lo),
        "PF2": call & ((req - r) >= float(pc["call_margin"])) & (q >= hi),
        "PF3": raise_ & (r < float(pc["light_r"])) & (q >= hi),
        "PF4": call & (r >= float(pc["strong_r"])) & (q < lo),
    }
    f = post[(post["street"] == "flop") & post["facing"] & post["q"].notna() & post["r"].notna()
             & post["req"].notna()]
    fq, fr, freq = f["q"], f["r"], f["req"]
    frules = {
        "FL1": f["is_fold"] & (fq < float(fc["fold_q_max"])) & ((fr - freq) >= float(fc["fold_margin"])),
        "FL2": f["is_cont"] & (fq >= 0.5) & ((freq - fr) >= float(fc["call_margin"])),
    }
    parts = [p[m.fillna(False)].assign(rule=k) for k, m in rules.items()]
    parts += [f[m.fillna(False)].assign(rule=k) for k, m in frules.items()]
    parts = [x for x in parts if len(x)]
    if not parts:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["rule_name"] = out["rule"].map(RULES)
    out["gap"] = (out["q"] - out["r"]).abs()
    out["tier"] = _tier(out["gap"])
    for c in FLAG_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out = out.sort_values(["player_id", "gap"], ascending=[True, False], kind="mergesort")
    return out[FLAG_COLUMNS].reset_index(drop=True)
