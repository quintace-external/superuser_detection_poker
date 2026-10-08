"""Stages 1 to 5 of the superuser screen: the reference implementation of step 05.

* Stage 1: win-rate outliers among comparable players (time window x hand-count band x style cluster);
* Stage 2: the top of the Stage-1 survivors on the hidden-information signals;
* Stage 3: hand tests, categories A-D with the labels Perfect / Good / Average, under two rule sets;
* Stage 4: a risk score per flagged hand (and a surprise version comparable across players);
* Stage 5: escalation over the last 15 days, by the article's fixed rule and by a population-calibrated rule;
* the win-rate percentile over same-length runs.

The functions take pandas frames built by ``05_detection/screen.py`` and the ``[step05]`` table of
``config/step05_screening.toml``. ``screen``, ``flag_hands`` and ``escalate`` form the replaceable interface
described in ``05_detection/README.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import hand_tiers, midrank_pct, normal_quantile

#: The five Stage-1 statistics and the direction an outlier takes (+1 high, -1 low).
STAGE1_STATS: dict[str, int] = {"bb100": 1, "wsd": 1, "wtsd": -1, "fold_river_bet": 1, "river_call_win": 1}
CATEGORY_NAMES = {"A": "River fold when beaten", "B": "River bluff-catch", "C": "Successful river bluff",
                  "D": "Early fold, near dead"}
CATEGORY_KIND = {"A": "fold", "B": "call", "C": "bluff", "D": "fold"}
LABEL_ORDER = {"Perfect": 0, "Good": 1, "Average": 2}


def band_labels(hands: pd.Series, edges: Sequence[int]) -> pd.Series:
    """Hand-count band: B1 for edges[0] <= hands < edges[1], ..., B5 at edges[4] or more; B0 below edges[0]."""
    idx = np.searchsorted(np.asarray(edges), hands.to_numpy(dtype=float), side="right")
    return pd.Series([f"B{i}" for i in idx], index=hands.index)


# --------------------------------------------------------------------------------------------- style clusters


def _kmeans(z: np.ndarray, k: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    n = len(z)
    k = max(1, min(k, n))
    centers = [z[rng.integers(n)]]
    for _ in range(1, k):                                     # k-means++ seeding
        d2 = np.min(((z[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        idx = rng.integers(n) if d2.sum() == 0 else rng.choice(n, p=d2 / d2.sum())
        centers.append(z[idx])
    c = np.asarray(centers, dtype=float)
    lab = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        lab = np.argmin(((z[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([z[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(len(c))])
        if np.allclose(new, c):
            break
        c = new
    order = np.lexsort((c[:, 1], c[:, 0]))                    # name clusters by their VPIP, then PFR, centre
    rename = np.empty(len(c), dtype=int)
    rename[order] = np.arange(len(c))
    return rename[lab]


def style_clusters(players: pd.DataFrame, sc: Mapping[str, Any], seed: int) -> pd.Series:
    """Style cluster of each row from VPIP and PFR: a quantile grid (``v1p1`` .. ``v3p3``) or k-means (``k1``..)."""
    x = players[["vpip", "pfr"]].astype(float)
    ok = x.notna().all(axis=1)
    out = pd.Series(None, index=players.index, dtype=object)
    if ok.sum() == 0:
        return out
    xv, xp = x.loc[ok, "vpip"].to_numpy(), x.loc[ok, "pfr"].to_numpy()
    if sc.get("cluster_method", "quantile_grid") == "kmeans":
        z = np.column_stack([xv, xp])
        sd = z.std(axis=0)
        z = (z - z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        lab = _kmeans(z, int(sc.get("kmeans_k", 3)), int(sc.get("kmeans_iter", 50)), np.random.default_rng(seed))
        out.loc[ok] = [f"k{i + 1}" for i in lab]
    else:
        g = int(sc.get("grid", 3))
        qs = np.linspace(0, 1, g + 1)[1:-1]
        iv = np.searchsorted(np.quantile(xv, qs), xv, side="right")
        ip = np.searchsorted(np.quantile(xp, qs), xp, side="right")
        out.loc[ok] = [f"v{a + 1}p{b + 1}" for a, b in zip(iv, ip)]
    return out


# --------------------------------------------------------------------------------------------- Stage 1 and 2


def _key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if df.empty:
        return pd.Series([], index=df.index, dtype=object)
    return df[list(cols)].astype(str).agg("|".join, axis=1)


def _leveled_pct(df: pd.DataFrame, value: pd.Series, levels: Sequence[Sequence[str]], min_group: int) -> tuple[
        pd.Series, pd.Series]:
    """Mid-rank percentile of ``value`` inside the finest level whose group holds at least ``min_group`` rows (the
    last level always applies); a small group is ranked against the whole coarser group. Returns (pct, level key)."""
    pct = pd.Series(np.nan, index=df.index)
    key_used = pd.Series(None, index=df.index, dtype=object)
    todo = pd.Series(True, index=df.index)
    for n, cols in enumerate(levels):
        key = _key(df, cols)
        size = key.map(key.value_counts())
        take = todo & ((size >= min_group) | (n == len(levels) - 1))
        p = midrank_pct(value, key)
        pct[take] = p[take]
        key_used[take] = key[take]
        todo &= ~take
    return pct, key_used


def stage1(players: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stage 1 on one row per (player_id, time_window) with ``hands``, ``vpip``, ``pfr`` and the five statistics.

    Adds ``s1_eligible`` (hands >= the window's minimum), ``band``, ``cluster``, ``pct_<stat>`` (directional
    mid-rank percentiles: high = outlier), ``s1_composite`` (their mean, at least ``min_stats`` of them),
    ``s1_composite_pct``, ``s1_rank_group``, ``s1_top10``, ``s1_p95`` and ``s1_flag`` (the configured rule).
    """
    sc = cfg["screening"]
    df = players.copy()
    df["s1_eligible"] = df["hands"] >= df["time_window"].map(sc["min_hands"]).astype(float)
    df["band"] = band_labels(df["hands"], sc["band_edges"]).where(df["s1_eligible"])
    df["cluster"] = pd.Series(None, index=df.index, dtype=object)
    for _, g in df[df["s1_eligible"]].groupby("time_window", sort=True):
        df.loc[g.index, "cluster"] = style_clusters(g, sc, int(cfg.get("seed", 0)))
    e = df[df["s1_eligible"]].copy()
    fine, mid, top = ["time_window", "band", "cluster"], ["time_window", "band"], ["time_window"]
    if sc.get("order", "cluster_then_outliers") == "outliers_then_cluster":
        stat_levels, cut_levels = [mid, top], [fine, mid, top]
    else:
        stat_levels, cut_levels = [fine, mid, top], [fine, mid, top]
    min_group = int(sc.get("min_group", 10))
    pcts = []
    for stat, direction in STAGE1_STATS.items():
        v = e[stat].astype(float) * direction
        e[f"pct_{stat}"], _ = _leveled_pct(e, v, stat_levels, min_group)
        pcts.append(f"pct_{stat}")
    avail = e[pcts].notna().sum(axis=1)
    e["s1_composite"] = e[pcts].mean(axis=1).where(avail >= int(sc.get("min_stats", 3)))
    e["s1_composite_pct"], e["s1_rank_group"] = _leveled_pct(e, e["s1_composite"], cut_levels, min_group)
    e["s1_top10"] = e["s1_composite_pct"] >= 1.0 - float(sc["top_share"]) - 1e-12
    e["s1_p95"] = (e[pcts] >= float(sc["p95"]) - 1e-12).sum(axis=1) >= int(sc["n_of_5"])
    rule = sc.get("rule", "top10_composite")
    e["s1_flag"] = e["s1_top10"] if rule == "top10_composite" else e["s1_p95"]
    new = [c for c in e.columns if c not in df.columns]
    df = df.join(e[new])
    for c in ("s1_top10", "s1_p95", "s1_flag"):
        df[c] = df[c].fillna(False).astype(bool)
    return df


def stage2(players: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stage 2: Stage-1 survivors in the top ``stage2_top_share`` of the population (the window's Stage-1 eligible
    players) on any of ``stage2_signals`` (a rate counts with at least ``rate_min_n`` qualifying decisions), or on
    the Stage-1 composite when ``stage2_basis = "composite"``. Adds ``pct2_<signal>`` and ``s2_flag``."""
    sc, min_n = cfg["screening"], int(cfg["scorecard"]["rate_min_n"])
    df = players.copy()
    cut = 1.0 - float(sc["stage2_top_share"]) - 1e-12
    e = df["s1_eligible"]
    hit = pd.Series(False, index=df.index)
    if sc.get("stage2_basis", "signals") == "composite":
        p = midrank_pct(df["s1_composite"].where(e), df["time_window"])
        df["pct2_composite"] = p
        hit |= p >= cut
    else:
        for sig in sc["stage2_signals"]:
            v = df[sig].where(e & (df[f"{sig}__n"] >= min_n))
            p = midrank_pct(v.astype(float), df["time_window"])
            df[f"pct2_{sig}"] = p
            hit |= p >= cut
    df["s2_flag"] = df["s1_flag"] & hit.fillna(False)
    return df


def screen(players: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stages 1 and 2 (the replaceable interface)."""
    return stage2(stage1(players, cfg), cfg)


# --------------------------------------------------------------------------------------------- win-rate runs


RUN_COLUMNS = ["player_id", "wr_hands", "wr_runs", "wr_run_bb100", "wr_pct_median", "wr_pct_lo", "wr_pct_hi",
               "wr_pct_pooled", "wr_at_or_above", "wr_ever_as_good", "wr_pct_best_run", "wr_field"]


def _bb100(v: np.ndarray) -> float:
    return 100.0 * float(v.sum()) / len(v)


def win_rate_runs(hands: pd.DataFrame, rc: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """Win-rate percentile of every eligible player's first run against random same-length runs of the others.

    ``hands`` holds player_id, started_at, hand_id, net_bb. A player with at least ``min_hands`` hands is cut into
    back-to-back runs of ``run_length`` hands in time order (the tail dropped; 500-719 hands = one shorter run).
    Each draw takes one random run per other player; the subject's mid-rank percentile is
    100 (#{v < s} + 0.5 #{v = s}) / n. Reported: median and 2.5 / 97.5 percentiles over the draws, the pooled
    version (every run weighted 1 / that player's run count), the median count of drawn runs at or above him, the
    players who ever had a run as good, and the best-run variant (each other player's best full run over the cut
    offsets; his first run when he has none)."""
    L, mn = int(rc["run_length"]), int(rc["min_hands"])
    h = hands.sort_values(["player_id", "started_at", "hand_id"])
    series = {pid: g["net_bb"].to_numpy(float) for pid, g in h.groupby("player_id", sort=True)}
    elig = [pid for pid, v in series.items() if len(v) >= mn]
    if len(elig) < 2:
        return pd.DataFrame(columns=RUN_COLUMNS)
    runs, first, best = [], [], []
    for pid in elig:
        v = series[pid]
        k = len(v) // L
        vals = 100.0 * v[:k * L].reshape(k, L).sum(axis=1) / L if k else np.array([_bb100(v)])
        runs.append(vals)
        first.append(vals[0])
        cand = []
        for o in rc.get("best_run_offsets", [0]):
            m = (len(v) - int(o)) // L
            if m >= 1:
                cand.append(float((100.0 * v[int(o):int(o) + m * L].reshape(m, L).sum(axis=1) / L).max()))
        best.append(max(cand) if cand else vals[0])
    counts = np.array([len(r) for r in runs])
    v_all = np.concatenate(runs)
    owner = np.repeat(np.arange(len(elig)), counts)
    w_all = np.repeat(1.0 / counts, counts)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_pl, n_draws = len(elig), int(rc["n_draws"])
    rng = np.random.default_rng(seed)
    drawn = v_all[start[:, None] + (rng.random((n_pl, n_draws)) * counts[:, None]).astype(int)]
    first_a, best_a = np.asarray(first), np.asarray(best)
    max_run = np.array([r.max() for r in runs])
    out = []
    for i, pid in enumerate(elig):
        s = first_a[i]
        nf = n_pl - 1
        less = (drawn < s).sum(axis=0) - (drawn[i] < s)
        ties = (drawn == s).sum(axis=0) - (drawn[i] == s)
        pct = 100.0 * (less + 0.5 * ties) / nf
        lo, med, hi = np.percentile(pct, [2.5, 50, 97.5])
        m = owner != i
        pooled = 100.0 * (w_all[m & (v_all < s)].sum() + 0.5 * w_all[m & (v_all == s)].sum()) / w_all[m].sum()
        above = np.median((drawn >= s).sum(axis=0) - (drawn[i] >= s))
        others = np.arange(n_pl) != i
        ever = int((max_run[others] >= s).sum())
        b = best_a[others]
        best_pct = 100.0 * ((b < s).sum() + 0.5 * (b == s).sum()) / nf
        out.append([pid, int(len(series[pid])), int(counts[i]), s, med, lo, hi, pooled, float(above), ever,
                    best_pct, nf])
    return pd.DataFrame(out, columns=RUN_COLUMNS)


# --------------------------------------------------------------------------------------------- Stage 3 and 4


def categories(dec: pd.DataFrame, rule_set: str, ht: Mapping[str, Any]) -> np.ndarray:
    """Category A-D of each decision row (None when none applies); the first that applies in A, B, C, D order."""
    q, street = dec["q"], dec["street"]
    river = street == "river"
    fold, cont, aggr, facing = dec["is_fold"], dec["is_cont"], dec["is_aggr"], dec["facing"]
    a = river & facing & fold & (q <= float(ht["a_q_max"]))
    b = river & facing & cont & (q >= float(ht["b_q_min"]))
    c = river & aggr & (q < float(ht["c_q_max"])) & dec["bluff_success"]
    d = street.isin(list(ht["d_streets"])) & fold & (q < float(ht["d_q_max"]))
    if rule_set == "range_conditioned":
        r, req = dec["r"], dec["req"]
        a &= (r - req) >= float(ht["range_fold_margin"])
        b &= (req - r) >= float(ht["range_call_margin"])
        c &= r < float(ht["range_bluff_r_max"])
        d &= (r - req) >= float(ht["range_fold_margin"])
    elif rule_set != "article":
        raise ValueError(f"unknown rule set {rule_set!r}")
    conds = [x.fillna(False).to_numpy(bool) for x in (a, b, c, d)]
    return np.select(conds, list("ABCD"), default=None)


def labels(f: pd.DataFrame, rule_set: str, ht: Mapping[str, Any]) -> np.ndarray:
    """Perfect / Good / Average of flagged rows (columns category, q, r, req, tier, opp_tier)."""
    cat, q, tier, opp = f["category"], f["q"], f["tier"], f["opp_tier"]
    fold, call, bluff = cat.isin(["A", "D"]), cat == "B", cat == "C"
    pq, gq = float(ht["label_fold_perfect_q"]), float(ht["label_fold_good_q"])
    if rule_set == "article":
        perfect = (fold & (q <= pq) & (tier >= 1)) | (call & (tier <= 1)) | (bluff & (tier == 0) & (opp >= 1))
        good = (fold & ((q <= pq) | ((q <= gq) & (tier >= 1)))) | (call & (tier == 2)) | (bluff & (opp >= 1))
    else:
        r, req = f["r"], f["req"]
        pr, cg = float(ht["label_range_perfect_r"]), float(ht["label_range_call_gap"])
        perfect = ((fold & (q <= pq) & (r >= pr) & (tier >= 1)) | (call & (q >= 0.5) & (r <= req - cg) & (tier <= 1))
                   | (bluff & (tier == 0) & (opp >= 1)))
        good = (fold & (q <= gq) & (r >= req)) | (call & (q >= 0.5) & (r < req)) | (bluff & (tier <= 1) & (opp >= 1))
    return np.select([perfect.fillna(False).to_numpy(bool), good.fillna(False).to_numpy(bool)],
                     ["Perfect", "Good"], default="Average")


def flag_hands(dec: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stages 3 and 4 over every player's heads-up decisions (the replaceable interface).

    One row per (rule set, player, hand): the hand's first qualifying decision in priority A > B > C > D (the
    earliest when one category occurs twice), its label, and the risk scores: ``risk_raw`` = category weight x
    hand-strength factor x ln(1 + pot in bb) x snap; ``risk`` = 100 x risk_raw / the player's highest risk_raw in
    the rule set; ``risk_surprise_raw`` = -ln(field probability of the action in its street x range decile)
    x ln(1 + pot) x snap (category C keeps its weight), comparable across players. ``review_rank`` orders a
    player's hands Perfect first, then bigger pots.
    """
    ht, rk, own = cfg["hand_tests"], cfg["risk"], bool(cfg["scorecard"]["pair_plus_own_cards"])
    parts = []
    for rule_set in ht["rule_sets"]:
        cat = categories(dec, rule_set, ht)
        sel = dec.assign(category=cat)[pd.notna(cat)]
        sel = sel.sort_values(["player_id", "hand_id", "category", "action_index"], kind="mergesort")
        parts.append(sel.drop_duplicates(["player_id", "hand_id"], keep="first").assign(rule_set=rule_set))
    cols = ["rule_set", "player_id", "hand_id", "action_index", "hand_date", "started_at", "street", "action",
            "category", "category_name", "label", "decision_label", "opponent_id", "q", "r", "req", "pot_bb",
            "hole_cards", "opp_hole_cards", "board_at_action", "tier", "opp_tier", "decision_s", "snap",
            "risk_raw", "risk", "risk_surprise_raw", "review_rank"]
    f = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if f.empty:
        return pd.DataFrame(columns=cols)
    full, own_t = hand_tiers(f["hole_cards"].tolist(), f["board_at_action"].tolist())
    ofull, oown = hand_tiers(f["opp_hole_cards"].tolist(), f["board_at_action"].tolist())
    f["tier"] = np.maximum(own_t if own else full, 0)
    f["opp_tier"] = np.maximum(oown if own else ofull, 0)
    f["category_name"] = f["category"].map(CATEGORY_NAMES)
    f["label"] = ""
    for rule_set, idx in f.groupby("rule_set").groups.items():
        f.loc[idx, "label"] = labels(f.loc[idx], rule_set, ht)
    f["decision_label"] = f["label"] + " " + f["category"].map(CATEGORY_KIND)
    pot = f["pot_bb"].astype(float)
    t = f["tier"].clip(0, 3).astype(int).to_numpy()
    sf = np.select([f["category"].isin(["A", "D"]).to_numpy(), (f["category"] == "B").to_numpy()],
                   [np.asarray(rk["strength_fold"])[t], np.asarray(rk["strength_call"])[t]],
                   default=float(rk["strength_bluff"]))
    snap = ((f["label"] == "Perfect") & (f["decision_s"] <= float(rk["snap_seconds"]))
            & (pot >= float(rk["snap_pot_bb"])))
    f["snap"] = snap.fillna(False).astype(bool)
    snap_mult = np.where(f["snap"], float(rk["snap_factor"]), 1.0)
    weight = f["category"].map(rk["weights"]).astype(float).to_numpy()
    gap = np.ones(len(f))
    if rk.get("info_gap_factor", False):
        rng_set = (f["rule_set"] == "range_conditioned").to_numpy()
        delta = np.where(f["category"].isin(["A", "D"]), f["r"] - f["q"],
                         np.where(f["category"] == "B", f["q"] - f["r"], 0.0))
        gap = np.where(rng_set & np.isfinite(delta), 1.0 + delta, 1.0)
    f["risk_raw"] = weight * sf * np.log1p(pot) * snap_mult * gap
    top = f.groupby(["player_id", "rule_set"])["risk_raw"].transform("max")
    f["risk"] = 100.0 * f["risk_raw"] / top.where(top > 0)
    p_fold = f["p_all"].astype(float)
    prob = np.where(f["category"] == "B", 1.0 - p_fold, p_fold)
    s = -np.log(np.maximum(prob, float(rk["surprise_eps"])))
    s = np.where(f["category"] == "C", float(rk["weights"]["C"]), s)
    f["risk_surprise_raw"] = s * np.log1p(pot) * snap_mult
    f["_lab"] = f["label"].map(LABEL_ORDER)
    f = f.sort_values(["rule_set", "player_id", "_lab", "pot_bb", "hand_id"],
                      ascending=[True, True, True, False, True], kind="mergesort")
    f["review_rank"] = f.groupby(["rule_set", "player_id"]).cumcount() + 1
    return f[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- Stage 5


def escalate(flagged: pd.DataFrame, hands: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stage 5 (the replaceable interface): one row per (player, rule set, rule).

    The window is the last ``days`` days of the player's heads-up hands, ending at his last hand
    (``anchor = "player_last_hand"``, the module's rule) or at the dataset's last hand. Rule ``article_fixed``:
    flagged / hands > ``min_ratio`` and flagged > ``min_count``. Rule ``population_calibrated``: flagged / hands
    above the population's ``calibrated_quantile`` among players of the same hand-count band (bands with fewer than
    ``calibrated_min_band`` players use every player), and flagged > ``min_count``. The quantile is
    median + z_q x 1.4826 MAD of the shares (``robust``; falls back to the empirical quantile when the MAD is 0) or
    the empirical quantile (``empirical``)."""
    es = cfg["escalation"]
    cols = ["player_id", "rule_set", "rule", "window_start", "window_end", "hands", "flagged", "ratio", "band",
            "threshold_ratio", "threshold_count", "escalate"]
    if hands.empty:
        return pd.DataFrame(columns=cols)
    h = hands[["player_id", "hand_id", "started_at"]].copy()
    end = (h.groupby("player_id")["started_at"].transform("max") if es.get("anchor") == "player_last_hand"
           else pd.Series(h["started_at"].max(), index=h.index))
    h["window_end"] = end
    h["window_start"] = end - pd.Timedelta(days=float(es["days"]))
    h = h[h["started_at"] >= h["window_start"]]
    base = h.groupby("player_id").agg(window_start=("window_start", "first"), window_end=("window_end", "first"),
                                      hands=("hand_id", "size")).reset_index()
    base["band"] = band_labels(base["hands"], cfg["screening"]["band_edges"])
    min_ratio, min_count = float(es["min_ratio"]), int(es["min_count"])
    zq = normal_quantile(float(es["calibrated_quantile"]))
    out = []
    for rule_set in cfg["hand_tests"]["rule_sets"]:
        f = flagged[flagged["rule_set"] == rule_set][["player_id", "hand_id"]].drop_duplicates()
        k = f.merge(h[["player_id", "hand_id"]], on=["player_id", "hand_id"]).groupby("player_id").size()
        b = base.copy()
        b["rule_set"] = rule_set
        b["flagged"] = b["player_id"].map(k).fillna(0).astype(int)
        b["ratio"] = b["flagged"] / b["hands"]
        fixed = b.assign(rule="article_fixed", threshold_ratio=min_ratio, threshold_count=min_count)
        fixed["escalate"] = (fixed["ratio"] > min_ratio) & (fixed["flagged"] > min_count)
        cal = b.assign(rule="population_calibrated", threshold_count=min_count)
        size = cal["band"].map(cal["band"].value_counts())
        group = cal["band"].where(size >= int(es["calibrated_min_band"]), "all")
        thr = {}
        for g, x in cal.groupby(group)["ratio"]:
            ref = cal["ratio"] if g == "all" else x
            thr[g] = _quantile_threshold(ref.to_numpy(float), float(es["calibrated_quantile"]), zq,
                                         es.get("calibrated_method", "robust"))
        cal["threshold_ratio"] = group.map(thr).astype(float)
        cal["escalate"] = (cal["ratio"] > cal["threshold_ratio"]) & (cal["flagged"] > min_count)
        out += [fixed, cal]
    return pd.concat(out, ignore_index=True)[cols]


def _quantile_threshold(x: np.ndarray, q: float, zq: float, method: str) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    if method == "robust":
        med = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - med)))
        if mad > 0:
            return med + zq * mad
    return float(np.quantile(x, q))
