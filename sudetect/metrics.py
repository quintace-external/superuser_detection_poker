"""Step 05 metrics: the article's scorecard, the range-adjusted test (method b), the frequency test (method a), the
single-victim pair test and the onset series.

Every function works on pandas frames of decision rows, one row per heads-up decision after the flop with the
columns of ``05_detection/sql/scored_decisions.sql`` plus the flags added by ``05_detection/screen.py``
(``is_fold``, ``is_cont``, ``is_aggr``, ``rbin``, ``p``, ...), and returns one row per value of the grouping columns
it is given. The warehouse is read and written by ``05_detection/screen.py``; nothing here touches it.

Notation (the article's terms in brackets):

* q = the decider's equity against the opponent's real cards (equity vs the opponent's hand);
* r = his equity against the opponent's predicted range (step 04);
* req = pot odds B / (P + 2B): the amount to call over the pot after the call;
* y = 1 for a fold, 0 for a call or a raise (a continue);
* dead: q < 0.02; winning: q >= 0.5 (a river split counts as winning); behind: q < 0.5;
* p = the field's fold rate in the decision's (street, range decile), fitted on the same window and decision set.

Counts carry the suffixes ``__k`` (numerator) and ``__n`` (denominator) next to each rate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import evaluator, handclass

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD_ID = {r + s: 4 * i + j for i, r in enumerate(_RANKS) for j, s in enumerate(_SUITS)}
_NORMAL = NormalDist()

# --------------------------------------------------------------------------------------------- small helpers


def rate(k: pd.Series | np.ndarray, n: pd.Series | np.ndarray) -> np.ndarray:
    """k / n as floats, NaN where n is 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full(k.shape, np.nan)
    ok = n > 0
    out[ok] = k[ok] / n[ok]
    return out


def flag(s: pd.Series) -> pd.Series:
    """A boolean column with NULL read as False."""
    return s.fillna(False).astype(bool)


def midrank_pct(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Mid-rank percentile in [0, 1]: (#{v < x} + 0.5 #{v = x}) / n inside each group; NaN values stay NaN."""
    v = values.astype(float)
    if groups is None:
        ranks = v.rank(method="average")
        n = v.notna().sum()
        return (ranks - 0.5) / n if n else ranks
    ranks = v.groupby(groups).rank(method="average")
    n = v.notna().groupby(groups).transform("sum")
    return (ranks - 0.5) / n.where(n > 0)


def upper_tail_p(z: np.ndarray) -> np.ndarray:
    """One-sided p-value P(Z >= z) of a standard normal; NaN stays NaN."""
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z)
    out[ok] = [0.5 * math.erfc(x / math.sqrt(2.0)) for x in z[ok]]
    return out


def normal_quantile(q: float) -> float:
    """Inverse of the standard normal distribution function."""
    return _NORMAL.inv_cdf(q)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up, capped at 1); NaN p-values are left out and stay NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return q
    order = ok[np.argsort(p[ok], kind="mergesort")]
    scaled = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return q


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD) over the finite values; NaN everywhere when the MAD is 0."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if not ok.any():
        return out
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    if mad > 0:
        out[ok] = (x[ok] - med) / mad
    return out


# --------------------------------------------------------------------------------------------- cards and tiers


def cards_array(texts: Iterable[str | None], width: int) -> np.ndarray:
    """('AhKd', 'Th9c2s', ...) -> an (n, width) int array of card ids (rank * 4 + suit), -1 padded."""
    rows = []
    for t in texts:
        ids = [] if not isinstance(t, str) else [_CARD_ID[t[i:i + 2]] for i in range(0, len(t) - 1, 2)]
        rows.append((ids + [-1] * width)[:width])
    return np.asarray(rows, dtype=np.int64).reshape(-1, width)


_OWN_TIER = np.array([0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3])   # handclass.MADE_CLASSES -> tier


def hand_tiers(hole: Sequence[str | None], board: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Made-hand tiers 0 high card, 1 pair, 2 two pair, 3 trips or better, of each (hole, board) row.

    Returns ``(tier_full, tier_own)``: ``tier_full`` from the best five of all the cards (a pair on the board
    counts), ``tier_own`` from the made-hand class of :mod:`sudetect.handclass` (a pair needs a hole card).
    -1 where the hole cards are unknown or fewer than three board cards are dealt.
    """
    h = cards_array(hole, 2)
    b = cards_array(board, 5)
    n = len(h)
    full = np.full(n, -1, dtype=np.int64)
    own = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return full, own
    n_board = (b >= 0).sum(axis=1)
    known = (h >= 0).all(axis=1) & (n_board >= 3)
    for nb in (3, 4, 5):
        rows = np.flatnonzero(known & (n_board == nb))
        if rows.size:
            cards = np.concatenate([h[rows], b[rows, :nb]], axis=1)
            full[rows] = np.minimum(evaluator.eval_hands(cards) >> 20, 3)
    rows = np.flatnonzero(known)
    if rows.size:
        made, _ = handclass.classify_batch(h[rows], b[rows])
        own[rows] = np.where(made >= 0, _OWN_TIER[np.clip(made, 0, None)], -1)
    return full, own


# --------------------------------------------------------------------------------------------- article scorecard

SCORECARD_RATES = ("ofr_post", "ofr_pair", "ofr_turn", "ofr_river", "ofr_tr", "fold_to_value", "fold_to_bluff",
                   "bluff_index", "river_accuracy", "fold_river_bet", "river_call_win")


def scorecard(dec: pd.DataFrame, keys: list[str], *, pair_plus_own_cards: bool = True,
              bluff_includes_raises: bool = True, behind_q: float = 0.5, value_bettor_q: float = 0.5,
              bluff_q: float = 0.5, gap_min_n: int = 20) -> pd.DataFrame:
    """The article's per-player rates on heads-up decisions after the flop.

    * ``ofr_post``: folds with q < 0.5 / folds with q known (the oracle or "perfect" fold ratio);
      ``ofr_pair`` the same over folds holding pair or better; ``ofr_turn``, ``ofr_river``, ``ofr_tr`` (turn and
      river folds pooled) over every fold of the street;
    * ``fold_to_value`` / ``fold_to_bluff``: folds / responses to a bet whose bettor had q >= 0.5 / q < 0.5 at the
      bet; ``disc_gap`` = their difference (a fraction; x 100 for points), NULL unless both denominators reach
      ``gap_min_n``;
    * ``bluff_index``: bluffs (bet or raise with q < ``bluff_q``; bets made when not facing a bet only, if
      ``bluff_includes_raises`` is false) after which the opponent folded on the street / bluffs;
    * ``river_accuracy``: river responses to a bet that were right on the real cards (fold with q < 0.5, call or
      raise with q >= 0.5) / river responses with q known;
    * ``fold_river_bet``: river folds / river responses to a bet; ``river_call_win``: river calls of a bet that won
      at showdown / river calls of a bet.
    """
    d = dec
    q, street = d["q"], d["street"]
    has_q = q.notna()
    fold, cont, aggr, facing = d["is_fold"], d["is_cont"], d["is_aggr"], d["facing"]
    behind = has_q & (q < behind_q)
    fq = fold & has_q
    pair = d["pair_own"] if pair_plus_own_cards else d["pair_full"]
    bluff = has_q & aggr & (q < bluff_q)
    if not bluff_includes_raises:
        bluff = bluff & (d["action"] == "bet") & ~facing
    faced = facing & d["bettor_q"].notna()
    value = faced & (d["bettor_q"] >= value_bettor_q)
    bluffed = faced & (d["bettor_q"] < value_bettor_q)
    river = street == "river"
    rv = river & facing & has_q & (fold | cont)
    correct = rv & ((fold & (q < behind_q)) | (cont & (q >= behind_q)))
    rvf = river & facing
    rcall = rvf & (d["action"] == "call")
    tr = street.isin(["turn", "river"])
    parts = {
        "ofr_post": (fq & behind, fq),
        "ofr_pair": (fq & pair & behind, fq & pair),
        "ofr_turn": (fq & (street == "turn") & behind, fq & (street == "turn")),
        "ofr_river": (fq & river & behind, fq & river),
        "ofr_tr": (fq & tr & behind, fq & tr),
        "fold_to_value": (value & fold, value),
        "fold_to_bluff": (bluffed & fold, bluffed),
        "bluff_index": (bluff & d["bluff_success"], bluff),
        "river_accuracy": (correct, rv),
        "fold_river_bet": (rvf & fold, rvf),
        "river_call_win": (rcall & d["won_at_showdown"], rcall),
    }
    cols = {k: d[k] for k in keys}
    for name, (num, den) in parts.items():
        cols[f"{name}__k"] = num.to_numpy(dtype=np.int64)
        cols[f"{name}__n"] = den.to_numpy(dtype=np.int64)
    g = pd.DataFrame(cols).groupby(keys, sort=True).sum().reset_index()
    for name in parts:
        g[name] = rate(g[f"{name}__k"], g[f"{name}__n"])
    both = (g["fold_to_value__n"] >= gap_min_n) & (g["fold_to_bluff__n"] >= gap_min_n)
    g["disc_gap"] = np.where(both, g["fold_to_value"] - g["fold_to_bluff"], np.nan)
    return g


def hand_stats(hands: pd.DataFrame, keys: list[str], *, vpip_excludes_walks: bool = True) -> pd.DataFrame:
    """Volume, win rate and style per group of hand rows (one row per player per hand).

    ``bb100`` = 100 x net bb / hands (after rake); ``wtsd`` = showdowns / hands that saw the flop; ``wsd`` =
    showdowns won / showdowns (both over hands that saw the flop); ``vpip``, ``pfr`` over dealt hands, a walk (the
    big blind with no decision) left out when ``vpip_excludes_walks``.
    """
    h = hands
    sd = h["saw_flop"] & h["went_to_showdown"]
    frame = pd.DataFrame({
        **{k: h[k] for k in keys},
        "hands": 1, "net_bb": h["net_bb"].astype(float), "walks": h["walk"].astype(np.int64),
        "vpip__k": h["vpip"].astype(np.int64), "pfr__k": h["pfr"].astype(np.int64),
        "saw_flop": h["saw_flop"].astype(np.int64), "sd": sd.astype(np.int64),
        "sd_won": (sd & h["won_at_showdown"]).astype(np.int64),
    })
    g = frame.groupby(keys, sort=True).sum().reset_index()
    days = h.groupby(keys, sort=True)["hand_date"].nunique().reset_index(name="active_days")
    g = g.merge(days, on=keys, how="left")
    g["bb100"] = 100.0 * rate(g["net_bb"], g["hands"])
    den = g["hands"] - g["walks"] if vpip_excludes_walks else g["hands"]
    g["vpip__n"] = den
    g["vpip"] = rate(g["vpip__k"], den)
    g["pfr__n"] = den
    g["pfr"] = rate(g["pfr__k"], den)
    g["wtsd"] = rate(g["sd"], g["saw_flop"])
    g["wsd"] = rate(g["sd_won"], g["sd"])
    return g


# --------------------------------------------------------------------------------------------- method (b)


def range_bin(r: pd.Series, n_bins: int = 10) -> pd.Series:
    """min(n_bins - 1, floor(n_bins r)); NaN where r is unknown."""
    return np.minimum(n_bins - 1, np.floor(r.astype(float) * n_bins))


def fit_expected_fold(resp: pd.DataFrame) -> pd.DataFrame:
    """The field's fold rate p and count n per (street, rbin) of response rows (columns street, rbin, y)."""
    return (resp.groupby(["street", "rbin"], sort=True)["y"].agg(p="mean", n="size").reset_index())


def attach_expected(resp: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Adds the column ``p`` of ``table`` (street, rbin, p) to each row."""
    out = resp.drop(columns=[c for c in ("p", "p_n") if c in resp.columns])
    return out.merge(table.rename(columns={"n": "p_n"}), on=["street", "rbin"], how="left")


COUNTERS = ("nd", "fd", "ed", "vd", "nw", "fw", "ew", "vw")


def range_counters(resp: pd.DataFrame, keys: list[str], *, dead_q: float = 0.02, win_q: float = 0.5) -> pd.DataFrame:
    """Per group: dead decisions nd, their folds fd, expected folds ed = sum p, variance vd = sum p (1 - p); the
    same for winning decisions (nw, fw, ew, vw). Rows in the middle class (0.02 <= q < 0.5) are dropped."""
    q, y, p = resp["q"], resp["y"].astype(float), resp["p"].astype(float)
    dead = (q < dead_q).astype(float)
    win = (q >= win_q).astype(float)
    frame = pd.DataFrame({
        **{k: resp[k] for k in keys},
        "nd": dead, "fd": dead * y, "ed": dead * p, "vd": dead * p * (1 - p),
        "nw": win, "fw": win * y, "ew": win * p, "vw": win * p * (1 - p),
    })
    return frame.groupby(keys, sort=True).sum().reset_index()


def centring(c: pd.DataFrame, *, min_dead: int = 20, min_win: int = 20) -> tuple[float, float, int]:
    """Population excess per decision over the eligible groups (nd >= min_dead, nw >= min_win), decision-weighted:
    delta_D = sum(fd - ed) / sum nd, delta_W = sum(ew - fw) / sum nw. Returns (delta_D, delta_W, n eligible)."""
    e = c[(c["nd"] >= min_dead) & (c["nw"] >= min_win)]
    if e.empty:
        return math.nan, math.nan, 0
    dd = float((e["fd"] - e["ed"]).sum() / e["nd"].sum())
    dw = float((e["ew"] - e["fw"]).sum() / e["nw"].sum())
    return dd, dw, len(e)


def range_scores(c: pd.DataFrame, dd: float, dw: float) -> pd.DataFrame:
    """X_dead, X_win, S, z, z_c, z_b and Var S of counter rows (NaN where a class is empty)."""
    out = c.copy()
    nd, fd, ed, vd = (out[k].astype(float) for k in ("nd", "fd", "ed", "vd"))
    nw, fw, ew, vw = (out[k].astype(float) for k in ("nw", "fw", "ew", "vw"))
    ok = (nd > 0) & (nw > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_dead = (fd - ed) / nd
        x_win = (ew - fw) / nw
        score = (fd - ed) / nd + (ew - fw) / nw
        sd_cnt = np.sqrt(vd + vw)
        var_s = vd / nd ** 2 + vw / nw ** 2
        out["x_dead"] = np.where(nd > 0, x_dead, np.nan)
        out["x_win"] = np.where(nw > 0, x_win, np.nan)
        out["s"] = np.where(ok, score, np.nan)
        out["var_s"] = np.where(ok, var_s, np.nan)
        out["z"] = np.where(ok & (sd_cnt > 0), ((fd - ed) + (ew - fw)) / sd_cnt, np.nan)
        out["z_c"] = np.where(ok & (sd_cnt > 0), ((fd - ed - dd * nd) + (ew - fw - dw * nw)) / sd_cnt, np.nan)
        out["z_b"] = np.where(ok & (var_s > 0), (score - dd - dw) / np.sqrt(var_s), np.nan)
    return out


def excess_discrimination(resp: pd.DataFrame, keys: list[str], *, dead_q: float = 0.02,
                          win_q: float = 0.5) -> pd.DataFrame:
    """E (the discrimination gap inside (street, range decile) strata, net of the population) and the CMH z.

    In stratum k the group has d dead decisions with fd folds and w winning ones with fw folds; the population's
    fold rates fD_k, fW_k come from every row of ``resp``. Strata with d = 0 or w = 0 are skipped.
    h = d w / (d + w); E = sum h (fd/d - fw/w - (fD_k - fW_k)) / sum h; CMH: o = sum fd, e = sum d m / n,
    v = sum d w m (n - m) / (n^2 (n - 1)) with n = d + w, m = fd + fw; z = (o - e) / sqrt(v).
    """
    q = resp["q"]
    cls = np.where(q < dead_q, "dead", np.where(q >= win_q, "win", ""))
    r = resp.assign(cls=cls)
    r = r[r["cls"] != ""]
    out_cols = list(keys) + ["e_disc", "cmh_z", "e_strata"]
    if r.empty:
        return pd.DataFrame(columns=out_cols)
    pop = r.groupby(["street", "rbin", "cls"])["y"].mean().unstack("cls")
    pop = pop.reindex(columns=["dead", "win"]).rename(columns={"dead": "pop_d", "win": "pop_w"}).reset_index()
    frame = pd.DataFrame({
        **{k: r[k] for k in keys}, "street": r["street"], "rbin": r["rbin"],
        "d": (r["cls"] == "dead").astype(float), "w": (r["cls"] == "win").astype(float),
    })
    frame["fd"] = frame["d"] * r["y"].astype(float)
    frame["fw"] = frame["w"] * r["y"].astype(float)
    strata = list(keys) + [c for c in ("street", "rbin") if c not in keys]
    st = frame.groupby(strata, sort=True).sum().reset_index()
    st = st[(st["d"] > 0) & (st["w"] > 0)].merge(pop, on=["street", "rbin"], how="left")
    if st.empty:
        return pd.DataFrame(columns=out_cols)
    d, w, fd, fw = st["d"], st["w"], st["fd"], st["fw"]
    h = d * w / (d + w)
    n, m = d + w, fd + fw
    st["num"] = h * (fd / d - fw / w)
    st["pnum"] = h * (st["pop_d"] - st["pop_w"])
    st["den"] = h
    st["o"] = fd
    st["e"] = d * m / n
    st["v"] = np.where(n > 1, d * w * m * (n - m) / (n * n * (n - 1)), 0.0)
    st["e_strata"] = 1
    g = st.groupby(list(keys), sort=True)[["num", "pnum", "den", "o", "e", "v", "e_strata"]].sum().reset_index()
    g["e_disc"] = (g["num"] - g["pnum"]) / g["den"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g["cmh_z"] = np.where(g["v"] > 0, (g["o"] - g["e"]) / np.sqrt(g["v"]), np.nan)
    return g[out_cols]


def oracle_events(resp: pd.DataFrame, keys: list[str], *, dead_q: float = 0.02, win_q: float = 0.5,
                  fold_margin: float = 0.24, call_margin: float = 0.20, line: float = 0.5) -> pd.DataFrame:
    """Single-decision events by range.

    Pot-odds margins: an oracle fold is a fold of a dead hand where r - req >= ``fold_margin`` (the range said
    continue); an oracle call is a continue of a winning hand where req - r >= ``call_margin`` (the range said
    fold). ``or_share`` = events / such spots, ``or_expected`` = its value had the group folded at the field's rate
    p, ``g_fold`` / ``g_call`` = the event rate minus the same rate of the mirror class (winning hands for folds,
    dead hands for calls), which a pure style (nit, calling station) does not move.
    At the 0.5 line: ``or_fold_range`` = folds with r >= 0.5 and q < 0.5 / folds with r >= 0.5;
    ``or_call_range`` = continues with r < 0.5 and q >= 0.5 / continues with r < 0.5.
    """
    q, r, req = resp["q"], resp["r"], resp["req"]
    fold = resp["y"].astype(bool)
    cont = ~fold
    p = resp["p"].astype(float)
    dead, win = q < dead_q, q >= win_q
    rc = (r - req) >= fold_margin
    rf = (req - r) >= call_margin
    parts = {
        "n_rc_dead": rc & dead, "k_or_fold": rc & dead & fold,
        "n_rf_win": rf & win, "k_or_call": rf & win & cont,
        "n_rc_win": rc & win, "k_rc_win_fold": rc & win & fold,
        "n_rf_dead": rf & dead, "k_rf_dead_cont": rf & dead & cont,
        "n_fold_r_ahead": fold & (r >= line), "k_fold_r_ahead_q_behind": fold & (r >= line) & (q < line),
        "n_cont_r_behind": cont & (r < line), "k_cont_r_behind_q_ahead": cont & (r < line) & (q >= line),
    }
    frame = pd.DataFrame({**{k: resp[k] for k in keys}, **{k: v.astype(np.int64) for k, v in parts.items()}})
    frame["e_or"] = np.where(rc & dead, p, 0.0) + np.where(rf & win, 1.0 - p, 0.0)
    g = frame.groupby(keys, sort=True).sum().reset_index()
    spots = g["n_rc_dead"] + g["n_rf_win"]
    g["or_share"] = rate(g["k_or_fold"] + g["k_or_call"], spots)
    g["or_expected"] = rate(g["e_or"], spots)
    g["g_fold"] = rate(g["k_or_fold"], g["n_rc_dead"]) - rate(g["k_rc_win_fold"], g["n_rc_win"])
    g["g_call"] = rate(g["k_or_call"], g["n_rf_win"]) - rate(g["k_rf_dead_cont"], g["n_rf_dead"])
    g["or_fold_range"] = rate(g["k_fold_r_ahead_q_behind"], g["n_fold_r_ahead"])
    g["or_call_range"] = rate(g["k_cont_r_behind_q_ahead"], g["n_cont_r_behind"])
    return g.drop(columns=["e_or"])


def two_way_fit(s: np.ndarray, w: np.ndarray, i: np.ndarray, j: np.ndarray, *, passes: int = 3,
                prior_weight: float = 1.0) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """S_ij = mu + a_i + b_j + eps_ij by weighted alternating means.

    ``w`` are the pair weights (1 / Var S_ij), ``i`` and ``j`` the row's player and opponent codes. a and b are
    shrunk toward 0 with a prior weight of ``prior_weight`` x the median pair weight, so a player seen in one pair
    only does not absorb that pair's whole signal. Returns (mu, a_i, b_j, eps) per row.
    """
    s, w = np.asarray(s, float), np.asarray(w, float)
    i, j = np.asarray(i), np.asarray(j)
    if s.size == 0:
        z = np.zeros(0)
        return math.nan, z, z, z
    mu = float(np.sum(w * s) / np.sum(w))
    w0 = prior_weight * float(np.median(w))
    ui, ii = np.unique(i, return_inverse=True)
    uj, jj = np.unique(j, return_inverse=True)
    a, b = np.zeros(ui.size), np.zeros(uj.size)
    for _ in range(max(1, passes)):
        res = s - mu - b[jj]
        a = np.bincount(ii, w * res, ui.size) / (np.bincount(ii, w, ui.size) + w0)
        res = s - mu - a[ii]
        b = np.bincount(jj, w * res, uj.size) / (np.bincount(jj, w, uj.size) + w0)
    eps = s - mu - a[ii] - b[jj]
    return mu, a[ii], b[jj], eps


def split_z(pair: pd.DataFrame, total: pd.DataFrame, keys_total: list[str]) -> pd.DataFrame:
    """z_ij = (S(A_ij) - S(A_i,-j)) / sqrt(Var S(A_ij) + Var S(A_i,-j)) and the directional z_dead, z_win.

    ``pair`` holds the counters of i's decisions against j, ``total`` of all i's decisions; the rest of i's
    decisions is total - pair."""
    m = pair.merge(total, on=keys_total, how="left", suffixes=("", "_tot"))
    for c in COUNTERS:
        m[f"{c}_rest"] = m[f"{c}_tot"] - m[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        xd, xw = (m["fd"] - m["ed"]) / m["nd"], (m["ew"] - m["fw"]) / m["nw"]
        xdr, xwr = (m["fd_rest"] - m["ed_rest"]) / m["nd_rest"], (m["ew_rest"] - m["fw_rest"]) / m["nw_rest"]
        vd, vw = m["vd"] / m["nd"] ** 2, m["vw"] / m["nw"] ** 2
        vdr, vwr = m["vd_rest"] / m["nd_rest"] ** 2, m["vw_rest"] / m["nw_rest"] ** 2
        m["s_rest"] = xdr + xwr
        m["z_split"] = ((xd + xw) - (xdr + xwr)) / np.sqrt(vd + vw + vdr + vwr)
        m["z_split_dead"] = (xd - xdr) / np.sqrt(vd + vdr)
        m["z_split_win"] = (xw - xwr) / np.sqrt(vw + vwr)
    for c in ("s_rest", "z_split", "z_split_dead", "z_split_win"):
        m[c] = m[c].where(np.isfinite(m[c]))
    return m.drop(columns=[f"{c}_tot" for c in COUNTERS] + [f"{c}_rest" for c in COUNTERS])


# --------------------------------------------------------------------------------------------- onset


def changepoint(weekly: pd.DataFrame, *, min_dead: int = 10, min_win: int = 10) -> pd.DataFrame:
    """Largest two-sample z of S between the weeks after a split and the weeks before it, per player.

    ``weekly`` holds the counters per (player_id, week_start), weeks in time order. For every split point with at
    least ``min_dead`` dead and ``min_win`` winning decisions on both sides:
    z = (S_after - S_before) / sqrt(Var S_after + Var S_before). Returns player_id, cp_z, cp_week_start (the
    first week after the split) and cp_splits (split points tested)."""
    out = []
    for pid, g in weekly.sort_values(["player_id", "week_start"]).groupby("player_id", sort=True):
        c = g[list(COUNTERS)].to_numpy(float)
        tot = c.sum(axis=0)
        before = np.cumsum(c, axis=0)[:-1]
        after = tot - before
        best, week, tested = math.nan, None, 0
        for t in range(before.shape[0]):
            b, a = before[t], after[t]
            if min(b[0], a[0]) < min_dead or min(b[4], a[4]) < min_win:
                continue
            sb = (b[1] - b[2]) / b[0] + (b[6] - b[5]) / b[4]
            sa = (a[1] - a[2]) / a[0] + (a[6] - a[5]) / a[4]
            var = b[3] / b[0] ** 2 + b[7] / b[4] ** 2 + a[3] / a[0] ** 2 + a[7] / a[4] ** 2
            if var <= 0:
                continue
            tested += 1
            z = (sa - sb) / math.sqrt(var)
            if math.isnan(best) or z > best:
                best, week = z, g["week_start"].iloc[t + 1]
        out.append({"player_id": pid, "cp_z": best, "cp_week_start": week, "cp_splits": tested})
    return pd.DataFrame(out, columns=["player_id", "cp_z", "cp_week_start", "cp_splits"])
