"""Pair outliers: the opponents a player beats beyond his own level and beyond what everyone else does against them.

A superuser who sees the cards of only a few opponents looks ordinary in his overall numbers: a few thousand hands
against his victims disappear inside many thousands of honest ones. The edge sits in a few pairs, so this module
scores every ordered pair "A against B" of the dataset and ranks them against each other.

Pair hands: in every hand, the two players who stayed in longest (both to the showdown, or the winner and the last
player to fold), each with at least one voluntary action; hands with three or more players still in at the end are
left out. Each hand gives two rows, one from each player's side.

Parts of a pair's score, from A's side (minimum sample in brackets, :class:`PairConfig`):

* ``wr``: A's net per pair hand, bb (30 hands);
* ``sd``: A's showdowns won against B, a split = 1/2; needs no hidden cards (7 showdowns);
* ``call``: A's calls facing B's bet on flop, turn or river that were ahead of B's real hand, minus the pool's rate at
  the same street and decile of equity against B's predicted range (7);
* ``bet``: A's bets and raises on flop, turn or river that were ahead, the same way (7);
* ``allin``: A's equity against B's real hand when the money went in (his all-in, or his call of one; preflop
  included), minus the pool at the same street and range band (7);
* ``pre``: A's preflop calls and raises facing B's raise that were ahead (q >= 0.5), minus the pool at the same
  situation and 5% band of equity against B's line (7);
* ``fold``: A's folds facing B's bet that were behind B's real hand, minus the pool at the same strength; it needs
  the cards behind a fold, so it exists on full-information data only (7).

The decision parts use the decisions with both hands known: every decision on full-information data, the showdowns
only on mined data. Each part is a leave-pair-out residual

    S(A vs B) - S(A vs everyone else) - S(everyone else vs B) + S(all)

over its standard error, so an opponent who loses to everybody, or a bluffer whose bets anyone calls right, does not
make a pair stand out by itself: what stands out is A doing to B what neither A does to others nor others do to B.
Three corrections keep the z honest:

* a hand counts once: A's calls (or bets) on several streets of one hand share the same cards, so each decision
  part's variance is scaled by its design effect (sum over hands of the squared per-hand residual sum, over the sum
  of the case variances; at least 1);
* the parts read the same hands (win rate and showdowns above all), so ``score`` = sum of the parts' z over
  sqrt(sum of their correlations), the correlation matrix estimated over every pair of the dataset; an honest pair's
  score then has a standard deviation close to 1;
* one strong pair would drag the "rest" of both its players (the victim's other opponents would seem to beat him
  less, the cheat would seem to beat everybody): the pairs are scored again with the rests leaving out every pair
  the previous pass put at ``victim_min_score`` or above, in either direction, until that set stops changing (a few
  passes; a pair flagged only because of another one drops out, and the true pair gets its own rests back).

Pairs with at least two parts are ranked (``field_rank``, 1 = most suspicious, of ``n_ranked``).

Pot-size columns (descriptive, the observer's view of the same pair, see :mod:`sudetect.potsize`): showdowns won in
big pots (``big_pot``, 30bb by default), their growth over smaller pots, river bets and calls won, stack-offs and
net in big pots; the rates with a binomial z against the pool of all pairs, the growth and the net against zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from . import potsize
from .warehouse import Connection, query_df, table_columns, table_exists

PARTS = ("wr", "sd", "call", "bet", "allin", "pre", "fold")


@dataclass(frozen=True)
class PairConfig:
    min_hands_wr: int = 30            # hands before the win-rate part counts
    min_cases: int = 7                # cases before any other part counts
    prior: int = 20                   # pseudo-cases pulling a thin "A vs the rest" or "the rest vs B" to the mean
    score_parts: tuple[str, ...] = PARTS
    victim_min_score: float = 3.0     # a victim: the pair's score is at least this ...
    victim_rank_share: float = 0.01   # ... and its rank is inside the top 1% of ranked pairs ...
    victim_min_rank: int = 10         # ... or the top 10, whichever is larger
    big_pot: float = potsize.BIG_POT  # bb, the pot-size columns' "big pot"

    @classmethod
    def from_toml(cls, cfg: Mapping[str, Any] | None) -> "PairConfig":
        c = dict(cfg or {})
        kw = {f.name: c[f.name] for f in fields(cls) if f.name in c}
        if "score_parts" in kw:
            kw["score_parts"] = tuple(kw["score_parts"])
        return cls(**kw)


_PAIR_SQL = """,
a AS (
    SELECT hand_id, player_id,
           min(CASE WHEN action = 'fold' THEN action_index END) AS fold_at,
           count(*) FILTER (WHERE action IN ('call', 'bet', 'raise', 'check')) AS active
    FROM actions WHERE NOT is_post GROUP BY 1, 2
),
alive AS (SELECT hand_id, count(*) FILTER (WHERE fold_at IS NULL) AS n_alive FROM a GROUP BY 1),
ranked AS (
    SELECT hand_id, player_id, active,
           row_number() OVER (PARTITION BY hand_id ORDER BY fold_at IS NULL DESC, fold_at DESC, player_id) AS rk
    FROM a
),
pr AS (
    SELECT x.hand_id, x.player_id AS p1, y.player_id AS p2
    FROM ranked x JOIN ranked y ON y.hand_id = x.hand_id AND x.rk = 1 AND y.rk = 2
    JOIN alive al ON al.hand_id = x.hand_id
    WHERE x.active > 0 AND y.active > 0 AND al.n_alive <= 2
)
SELECT pr.hand_id, h.table_id, h.hand_date, hx.player_id AS A,
       CASE WHEN hx.player_id = pr.p1 THEN pr.p2 ELSE pr.p1 END AS B,
       hx.net, hx.stack, hx.saw_river, hx.sd, hx.won, hx.pot, hx.river_in, hx.raggr, hx.rcall_big
FROM pr
JOIN hx ON hx.hand_id = pr.hand_id AND (hx.player_id = pr.p1 OR hx.player_id = pr.p2)
JOIN hands h ON h.hand_id = pr.hand_id
"""

_DEC_SQL = """
WITH last_aggr AS (
    SELECT hand_id, action_index,
           LAST_VALUE(CASE WHEN action IN ('bet', 'raise') THEN is_allin END IGNORE NULLS) OVER (
               PARTITION BY hand_id, street ORDER BY action_index ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
               AS faced_allin
    FROM actions WHERE NOT is_post
)
SELECT d.hand_id, d.player_id AS A, d.live_opponents[1].player_id AS B, d.street, d.situation, d.action,
       COALESCE(d.is_allin, FALSE) AS is_allin, d.to_call_bb, e.eq_vs_real AS q, {r} AS r,
       COALESCE(la.faced_allin, FALSE) AS faced_allin
FROM decisions d
JOIN hands h ON h.hand_id = d.hand_id
JOIN action_equity e ON e.hand_id = d.hand_id AND e.action_index = d.action_index AND e.eq_real_known
{joins}
LEFT JOIN last_aggr la ON la.hand_id = d.hand_id AND la.action_index = d.action_index
WHERE d.n_live = 2 {std} {extra}
"""


def load_inputs(con: Connection, *, showdown_only: bool = False, big_pot: float = potsize.BIG_POT
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair rows (two per pair hand) and the decisions with one live opponent and both hands known.

    ``showdown_only`` keeps the decisions of hands that reached a showdown: the observer's view of a
    full-information dataset (for evaluation). Without step 02 the decision frame is empty; without step 04 its ``r``
    is NULL and only the parts that need no range (``wr``, ``sd``) score."""
    rows = query_df(con, "WITH " + potsize.hx_sql(con) + _PAIR_SQL, big=float(big_pot))
    if not table_exists(con, "action_equity"):
        return rows, pd.DataFrame(columns=["hand_id", "A", "B", "street", "situation", "action", "is_allin",
                                           "to_call_bb", "q", "r", "faced_allin"])
    post, pre = table_exists(con, "action_range_equity"), table_exists(con, "action_range_equity_pre")
    joins, rs = [], []
    if post:
        joins.append("LEFT JOIN action_range_equity re ON re.hand_id = d.hand_id AND re.action_index = d.action_index")
        rs.append("re.eq_vs_range")
    if pre:
        joins.append("LEFT JOIN action_range_equity_pre rp ON rp.hand_id = d.hand_id "
                     "AND rp.action_index = d.action_index")
        rs.append("rp.eq_vs_range")
    r = ("COALESCE(" + ", ".join(rs) + ")") if len(rs) > 1 else (rs[0] if rs else "CAST(NULL AS DOUBLE)")
    hc = set(table_columns(con, "hands"))
    std = " ".join(c for c, ok in (("AND h.variant = 'standard'", "variant" in hc),
                                   ("AND NOT h.error_flag", "error_flag" in hc)) if ok)
    sql = _DEC_SQL.format(r=r, joins="\n".join(joins), std=std,
                          extra="AND h.went_to_showdown" if showdown_only else "")
    return rows, query_df(con, sql)


def merge_aliases(rows: pd.DataFrame, dec: pd.DataFrame, aliases: Mapping[str, str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treat several accounts of one person as one player (``aliases``: account -> person); hands between two of his
    own accounts are dropped."""
    if not aliases:
        return rows, dec
    rows, dec = rows.copy(), dec.copy()
    for f in (rows, dec):
        f["A"] = f["A"].replace(aliases)
        f["B"] = f["B"].replace(aliases)
    return rows[rows["A"] != rows["B"]], dec[dec["A"] != dec["B"]]


def _parts_rows(a, b, part: str, x, v, hand=None) -> pd.DataFrame:
    a = np.asarray(a)
    return pd.DataFrame({"hand_id": np.asarray(hand) if hand is not None else np.full(len(a), None, dtype=object),
                         "A": a, "B": np.asarray(b), "part": part, "x": np.asarray(x, float), "v": np.asarray(v, float)})


def hand_parts(rows: pd.DataFrame) -> pd.DataFrame:
    """``wr`` (x = net, v = net squared) and ``sd`` (x = showdown won, v = 1/4) from the pair rows."""
    hid = rows["hand_id"] if "hand_id" in rows else None
    out = [_parts_rows(rows["A"], rows["B"], "wr", rows["net"], rows["net"].astype(float) ** 2, hid)]
    s = rows[rows["sd"].astype(bool)]
    out.append(_parts_rows(s["A"], s["B"], "sd", s["won"], 0.25, s["hand_id"] if "hand_id" in s else None))
    return pd.concat(out, ignore_index=True)


def decision_parts(dec: pd.DataFrame) -> pd.DataFrame:
    """``call``, ``bet``, ``allin``, ``pre`` and ``fold`` from the decisions with both hands known and r known."""
    d = dec[dec["r"].notna() & dec["q"].notna() & dec["B"].notna()].copy()
    if d.empty:
        return _parts_rows([], [], "call", [], [])
    d["q"] = d["q"].astype(float)
    d["r"] = d["r"].astype(float)
    pre = d["street"].eq("preflop")
    d["rb"] = np.where(pre, np.minimum(19, np.floor(d["r"] * 20)), np.minimum(9, np.floor(d["r"] * 10)))
    d["stratum"] = np.where(pre, d["situation"].fillna("na").astype(str), d["street"].astype(str))
    d["ahead"] = (d["q"] >= 0.5).astype(float)
    facing = d["to_call_bb"].fillna(0) > 0
    out = []

    def rate_part(name, t, y):
        if t.empty:
            return
        y = y.astype(float)
        p = y.groupby([t["stratum"], t["rb"]]).transform("mean")
        out.append(_parts_rows(t["A"], t["B"], name, y - p, p * (1 - p), t["hand_id"]))

    post = d[~pre]
    sel = (post["action"] == "call") & (post["to_call_bb"].fillna(0) > 0)
    rate_part("call", post[sel], post.loc[sel, "ahead"])
    sel = post["action"].isin(["bet", "raise"])
    rate_part("bet", post[sel], post.loc[sel, "ahead"])
    t = d[pre & facing & d["action"].isin(["call", "raise"]) & d["situation"].isin(["vs_open", "vs_3bet",
                                                                                     "vs_4bet_plus"])]
    rate_part("pre", t, t["ahead"])
    t = d[facing & (d["action"] == "fold")]
    rate_part("fold", t, 1.0 - t["ahead"])
    t = d[d["is_allin"].astype(bool) | ((d["action"] == "call") & d["faced_allin"].astype(bool))]
    if not t.empty:
        g = t.groupby([t["stratum"], t["rb"]])["q"]
        mu = g.transform("mean")
        var = g.transform("var").fillna(0.0625).clip(lower=0.0025)
        out.append(_parts_rows(t["A"], t["B"], "allin", t["q"] - mu, var, t["hand_id"]))
    if not out:
        return _parts_rows([], [], "call", [], [])
    return pd.concat(out, ignore_index=True)


DEFF_PARTS = ("call", "bet", "allin", "pre", "fold")   # centred decision parts: several cases may share a hand
MAX_PASSES = 6                                         # leave-out passes until the flagged set stops changing


def design_effect(df: pd.DataFrame) -> float:
    """Sum over hands of (sum of the hand's x)^2 over the sum of the case variances, at least 1: how much the cases of
    one hand move together (x must be centred). 1 when the frame carries no hand ids."""
    if df.empty or df["hand_id"].isna().all() or float(df["v"].sum()) <= 0:
        return 1.0
    per_hand = df.groupby(["hand_id", "A", "B"], sort=False)["x"].sum()
    return max(1.0, float((per_hand ** 2).sum()) / float(df["v"].sum()))


def split_resid(df: pd.DataFrame, prior: int = 20, leave_out: set | None = None) -> pd.DataFrame:
    """Leave-pair-out residual of one part per ordered pair, with its z.

    resid = mean(A vs B) - mean(A vs rest) - mean(rest vs B) + mean(all); the two rests are pulled toward the
    global mean with ``prior`` pseudo-cases; Var = the three means' variances, each floored at a quarter of the
    average case variance. ``leave_out``: unordered pairs (frozensets of two ids) whose cases stay out of every rest
    as well (the later passes, see the module docstring)."""
    m = float(df["x"].mean())
    vf = 0.25 * float(df["v"].mean())
    pair = df.groupby(["A", "B"], sort=False).agg(n=("x", "size"), sx=("x", "sum"), sv=("v", "sum")).reset_index()
    ga = df.groupby("A").agg(nA=("x", "size"), sxA=("x", "sum"), svA=("v", "sum"))
    gb = df.groupby("B").agg(nB=("x", "size"), sxB=("x", "sum"), svB=("v", "sum"))
    t = pair.join(ga, on="A").join(gb, on="B")
    cols = ["n", "sx", "sv"]
    # the cases that leave the rests: the pair itself, plus every flagged pair of A (as A) and of B (as B)
    out_a = t[cols].astype(float).copy()
    out_b = t[cols].astype(float).copy()
    if leave_out:
        flag = pd.Series([frozenset((a, b)) in leave_out for a, b in zip(t["A"], t["B"])], index=t.index)
        fl = t[flag]
        fa = fl.groupby("A")[cols].sum()
        fb = fl.groupby("B")[cols].sum()
        own = t[cols].astype(float).where(flag, 0.0).to_numpy()   # a flagged pair is already left out once
        out_a = out_a + t[["A"]].join(fa, on="A")[cols].fillna(0.0).to_numpy() - own
        out_b = out_b + t[["B"]].join(fb, on="B")[cols].fillna(0.0).to_numpy() - own
    nar, nrb = t["nA"] - out_a["n"] + prior, t["nB"] - out_b["n"] + prior
    t["mean"] = t["sx"] / t["n"]
    t["resid"] = (t["mean"] - (t["sxA"] - out_a["sx"] + prior * m) / nar
                  - (t["sxB"] - out_b["sx"] + prior * m) / nrb + m)
    var = (np.maximum(t["sv"], t["n"] * vf) / t["n"] ** 2
           + np.maximum(t["svA"] - out_a["sv"] + prior * vf, nar * vf) / nar ** 2
           + np.maximum(t["svB"] - out_b["sv"] + prior * vf, nrb * vf) / nrb ** 2)
    t["z"] = t["resid"] / np.sqrt(var)
    return t[["A", "B", "n", "mean", "resid", "z"]]


def pot_columns(rows: pd.DataFrame, cfg: PairConfig, min_cases: int | None = None) -> pd.DataFrame:
    """The pot-size view of every ordered pair, each rate with a z against the pool of all pairs."""
    k_min = cfg.min_cases if min_cases is None else min_cases
    sd = rows["sd"].astype(bool)
    big = rows["pot"] >= cfg.big_pot
    raggr = rows["raggr"].astype(bool)
    rcall = rows["rcall_big"].astype(bool)
    so = rows["saw_river"].astype(bool) & (rows["river_in"] > cfg.big_pot)
    won = rows["won"].astype(float)
    x = pd.DataFrame({
        "A": rows["A"], "B": rows["B"],
        "sdbig_n": (sd & big).astype(int), "sdbig_k": won * (sd & big),
        "sdlt_n": (sd & ~big).astype(int), "sdlt_k": won * (sd & ~big),
        "rbet_n": (sd & raggr & (rows["pot"] > cfg.big_pot)).astype(int),
        "rbet_k": won * (sd & raggr & (rows["pot"] > cfg.big_pot)),
        "rcall_n": (sd & rcall).astype(int), "rcall_k": won * (sd & rcall),
        "stackoff_n": so.astype(int),
        "stackoff_k": (so & (rows["net"] <= -potsize.STACKOFF_SHARE * rows["stack"])).astype(int),
        "net_big_n": big.astype(int), "net_big": rows["net"] * big, "net_big_sq": rows["net"].astype(float) ** 2 * big,
    })
    g = x.groupby(["A", "B"], sort=False).sum().reset_index()
    out = g[["A", "B"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, k, n, sign in (("wsd_big", "sdbig_k", "sdbig_n", 1), ("rbet_win", "rbet_k", "rbet_n", 1),
                                 ("rcall_win", "rcall_k", "rcall_n", 1), ("stackoff", "stackoff_k", "stackoff_n", -1)):
            tot = g[n].sum()
            p0 = g[k].sum() / tot if tot else np.nan
            out[name] = g[k] / g[n].where(g[n] > 0)
            out[f"{name}_n"] = g[n]
            z = sign * (g[k] - g[n] * p0) / np.sqrt(g[n] * p0 * (1 - p0))
            out[f"{name}_z"] = z.where(g[n] >= k_min)
        rbig = g["sdbig_k"] / g["sdbig_n"].where(g["sdbig_n"] > 0)
        rlt = g["sdlt_k"] / g["sdlt_n"].where(g["sdlt_n"] > 0)
        out["growth"] = rbig - rlt
        out["growth_z"] = (out["growth"] / np.sqrt(0.25 / g["sdbig_n"] + 0.25 / g["sdlt_n"])).where(
            (g["sdbig_n"] >= k_min) & (g["sdlt_n"] >= k_min))
        out["net_big"] = g["net_big"]
        out["net_big_n"] = g["net_big_n"]
        out["net_big_z"] = (g["net_big"] / np.sqrt(g["net_big_sq"].where(g["net_big_sq"] > 0))).where(
            g["net_big_n"] >= k_min)
    return out


FIRST = ["A", "B", "hands", "bb100", "sd_won", "k", "score", "field_rank", "n_ranked", "top_pct"]


def _parts_table(parts: pd.DataFrame, cfg: PairConfig, leave_out: set | None) -> pd.DataFrame:
    res = []
    for name, g in parts.groupby("part"):
        t = split_resid(g, cfg.prior, leave_out)
        t.loc[t["n"] < (cfg.min_hands_wr if name == "wr" else cfg.min_cases), "z"] = np.nan
        res.append(t.rename(columns={c: f"{name}_{c}" for c in ("n", "mean", "resid", "z")}).set_index(["A", "B"]))
    pairs = pd.concat(res, axis=1).reset_index()
    for p in PARTS:
        for c in ("n", "mean", "resid", "z"):
            if f"{p}_{c}" not in pairs:
                pairs[f"{p}_{c}"] = np.nan
    return pairs


def _score(pairs: pd.DataFrame, cfg: PairConfig) -> tuple[pd.Series, pd.Series]:
    """(k, score): the sum of the parts' z over sqrt(the sum of their correlations over the parts present)."""
    zs = pairs[[f"{p}_z" for p in cfg.score_parts]].astype(float)
    k = zs.notna().sum(axis=1)
    r = np.array(zs.corr(min_periods=30).fillna(0.0), dtype=float)
    np.fill_diagonal(r, 1.0)
    m = zs.notna().to_numpy(dtype=float)
    denom = np.einsum("ni,ij,nj->n", m, r, m)
    score = zs.sum(axis=1, min_count=1) / np.sqrt(np.clip(denom, 1e-9, None))
    return k, score.where(k > 0)


def pair_table(rows: pd.DataFrame, dec: pd.DataFrame, cfg: PairConfig | None = None) -> pd.DataFrame:
    """One row per ordered pair (A, B): the parts, the score, the rank among ranked pairs and the pot-size columns."""
    cfg = cfg or PairConfig()
    parts = pd.concat([hand_parts(rows), decision_parts(dec)], ignore_index=True)
    if parts.empty:
        return pd.DataFrame(columns=FIRST + [f"{p}_{c}" for p in PARTS for c in ("n", "mean", "resid", "z")])
    for name in DEFF_PARTS:
        sel = parts["part"] == name
        if sel.any():
            parts.loc[sel, "v"] *= design_effect(parts[sel])
    pairs = _parts_table(parts, cfg, None)
    leave_out: set = set()
    for _ in range(MAX_PASSES):
        _, score = _score(pairs, cfg)
        flagged = pairs[score >= cfg.victim_min_score]
        now = {frozenset((a, b)) for a, b in zip(flagged["A"], flagged["B"])}
        if now == leave_out:
            break
        leave_out = now
        pairs = _parts_table(parts, cfg, leave_out)
    pairs["k"], pairs["score"] = _score(pairs, cfg)
    pairs["hands"] = pairs["wr_n"].fillna(0).astype(int)
    pairs["bb100"] = 100 * pairs["wr_mean"]
    pairs["sd_won"] = pairs["sd_mean"]
    ranked = pairs["k"] >= 2
    n_ranked = int(ranked.sum())
    pairs["field_rank"] = pairs["score"].where(ranked).rank(ascending=False, method="min")
    pairs["n_ranked"] = n_ranked
    pairs["top_pct"] = 100 * pairs["field_rank"] / n_ranked if n_ranked else np.nan
    pairs = pairs.merge(pot_columns(rows, cfg), on=["A", "B"], how="left")
    return pairs[FIRST + [c for c in pairs.columns if c not in FIRST]].sort_values(
        ["field_rank", "score"], ascending=[True, False], na_position="last").reset_index(drop=True)


def victims(pairs: pd.DataFrame, player: str, cfg: PairConfig | None = None) -> pd.DataFrame:
    """The player's victims: his pairs (as A) ranked with at least two parts, ``score >= victim_min_score`` and a rank
    inside ``max(victim_min_rank, ceil(victim_rank_share x n_ranked))``, most suspicious first."""
    cfg = cfg or PairConfig()
    if pairs.empty:
        return pairs.copy()
    n = int(pairs["n_ranked"].iloc[0])
    cut = max(cfg.victim_min_rank, math.ceil(cfg.victim_rank_share * n))
    v = pairs[(pairs["A"] == player) & (pairs["k"] >= 2) & (pairs["score"] >= cfg.victim_min_score)
              & (pairs["field_rank"] <= cut)]
    return v.sort_values("score", ascending=False).reset_index(drop=True)


def pair_label(row: Mapping[str, Any], with_n: bool = True) -> str:
    """'pair #k of N (top x%)' for a ranked pair; with ``with_n`` False only 'top x% of the field's pairs' (for a report
    published from a private field, whose size is nobody else's business); '' when the pair is not ranked."""
    k, n = row.get("field_rank"), row.get("n_ranked")
    if k is None or n is None or pd.isna(k) or not n:
        return ""
    share = 100 * float(k) / float(n)
    top = f"{share:.1f}%" if share >= 0.1 else (f"{share:.2f}%" if share >= 0.01 else "<0.01%")
    return f"pair #{int(k):,} of {int(n):,} (top {top})" if with_n else f"top {top} of the field's pairs"
