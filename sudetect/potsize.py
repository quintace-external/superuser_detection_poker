"""Observer metrics: what someone who only reads the hand histories can measure without the hidden cards.

Mined hand histories (and any history the operator did not deal) show hole cards only at showdowns, so every test
that needs the real cards behind a fold (method b, the preflop and flop tests, flags A and D) is blind there. Money
and showdowns survive. A player who sees the cards wins an unusual share of the big pots, wins the showdowns he
chooses to reach, skips the ones he would lose and almost never loses his stack on the river; these metrics look at
exactly that, split by pot size, because the edge of seeing the cards grows with what is at stake.

One row per hand x player (:data:`HX_SQL`), then one row per player (:func:`player_metrics`), columns ``obs_*``:

* ``obs_bb100``: net per 100 hands; ``obs_net_b1`` .. ``obs_net_b4``: net bb in pots under 30bb, 30-100, 100-200 and
  200bb or more (the final contested pot, uncalled bets excluded); ``obs_big_net100``: net in 200bb+ pots per 100
  hands;
* ``obs_wtsd``: went to showdown among the hands that saw the flop, and ``obs_wtsd_srp`` / ``_3bp`` / ``_4bp`` by the
  pot at the flop (under 15bb, 15-40bb, 40bb or more: roughly single-raised, 3-bet and 4-bet pots); lower = more
  suspicious: a player who sees the cards skips the showdowns he would lose;
* ``obs_wsd``: showdowns won (exactly two players at the showdown; a split counts 1/2) and ``obs_wsd_b1`` ..
  ``obs_wsd_b4`` by pot size; ``obs_wsd_growth`` = ``obs_wsd_b4`` - ``obs_wsd_b1`` (how much better he does when
  the pot is big);
* ``obs_rbet_win``: he bet or raised the river, the pot was over 30bb and the hand reached a showdown: share won;
* ``obs_rcall_win``: he called a river bet heads-up and the pot after the call was over 30bb: share won;
  ``obs_rcall_freq``: those calls over the river bets he faced that way; lower = more suspicious (he folds the bets
  he would lose), so a low rate next to a high ``obs_rcall_win`` is the pattern;
* ``obs_stackoff``: he saw the river and more than 30bb went in on it: share of those hands in which he lost 90% or
  more of his effective starting stack (the smaller of his stack and the biggest other stack; lower = more
  suspicious).

Only standard hands without a parsing error count (no bomb pots, as everywhere in step 05). ``obs_wsd_b1`` (small
pots) is shown but not ranked: on the simulator a player who sees the cards wins fewer of the small showdowns (he
reaches them with the hands he can check down) and more of the big ones, which is what ``obs_wsd_growth`` measures.
A two-player showdown in which both collect (a side pot built by a third player who folded) counts 1/2 each; that is
rare and left as is.

Every rate has its count ``<name>_n``. :func:`field_ranks` adds ``<name>_pct`` (mid-rank percentile, 0-100, higher =
more suspicious) and ``<name>_rank`` / ``<name>_of`` among players with enough hands and cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .warehouse import Connection, query_df, table_columns

BIG_POT = 30.0                    # bb: the "big pot" line of the river metrics and of the pair columns
POT_EDGES = (30.0, 100.0, 200.0)  # bb, final pot: b1 < 30 <= b2 < 100 <= b3 < 200 <= b4
FLOP_EDGES = (15.0, 40.0)         # bb, pot at the flop: single-raised < 15 <= 3-bet < 40 <= 4-bet and more
STACKOFF_SHARE = 0.9              # a stack-off: net <= -0.9 x the effective starting stack

# one row per hand x player; :big is BIG_POT; {stack} and {where} are filled by hx_sql() from the warehouse's columns
HX_SQL = """
win AS (
    SELECT hand_id, count(*) FILTER (WHERE went_to_showdown AND won_at_showdown) AS n_win
    FROM hand_players GROUP BY 1
),
rv AS (
    SELECT hand_id, player_id, bool_or(action IN ('bet', 'raise')) AS raggr
    FROM actions WHERE street = 'river' AND NOT is_post GROUP BY 1, 2
),
rc AS (
    SELECT hand_id, player_id,
           bool_or(action = 'call' AND pot_before_bb + to_call_bb > :big) AS rcall_big,
           bool_or(pot_before_bb + to_call_bb > :big) AS rfaced_big
    FROM decisions WHERE street = 'river' AND n_live = 2 AND to_call_bb > 0 GROUP BY 1, 2
),
hx AS (
    SELECT p.hand_id, p.player_id, p.net_bb AS net, {stack} AS stack,
           COALESCE(p.saw_flop, FALSE) AS saw_flop, COALESCE(p.saw_river, FALSE) AS saw_river,
           COALESCE(p.went_to_showdown, FALSE) AS went_sd,
           COALESCE(p.went_to_showdown, FALSE) AND h.n_showdown = 2 AS sd,
           CASE WHEN p.won_at_showdown THEN 1.0 / greatest(w.n_win, 1) ELSE 0.0 END AS won,
           h.final_pot_bb AS pot, h.pot_flop_bb AS pot_flop, h.final_pot_bb - h.pot_river_bb AS river_in,
           COALESCE(rv.raggr, FALSE) AS raggr, COALESCE(rc.rcall_big, FALSE) AS rcall_big,
           COALESCE(rc.rfaced_big, FALSE) AS rfaced_big
    FROM hand_players p
    JOIN hands h ON h.hand_id = p.hand_id
    JOIN win w ON w.hand_id = p.hand_id
    LEFT JOIN rv ON rv.hand_id = p.hand_id AND rv.player_id = p.player_id
    LEFT JOIN rc ON rc.hand_id = p.hand_id AND rc.player_id = p.player_id
    {where}
)"""

_PLAYER_SQL = """
SELECT player_id, count(*) AS obs_hands, sum(net) AS net,
       sum(net) FILTER (WHERE pot < :e1) AS obs_net_b1,
       sum(net) FILTER (WHERE pot >= :e1 AND pot < :e2) AS obs_net_b2,
       sum(net) FILTER (WHERE pot >= :e2 AND pot < :e3) AS obs_net_b3,
       sum(net) FILTER (WHERE pot >= :e3) AS obs_net_b4,
       count(*) FILTER (WHERE saw_flop) AS sf_n, count(*) FILTER (WHERE saw_flop AND went_sd) AS sf_k,
       count(*) FILTER (WHERE saw_flop AND pot_flop < :f1) AS srp_n,
       count(*) FILTER (WHERE saw_flop AND pot_flop < :f1 AND went_sd) AS srp_k,
       count(*) FILTER (WHERE saw_flop AND pot_flop >= :f1 AND pot_flop < :f2) AS tbp_n,
       count(*) FILTER (WHERE saw_flop AND pot_flop >= :f1 AND pot_flop < :f2 AND went_sd) AS tbp_k,
       count(*) FILTER (WHERE saw_flop AND pot_flop >= :f2) AS fbp_n,
       count(*) FILTER (WHERE saw_flop AND pot_flop >= :f2 AND went_sd) AS fbp_k,
       count(*) FILTER (WHERE sd) AS sd_n, sum(won) FILTER (WHERE sd) AS sd_k,
       count(*) FILTER (WHERE sd AND pot < :e1) AS sd1_n, sum(won) FILTER (WHERE sd AND pot < :e1) AS sd1_k,
       count(*) FILTER (WHERE sd AND pot >= :e1 AND pot < :e2) AS sd2_n,
       sum(won) FILTER (WHERE sd AND pot >= :e1 AND pot < :e2) AS sd2_k,
       count(*) FILTER (WHERE sd AND pot >= :e2 AND pot < :e3) AS sd3_n,
       sum(won) FILTER (WHERE sd AND pot >= :e2 AND pot < :e3) AS sd3_k,
       count(*) FILTER (WHERE sd AND pot >= :e3) AS sd4_n, sum(won) FILTER (WHERE sd AND pot >= :e3) AS sd4_k,
       count(*) FILTER (WHERE sd AND raggr AND pot > :big) AS rbet_n,
       sum(won) FILTER (WHERE sd AND raggr AND pot > :big) AS rbet_k,
       count(*) FILTER (WHERE sd AND rcall_big) AS rcall_n, sum(won) FILTER (WHERE sd AND rcall_big) AS rcall_k,
       count(*) FILTER (WHERE rfaced_big) AS rfaced_n, count(*) FILTER (WHERE rcall_big) AS rfaced_k,
       count(*) FILTER (WHERE saw_river AND river_in > :big) AS so_n,
       count(*) FILTER (WHERE saw_river AND river_in > :big AND net <= -:so * stack) AS so_k
FROM hx GROUP BY 1
"""


def hx_sql(con: Connection) -> str:
    """:data:`HX_SQL` for this warehouse: the effective stack when step 01 wrote it (else the own stack), and only
    standard hands without a parsing error when the hands table carries those columns."""
    hp, hc = set(table_columns(con, "hand_players")), set(table_columns(con, "hands"))
    stack = "COALESCE(p.eff_stack_start_bb, p.stack_start_bb)" if "eff_stack_start_bb" in hp else "p.stack_start_bb"
    where = [c for c, ok in (("h.variant = 'standard'", "variant" in hc), ("NOT h.error_flag", "error_flag" in hc))
             if ok]
    return HX_SQL.format(stack=stack, where=("WHERE " + " AND ".join(where)) if where else "")

# rate column -> (numerator, denominator, direction: +1 when higher is more suspicious, -1 when lower, 0 = shown
# but not ranked)
RATES: dict[str, tuple[str, str, int]] = {
    "obs_wtsd": ("sf_k", "sf_n", -1),
    "obs_wtsd_srp": ("srp_k", "srp_n", -1),
    "obs_wtsd_3bp": ("tbp_k", "tbp_n", -1),
    "obs_wtsd_4bp": ("fbp_k", "fbp_n", -1),
    "obs_wsd": ("sd_k", "sd_n", 1),
    "obs_wsd_b1": ("sd1_k", "sd1_n", 0),
    "obs_wsd_b2": ("sd2_k", "sd2_n", 1),
    "obs_wsd_b3": ("sd3_k", "sd3_n", 1),
    "obs_wsd_b4": ("sd4_k", "sd4_n", 1),
    "obs_rbet_win": ("rbet_k", "rbet_n", 1),
    "obs_rcall_win": ("rcall_k", "rcall_n", 1),
    "obs_rcall_freq": ("rfaced_k", "rfaced_n", -1),
    "obs_stackoff": ("so_k", "so_n", -1),
}
# the per-hands metrics, ranked among players with enough hands
PER_HAND = ("obs_bb100", "obs_big_net100")


def player_metrics(con: Connection, *, big_pot: float = BIG_POT, pot_edges=POT_EDGES, flop_edges=FLOP_EDGES,
                   stackoff_share: float = STACKOFF_SHARE) -> pd.DataFrame:
    """One row per player (index ``player_id``) with the ``obs_*`` rates, their ``_n`` counts and the net columns."""
    t = query_df(con, "WITH " + hx_sql(con) + _PLAYER_SQL, big=float(big_pot), e1=float(pot_edges[0]), e2=float(pot_edges[1]),
                 e3=float(pot_edges[2]), f1=float(flop_edges[0]), f2=float(flop_edges[1]),
                 so=float(stackoff_share)).set_index("player_id")
    return metrics_from_counts(t)


def metrics_from_counts(t: pd.DataFrame) -> pd.DataFrame:
    """The rates of :func:`player_metrics` from its count columns (also used on sums of several accounts)."""
    out = pd.DataFrame(index=t.index)
    hands = t["obs_hands"].astype(float)
    out["obs_hands"] = t["obs_hands"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["obs_bb100"] = 100 * t["net"] / hands.where(hands > 0)
        for b in range(1, 5):
            out[f"obs_net_b{b}"] = t[f"obs_net_b{b}"].fillna(0.0)
        out["obs_big_net100"] = 100 * out["obs_net_b4"] / hands.where(hands > 0)
        for name, (k, n, _) in RATES.items():
            den = t[n].astype(float)
            out[name] = t[k].astype(float) / den.where(den > 0)
            out[f"{name}_n"] = t[n].fillna(0).astype(int)
    out["obs_wsd_growth"] = out["obs_wsd_b4"] - out["obs_wsd_b1"]
    out["obs_growth_n"] = np.minimum(out["obs_wsd_b4_n"], out["obs_wsd_b1_n"])
    return out


def field_ranks(m: pd.DataFrame, *, min_hands: int = 1000, min_cases: int = 20) -> pd.DataFrame:
    """``<name>_pct`` (0-100, higher = more suspicious), ``<name>_rank`` and ``<name>_of`` for every ``obs_*`` metric,
    among the players with at least ``min_hands`` hands and, for a rate, ``min_cases`` cases; others get NaN."""
    out = m.copy()
    base = out["obs_hands"] >= min_hands
    cols = [(c, "obs_hands", 1) for c in PER_HAND] + [(c, f"{c}_n", d) for c, (_, _, d) in RATES.items() if d]
    cols.append(("obs_wsd_growth", "obs_growth_n", 1))
    for col, ncol, d in cols:
        ok = base & (out[ncol] >= (min_cases if ncol != "obs_hands" else min_hands)) & out[col].notna()
        v = out.loc[ok, col] * d
        out[f"{col}_pct"] = np.nan
        out[f"{col}_rank"] = np.nan
        out[f"{col}_of"] = int(ok.sum())
        if ok.any():
            out.loc[ok, f"{col}_pct"] = 100 * (v.rank(method="average") - 0.5) / len(v)
            out.loc[ok, f"{col}_rank"] = v.rank(method="min", ascending=False)
    return out


def from_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keyword arguments for :func:`player_metrics` and :func:`field_ranks` from a ``[step05.observer]`` table."""
    c = dict(cfg or {})
    return {"big_pot": float(c.get("big_pot_bb", BIG_POT)), "pot_edges": tuple(c.get("pot_edges_bb", POT_EDGES)),
            "flop_edges": tuple(c.get("flop_edges_bb", FLOP_EDGES)),
            "stackoff_share": float(c.get("stackoff_share", STACKOFF_SHARE)),
            "min_hands": int(c.get("min_hands", 1000)), "min_cases": int(c.get("min_cases", 20))}
