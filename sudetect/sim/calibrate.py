"""HUD statistics of a simulated field, computed straight from hand records.

:class:`HudCounter` reads :class:`~sudetect.sim.engine.HandRecord` objects
as the engine produces them (no text, no parser) and keeps, per player,
table size and number of players dealt, the numerators and denominators of
the usual HUD statistics. :func:`summarize` pools them over groups of
players (the honest field, each archetype, each planted player) and
:func:`format_calibration` prints them next to the target ranges of
``[sim.calibration]``.

Definitions (the usual HUD conventions):

* VPIP, PFR: hands with a voluntary call or raise (a raise) before the flop,
  over hands with at least one preflop decision (walks are left out; the big
  blind's free check counts in the denominator only).
* RFI: open raises over first-in opportunities (no raise or limp before).
* 3-bet: raises over decisions facing exactly one raise made by someone else
  with chips behind to raise. Fold to 3-bet: folds over the opener's first
  decision facing a 3-bet.
* Flop c-bet: bets over the preflop aggressor's first flop decision with no
  bet in front of him. Fold to flop c-bet: folds over the first decision of
  another player facing that c-bet before any raise.
* WTSD: hands that reached the showdown over hands that saw the flop.
  W$SD: showdowns where the player collected chips over showdowns.
* AF = (bets + raises) / calls after the flop; AFq = (bets + raises) /
  (bets + raises + calls + folds) after the flop.
* bb/100: net result (after rake) in big blinds per 100 hands dealt; the
  pre-rake version adds back the rake of the pots the player won.

:func:`duel` pits one policy against a sample of the field with shared decks,
which is how the strong regular and a planted superuser are compared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .engine import HandRecord

if TYPE_CHECKING:  # pandas is imported where tables are built
    import pandas as pd

    from .policies import HonestPolicy
    from .population import Population, SimConfig

__all__ = ["FIELDS", "HudCounter", "STATS", "summarize", "player_table", "by_players_dealt", "format_calibration",
           "check_targets", "DuelResult", "duel"]

FIELDS: tuple[str, ...] = (
    "hands", "pf_dec", "vpip", "pfr", "rfi_opp", "rfi", "tb_opp", "tb", "f3b_opp", "f3b",
    "saw_flop", "cbet_opp", "cbet", "fcb_opp", "fcb", "wtsd", "wsd",
    "p_bet", "p_raise", "p_call", "p_fold", "p_check", "net", "won_hands", "rake_won",
)
_F = {name: i for i, name in enumerate(FIELDS)}

STATS: dict[str, tuple[tuple[str, ...], tuple[str, ...], float]] = {
    # name: (numerator fields, denominator fields, scale)
    "VPIP": (("vpip",), ("pf_dec",), 100.0),
    "PFR": (("pfr",), ("pf_dec",), 100.0),
    "RFI": (("rfi",), ("rfi_opp",), 100.0),
    "3bet": (("tb",), ("tb_opp",), 100.0),
    "fold_to_3bet": (("f3b",), ("f3b_opp",), 100.0),
    "flop_cbet": (("cbet",), ("cbet_opp",), 100.0),
    "fold_to_flop_cbet": (("fcb",), ("fcb_opp",), 100.0),
    "saw_flop": (("saw_flop",), ("hands",), 100.0),
    "WTSD": (("wtsd",), ("saw_flop",), 100.0),
    "W$SD": (("wsd",), ("wtsd",), 100.0),
    "AF": (("p_bet", "p_raise"), ("p_call",), 1.0),
    "AFq": (("p_bet", "p_raise"), ("p_bet", "p_raise", "p_call", "p_fold"), 100.0),
}


class HudCounter:
    """Numerators and denominators per ``(player, max_seats, players dealt)``."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int, int], list[int]] = {}

    def _row(self, key: tuple[str, int, int]) -> list[int]:
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = [0] * len(FIELDS)
        return row

    def merge(self, other: "HudCounter") -> "HudCounter":
        for key, vals in other.rows.items():
            row = self._row(key)
            for i, v in enumerate(vals):
                row[i] += v
        return self

    def add_many(self, records: Iterable[HandRecord]) -> None:
        for r in records:
            self.add(r)

    def add(self, rec: HandRecord) -> None:
        """Count one hand."""
        names = {s.seat: s.player for s in rec.seats}
        n = len(names)
        rows = {seat: self._row((name, rec.max_seats, n)) for seat, name in names.items()}
        F = _F
        for row in rows.values():
            row[F["hands"]] += 1

        raises = 0
        limped = False
        first_raiser: Optional[int] = None
        last_raiser: Optional[int] = None
        decided: set[int] = set()
        vpip: set[int] = set()
        pfr: set[int] = set()
        rfi_opp: set[int] = set()
        tb_opp: set[int] = set()
        f3b_opp: set[int] = set()
        actions = rec.actions
        k = 0
        for k, a in enumerate(actions):
            if a.street != "preflop":
                break
            kind = a.kind
            if kind.startswith("post"):
                continue
            s = a.seat
            row = rows[s]
            if s not in decided:
                decided.add(s)
                row[F["pf_dec"]] += 1
            if raises == 0 and not limped and s not in rfi_opp:
                rfi_opp.add(s)
                row[F["rfi_opp"]] += 1
                if kind == "raise":
                    row[F["rfi"]] += 1
            if raises == 1 and last_raiser != s and a.stack_before > a.to_call_before and s not in tb_opp:
                tb_opp.add(s)
                row[F["tb_opp"]] += 1
                if kind == "raise":
                    row[F["tb"]] += 1
            if raises == 2 and first_raiser == s and s not in f3b_opp:
                f3b_opp.add(s)
                row[F["f3b_opp"]] += 1
                if kind == "fold":
                    row[F["f3b"]] += 1
            if kind in ("call", "raise", "bet") and s not in vpip:
                vpip.add(s)
                row[F["vpip"]] += 1
            if kind == "raise":
                if s not in pfr:
                    pfr.add(s)
                    row[F["pfr"]] += 1
                raises += 1
                if raises == 1:
                    first_raiser = s
                last_raiser = s
            elif kind == "call" and raises == 0:
                limped = True
        else:
            k = len(actions)

        flop = len(rec.board) >= 3
        folded = rec.folded_on
        saw = [s for s in names if flop and folded.get(s) != "preflop"]
        for s in saw:
            row = rows[s]
            row[F["saw_flop"]] += 1
            if rec.showdown and s not in folded:
                row[F["wtsd"]] += 1
                if rec.collected.get(s, 0) > 0:
                    row[F["wsd"]] += 1

        # postflop counts, c-bets and folds to c-bets
        aggressor = last_raiser
        cbet_seat: Optional[int] = None
        flop_bets = 0
        first_flop: set[int] = set()
        fcb_done: set[int] = set()
        for a in actions[k:]:
            kind = a.kind
            row = rows[a.seat]
            if kind == "bet":
                row[F["p_bet"]] += 1
            elif kind == "raise":
                row[F["p_raise"]] += 1
            elif kind == "call":
                row[F["p_call"]] += 1
            elif kind == "fold":
                row[F["p_fold"]] += 1
            elif kind == "check":
                row[F["p_check"]] += 1
            if a.street != "flop":
                continue
            s = a.seat
            if s == aggressor and s not in first_flop and flop_bets == 0:
                row[F["cbet_opp"]] += 1
                if kind == "bet":
                    row[F["cbet"]] += 1
                    cbet_seat = s
            elif cbet_seat is not None and s != cbet_seat and flop_bets == 1 and s not in fcb_done:
                fcb_done.add(s)
                row[F["fcb_opp"]] += 1
                if kind == "fold":
                    row[F["fcb"]] += 1
            first_flop.add(s)
            if kind in ("bet", "raise"):
                flop_bets += 1

        for s, row in rows.items():
            row[F["net"]] += rec.net[s]
            if rec.collected.get(s, 0) > 0:
                row[F["won_hands"]] += 1
        for pot in rec.pots:  # the rake of a pot, split like its winnings (for pre-rake results)
            if pot.rake:
                paid = sum(pot.shares.values())
                for s, share in pot.shares.items():
                    rows[s][F["rake_won"]] += pot.rake * share / paid if paid else pot.rake / len(pot.shares)

    # ------------------------------------------------------------------ views

    def frame(self) -> pd.DataFrame:
        """All rows as a pandas DataFrame (player, max_seats, n_dealt, fields...)."""
        import pandas as pd

        data = [(*key, *vals) for key, vals in self.rows.items()]
        return pd.DataFrame(data, columns=["player", "max_seats", "n_dealt", *FIELDS])


def _stat(totals: Mapping[str, float], name: str) -> float:
    num, den, scale = STATS[name]
    d = sum(totals[f] for f in den)
    if d <= 0:
        return float("nan")
    return scale * sum(totals[f] for f in num) / d


def _pool(df: pd.DataFrame, bb: int) -> dict[str, float]:
    totals = {f: float(df[f].sum()) for f in FIELDS}
    out = {name: _stat(totals, name) for name in STATS}
    out["hands"] = totals["hands"]
    out["bb/100"] = 100.0 * totals["net"] / bb / totals["hands"] if totals["hands"] else float("nan")
    out["bb/100 pre-rake"] = (100.0 * (totals["net"] + totals["rake_won"]) / bb / totals["hands"]
                              if totals["hands"] else float("nan"))
    return out


def summarize(counter: HudCounter, groups: Mapping[str, str], bb: int = 100) -> pd.DataFrame:
    """Pooled statistics per group and table kind.

    Args:
        counter: Counts from the run.
        groups: Player name -> group label (archetype, plant id, ...). The
            label ``honest`` is added for every player whose label is an
            archetype; players missing from ``groups`` are ignored.
        bb: Big blind in chip units.

    Returns:
        DataFrame indexed by (kind, group) with one column per statistic,
        ``hands`` and ``bb/100``; kind is ``hu`` (2-seat tables) or ``ring``.
    """
    import pandas as pd

    df = counter.frame()
    df = df[df["player"].isin(groups.keys())].copy()
    df["group"] = df["player"].map(groups)
    df["kind"] = np.where(df["max_seats"] == 2, "hu", "ring")
    rows = []
    for kind, part in df.groupby("kind"):
        honest = part[~part["group"].str.startswith("SU_")]
        rows.append({"kind": kind, "group": "honest", **_pool(honest, bb)})
        for group, g in part.groupby("group"):
            rows.append({"kind": kind, "group": group, **_pool(g, bb)})
    return pd.DataFrame(rows).set_index(["kind", "group"]).sort_index()


def player_table(counter: HudCounter, groups: Mapping[str, str], bb: int = 100) -> pd.DataFrame:
    """Per-player statistics over all tables plus per-kind hands and bb/100."""
    import pandas as pd

    df = counter.frame()
    df = df[df["player"].isin(groups.keys())]
    rows = []
    for player, part in df.groupby("player"):
        rec = {"player": player, "group": groups[player], **_pool(part, bb)}
        for kind, cond in (("hu", part["max_seats"] == 2), ("ring", part["max_seats"] != 2)):
            sub = part[cond]
            rec[f"{kind}_hands"] = float(sub["hands"].sum())
            rec[f"{kind}_bb/100"] = (100.0 * float(sub["net"].sum()) / bb / float(sub["hands"].sum())
                                     if len(sub) and sub["hands"].sum() else float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("player")


def by_players_dealt(counter: HudCounter, honest: Iterable[str]) -> pd.DataFrame:
    """VPIP and PFR of the honest field by number of players dealt."""
    import pandas as pd

    df = counter.frame()
    df = df[df["player"].isin(set(honest))]
    rows = []
    for n, part in df.groupby("n_dealt"):
        t = {f: float(part[f].sum()) for f in FIELDS}
        rows.append({"n_dealt": int(n), "hands": t["hands"], "VPIP": _stat(t, "VPIP"), "PFR": _stat(t, "PFR")})
    return pd.DataFrame(rows).set_index("n_dealt")


def check_targets(summary: pd.DataFrame, targets: Mapping[str, Mapping[str, Sequence[float]]]
                  ) -> list[tuple[str, str, float, float, float, bool]]:
    """``(kind, stat, low, high, achieved, inside)`` for every configured target."""
    out = []
    for kind, stats in targets.items():
        for stat, (lo, hi) in stats.items():
            try:
                value = float(summary.loc[(kind, "honest"), stat])
            except KeyError:
                value = float("nan")
            out.append((kind, stat, float(lo), float(hi), value, bool(lo <= value <= hi)))
    return out


def format_calibration(summary: pd.DataFrame, targets: Mapping[str, Mapping[str, Sequence[float]]],
                       players: Optional[pd.DataFrame] = None, dealt: Optional[pd.DataFrame] = None,
                       min_hands: int = 5000) -> str:
    """Text report: targets vs achieved, the per-group table and the bb/100 spread."""
    lines = ["Honest field vs target ranges", f"{'table':6} {'stat':18} {'target':>13} {'achieved':>9}  ok"]
    for kind, stat, lo, hi, value, ok in check_targets(summary, targets):
        lines.append(f"{kind:6} {stat:18} {lo:6.1f}-{hi:<6.1f} {value:9.2f}  {'yes' if ok else 'NO'}")
    cols = ["hands", "VPIP", "PFR", "RFI", "3bet", "fold_to_3bet", "flop_cbet", "fold_to_flop_cbet",
            "saw_flop", "WTSD", "W$SD", "AF", "AFq", "bb/100", "bb/100 pre-rake"]
    for kind in ("hu", "ring"):
        if kind not in summary.index.get_level_values(0):
            continue
        lines.append("")
        lines.append(f"{'heads-up tables' if kind == 'hu' else '6-max tables'}, pooled by group")
        heads = [c if c != "bb/100 pre-rake" else "pre-rake" for c in cols]
        lines.append(f"{'group':16}" + "".join(f"{c[:9]:>10}" for c in heads))
        part = summary.loc[kind]
        for group, row in part.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                cells.append(f"{v:10.0f}" if c == "hands" else f"{v:10.2f}")
            lines.append(f"{group:16}" + "".join(cells))
    if dealt is not None and len(dealt):
        lines.append("")
        lines.append("Honest VPIP / PFR by players dealt: " + ", ".join(
            f"{n}: {r['VPIP']:.1f}/{r['PFR']:.1f}" for n, r in dealt.iterrows()))
    if players is not None and len(players):
        lines.append("")
        big = players[players["hands"] >= min_hands]
        lines.append(f"bb/100 per player (players with >= {min_hands} hands: {len(big)} of {len(players)})")
        if len(big):
            honest = big[~big["group"].str.startswith("SU_")]
            inside = (honest["bb/100"] >= -30) & (honest["bb/100"] <= 30)
            within = float(inside.mean()) if len(honest) else float("nan")
            lines.append(f"  honest within +-30 bb/100: {100 * within:.0f}%")
            for group, g in big.groupby("group"):
                q = np.percentile(g["bb/100"], [10, 50, 90])
                lines.append(f"  {group:16} n={len(g):3d}  p10 {q[0]:7.1f}  p50 {q[1]:7.1f}  p90 {q[2]:7.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- duels


@dataclass
class DuelResult:
    """A hero's results against a sample of the field.

    Attributes:
        hands: Hands played.
        bb100: Net bb/100 (after rake) and its standard error.
        bb100_pre_rake: bb/100 with the rake of won pots added back.
        by_archetype: Opponent archetype -> (hands, net bb/100) (heads-up only).
        nets: Net result of every hand in big blinds (for paired comparisons).
    """

    hands: int
    bb100: float
    se: float
    bb100_pre_rake: float
    by_archetype: dict[str, tuple[int, float]]
    nets: np.ndarray


def duel(cfg: SimConfig, pop: Population, hero: HonestPolicy, *, n_hands: int, seed: int, max_seats: int = 2,
         block: int = 100, start: int = 0, info: Optional[str] = None) -> DuelResult:
    """Play ``hero`` (a policy object) against honest players of ``pop`` with shared decks.

    The hero meets the population's honest players in turn, ``block`` hands
    each (heads-up), or sits with five of them drawn per block (6-max). Every
    hand starts with 100 bb stacks; hand ``k``'s deck depends only on
    ``(seed, start + k)``, so two heroes run with the same arguments see the
    same cards and the same opponents, and their per-hand results can be
    compared pair by pair.

    Args:
        cfg: Simulator configuration (:class:`~sudetect.sim.population.SimConfig`).
        pop: The population the opponents come from.
        hero: A policy object; its ``name`` must not collide with a player of ``pop``.
        n_hands: Hands to play.
        seed: Seed of the decks and the opponents' decisions.
        max_seats: 2 for heads-up, 6 for 6-max.
        block: Hands per opponent (or per table line-up).
        start: Offset of the first hand number (to split a duel into chunks).
        info: ``"all"`` to let the hero see every opponent's cards.
    """
    from datetime import datetime

    from .engine import SeatState, TableConfig, play_hand
    from .run import make_agents, table_config

    table: TableConfig = table_config(cfg, max_seats)
    agents = make_agents(cfg, pop, seed)
    honest = [p for p in pop.players if p.plant_id is None]
    order = np.random.default_rng([seed, 77]).permutation(len(honest))
    stack = 100 * cfg.bb_units
    t0 = datetime(2001, 1, 1)
    nets = np.zeros(n_hands)
    pre = 0.0
    by: dict[str, list[float]] = {}
    grants = {hero.name: "all"} if info == "all" else None
    for k in range(n_hands):
        hand_no = start + k
        b = hand_no // block
        if max_seats == 2:
            opp = [honest[order[b % len(honest)]]]
        else:
            pick = np.random.default_rng([seed, 78, b]).choice(len(honest), size=max_seats - 1, replace=False)
            opp = [honest[i] for i in pick]
        names = [hero.name] + [p.name for p in opp]
        seats = [SeatState(i + 1, names[i], stack) for i in range(len(names))]
        button = 1 + (hand_no % len(names))
        amap = {hero.name: hero, **{p.name: agents[p.index] for p in opp}}
        rec = play_hand(table, seats, button, amap, 10_000_000 + hand_no, "D", t0,
                        np.random.default_rng([seed, 10_000_000 + hand_no]), grants)
        net = rec.net[1] / cfg.bb_units
        nets[k] = net
        for pot in rec.pots:
            if pot.rake and 1 in pot.shares:
                pre += pot.rake * pot.shares[1] / max(1, sum(pot.shares.values())) / cfg.bb_units
        if max_seats == 2:
            by.setdefault(opp[0].archetype, []).append(net)
    se = float(nets.std(ddof=1) / np.sqrt(n_hands) * 100) if n_hands > 1 else float("nan")
    return DuelResult(
        hands=n_hands, bb100=float(nets.mean() * 100), se=se,
        bb100_pre_rake=float((nets.sum() + pre) / n_hands * 100),
        by_archetype={a: (len(v), float(np.mean(v) * 100)) for a, v in sorted(by.items())},
        nets=nets,
    )
