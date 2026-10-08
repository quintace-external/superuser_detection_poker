"""Predicted range: the hands an opponent could hold, given what he has done in the hand.

The range model answers, for one decision of the hero and one opponent still in
the hand, "which of his 1,326 possible holdings are likely, given his preflop
line and every postflop action he has taken so far?". The hero's equity against
that weighted range is the reference that detection method (b) compares a
player's decisions with: an honest player can act on it, only a player who sees
the cards can act on the equity against the real hand.

Everything the model knows comes from three field tables fitted on a
full-information dataset (``04_predicted_range/sql``) and from two statistics
of the opponent. The steps, for one opponent:

1. **Prior.** The distribution over the 169 starting-hand classes of the hands
   that players in the same preflop spot took to the flop (table
   ``rng_prior_pre``). The key is (table-size class, situation of his last
   preflop betting line, position class, action of that line). The lookup backs
   off P4 -> P3 -> P2 -> P1 -> P1 with call and raise merged -> P0 and uses the
   first cell holding at least ``prior_min_support`` hands.
2. **Looseness tilt.** With at least ``loose_min_n`` postflop actions on
   record, ``r = clip(stat / field_stat, 0.5, 2)``, where ``stat`` is his PFR if
   he raised before the flop and his VPIP otherwise, reshapes the prior::

       pi1(c) ~ pi0(c) * r ** (beta * (0.5 - s_c))

   ``s_c`` is the class's strength percentile (:data:`sudetect.handclass.CHEN_PERCENTILE`,
   1 for AA, 0 for 32o) and ``beta = 1.4``. A loose player (``r > 1``) gets
   more weak hands, a tight one more strong hands.
3. **Combos.** Classes with ``pi1 >= class_floor`` and a positive weight expand
   to their live combos (no card shared with the hero's hand or the board). With
   ``combo_weight = "class_size"`` (the default) every live combo gets
   ``pi1(c) / |c|``, ``|c|`` = 6, 4 or 12 combos: the probability of that combo
   once the dead cards are known. With ``"live_combos"`` (the design this module
   ports) a class's weight is split over its live combos only, so a class the
   board half blocks keeps its whole weight on the combos left and moves up the
   order a cap cuts by.
4. **Narrowing.** For each of his postflop actions so far, in order::

       w(h) <- w(h) * (P(a | street, situation, pose, pot type, made(h), draw(h)) + eps)

   ``made(h)`` and ``draw(h)`` are the classes of combo ``h`` on the board of
   that action (:func:`sudetect.handclass.classes_for_all_combos`). ``P`` comes
   from ``rng_lik_post`` at the finest of the levels K6 ... K1 whose cell holds at
   least ``lik_min_support`` actions, as ``(N_a + eps) / (N + 5 eps)``, and is
   0.2 when no level qualifies.
5. **Aggression tilt.** On his bets and raises ``P`` is multiplied by
   ``(af ** b_g) ** omega`` with ``af = clip(agg / field_agg, 0.5, 2)``,
   ``omega = clip((n - 8) / (50 - 8), 0, 1)`` and ``b_g`` set by the combo's
   made-hand group (0 for a full house or better up to 1 for no pair). The bets
   of an aggressive player keep more air in his range.
6. **Pruning.** After step 3 and after every update, combos lighter than
   ``prune_frac`` times the heaviest are dropped, the ``cap`` heaviest are kept
   and the weights are normalised to sum to 1. The defaults (``cap = 1326``,
   ``prune_frac = 0``, ``class_floor = 0``) keep every live combo with a
   positive weight, i.e. the whole posterior; the narrow-setting constants (240 combos,
   ``2e-4``, ``0.0008``) are a variant of the step's configuration. With
   ``prune_each_step = False`` every live combo is updated and the range is cut
   once, at the end: classes holding less than ``class_floor`` of the range are
   dropped, then the same pruning applies.

The opponent's actions are read from the action table, so the spot of every
update (situation, pose, pot type) comes from the same functions that keyed the
field tables when they were fitted (:mod:`sudetect.situations`).

Several opponents get one range each, built independently with the hero's cards
and the board as the only dead cards; :func:`equity_vs_ranges_mc` resolves card
clashes between them when it deals. Heads-up the equity is exact
(:func:`sudetect.equity.equity_vs_range_batch`).

The worker functions at the end (:func:`init_worker`, :func:`score_chunk`) run
the model over parquet chunks in a process pool; ``04_predicted_range/run.py``
stages the chunks and writes the results to the warehouse.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from .cards import CLASS_LABELS, COMBO_INDEX, COMBO_TO_CLASS, COMBOS, N_CLASSES, N_COMBOS, RANKS, SUITS
from .handclass import CHEN_PERCENTILE, DRAW_CLASSES, MADE_CLASSES, classes_for_all_combos

__all__ = [
    "ACTIONS",
    "LIK_LEVELS",
    "PRIOR_LEVELS",
    "GROUPS",
    "MADE_GROUP",
    "RangeConfig",
    "FitTables",
    "Node",
    "OpponentLine",
    "RangeResult",
    "size_class",
    "cards_of",
    "looseness_tilt",
    "aggression_tilt",
    "expand_prior",
    "prune",
    "floor_and_prune",
    "predict_range",
    "range_dense",
    "range_classes",
    "top_class",
    "group_shares",
    "class_grid",
    "equity_vs_ranges_mc",
    "row_seed",
    "init_worker",
    "score_chunk",
]

#: The five actions of the likelihood tables, in column order.
ACTIONS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise")
ACTION_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
#: Levels of ``rng_lik_post``, finest first.
LIK_LEVELS: tuple[str, ...] = ("K6", "K5", "K4", "K3", "K2", "K1")
#: Upper edges of the size tiers t1 ... t4, as a fraction of the pot (effective size, all-ins included); t5 is the rest.
SIZE_EDGES: tuple[float, ...] = (0.40, 0.60, 0.90, 1.10)
SIZE_TIERS: tuple[str, ...] = ("t1", "t2", "t3", "t4", "t5")
#: Faced tier of an action that faces no bet (a check or a bet).
FACED_NONE = "none"
#: Outcomes of ``rng_lik_size``, in column order: every bet and raise split by the tier of its size.
SIZED_ACTIONS: tuple[str, ...] = (("fold", "check", "call") + tuple(f"bet_{t}" for t in SIZE_TIERS)
                                  + tuple(f"raise_{t}" for t in SIZE_TIERS))
SIZED_INDEX: dict[str, int] = {a: i for i, a in enumerate(SIZED_ACTIONS)}
#: Levels of ``rng_lik_size``: the levels of LIK_LEVELS with the faced tier added (same order), then the field's
#: size mix per (street, faced tier) and per street, the last back-off of the tier distribution.
SIZED_LEVELS: tuple[str, ...] = ("S6", "S5", "S4", "S3", "S2", "S1")
FIELD_SIZE_LEVELS: tuple[str, ...] = ("F1", "F0")
_TIER_SLICE = {"bet": slice(3, 8), "raise": slice(8, 13)}
#: Levels of ``rng_prior_pre``, finest first; ``P1m`` is P1 with call and raise merged (built at lookup time).
PRIOR_LEVELS: tuple[str, ...] = ("P4", "P3", "P2", "P1", "P1m", "P0")
#: Made-hand groups of the aggression tilt, strongest first.
GROUPS: tuple[str, ...] = ("monster", "strong", "top_pair", "middle_pair", "weak_pair", "air")
#: Group of each made-hand class of :mod:`sudetect.handclass`.
MADE_GROUP: dict[str, str] = {
    "nothing": "air",
    "underpair": "weak_pair",
    "weak_pair": "weak_pair",
    "middle_pair": "middle_pair",
    "top_pair_weak": "top_pair",
    "top_pair_good": "top_pair",
    "overpair": "strong",
    "two_pair": "strong",
    "trips_set": "strong",
    "straight": "strong",
    "flush": "strong",
    "full_house_plus": "monster",
}
_GROUP_OF_MADE = np.array([GROUPS.index(MADE_GROUP[m]) for m in MADE_CLASSES], dtype=np.intp)
_N_MADE = len(MADE_CLASSES)
_N_DRAW = len(DRAW_CLASSES)
_UNIFORM_LEVEL = len(LIK_LEVELS)          # index used when no likelihood level qualifies
_STRENGTH = np.asarray(CHEN_PERCENTILE, dtype=np.float64)
_C0 = COMBOS[:, 0].astype(np.intp)
_C1 = COMBOS[:, 1].astype(np.intp)
_CLASS_OF = COMBO_TO_CLASS.astype(np.intp)
_CLASS_SIZE = np.bincount(_CLASS_OF, minlength=N_CLASSES).astype(np.float64)   # 6, 4 or 12 combos per class
_CARD_ID: dict[str, int] = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeConfig:
    """Constants of the range model: the [prior], [likelihood], [tilt] and [range] tables of
    ``config/step04_range.toml`` (and ``fit.table_size_classes``)."""

    prior_min_support: int = 30
    class_floor: float = 0.0
    lik_min_support: int = 25
    eps: float = 1e-4
    loose_beta: float = 1.4
    loose_min_n: int = 15
    agg_min_n: int = 8
    agg_full_n: int = 50
    ratio_min: float = 0.5
    ratio_max: float = 2.0
    bluff_exponent: tuple[float, ...] = (0.0, 0.05, 0.15, 0.45, 0.8, 1.0)
    cap: int = 1326
    prune_frac: float = 0.0
    prune_each_step: bool = True
    combo_weight: str = "class_size"
    use_size_class: bool = True
    sizing: bool = True

    def __post_init__(self) -> None:
        if self.combo_weight not in ("class_size", "live_combos"):
            raise ValueError(f"combo_weight must be 'class_size' or 'live_combos', not {self.combo_weight!r}")
        if len(self.bluff_exponent) != len(GROUPS):
            raise ValueError(f"bluff_exponent needs one value per group {GROUPS}")
        if self.cap < 1:
            raise ValueError("cap must be at least 1")

    @classmethod
    def from_toml(cls, cfg: Mapping[str, Any]) -> "RangeConfig":
        """Build from the parsed TOML file (the whole document)."""
        pr, lk, tl, rg = cfg["prior"], cfg["likelihood"], cfg["tilt"], cfg["range"]
        bluff = tl["bluff_exponent"]
        return cls(
            prior_min_support=int(pr["min_support"]),
            class_floor=float(pr["class_floor"]),
            lik_min_support=int(lk["min_support"]),
            eps=float(lk["eps"]),
            loose_beta=float(tl["loose_beta"]),
            loose_min_n=int(tl["loose_min_n"]),
            agg_min_n=int(tl["agg_min_n"]),
            agg_full_n=int(tl["agg_full_n"]),
            ratio_min=float(tl["ratio_min"]),
            ratio_max=float(tl["ratio_max"]),
            bluff_exponent=tuple(float(bluff[g]) for g in GROUPS),
            cap=int(rg["cap"]),
            prune_frac=float(rg["prune_frac"]),
            prune_each_step=bool(rg.get("prune_each_step", True)),
            combo_weight=str(rg.get("combo_weight", "class_size")),
            use_size_class=bool(cfg["fit"]["table_size_classes"]),
            sizing=bool(rg.get("sizing", False)),
        )


def size_class(n_dealt: int) -> str:
    """Table-size class of a hand: ``'2'``, ``'3-6'`` or ``'7+'`` players dealt in."""
    return "2" if n_dealt <= 2 else ("3-6" if n_dealt <= 6 else "7+")


def size_tier(frac: Any) -> str:
    """Tier of a bet or raise from its effective size as a fraction of the pot (``actions.size_pot``):
    t1 below 0.40, t2 from 0.40, t3 from 0.60, t4 from 0.90, t5 from 1.10 (all-ins by their effective size).
    ``''`` when the size is unknown (None, NaN, zero or negative)."""
    try:
        f = float(frac)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f) or f <= 0:
        return ""
    for edge, name in zip(SIZE_EDGES, SIZE_TIERS):
        if f < edge:
            return name
    return SIZE_TIERS[-1]


def faced_tier(frac: Any) -> str:
    """Tier of the bet or raise an action faces (``actions.faced_size_pot``); ``'none'`` when it faces none."""
    return size_tier(frac) or FACED_NONE


def sized_outcome(action: str, tier: str) -> Optional[int]:
    """Column of :data:`SIZED_ACTIONS` for an action and its own size tier (None: a bet or raise of unknown size)."""
    if action in ("fold", "check", "call"):
        return SIZED_INDEX[action]
    if action in ("bet", "raise") and tier in SIZE_TIERS:
        return SIZED_INDEX[f"{action}_{tier}"]
    return None


@lru_cache(maxsize=65536)
def cards_of(text: str) -> tuple[int, ...]:
    """Card ids of a compact card string as stored in the warehouse (``'AhKd'``, ``'AsKd7c'``; ``''`` -> ())."""
    if not text:
        return ()
    if len(text) % 2:
        raise ValueError(f"not a compact card string: {text!r}")
    return tuple(_CARD_ID[text[i:i + 2]] for i in range(0, len(text), 2))


# ----------------------------------------------------------------------------
# Field tables
# ----------------------------------------------------------------------------

def _clean(value: Any) -> Optional[str]:
    """A key part as stored: None for SQL NULL / NaN, else the string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


@dataclass
class FitTables:
    """The fitted field tables, as dictionaries keyed the way the model looks them up.

    Attributes:
        lik: level (``K6`` ... ``K1``) -> key tuple -> five action counts (fold, check, call, bet, raise).
        prior: level (``P4`` ... ``P0``) -> key tuple -> 169 class counts in :data:`sudetect.cards.CLASS_LABELS` order.
        field: table-size class (``'2'``, ``'3-6'``, ``'7+'``, ``'all'``) -> (VPIP, PFR, aggression) of the field.
        lik_size: level (``S6`` ... ``S1``, ``F1``, ``F0``) -> key tuple (faced tier last) -> 13 outcome counts in
            :data:`SIZED_ACTIONS` order (``rng_lik_size``); empty when the tables were fitted without it.
    """

    lik: dict[str, dict[tuple, np.ndarray]]
    prior: dict[str, dict[tuple, np.ndarray]]
    field: dict[str, tuple[Optional[float], Optional[float], Optional[float]]]
    lik_size: dict[str, dict[tuple, np.ndarray]] = field(default_factory=dict)
    _node_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _prior_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __getstate__(self) -> dict:
        return {"lik": self.lik, "prior": self.prior, "field": self.field, "lik_size": self.lik_size}

    def __setstate__(self, state: dict) -> None:
        self.lik, self.prior, self.field = state["lik"], state["prior"], state["field"]
        self.lik_size = state.get("lik_size", {})
        self._node_cache, self._prior_cache = {}, {}

    # -------------------------------------------------------------- building

    @classmethod
    def from_frames(cls, lik_df: Any, prior_df: Any, field_df: Any, size_df: Any = None) -> "FitTables":
        """Build from the warehouse tables ``rng_lik_post``, ``rng_prior_pre``, ``rng_field`` and (optional)
        ``rng_lik_size`` (pandas frames)."""
        lik: dict[str, dict[tuple, np.ndarray]] = {lv: {} for lv in LIK_LEVELS}
        key_cols = {
            "K6": ("street", "situation", "pose", "pot_type", "made_hand", "draw"),
            "K5": ("street", "situation", "pose", "made_hand", "draw"),
            "K4": ("street", "situation", "made_hand", "draw"),
            "K3": ("street", "situation", "made_hand"),
            "K2": ("street", "situation", "pose", "pot_type"),
            "K1": ("street", "situation"),
        }
        counts = lik_df[["c_fold", "c_check", "c_call", "c_bet", "c_raise"]].to_numpy(dtype=np.float64)
        for i, row in enumerate(lik_df[["level", "street", "situation", "pose", "pot_type", "made_hand",
                                        "draw"]].itertuples(index=False)):
            level = row.level
            key = tuple(_clean(getattr(row, c)) for c in key_cols[level])
            lik[level][key] = counts[i]
        prior: dict[str, dict[tuple, np.ndarray]] = {lv: {} for lv in ("P4", "P3", "P2", "P1", "P0")}
        p_cols = {
            "P4": ("size_class", "preflop_situation", "position_class", "action"),
            "P3": ("preflop_situation", "position_class", "action"),
            "P2": ("preflop_situation", "position_class"),
            "P1": ("position_class", "action"),
            "P0": (),
        }
        class_pos = {label: i for i, label in enumerate(CLASS_LABELS)}
        for row in prior_df[["level", "size_class", "preflop_situation", "position_class", "action", "hand_class",
                             "n"]].itertuples(index=False):
            key = tuple(_clean(getattr(row, c)) for c in p_cols[row.level])
            arr = prior[row.level].get(key)
            if arr is None:
                arr = prior[row.level][key] = np.zeros(N_CLASSES)
            arr[class_pos[row.hand_class]] += float(row.n)
        fld: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {}
        for row in field_df.itertuples(index=False):
            fld[str(row.size_class)] = tuple(  # type: ignore[assignment]
                None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                for v in (row.field_vpip, row.field_pfr, row.field_agg)
            )
        size: dict[str, dict[tuple, np.ndarray]] = {lv: {} for lv in SIZED_LEVELS + FIELD_SIZE_LEVELS}
        if size_df is not None and len(size_df):
            s_cols = {
                "S6": ("street", "situation", "pose", "pot_type", "made_hand", "draw", "faced_tier"),
                "S5": ("street", "situation", "pose", "made_hand", "draw", "faced_tier"),
                "S4": ("street", "situation", "made_hand", "draw", "faced_tier"),
                "S3": ("street", "situation", "made_hand", "faced_tier"),
                "S2": ("street", "situation", "pose", "pot_type", "faced_tier"),
                "S1": ("street", "situation", "faced_tier"),
                "F1": ("street", "faced_tier"),
                "F0": ("street",),
            }
            s_counts = size_df[["c_" + a for a in SIZED_ACTIONS]].to_numpy(dtype=np.float64)
            for i, row in enumerate(size_df[["level", "street", "situation", "pose", "pot_type", "made_hand", "draw",
                                             "faced_tier"]].itertuples(index=False)):
                size[row.level][tuple(_clean(getattr(row, c)) for c in s_cols[row.level])] = s_counts[i]
        return cls(lik=lik, prior=prior, field=fld, lik_size=size)

    # -------------------------------------------------------------- lookups

    def node_table(self, street: str, situation: str, pose: str, pot_type: str,
                   cfg: RangeConfig) -> tuple[np.ndarray, np.ndarray]:
        """Action probabilities of every (made hand, draw) cell of one spot.

        Returns ``(P, level)``: ``P[m, d, a]`` = P(action a | spot, made class m, draw class d), shape (12, 5, 5),
        and the index into :data:`LIK_LEVELS` of the level each cell came from (6 = no level qualified, uniform).
        Cached per spot.
        """
        key = (street, situation, pose, pot_type)
        hit = self._node_cache.get(key)
        if hit is not None:
            return hit
        probs = np.full((_N_MADE, _N_DRAW, len(ACTIONS)), 1.0 / len(ACTIONS))
        level = np.full((_N_MADE, _N_DRAW), _UNIFORM_LEVEL, dtype=np.int8)
        k2 = self.lik["K2"].get((street, situation, pose, pot_type))
        k1 = self.lik["K1"].get((street, situation))
        for mi, made in enumerate(MADE_CLASSES):
            k3 = self.lik["K3"].get((street, situation, made))
            for di, drw in enumerate(DRAW_CLASSES):
                for li, arr in enumerate((
                    self.lik["K6"].get((street, situation, pose, pot_type, made, drw)),
                    self.lik["K5"].get((street, situation, pose, made, drw)),
                    self.lik["K4"].get((street, situation, made, drw)),
                    k3, k2, k1,
                )):
                    if arr is not None and arr.sum() >= cfg.lik_min_support:
                        probs[mi, di] = (arr + cfg.eps) / (arr.sum() + len(ACTIONS) * cfg.eps)
                        level[mi, di] = li
                        break
        out = (probs, level)
        self._node_cache[key] = out
        return out

    def sized_node_table(self, street: str, situation: str, pose: str, pot_type: str, faced: str,
                         cfg: RangeConfig) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities of the 13 sized outcomes in every (made hand, draw) cell of one spot and faced tier.

        Returns ``(P, direct)``: ``P[m, d, o]`` over :data:`SIZED_ACTIONS`, shape (12, 5, 13), each cell summing
        to 1, and whether the cell came from a populated sized cell. A cell reads the ``rng_lik_size`` cell at the
        level its plain likelihood comes from (:meth:`node_table`), with the faced tier added to the key, when that
        cell holds at least ``lik_min_support`` actions. Otherwise fold, check and call keep their plain
        probability and a bet (raise) gets its plain probability times the tier mix of the bets (raises) at the
        first sized level from there on holding ``lik_min_support`` of them, else of the field on that street and
        faced tier, else of the street, else 1/5 per tier. Cached per spot and faced tier.
        """
        key = ("sized", street, situation, pose, pot_type, faced)
        hit = self._node_cache.get(key)
        if hit is not None:
            return hit
        plain, plain_level = self.node_table(street, situation, pose, pot_type, cfg)
        n_out = len(SIZED_ACTIONS)
        sz = self.lik_size
        empty: dict[tuple, np.ndarray] = {}
        field_cells = (sz.get("F1", empty).get((street, faced)), sz.get("F0", empty).get((street,)))
        probs = np.empty((_N_MADE, _N_DRAW, n_out))
        direct = np.zeros((_N_MADE, _N_DRAW), dtype=bool)
        for mi, made in enumerate(MADE_CLASSES):
            for di, drw in enumerate(DRAW_CLASSES):
                cells = (
                    sz.get("S6", empty).get((street, situation, pose, pot_type, made, drw, faced)),
                    sz.get("S5", empty).get((street, situation, pose, made, drw, faced)),
                    sz.get("S4", empty).get((street, situation, made, drw, faced)),
                    sz.get("S3", empty).get((street, situation, made, faced)),
                    sz.get("S2", empty).get((street, situation, pose, pot_type, faced)),
                    sz.get("S1", empty).get((street, situation, faced)),
                )
                lp = int(plain_level[mi, di])
                own = cells[lp] if lp < len(cells) else None
                if own is not None and own.sum() >= cfg.lik_min_support:
                    probs[mi, di] = (own + cfg.eps) / (own.sum() + n_out * cfg.eps)
                    direct[mi, di] = True
                    continue
                p = plain[mi, di]
                probs[mi, di, :3] = p[:3]
                for act, sl in _TIER_SLICE.items():
                    mix = np.full(len(SIZE_TIERS), 1.0 / len(SIZE_TIERS))
                    for cell in cells[lp:] + field_cells:
                        if cell is not None and cell[sl].sum() >= cfg.lik_min_support:
                            mix = (cell[sl] + cfg.eps) / (cell[sl].sum() + len(SIZE_TIERS) * cfg.eps)
                            break
                    probs[mi, di, sl] = p[ACTION_INDEX[act]] * mix
        out = (probs, direct)
        self._node_cache[key] = out
        return out

    def prior_for(self, size_cls: str, situation: Optional[str], position_class: str, action: str,
                  cfg: RangeConfig) -> tuple[Optional[np.ndarray], str]:
        """Normalised 169-class prior of a preflop key and the level it came from (None, '' when no data at all)."""
        key = (size_cls, situation, position_class, action, cfg.use_size_class, cfg.prior_min_support)
        hit = self._prior_cache.get(key)
        if hit is not None:
            return hit
        degenerate = situation in (None, "", "none")
        cands: list[tuple[str, Optional[np.ndarray]]] = []
        if not degenerate:
            if cfg.use_size_class:
                cands.append(("P4", self.prior["P4"].get((size_cls, situation, position_class, action))))
            cands.append(("P3", self.prior["P3"].get((situation, position_class, action))))
            cands.append(("P2", self.prior["P2"].get((situation, position_class))))
        cands.append(("P1", self.prior["P1"].get((position_class, action))))
        merged = None
        for act in ("call", "raise"):
            arr = self.prior["P1"].get((position_class, act))
            if arr is not None:
                merged = arr.copy() if merged is None else merged + arr
        cands.append(("P1m", merged))
        out: tuple[Optional[np.ndarray], str] = (None, "")
        for level, arr in cands:
            if arr is not None and arr.sum() >= cfg.prior_min_support:
                out = (arr / arr.sum(), level)
                break
        else:
            p0 = self.prior["P0"].get(())
            if p0 is not None and p0.sum() > 0:
                out = (p0 / p0.sum(), "P0")
        self._prior_cache[key] = out
        return out

    def field_stats(self, size_cls: str, cfg: RangeConfig) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """(VPIP, PFR, aggression) of the field for a table-size class (the pooled row when classes are off)."""
        return self.field.get(size_cls if cfg.use_size_class else "all", (None, None, None))


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One postflop action of the opponent: the spot as the field tables key it, and the board he saw."""

    street: str
    situation: str
    pose: str
    pot_type: str
    action: str
    board: tuple[int, ...]
    label: str = ""
    size_tier: str = ""             # his bet or raise: tier of its size (t1 ... t5); '' otherwise or unknown
    faced_tier: str = FACED_NONE    # tier of the bet or raise he faced; 'none' when he faced none


@dataclass(frozen=True)
class OpponentLine:
    """What the model needs to know about one opponent at one decision.

    Attributes:
        size_cls: Table-size class of the hand.
        pf_situation: Situation of his last preflop betting line (None / 'none' if he never acted).
        position_class: His position class.
        pf_action: Action of that line (check, call or raise).
        raised_preflop: He raised at some point before the flop (picks PFR over VPIP for the looseness tilt).
        n: His postflop checks, calls, bets and raises on record (None if unknown).
        vpip, pfr, agg: His statistics (agg smoothed toward the field), None if unknown.
        nodes: His postflop actions before the decision, in order.
    """

    size_cls: str
    pf_situation: Optional[str]
    position_class: str
    pf_action: str
    raised_preflop: bool = False
    n: Optional[float] = None
    vpip: Optional[float] = None
    pfr: Optional[float] = None
    agg: Optional[float] = None
    nodes: tuple[Node, ...] = ()


@dataclass
class RangeResult:
    """A predicted range: combo ids (rows of :data:`sudetect.cards.COMBOS`) and weights summing to 1.

    ``prior_level`` says which prior level was used, ``lik_mass[k]`` how much range weight (summed over the
    updates) took its likelihood from level k of :data:`LIK_LEVELS` (index 6: uniform). ``steps`` holds
    ``(label, ids, w)`` after the prior and after each update when the range was built with ``trace=True``.
    """

    ids: np.ndarray
    w: np.ndarray
    prior_level: str
    lik_mass: np.ndarray
    steps: list[tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)
    sized_mass: float = 0.0


def looseness_tilt(prior: np.ndarray, stat: Optional[float], field_stat: Optional[float],
                   n: Optional[float], cfg: RangeConfig) -> np.ndarray:
    """Reshape a 169-class prior toward weak (loose player) or strong (tight player) classes.

    ``pi1(c) ~ pi0(c) * r ** (beta * (0.5 - s_c))``, ``r = clip(stat / field_stat, ratio_min, ratio_max)``;
    the prior is returned unchanged below ``loose_min_n`` postflop actions, without a statistic or a field
    value, or when ``|r - 1| < 0.001``.
    """
    if cfg.loose_beta <= 0 or n is None or n < cfg.loose_min_n or stat is None or not field_stat:
        return prior
    ratio = min(cfg.ratio_max, max(cfg.ratio_min, stat / field_stat))
    if abs(ratio - 1.0) < 1e-3:
        return prior
    out = prior * np.power(ratio, cfg.loose_beta * (0.5 - _STRENGTH))
    total = out.sum()
    return out / total if total > 0 else prior


def aggression_tilt(n: Optional[float], agg: Optional[float], field_agg: Optional[float],
                    cfg: RangeConfig) -> tuple[float, float]:
    """``(af, omega)`` of the aggression tilt; ``(1, 0)`` (no tilt) below ``agg_min_n`` actions or without data."""
    if n is None or n < cfg.agg_min_n or agg is None or field_agg is None:
        return 1.0, 0.0
    af = min(cfg.ratio_max, max(cfg.ratio_min, agg / max(field_agg, 1e-3)))
    omega = min(1.0, max(0.0, (n - cfg.agg_min_n) / (cfg.agg_full_n - cfg.agg_min_n)))
    return af, omega


def _blocked(dead: Sequence[int]) -> np.ndarray:
    mask = np.zeros(52, dtype=bool)
    if len(dead):
        mask[np.asarray(dead, dtype=np.intp)] = True
    return mask[_C0] | mask[_C1]


def prune(ids: np.ndarray, w: np.ndarray, cfg: RangeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Drop combos below ``prune_frac`` x the heaviest, then keep the ``cap`` heaviest (ties: lower combo id first).

    The kept combos stay in ascending id order; the weights are not normalised here.
    """
    if ids.size == 0:
        return ids, w
    keep = w >= w.max() * cfg.prune_frac
    if not keep.all():
        ids, w = ids[keep], w[keep]
    if ids.size > cfg.cap:
        order = np.sort(np.argsort(-w, kind="stable")[:cfg.cap])
        ids, w = ids[order], w[order]
    return ids, w


def expand_prior(prior: np.ndarray, dead: Sequence[int], cfg: RangeConfig,
                 prune_now: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """169-class weights -> normalised combo weights with every combo touching a dead card removed.

    With ``prune_now`` (the default) classes below ``class_floor`` are left out and the combos are pruned
    (:func:`prune`); without it every class with a positive weight expands and nothing is pruned.
    """
    keep_class = ((prior >= cfg.class_floor) if prune_now else True) & (prior > 0)
    z = prior[keep_class].sum()
    if z <= 0:
        return np.empty(0, dtype=np.intp), np.empty(0)
    alive = ~_blocked(dead)
    ids = np.flatnonzero(alive & keep_class[_CLASS_OF])
    if ids.size == 0:
        return ids, np.empty(0)
    cls = _CLASS_OF[ids]
    if cfg.combo_weight == "live_combos":
        per_class = np.bincount(_CLASS_OF[alive], minlength=N_CLASSES)
    else:
        per_class = _CLASS_SIZE
    w = prior[cls] / z / per_class[cls]
    if prune_now:
        ids, w = prune(ids, w, cfg)
    return ids, w / w.sum()


def floor_and_prune(ids: np.ndarray, w: np.ndarray, cfg: RangeConfig) -> tuple[np.ndarray, np.ndarray]:
    """The end step when ``prune_each_step`` is off: drop the classes holding less than ``class_floor`` of the
    range, prune (:func:`prune`) and normalise."""
    if ids.size == 0:
        return ids, w
    mass = np.bincount(_CLASS_OF[ids], weights=w, minlength=N_CLASSES)
    keep = mass[_CLASS_OF[ids]] >= cfg.class_floor * w.sum()
    if keep.any():
        ids, w = ids[keep], w[keep]
    ids, w = prune(ids, w, cfg)
    return ids, w / w.sum()


def predict_range(tables: FitTables, cfg: RangeConfig, line: OpponentLine, dead: Sequence[int],
                  trace: bool = False) -> Optional[RangeResult]:
    """The opponent's range at a decision: prior, looseness tilt, combos, one update per postflop action.

    With ``cfg.sizing`` an update reads :meth:`FitTables.sized_node_table`: a bet or raise with the tier of its size,
    every action under the key of the tier he faced. A bet or raise of unknown size is read size-blind.

    Args:
        tables: Fitted field tables.
        cfg: Model constants.
        line: The opponent's preflop key, statistics and postflop actions before the decision.
        dead: Cards no combo of his may hold: the hero's hole cards and the board at the decision.
        trace: Keep a copy of the range after the prior and after every update (for figures).

    Returns:
        The range, or None when no prior exists (empty tables).
    """
    prior, level = tables.prior_for(line.size_cls, line.pf_situation, line.position_class, line.pf_action, cfg)
    if prior is None:
        return None
    fv, fp, fa = tables.field_stats(line.size_cls, cfg)
    if line.raised_preflop:
        prior = looseness_tilt(prior, line.pfr, fp, line.n, cfg)
    else:
        prior = looseness_tilt(prior, line.vpip, fv, line.n, cfg)
    each = cfg.prune_each_step
    ids, w = expand_prior(prior, dead, cfg, prune_now=each)
    lik_mass = np.zeros(len(LIK_LEVELS) + 1)
    res = RangeResult(ids=ids, w=w, prior_level=level, lik_mass=lik_mass)
    if trace:
        res.steps.append(("prior", ids.copy(), w.copy()))
    if ids.size == 0:
        return res
    af, omega = aggression_tilt(line.n, line.agg, fa, cfg)
    exponent = np.asarray(cfg.bluff_exponent)[_GROUP_OF_MADE]          # (12,) b_g per made class
    for node in line.nodes:
        a = ACTION_INDEX.get(node.action)
        if a is None or not node.situation:
            continue
        made, drw = classes_for_all_combos(node.board)
        m = made[ids].astype(np.intp)
        d = drw[ids].astype(np.intp)
        if (m < 0).any():                       # cannot happen: dead cards include every board card
            raise ValueError("a range combo shares a card with the board")
        probs, levels = tables.node_table(node.street, node.situation, node.pose, node.pot_type, cfg)
        o = sized_outcome(node.action, node.size_tier) if cfg.sizing else None
        if o is None:                           # size-blind model, or a bet / raise of unknown size
            pa = probs[m, d, a]
        else:
            sprobs, direct = tables.sized_node_table(node.street, node.situation, node.pose, node.pot_type,
                                                     node.faced_tier or FACED_NONE, cfg)
            pa = sprobs[m, d, o]
            res.sized_mass += float(w[direct[m, d]].sum())
        np.add.at(lik_mass, levels[m, d], w)
        if omega > 0 and node.action in ("bet", "raise"):
            pa = pa * np.power(af, exponent[m] * omega)
        nw = w * (pa + cfg.eps)
        keep = nw > 0
        ids, w = ids[keep], nw[keep]
        if each:
            ids, w = prune(ids, w, cfg)
        w = w / w.sum()
        if trace:
            res.steps.append((node.label or f"{node.street}:{node.action}", ids.copy(), w.copy()))
    if not each:
        ids, w = floor_and_prune(ids, w, cfg)
    res.ids, res.w = ids, w
    return res


# ----------------------------------------------------------------------------
# Views of a range
# ----------------------------------------------------------------------------

def range_dense(ids: np.ndarray, w: np.ndarray) -> np.ndarray:
    """1,326 weights in :data:`sudetect.cards.COMBOS` order (0 outside the range)."""
    out = np.zeros(N_COMBOS)
    out[ids] = w
    return out


def range_classes(ids: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weight of each of the 169 classes (sums to the range's total weight)."""
    return np.bincount(_CLASS_OF[ids], weights=w, minlength=N_CLASSES)


def top_class(ids: np.ndarray, w: np.ndarray) -> Optional[str]:
    """The heaviest of the 169 classes (ties: the first in CLASS_LABELS order), None for an empty range."""
    if ids.size == 0:
        return None
    return CLASS_LABELS[int(np.argmax(range_classes(ids, w)))]


def group_shares(ids: np.ndarray, w: np.ndarray, board: Sequence[int]) -> dict[str, float]:
    """Share of the range in each made-hand group of :data:`GROUPS` on ``board`` (3 to 5 cards)."""
    if ids.size == 0 or len(board) < 3:
        return {}
    made, _ = classes_for_all_combos(tuple(board))
    g = _GROUP_OF_MADE[made[ids].astype(np.intp)]
    s = np.bincount(g, weights=w, minlength=len(GROUPS))
    total = s.sum()
    return {name: float(s[i] / total) for i, name in enumerate(GROUPS)} if total > 0 else {}


def class_grid(class_weights: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
    """Arrange 169 class values on the usual 13 x 13 grid.

    Row and column ``i`` stand for the rank ``A, K, ..., 2``; pairs lie on the diagonal, suited classes above it
    (row = higher rank) and offsuit classes below it (column = higher rank). Returns the values and the labels.
    """
    order = RANKS[::-1]
    grid = np.zeros((13, 13))
    labels = [["" for _ in range(13)] for _ in range(13)]
    pos = {label: i for i, label in enumerate(CLASS_LABELS)}
    for i, r1 in enumerate(order):
        for j, r2 in enumerate(order):
            if i == j:
                label = r1 + r2
            elif i < j:
                label = r1 + r2 + "s"
            else:
                label = r2 + r1 + "o"
            grid[i, j] = class_weights[pos[label]]
            labels[i][j] = label
    return grid, labels


# ----------------------------------------------------------------------------
# Multiway equity against ranges (Monte Carlo)
# ----------------------------------------------------------------------------

def row_seed(seed: int, *parts: Any) -> int:
    """A 63-bit seed derived from a base seed and identifiers (stable across processes and runs)."""
    h = hashlib.blake2b(repr((int(seed),) + tuple(str(p) for p in parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") >> 1


def equity_vs_ranges_mc(hero: Sequence[int], board: Sequence[int],
                        ranges: Sequence[tuple[np.ndarray, np.ndarray]],
                        n_samples: int = 4096, seed: int = 0) -> tuple[float, float]:
    """The hero's equity against one or more opponents' ranges by Monte Carlo.

    Each sample draws one combo per opponent from his weights, rejects the draw when two opponents' combos share
    a card, deals the rest of the board from the cards left and splits the pot among the best hands. Ranges
    must already exclude the hero's cards and the board.

    Returns:
        ``(equity, standard error)``; ``(nan, nan)`` when no sample survives.
    """
    from .evaluator import eval_hands

    rng = np.random.default_rng(seed)
    hero = list(hero)
    board = list(board)
    k = len(ranges)
    picks = np.empty((k, n_samples), dtype=np.intp)
    for i, (ids, w) in enumerate(ranges):
        p = np.asarray(w, dtype=np.float64)
        picks[i] = np.asarray(ids)[rng.choice(len(ids), size=n_samples, p=p / p.sum())]
    cards = np.stack([_C0[picks], _C1[picks]], axis=-1)                # (k, n, 2)
    flat = cards.transpose(1, 0, 2).reshape(n_samples, 2 * k)
    srt = np.sort(flat, axis=1)
    ok = (np.diff(srt, axis=1) != 0).all(axis=1) if k > 1 else np.ones(n_samples, dtype=bool)
    flat = flat[ok]
    n = flat.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    need = 5 - len(board)
    if need > 0:
        u = rng.random((n, 52))
        u[:, hero + board] = 2.0
        u[np.arange(n)[:, None], flat] = 2.0
        runout = np.argpartition(u, need, axis=1)[:, :need]
    else:
        runout = np.empty((n, 0), dtype=np.intp)
    full_board = np.concatenate([np.broadcast_to(np.asarray(board, dtype=np.intp), (n, len(board))), runout], axis=1)
    hero_val = eval_hands(np.concatenate([np.broadcast_to(np.asarray(hero, dtype=np.intp), (n, 2)), full_board],
                                         axis=1), check=False)
    best = hero_val.copy()
    vals = []
    for i in range(k):
        v = eval_hands(np.concatenate([flat[:, 2 * i:2 * i + 2], full_board], axis=1), check=False)
        vals.append(v)
        best = np.maximum(best, v)
    n_best = (hero_val == best).astype(np.float64)
    for v in vals:
        n_best += (v == best)
    share = np.where(hero_val == best, 1.0 / n_best, 0.0)
    eq = float(share.mean())
    se = float(share.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return eq, se


# ----------------------------------------------------------------------------
# Process-pool workers (spawn-safe: module-level functions, state set by the initializer)
# ----------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def init_worker(fit_path: str, settings: Mapping[str, Any]) -> None:
    """Load the pickled :class:`FitTables` and the run settings once per worker process."""
    import pickle

    with open(fit_path, "rb") as fh:
        tables = pickle.load(fh)
    _WORKER.clear()
    _WORKER["tables"] = tables
    _WORKER["cfg"] = RangeConfig(**settings["range"])
    _WORKER["settings"] = dict(settings)


def _opt(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def score_chunk(job: Mapping[str, str]) -> dict[str, Any]:
    """Score one staged chunk: predicted range and equity for every decision, written to ``job['out']``.

    ``job`` holds the paths ``dec``, ``opp`` and ``nodes`` (parquet files written by the staging step) and the
    output path ``out``. Returns counters for the run summary.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .equity import equity_vs_hand_batch, equity_vs_range_batch, multiway_equity

    t0 = time.perf_counter()
    tables: FitTables = _WORKER["tables"]
    cfg: RangeConfig = _WORKER["cfg"]
    st = _WORKER["settings"]
    dec = pq.read_table(job["dec"]).to_pandas()
    opp = pq.read_table(job["opp"]).to_pandas()
    nodes = pq.read_table(job["nodes"]).to_pandas()

    node_map: dict[tuple[str, str], list[tuple[int, Node]]] = {}
    for r in nodes.itertuples(index=False):
        node_map.setdefault((r.hand_id, r.player_id), []).append((int(r.action_index), Node(
            street=r.street, situation=r.situation, pose=r.pose, pot_type=r.pot_type, action=r.action,
            board=cards_of(r.board), size_tier=size_tier(getattr(r, "size_pot", None)),
            faced_tier=faced_tier(getattr(r, "faced_size_pot", None)))))
    opp_map: dict[tuple[str, int], list[Any]] = {}
    for r in opp.itertuples(index=False):
        opp_map.setdefault((r.hand_id, int(r.action_index)), []).append(r)

    n = len(dec)
    eq_r = np.full(n, np.nan)
    se_r = np.zeros(n)
    method = np.array(["exact"] * n, dtype=object)
    n_combos = np.zeros(n, dtype=np.int64)
    top = np.empty(n, dtype=object)
    strong = np.full(n, np.nan)
    air = np.full(n, np.nan)
    eq_q = np.full(n, np.nan)
    n_eff = np.full(n, np.nan)
    covered = np.full(n, np.nan)
    real_w = np.full(n, np.nan)
    prior_lv = np.empty(n, dtype=object)
    lik_mass = np.zeros(len(LIK_LEVELS) + 1)
    sized_mass = 0.0
    prior_counts: dict[str, int] = {}
    hu_rows: list[int] = []
    hu_weights: list[np.ndarray] = []
    hu_villain: list[Optional[tuple[int, ...]]] = []
    t_range = 0.0
    for i, r in enumerate(dec.itertuples(index=False)):
        hero = cards_of(r.hole_cards)
        board = cards_of(r.board)
        dead = hero + board
        aidx = int(r.action_index)
        ranges: list[RangeResult] = []
        real: list[Optional[tuple[int, ...]]] = []
        t1 = time.perf_counter()
        for o in opp_map.get((r.hand_id, aidx), []):
            line = OpponentLine(
                size_cls=r.size_class,
                pf_situation=o.pf_sit,
                position_class=o.position_class,
                pf_action=o.pf_act,
                raised_preflop=bool(o.raised_pf),
                n=_opt(o.t_n), vpip=_opt(o.t_vpip), pfr=_opt(o.t_pfr), agg=_opt(o.t_agg),
                nodes=tuple(nd for ai, nd in node_map.get((r.hand_id, o.opp_id), []) if ai < aidx),
            )
            res = predict_range(tables, cfg, line, dead)
            if res is None or res.ids.size == 0:
                ranges = []
                break
            ranges.append(res)
            real.append(cards_of(o.opp_cards) if isinstance(o.opp_cards, str) and o.opp_cards else None)
        t_range += time.perf_counter() - t1
        if not ranges or len(ranges) != int(r.n_live_opp):
            method[i] = None
            se_r[i] = np.nan
            continue
        for res in ranges:
            lik_mass += res.lik_mass
            sized_mass += res.sized_mass
            prior_counts[res.prior_level] = prior_counts.get(res.prior_level, 0) + 1
        smallest = min(ranges, key=lambda x: x.ids.size)
        n_combos[i] = smallest.ids.size
        top[i] = top_class(smallest.ids, smallest.w)
        prior_lv[i] = ranges[0].prior_level
        n_eff[i] = 1.0 / float(np.sum(ranges[0].w ** 2))
        if real[0] is not None:                 # hindsight: the opponent's real combo and its weight
            hit = ranges[0].ids == int(COMBO_INDEX[real[0][0], real[0][1]])
            covered[i] = float(hit.any())
            real_w[i] = float(ranges[0].w[hit].sum())
        if len(board) >= 3:
            gs = group_shares(smallest.ids, smallest.w, board)
            strong[i] = gs.get("monster", 0.0) + gs.get("strong", 0.0)
            air[i] = gs.get("air", 0.0)
        if len(ranges) == 1:
            hu_rows.append(i)
            hu_weights.append(range_dense(ranges[0].ids, ranges[0].w))
            hu_villain.append(real[0])
        else:
            eq, se = equity_vs_ranges_mc(hero, board, [(x.ids, x.w) for x in ranges],
                                         n_samples=int(st["mc_samples"]),
                                         seed=row_seed(int(st["seed"]), r.hand_id, aidx))
            eq_r[i], se_r[i], method[i] = eq, se, "mc"
            if all(x is not None for x in real):
                eq_q[i] = multiway_equity([hero] + [list(x) for x in real], list(board))[0]

    t_eq = time.perf_counter()
    if hu_rows:
        rows = np.asarray(hu_rows)
        hero_arr = np.array([cards_of(dec.hole_cards.iat[i]) for i in rows], dtype=np.int64)
        board_arr = np.full((rows.size, 5), -1, dtype=np.int64)
        for k, i in enumerate(rows):
            b = cards_of(dec.board.iat[i])
            board_arr[k, :len(b)] = b
        exact = bool(st["exact"])
        eq_r[rows] = equity_vs_range_batch(hero_arr, board_arr, np.vstack(hu_weights),
                                           exact=True if exact else None,
                                           n_samples=int(st["mc_samples"]), seed=int(st["seed"]))
        if not exact:
            flop = board_arr[:, 3] < 0
            method[rows[flop]] = "mc"
            se_r[rows[flop]] = 0.5 / math.sqrt(int(st["mc_samples"]))
        known = np.array([v is not None for v in hu_villain])
        if known.any():
            vil = np.array([v for v in hu_villain if v is not None], dtype=np.int64)
            eq_q[rows[known]] = equity_vs_hand_batch(hero_arr[known], vil, board_arr[known])
    t_eq = time.perf_counter() - t_eq
    eq_r = np.clip(eq_r, 0.0, 1.0)              # enumeration sums can land a few ulps outside [0, 1]
    eq_q = np.clip(eq_q, 0.0, 1.0)

    out = pd.DataFrame({
        "hand_id": dec.hand_id.to_numpy(),
        "action_index": dec.action_index.to_numpy().astype(np.int16),
        "player_id": dec.player_id.to_numpy(),
        "street": dec.street.to_numpy(),
        "n_live_opp": dec.n_live_opp.to_numpy().astype(np.int16),
        "eq_vs_range": eq_r,
        "eq_range_method": method,
        "eq_range_se": se_r,
        "n_combos_opp": n_combos.astype(np.int32),
        "range_top_class": top,
        "range_strong_share": strong,
        "range_air_share": air,
        "prior_level": prior_lv,
        "n_eff_opp": n_eff,
        "real_in_range": covered,
        "real_combo_weight": real_w,
        "eq_vs_real": eq_q,
        "action": dec.action.to_numpy(),
        "pot_odds": dec.pot_odds.to_numpy(dtype=np.float64),
    })
    tmp = job["out"] + ".tmp"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), tmp)
    import os

    os.replace(tmp, job["out"])
    return {
        "chunk": job["out"],
        "rows": n,
        "scored": int(np.isfinite(eq_r).sum()),
        "seconds": time.perf_counter() - t0,
        "seconds_range": t_range,
        "seconds_equity": t_eq,
        "lik_mass": lik_mass.tolist(),
        "sized_mass": sized_mass,
        "prior_levels": prior_counts,
    }
