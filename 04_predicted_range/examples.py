"""The narrowing example: one simulator hand, the opponent's range after each of his actions.

Picks a heads-up hand in which the opponent bets the flop, bets the turn and raises the river (config
``[examples]``; ``hand_id`` pins one), rebuilds his predicted range before the flop and after every postflop
action, and computes the hero's three equities at each of the hero's postflop decisions:

* ``figures/range_narrowing.png``   the range as a 13 x 13 grid after each step;
* ``figures/equity_by_decision.png`` equity vs a random hand, vs the predicted range and vs the real hand;
* ``figures/range_narrowing.json``  every number behind both figures, and the hand itself;
* table ``action_range_opp``        the 169 class weights of each step.

Usage::

    python 04_predicted_range/examples.py --dataset <name>
    python 04_predicted_range/examples.py --dataset <name> --hand-id <hand_id>

The opponent's real cards outline one cell of each grid: hindsight, shown so the reader can see where the
real hand sits in the range. The model never sees them.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import rangemodel, warehouse  # noqa: E402
from sudetect.cards import CLASS_LABELS, COMBO_TO_CLASS, format_cards, hand_class  # noqa: E402
from sudetect.equity import equity_vs_hand, equity_vs_random, equity_vs_range  # noqa: E402

FIG_DIR = HERE / "figures"

# Chart colours (light surface): a one-hue sequential ramp for the grids, three categorical slots for the lines.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
OUT_OF_RANGE = "#f0efec"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = {"eq_vs_random": "#2a78d6", "eq_vs_range": "#eb6834", "eq_vs_real": "#1baf7a"}
SERIES_LABEL = {"eq_vs_random": "vs a random hand", "eq_vs_range": "vs the predicted range",
                "eq_vs_real": "vs the real hand"}

PREFLOP_PHRASES = {
    ("unopened", "raise"): "raised first in",
    ("unopened", "call"): "limped",
    ("unopened", "check"): "checked",
    ("vs_limp", "check"): "checked his option after a limp",
    ("vs_limp", "call"): "limped behind",
    ("vs_limp", "raise"): "raised a limp",
    ("vs_open", "call"): "called a raise",
    ("vs_open", "raise"): "3-bet",
    ("vs_open_call", "call"): "called a raise after a caller",
    ("vs_open_call", "raise"): "squeezed",
    ("vs_3bet", "call"): "called a 3-bet",
    ("vs_3bet", "raise"): "4-bet",
    ("vs_4bet_plus", "call"): "called a 4-bet",
    ("vs_4bet_plus", "raise"): "5-bet",
}


def _run_module() -> Any:
    spec = importlib.util.spec_from_file_location("step04_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ choosing the hand


def find_candidates(con: Any, pattern: Sequence[str]) -> list[tuple[str, str, str]]:
    """``(hand_id, opponent, hero)`` of the two-player hands where the opponent's postflop actions are exactly
    ``pattern`` (``street:action``) and the other player acts after the opponent's last action."""
    last_street, last_action = pattern[-1].split(":")
    df = warehouse.query_df(con, """
        SELECT a.hand_id, a.action_index, a.player_id, a.street, a.action
        FROM actions AS a
        JOIN hands AS h ON h.hand_id = a.hand_id
        WHERE h.n_dealt = 2
          AND a.street IN ('flop', 'turn', 'river')
          AND NOT a.is_post
          AND a.hand_id IN (SELECT hand_id FROM actions WHERE street = :st AND action = :act AND NOT is_post)
        ORDER BY a.hand_id, a.action_index""", st=last_street, act=last_action)
    out: list[tuple[str, str, str]] = []
    for hid, g in df.groupby("hand_id", sort=True):
        for pid, gp in g.groupby("player_id", sort=True):
            if [f"{s}:{a}" for s, a in zip(gp.street, gp.action)] != list(pattern):
                continue
            after = g[(g.action_index > gp.action_index.max()) & (g.player_id != pid)]
            if len(after):
                out.append((str(hid), str(pid), str(after.player_id.iat[0])))
    return out


# ------------------------------------------------------------------ the example


def _step_label(row: Any) -> str:
    size = f" {row.size_pot:.2f} pot" if row.action in ("bet", "raise") and pd.notna(row.size_pot) else ""
    return f"{row.street}: {row.action}{size}"


def _preflop_phrase(position: str, situation: Optional[str], action: str) -> str:
    phrase = PREFLOP_PHRASES.get((situation or "none", action), f"{situation}, {action}")
    return f"before the flop: {position} {phrase}"


def _class_view(ids: np.ndarray, w: np.ndarray, dead: Sequence[int]) -> dict[str, Any]:
    """Class shares, per-combo weights (relative to the heaviest class) and the top classes of a range."""
    mass = rangemodel.range_classes(ids, w)
    blocked = np.zeros(52, dtype=bool)
    blocked[list(dead)] = True
    from sudetect.cards import COMBOS

    alive = ~(blocked[COMBOS[:, 0]] | blocked[COMBOS[:, 1]])
    live = np.bincount(COMBO_TO_CLASS[alive], minlength=len(CLASS_LABELS)).astype(np.float64)
    per_combo = np.where(live > 0, mass / np.where(live > 0, live, 1), 0.0)
    rel = per_combo / per_combo.max() if per_combo.max() > 0 else per_combo
    order = np.argsort(-mass, kind="stable")[:8]
    return {
        "class_share": {CLASS_LABELS[i]: round(float(mass[i]), 6) for i in range(len(CLASS_LABELS)) if mass[i] > 0},
        "per_combo_relative": rel,
        "top_classes": [{"class": CLASS_LABELS[i], "share": round(float(mass[i]), 4)} for i in order if mass[i] > 0],
    }


def _stat(row: Optional[pd.Series], key: str) -> Optional[float]:
    """One tilt statistic of a ``rng_player_tilt`` row, None when the row or the value is missing."""
    if row is None or pd.isna(row[key]):
        return None
    return float(row[key])


def load_hand(con: Any, cfg: Mapping[str, Any], hand_id: str, opp_id: str, hero_id: str) -> dict[str, Any]:
    """The rows of one hand and the opponent's line as the range model reads it.

    Returns the hand's ``actions`` rows (``acts``), its ``hand_players`` rows indexed by player (``hps``), the
    ``hands`` row (``hand``), the opponent's :class:`sudetect.rangemodel.OpponentLine` without postflop actions
    (``line``), his postflop actions as ``(action_index, Node)`` pairs (``nodes``), his tilt row (``tilt``), and
    both players' cards (``hero``, ``opp_cards``).
    """
    rcfg = rangemodel.RangeConfig.from_toml(cfg)
    acts = warehouse.query_df(con, "SELECT * FROM actions WHERE hand_id = :h ORDER BY action_index", h=hand_id)
    hps = warehouse.query_df(con, "SELECT * FROM hand_players WHERE hand_id = :h", h=hand_id).set_index("player_id")
    hand = warehouse.query_df(con, "SELECT * FROM hands WHERE hand_id = :h", h=hand_id).iloc[0]
    size_cls = rangemodel.size_class(int(hand.n_dealt))
    tilt_size = size_cls if rcfg.use_size_class else "all"
    tilt = warehouse.query_df(con, "SELECT n, vpip, pfr, agg FROM rng_player_tilt WHERE player_id = :p "
                                   "AND size_class = :s", p=opp_id, s=tilt_size)
    t = tilt.iloc[0] if len(tilt) else None
    pre = acts[(acts.player_id == opp_id) & (acts.street == "preflop") & ~acts.is_post]
    pf_sit = str(pre.situation.iat[-1]) if len(pre) else "none"
    pf_act = str(pre.action.iat[-1]) if len(pre) else "check"
    opp_rows = acts[(acts.player_id == opp_id) & acts.street.isin(["flop", "turn", "river"]) & ~acts.is_post
                    & acts.is_decision & acts.action.isin(["check", "call", "bet", "raise"])]
    nodes = [(int(r.action_index), rangemodel.Node(
        street=r.street, situation=r.situation, pose="ip" if r.is_ip else "oop", pot_type=r.pot_type_now,
        action=r.action, board=rangemodel.cards_of(r.board_at_action), label=_step_label(r),
        size_tier=rangemodel.size_tier(r.size_pot), faced_tier=rangemodel.faced_tier(r.faced_size_pot)))
        for r in opp_rows.itertuples(index=False)]
    line = rangemodel.OpponentLine(
        size_cls=size_cls, pf_situation=pf_sit, position_class=str(hps.loc[opp_id, "position_class"]),
        pf_action=pf_act, raised_preflop=bool(hps.loc[opp_id, "pfr"]),
        n=_stat(t, "n"), vpip=_stat(t, "vpip"), pfr=_stat(t, "pfr"), agg=_stat(t, "agg"))
    return {"acts": acts, "hps": hps, "hand": hand, "line": line, "nodes": nodes, "tilt": t,
            "hero": rangemodel.cards_of(hps.loc[hero_id, "hole_cards"]),
            "opp_cards": rangemodel.cards_of(hps.loc[opp_id, "hole_cards"])}


def build_example(con: Any, cfg: Mapping[str, Any], tables: rangemodel.FitTables, hand_id: str, opp_id: str,
                  hero_id: str) -> dict[str, Any]:
    """Rebuild the opponent's range step by step and the hero's equities at each of his postflop decisions."""
    rcfg = rangemodel.RangeConfig.from_toml(cfg)
    h = load_hand(con, cfg, hand_id, opp_id, hero_id)
    acts, hps, hand, line, nodes, t = h["acts"], h["hps"], h["hand"], h["line"], h["nodes"], h["tilt"]
    hero, opp_cards = h["hero"], h["opp_cards"]
    size_cls, pf_sit, pf_act = line.size_cls, line.pf_situation, line.pf_action
    prior_label = _preflop_phrase(str(hps.loc[opp_id, "position"]), pf_sit, pf_act)

    steps: list[dict[str, Any]] = []
    for k in range(len(nodes) + 1):
        dead = hero + (nodes[k - 1][1].board if k else ())
        res = rangemodel.predict_range(tables, rcfg, dataclasses.replace(line, nodes=tuple(n for _, n in nodes[:k])),
                                       dead)
        if res is None:
            raise RuntimeError("no prior: the fit tables are empty")
        view = _class_view(res.ids, res.w, dead)
        board = nodes[k - 1][1].board if k else ()
        steps.append({
            "step": k,
            "label": prior_label if k == 0 else nodes[k - 1][1].label,
            "action_index": None if k == 0 else nodes[k - 1][0],
            "board": format_cards(board, sep=""),
            "prior_level": res.prior_level,
            "n_combos": int(res.ids.size),
            "effective_combos": round(float(1.0 / np.sum(res.w ** 2)), 1),
            "groups": {g: round(v, 4) for g, v in rangemodel.group_shares(res.ids, res.w, board).items()},
            "top_classes": view["top_classes"],
            "class_share": view["class_share"],
            "_grid": view["per_combo_relative"],
        })

    decisions: list[dict[str, Any]] = []
    hero_rows = acts[(acts.player_id == hero_id) & acts.street.isin(["flop", "turn", "river"]) & ~acts.is_post
                     & acts.is_decision]
    for r in hero_rows.itertuples(index=False):
        board = rangemodel.cards_of(r.board_at_action)
        before = tuple(n for ai, n in nodes if ai < int(r.action_index))
        res = rangemodel.predict_range(tables, rcfg, dataclasses.replace(line, nodes=before), hero + board)
        assert res is not None
        weights = rangemodel.range_dense(res.ids, res.w)
        decisions.append({
            "action_index": int(r.action_index),
            "street": r.street,
            "situation": r.situation,
            "action": r.action,
            "facing_size_pot": None if pd.isna(r.faced_size_pot) else round(float(r.faced_size_pot), 3),
            "pot_odds": None if pd.isna(r.pot_odds) else round(float(r.pot_odds), 4),
            "board": format_cards(board, sep=""),
            "opponent_actions_used": len(before),
            "n_combos": int(res.ids.size),
            "eq_vs_random": round(float(equity_vs_random(list(hero), list(board))), 4),
            "eq_vs_range": round(float(equity_vs_range(list(hero), list(board), weights)), 4),
            "eq_vs_real": round(float(equity_vs_hand(list(hero), list(opp_cards), list(board))), 4),
        })

    replay = []
    for r in acts[~acts.is_post].itertuples(index=False):
        replay.append({
            "street": r.street, "who": "hero" if r.player_id == hero_id else "opponent", "action": r.action,
            "amount_bb": round(float(r.amount_eff_bb), 2), "pot_before_bb": round(float(r.pot_before_bb), 2),
            "size_pot": None if pd.isna(r.size_pot) else round(float(r.size_pot), 3),
        })
    return {
        "hand_id": hand_id,
        "table_size": int(hand.n_dealt),
        "hero": {"player_id": hero_id, "position": str(hps.loc[hero_id, "position"]),
                 "cards": format_cards(hero, sep=""), "class": hand_class(list(hero))},
        "opponent": {"player_id": opp_id, "position": str(hps.loc[opp_id, "position"]),
                     "cards_hindsight": format_cards(opp_cards, sep=""), "class_hindsight": hand_class(list(opp_cards)),
                     "preflop_key": {"size_class": size_cls, "situation": pf_sit, "action": pf_act,
                                     "position_class": str(hps.loc[opp_id, "position_class"])},
                     "stats": None if t is None else {k: round(v, 4) for k in ("n", "vpip", "pfr", "agg")
                                                      if (v := _stat(t, k)) is not None}},
        "board": str(hand.board),
        "replay": replay,
        "steps": steps,
        "decisions": decisions,
    }


# ------------------------------------------------------------------ figures


def _ramp_cmap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ramp", RAMP)


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def plot_grids(example: Mapping[str, Any], path: Path, note: str = "") -> None:
    """The range after each step as 13 x 13 grids (2 x 2 panels for four steps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle

    steps = example["steps"]
    n = len(steps)
    cols = 2 if n == 4 else min(n, 3)
    rows = int(np.ceil(n / cols))
    cmap = _ramp_cmap()
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 6.6 * rows + 0.5), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    real_class = example["opponent"]["class_hindsight"]
    decisions = example["decisions"]
    for ax, st in zip(axes, steps):
        grid, labels = rangemodel.class_grid(np.asarray(st["_grid"]))
        ax.set_facecolor(SURFACE)
        for i in range(13):
            for j in range(13):
                v = grid[i, j]
                fill = OUT_OF_RANGE if v <= 0 else to_hex(cmap(min(1.0, v)))
                ax.add_patch(Rectangle((j + 0.04, 12 - i + 0.04), 0.92, 0.92, facecolor=fill, edgecolor="none"))
                ink = "#ffffff" if v > 0 and _luminance(fill) < 0.25 else (MUTED if v <= 0 else INK)
                ax.text(j + 0.5, 12 - i + 0.5, labels[i][j], ha="center", va="center", fontsize=6.3, color=ink)
                if labels[i][j] == real_class:
                    ax.add_patch(Rectangle((j + 0.02, 12 - i + 0.02), 0.96, 0.96, facecolor="none",
                                           edgecolor=INK, linewidth=1.6))
        groups = st["groups"]
        shape = ""
        if groups:
            strong = groups.get("monster", 0) + groups.get("strong", 0)
            shape = f" · two pair or better {strong:.0%} · no pair {groups.get('air', 0):.0%}"
        top = ", ".join(f"{c['class']} {c['share']:.0%}" for c in st["top_classes"][:3])
        nxt = [d for d in decisions if st["action_index"] is None or d["action_index"] > st["action_index"]]
        lines = [f"{st['n_combos']} combos ({st['effective_combos']:.0f} effective){shape}", f"heaviest: {top}"]
        if nxt and st["step"] > 0:
            d = nxt[0]
            lines.append(f"hero's next decision ({d['street']} {d['action']}): vs range {d['eq_vs_range']:.2f}, "
                         f"vs real {d['eq_vs_real']:.2f}")
        board = st["board"]
        board_txt = f"   [{' '.join(board[i:i + 2] for i in range(0, len(board), 2))}]" if board else ""
        ax.text(0, 16.0, f"{st['step']}. {st['label']}{board_txt}", fontsize=10.5, color=INK, va="bottom")
        for k, text in enumerate(lines):
            ax.text(0, 15.2 - 0.72 * k, text, fontsize=7.8, color=INK_2, va="bottom")
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 16.9)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    hero = example["hero"]
    opp = example["opponent"]
    fig.suptitle(f"Predicted range of the opponent ({opp['position']}) in hand {example['hand_id']}; the hero "
                 f"({hero['position']}) holds {hero['cards']}{note}", x=0.02, ha="left", fontsize=12.5, color=INK,
                 y=0.995)
    fig.text(0.02, 0.006, f"Colour: weight per combo relative to the heaviest class of the panel; grey cells are "
                          f"outside the range. Outlined: the opponent's real hand ({opp['cards_hindsight']}), "
                          f"shown in hindsight.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_equities(example: Mapping[str, Any], path: Path) -> None:
    """The hero's three equities at each of his postflop decisions, with the pot odds when he faced a bet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dec = example["decisions"]
    x = np.arange(len(dec))
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRID, linewidth=1, zorder=0)
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    for key, color in SERIES.items():
        ys = [d[key] for d in dec]
        ax.plot(x, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.scatter(x, ys, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(SERIES_LABEL[key], (x[-1], ys[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK_2)
    po = [(i, d["pot_odds"]) for i, d in enumerate(dec) if d["pot_odds"] is not None]
    if po:
        ax.scatter([p[0] for p in po], [p[1] for p in po], marker="_", s=420, linewidths=2, color=INK, zorder=4)
        i0, y0 = po[0]
        ax.annotate("pot odds", (i0, y0), xytext=(-12, -2), textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['street']}\n{d['action']}" for d in dec], fontsize=8.5, color=INK_2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.4, len(dec) - 0.4 + 1.6)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8.5, colors=INK_2, length=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    hero = example["hero"]
    ax.set_title(f"Hero's equity at each decision, hand {example['hand_id']} ({hero['cards']})", loc="left",
                 fontsize=11, color=INK)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE)
               for c in SERIES.values()]
    ax.legend(handles, list(SERIES_LABEL.values()), loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3,
              frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ outputs


def write_table(con: Any, example: Mapping[str, Any], create_output_table: Any) -> int:
    """Store the class weights of every step in ``action_range_opp``."""
    rows = []
    for st in example["steps"]:
        for cls, share in st["class_share"].items():
            rows.append({"hand_id": example["hand_id"], "step": st["step"], "action_index": st["action_index"],
                         "step_label": st["label"], "opp_player_id": example["opponent"]["player_id"],
                         "hero_player_id": example["hero"]["player_id"], "n_combos": st["n_combos"],
                         "hand_class": cls, "weight": share})
    create_output_table(con, "action_range_opp")
    df = pd.DataFrame(rows)
    df["action_index"] = df["action_index"].astype("Int16")
    warehouse.insert_df(con, "action_range_opp", df, mode="append")
    return len(df)


def run_example(dataset: str, cfg: Mapping[str, Any], hand_id: Optional[str] = None, suffix: str = "",
                write_db: bool = True) -> dict[str, Any]:
    """Find (or take) the example hand, build it, write the figures, the JSON and the table; returns a summary.

    ``suffix`` is appended to the file names (for a variant of the settings); ``write_db=False`` leaves the
    warehouse alone.
    """
    runmod = _run_module()
    ex = cfg["examples"]
    pattern = list(ex["pattern"])
    con = runmod.connect(dataset, read_only=not write_db)
    try:
        tables, _ = runmod.load_tables(con)
        cands = find_candidates(con, pattern)
        pinned = hand_id or str(ex.get("hand_id") or "")
        pick = [c for c in cands if c[0] == pinned] if pinned else cands[:1]
        if not pick:
            return {"hand_id": pinned or None, "candidates": len(cands), "written": False}
        hid, opp, hero = pick[0]
        example = build_example(con, cfg, tables, hid, opp, hero)
        n_rows = write_table(con, example, runmod.create_output_table) if write_db else 0
    finally:
        con.close()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    note = f" (variant {suffix.strip('_')})" if suffix else ""
    plot_grids(example, FIG_DIR / f"range_narrowing{suffix}.png", note)
    plot_equities(example, FIG_DIR / f"equity_by_decision{suffix}.png")
    out = dict(example)
    out["dataset"] = dataset
    out["pattern"] = pattern
    out["candidates"] = len(cands)
    out["steps"] = [{k: v for k, v in st.items() if k != "_grid"} for st in example["steps"]]
    (FIG_DIR / f"range_narrowing{suffix}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return {"hand_id": hid, "candidates": len(cands), "written": True, "action_range_opp_rows": n_rows,
            "steps": [{"label": s["label"], "n_combos": s["n_combos"], "top": s["top_classes"][:3]}
                      for s in out["steps"]],
            "decisions": out["decisions"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 04 example: the opponent's range narrowing street by street.")
    ap.add_argument("--dataset", default=None, help="dataset (config examples.dataset)")
    ap.add_argument("--hand-id", default=None, help="example hand (config examples.hand_id)")
    ap.add_argument("--config", default=None, help="TOML file overriding config/step04_range.toml")
    ap.add_argument("--list", action="store_true", help="print the candidate hands and exit")
    ap.add_argument("--suffix", default="", help="append to the output file names (a variant of the settings); "
                                                 "the warehouse table is then left alone")
    args = ap.parse_args(argv)
    runmod = _run_module()
    cfg = runmod.load_config(args.config)
    dataset = args.dataset or cfg["examples"]["dataset"]
    if args.list:
        con = runmod.connect(dataset, read_only=True)
        try:
            for c in find_candidates(con, cfg["examples"]["pattern"]):
                print(*c)
        finally:
            con.close()
        return 0
    summary = run_example(dataset, cfg, args.hand_id, suffix=args.suffix, write_db=not args.suffix)
    print(json.dumps(summary, indent=1))
    return 0 if summary.get("written") else 1


if __name__ == "__main__":
    sys.exit(main())
