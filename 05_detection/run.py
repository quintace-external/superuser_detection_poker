"""Step 05: the per-suspect forensic analysis on the warehouse tables, with the opponent's predicted range.

Runs the analysis of the forensic package (``superuser/``: Stages 1-5, statistics, signals, suspicious hands A-D,
the 15-day escalation rule) for one player through ``adapter.py``, then adds what steps 02 and 04 know about each
decision:

* every decision row gets ``eq_vs_all_hands`` (step 02, equity vs one random hand), ``eq_vs_real`` (step 02),
  ``eq_vs_range`` (step 04, equity vs the opponent's predicted range) and ``pot_odds`` (step 01);
* every suspicious hand gets the same four numbers for its key decision and a ``range_verdict``;
* up to ``--range-hands`` more hands that only the range rule finds are added as category ``R``: a fold with
  ``eq_vs_range - pot_odds >= 0.24`` while ``eq_vs_real < 0.02``, or a call or raise facing a bet with
  ``pot_odds - eq_vs_range >= 0.20`` while ``eq_vs_real >= 0.5``. They are not counted in the escalation rule.

and what the reference field F knows about him (``--field-dataset``, default: the dataset itself):

* rare decisions on the preflop and the flop (category ``S``): his decisions the opponent's real cards made right and
  that the pool (every other player of F) rarely makes in the same spot at the same strength against the opponent's
  line, ranked by -ln(the pool's rate); the top ``--top-hands`` of each street (``config/step05_report.toml``);
* the street tests of F's ``player_scores`` (his value, percentile and rank, the best other player), the street-tails
  verdict, the article's scorecard against F and the observer metrics;
* the opponents he beats beyond his own level and beyond what the rest of F does against them (F's
  ``pair_outliers``, ``sudetect/pairs.py``): the victims table, and "pair #k of N" next to a victim's id in every hand.

``--id-map`` (a CSV with the columns ``case_id`` and ``ref_id``) translates this dataset's player ids into F's when
the dataset is an anonymised extract of a larger field; every lookup in F goes through it.

The thresholds and definitions of A-D are those of the package (`superuser/`). Outputs go to
``$SU_DATA_DIR/<dataset>/report/``: ``report.md`` and ``report.html`` (verdict, metrics by street, the opponents he
beat, the article scorecard, the observer metrics, then the package's sections), ``suspicious_hands.html`` (one page,
tabs Postflop / Preflop / All, a predicted-range box per hand), ``suspicious_hands_all.html``,
``suspicious_hands_ranked.csv``, ``suspicious_hands.csv``, ``rare_decisions.csv``, ``pairs_victims.csv``,
``metrics_summary.csv``, ``decisions.csv``, ``decision_times.csv``, ``stats.csv``, ``signals.csv`` and
``summary.json``.

Usage::

    python 05_detection/run.py --dataset article_sample                 # the sample's subject
    python 05_detection/run.py --dataset <name> --player <id> [--opponent <id>] [--scope flop_hu|hu_tables|all]
    python 05_detection/run.py --dataset <case> --player <id> --field-dataset <field> [--id-map ids.csv]

Scope (``--scope``, default ``flop_hu``): the hands dealt to two players and, at larger tables, the hands the player
took to the flop with exactly one other player; ``hu_tables`` keeps two-player tables only (the article's scope);
``all`` (or ``--all-tables``) adds multiway pots. The flags outside the scope are listed after the others.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import html
import importlib.util
import json
import math
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import pairs, potsize, rangemodel, warehouse  # noqa: E402
from sudetect.cards import CLASS_LABELS, class_index, hand_class  # noqa: E402
from superuser.cards import CATEGORY_NAMES, evaluate, to_pretty  # noqa: E402
from superuser.config import DEFAULT_PLAYER  # noqa: E402
from superuser.html_report import suspicious_hands_html, tabbed_hands_html  # noqa: E402
from superuser.parser import BOARD_LEN  # noqa: E402
from superuser.report import _escalation_text, write_report  # noqa: E402
from superuser.suspicious import _category, _live_uids  # noqa: E402

RANGE_COLUMNS = ["eq_vs_all_hands", "eq_vs_range", "eq_vs_real", "pot_odds", "range_verdict"]
FOLD_MARGIN, CALL_MARGIN = 0.24, 0.20          # the range rule of category R
DEAD_BELOW, AHEAD_FROM = 0.02, 0.5
DEFAULT_RANGE_CONFIG = ROOT / "config" / "step04_range.toml"
REPORT_CONFIG = ROOT / "config" / "step05_report.toml"
SCREEN_CONFIG = ROOT / "config" / "step05_screening.toml"


def _module(name: str, path: Path) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def adapter() -> Any:
    return _module("step05_adapter", HERE / "adapter.py")


def _num(v: Any) -> Optional[float]:
    return None if v is None or pd.isna(v) else float(v)


def _hand_ids(df: pd.DataFrame) -> set:
    """The hand ids of a suspicious-hands frame (an empty frame may have no columns at all)."""
    return set(df["hand_uid"]) if "hand_uid" in df.columns else set()


def _stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The frames one under the other, the empty ones left out: an empty frame has untyped columns and must not decide
    a column's type (pandas 2.2 warns about it, pandas 3 turns the column into object)."""
    return pd.concat([f for f in frames if not f.empty] or frames[:1], ignore_index=True)


# ------------------------------------------------------------------ range columns


def range_frame(con: Any) -> pd.DataFrame:
    """(hand_uid, seq) -> pot odds and the three equities of every decision row."""
    has_eq = warehouse.table_exists(con, "action_equity")
    has_rng = warehouse.table_exists(con, "action_range_equity")
    sql = f"""
        SELECT a.hand_id AS hand_uid, CAST(a.action_index AS INTEGER) AS seq, a.pot_odds,
               {'e.eq_vs_random' if has_eq else 'NULL'} AS eq_vs_all_hands,
               {'e.eq_vs_real' if has_eq else 'NULL'} AS eq_vs_real,
               {'r.eq_vs_range' if has_rng else 'NULL'} AS eq_vs_range,
               {'r.n_combos_opp' if has_rng else 'NULL'} AS n_combos_opp
        FROM actions a
        {'LEFT JOIN action_equity e USING (hand_id, action_index)' if has_eq else ''}
        {'LEFT JOIN action_range_equity r USING (hand_id, action_index)' if has_rng else ''}
        WHERE NOT a.is_post"""
    return warehouse.query_df(con, sql)


def verdict(row: dict) -> str:
    eq_r, po = _num(row.get("eq_vs_range")), _num(row.get("pot_odds"))
    if eq_r is None:
        return "no range (not a heads-up postflop decision)"
    if row["is_fold"] and po is not None:
        m = eq_r - po
        if m >= FOLD_MARGIN:
            return f"range says call: range equity {m:+.0%} over the pot odds, folded"
        return "range says call (thin), folded" if m > 0 else "range agrees with the fold"
    if row["is_continue"] and row["facing_bet"] and po is not None:
        g = po - eq_r
        if g >= CALL_MARGIN:
            return f"range says fold: range equity {g:.0%} under the pot odds, continued"
        return "range says fold (thin), continued" if g > 0 else "range agrees with the call"
    return f"{row['action']} with {eq_r:.0%} equity against the range"


SEVERITY_EDGES = (0.30, 0.15)   # blatant >= 0.30 > strong >= 0.15 > thin > 0


def severity_note(row, is_fold: bool, sev: float) -> dict:
    """Tier and plain-words verdict of one suspicious hand for the review page."""
    eq_r, po, eq_q = _num(row.get("eq_vs_range")), _num(row.get("pot_odds")), _num(row.get("eq_vs_real"))
    opp = str(row.get("opponent_hand") or "").lower() or "a better hand"
    if row.get("category") == "C":
        return {"severity": 0.0, "tier": "context",
                "comment": "Successful bluff. The predicted range cannot show whether he knew the opponent was weak."}
    if eq_r is None or po is None or sev <= 0:
        return {"severity": 0.0, "tier": "agrees", "comment": "The predicted range agrees with the decision: ordinary play."}
    tier = "blatant" if sev >= SEVERITY_EDGES[0] else "strong" if sev >= SEVERITY_EDGES[1] else "thin"
    pts = int(sev * 100)                 # whole points, never rounded across a tier edge
    if is_fold:
        text = {"blatant": f"The betting gave every reason to call: {eq_r:.0%} equity against the predicted range at pot odds "
                           f"of only {po:.0%}. He folded anyway, and only the opponent's real hand ({opp}, leaving him "
                           f"{eq_q:.0%}) made the fold right. An honest player cannot see that hand.",
                "strong": f"The range favoured a call by {pts} points over the pot odds ({eq_r:.0%} against {po:.0%}). "
                          f"He folded into the opponent's real hand ({opp}).",
                "thin": f"The range was only {pts} points above the pot odds; weak evidence on its own."}[tier]
    else:
        text = {"blatant": f"The betting said fold: {eq_r:.0%} equity against the predicted range at pot odds of {po:.0%}. "
                           f"He called anyway, and the opponent's real hand ({opp}) made the call right.",
                "strong": f"The range favoured a fold by {pts} points under the pot odds ({eq_r:.0%} against {po:.0%}). "
                          f"He called into the opponent's weak real hand ({opp}).",
                "thin": f"The range was only {pts} points under the pot odds; weak evidence on its own."}[tier]
    return {"severity": round(sev, 4), "tier": tier, "comment": text}


FIELD_REF_DRAWS, FIELD_REF_SEED = 5, 20261007     # subsamples per field player for the matched comparison


def matched_field_shares(field_dataset: str, n: int) -> Optional[np.ndarray]:
    """Real-card share of every field player with at least n responses, measured on n of them (the mean of
    FIELD_REF_DRAWS random subsamples). The share is inflated on small samples, so a player's own number compares
    only with the field at the same n. Reads the field's scored decisions from its step-04 stage folder."""
    s04 = _module("step04_run", ROOT / "04_predicted_range" / "run.py")
    cfg = s04.load_config()
    streets = list(cfg["leakage"]["streets"])
    files = sorted((warehouse.data_dir() / f"{field_dataset}_step04" / "out").glob("*.parquet"))
    if not files:
        return None
    cols = ["player_id", "street", "n_live_opp", "eq_vs_range", "eq_vs_real", "action", "pot_odds"]
    resp = s04.responses(pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True), streets)
    rng = np.random.default_rng(FIELD_REF_SEED)
    counts = resp.player_id.value_counts()
    shares = []
    for pid in counts[counts >= n].index:
        d = resp[resp.player_id == pid]
        vals = [s04.leakage_split(s04.leak_matrix(d.iloc[rng.choice(len(d), n, replace=False)], cfg), streets)
                .get("real_share") for _ in range(FIELD_REF_DRAWS)]
        vals = [v for v in vals if v is not None]
        if vals:
            shares.append(float(np.mean(vals)))
    return np.asarray(shares) if shares else None


def field_note(dataset: str, field_dataset: str | None, player: str | None = None, with_n: bool = True) -> str:
    """One box at the top of the page: how much the opponents' real cards add to his folds, against field players
    measured on the same number of responses."""
    f = warehouse.data_dir() / f"{dataset}_step04" / "summary.json"
    try:
        own = json.loads(f.read_text(encoding="utf-8"))["leakage"]["split_by_player"][str(player)]
        share, n = float(own["real_share"]), int(own["n"])
    except Exception:
        return ""
    ref = None
    if field_dataset:
        try:
            ref = matched_field_shares(field_dataset, n)
        except Exception:
            ref = None
    if ref is None or not len(ref):
        return ""
    return ('<div class="box"><b>Real cards beyond the range.</b> Knowing the opponents\' real cards explains his '
            f"turn and river folds better than for {np.mean(ref < share):.0%} of field players measured on the same "
            f"number of responses ({n}): the real cards add {share:.1%} to the fit for him and {np.median(ref):.1%} "
            f"for the median player" + (f" ({len(ref)} players)" if with_n else "") + ".</div>")


TIER_ORDER = ("blatant", "strong", "thin", "context", "raise", "agrees", "mistake", "norange")
TIER_NAMES = {"blatant": "BLATANT", "strong": "STRONG", "thin": "THIN", "context": "BLUFF", "raise": "RAISE / OVERBET",
              "agrees": "RANGE AGREES", "mistake": "PLAIN MISTAKE", "norange": "NO RANGE"}
RULE_NAMES = {"A": "River fold when beaten", "B": "River bluff-catch", "C": "Successful river bluff",
              "D": "Early fold, near dead", "R": "Range rule"}
MULTIWAY_NOTE = "Outside the scope (a multiway pot): the predicted range covers heads-up pots only."


def faced_raise(hand: Any, row: dict) -> bool:
    """The bet he faced was a raise: somebody bet or raised earlier on the same street."""
    before = [a for a in hand.actions.get(row["street"], []) if a.seq < row["seq"]]
    return sum(bool(a.aggressive) for a in before) >= 2


def equity_note(row, is_fold: bool, sev: float, raised: bool) -> dict:
    """Tier and plain-words verdict of the Equity review for one flagged heads-up hand."""
    eq_r, po, btp = _num(row.get("eq_vs_range")), _num(row.get("pot_odds")), _num(row.get("bet_to_pot"))
    opp = str(row.get("opponent_hand") or "").lower() or "a better hand"
    if row.get("category") == "C":
        return severity_note(row, is_fold, 0.0)
    if eq_r is None or po is None:
        return {"severity": 0.0, "tier": "norange", "comment": "No predicted range for this decision."}
    if is_fold and btp is not None and btp >= 1.5:
        what = "a raise" if raised else "an overbet"
        size = f"{btp:.0%} of the pot" if btp < 2 else f"{btp:.1f} times the pot"
        return {"severity": round(sev, 4), "tier": "raise",
                "comment": f"He folded to {what} of {size}. The range model reads the bet size but not the betting "
                           f"line, so after raises and overbets its verdict ({eq_r:.0%} against the range, pot odds "
                           f"{po:.0%}) is weak evidence."}
    if sev > 0:
        return severity_note(row, is_fold, sev)
    if (eq_r < po) == is_fold:
        act = "folds" if is_fold else "calls"
        return {"severity": 0.0, "tier": "agrees",
                "comment": f"The predicted range agrees with the decision ({eq_r:.0%} against the range at pot odds of "
                           f"{po:.0%}): an honest player {act} here too. The package flags it because the opponent's "
                           f"real hand ({opp}) made the decision right."}
    return {"severity": 0.0, "tier": "mistake",
            "comment": f"The range ({eq_r:.0%} against pot odds of {po:.0%}) and the real cards both point the other "
                       f"way: a plain mistake, not a sign of seeing cards."}


def overview_html(ranked: pd.DataFrame, n_mw: int) -> str:
    """The box at the top of the ranked page: the two verdicts and the order."""
    agree = ranked[ranked["_tier"].eq("agrees") & ranked["category"].isin(["A", "D"])]
    mw = (f" The last {n_mw} come from hands outside the scope (multiway pots), which the range model does not cover."
          if n_mw else "")
    return (f'<div class="box"><b>Every flagged hand, ranked.</b> All {len(ranked)} hands the package flags are on this '
            "page, each with two verdicts. The coloured tag at the top is the package's hand-set rule (A-D), read from "
            "the opponent's real cards. The <b>Equity review</b> box under the hand is computed from the betting: his "
            "equity against the range the betting predicts for the opponent, set against the pot odds. The Equity "
            "review only orders and colours the hands; it removes none. Order: BLATANT, STRONG and THIN contradictions "
            "of the range, largest first (the betting pointed one way, the real cards the other, and he followed the "
            "real cards); then successful bluffs (BLUFF); then folds to a raise or an overbet (RAISE / OVERBET, where "
            "the range is weak evidence); then hands where the range agrees with his decision (RANGE AGREES) or "
            f"disagrees with it as much as the real cards do (PLAIN MISTAKE); then hands without a range (NO RANGE).{mw} "
            "Under each Equity review box: the opponent's predicted range as a 13 x 13 grid and, in its own toggle, as "
            "text (the weights), plus a JSON block (script.range-data) for scripts.</div>"
            '<div class="box"><b>Why the simple rules also flag ordinary folds.</b> Rules A and D read only the '
            "opponent's real cards: a fold counts whenever the bettor really had the better hand, whatever the bet "
            f"size or the pot odds. {len(agree)} of the A and D folds rank RANGE AGREES: an honest player folds there "
            "too. The rules find the candidates; the Equity review puts the strongest first.</div>")


def key_decisions(result: dict) -> dict[str, dict]:
    """The decision row the package used for each suspicious hand (first qualifying row, priority A > B > C > D);
    {} when he has none."""
    if result["suspicious_hands"].empty:
        return {}
    dec, hands, uid = result["decisions"], result["hands"], result["player"]
    mine = dec[dec["uid"] == uid]
    out = {}
    for gid in result["suspicious_hands"]["hand_uid"]:
        rows = mine[mine["hand_uid"] == gid].to_dict("records")
        found = [(c, r) for r in rows if (c := _category(hands[gid], r))]
        if found:
            out[gid] = min(found, key=lambda x: "ABCD".index(x[0]))[1]
    return out


def range_only_rows(result: dict, name_of: dict[int, str], limit: int) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Category R: hands the package does not flag where the range rule fires; the strongest first."""
    dec, hands, uid = result["decisions"], result["hands"], result["player"]
    flagged = set(result["suspicious_hands"]["hand_uid"]) if not result["suspicious_hands"].empty else set()
    m = dec[(dec["uid"] == uid) & dec["eq_vs_range"].notna() & dec["pot_odds"].notna()
            & dec["eq_vs_real"].notna() & ~dec["hand_uid"].isin(flagged)].copy()
    fold = m["is_fold"] & (m["eq_vs_range"] - m["pot_odds"] >= FOLD_MARGIN) & (m["eq_vs_real"] < DEAD_BELOW)
    cont = (m["is_continue"] & m["facing_bet"] & (m["pot_odds"] - m["eq_vs_range"] >= CALL_MARGIN)
            & (m["eq_vs_real"] >= AHEAD_FROM))
    m["score"] = np.where(fold, m["eq_vs_range"] - m["pot_odds"], m["pot_odds"] - m["eq_vs_range"])
    m = m[fold | cont].sort_values("score", ascending=False).drop_duplicates("hand_uid").head(limit)
    rows, keys = [], {}
    for row in m.to_dict("records"):
        hand, gid = hands[row["hand_uid"]], row["hand_uid"]
        seat = hand.seat_of(uid)
        board = hand.board[:BOARD_LEN[row["street"]]]
        opp = row["bettor_uid"] if row.get("bettor_uid") is not None and not pd.isna(row["bettor_uid"]) else next(
            (s.uid for s in hand.seats.values() if s.uid != uid), None)
        opp_seat = hand.seat_of(int(opp)) if opp is not None else None
        hero_cat = evaluate(board + seat.hole)[0]
        opp_cat = evaluate(board + opp_seat.hole)[0] if opp_seat and opp_seat.hole else None
        bet = row.get("bet_faced") or 0.0
        pot_pre = max(row["pot_before"] - bet, 1)
        verb = "folded" if row["is_fold"] else f"{row['action']} {row['amount']:,.0f}"
        rows.append({
            "hand_uid": gid, "start_utc": pd.to_datetime(hand.start_ts, unit="s"), "category": "R",
            "category_name": "Against the predicted range",
            "decision_label": "Range fold" if row["is_fold"] else "Range call", "street": row["street"],
            "key_action": f"faced {bet:,.0f} into ~{pot_pre:,.0f} ({bet / pot_pre:.0%} pot), {verb}",
            "bet_to_pot": round(bet / pot_pre, 2), "decision_time_s": row["think_time"], "snap_decision": False,
            "equity_at_decision": round(float(row["eq_vs_real"]), 3), "player_cards": to_pretty(seat.hole),
            "opponent_uid": name_of.get(int(opp), opp) if opp is not None else None,
            "opponent_cards": to_pretty(opp_seat.hole) if opp_seat else "", "board": to_pretty(hand.board),
            "player_hand": CATEGORY_NAMES[hero_cat], "opponent_hand": CATEGORY_NAMES[opp_cat] if opp_cat is not None else "",
            "pot_bb": round(hand.contested_pot() / hand.blind, 1) if hand.blind else 0.0,
            "net_win": hand.settlement.get(uid, 0.0),
        })
        keys[gid] = row
    return pd.DataFrame(rows), keys


def _range_view(ids: np.ndarray, w: np.ndarray, opp_cards: Any = None) -> dict:
    """What a box shows of a range: its size, the effective size 1 / sum(p^2) (a range of n equal combos gives n;
    light tails count for little), the 13 x 13 grid, the ten heaviest classes and, with the opponent's real cards,
    his class's share of the range and its rank among the 169 classes (1 = the heaviest)."""
    mass = rangemodel.range_classes(ids, w)
    order = np.argsort(-mass, kind="stable")[:10]
    grid, labels = rangemodel.class_grid(mass)
    p = np.asarray(w, dtype=float) / float(np.sum(w))
    out = dict(n_combos=int(ids.size), n_effective=int(round(1.0 / float(np.sum(p * p)))),
               grid=grid.tolist(), grid_labels=labels,
               top_classes=[(CLASS_LABELS[i], float(mass[i])) for i in order if mass[i] > 0])
    if opp_cards is not None and len(opp_cards) == 2:
        k = class_index(hand_class(list(opp_cards)))
        total = float(mass.sum())
        out.update(actual_class=CLASS_LABELS[k], actual_share=float(mass[k]) / total if total > 0 else 0.0,
                   actual_rank=int((mass > mass[k]).sum()) + 1)
    return out


def range_box(con: Any, cfg: dict, tables: Any, examples: Any, hand_id: str, row: dict, hero_id: str,
              opp_id: str) -> dict:
    """The opponent's predicted range before the key decision, as the HTML box shows it."""
    box = {c: _num(row.get(c)) for c in RANGE_COLUMNS[:-1]}
    box["range_verdict"] = verdict(row)
    box["opponent"] = opp_id
    if box["eq_vs_range"] is None:
        return box
    h = examples.load_hand(con, cfg, hand_id, opp_id, hero_id)
    ai = int(row["seq"])
    act = h["acts"][h["acts"]["action_index"] == ai].iloc[0]
    board = rangemodel.cards_of(act["board_at_action"])
    before = tuple(n for k, n in h["nodes"] if k < ai)
    res = rangemodel.predict_range(tables, rangemodel.RangeConfig.from_toml(cfg),
                                   dataclasses.replace(h["line"], nodes=before), h["hero"] + board)
    if res is None or res.ids.size == 0:
        return box
    box.update(_range_view(res.ids, res.w, h["opp_cards"]))
    return box


def preflop_box(con: Any, cfg: dict, tables: Any, examples: Any, hand_id: str, row: dict, hero_id: str,
                opp_id: str) -> dict:
    """The opponent's predicted range before a preflop decision: the field's range for his last preflop action
    before it (``FitTables.prior_for`` on that action's situation, his position class and the action, then
    ``expand_prior`` without the hero's cards), the range ``04_predicted_range/preflop.py`` measures r against. No
    range when he has not acted yet: his range is every hand."""
    box = {c: _num(row.get(c)) for c in RANGE_COLUMNS[:-1]}
    box["range_verdict"] = verdict(row)
    box["opponent"] = opp_id
    h = examples.load_hand(con, cfg, hand_id, opp_id, hero_id)
    acts = h["acts"]
    prev = acts[(acts["player_id"] == opp_id) & (acts["street"] == "preflop") & ~acts["is_post"].astype(bool)
                & (acts["action_index"] < int(row["seq"]))]
    if prev.empty:
        return box
    rcfg = rangemodel.RangeConfig.from_toml(cfg)
    act = str(prev["action"].iat[-1])
    prior, _ = tables.prior_for(h["line"].size_cls, str(prev["situation"].iat[-1]), h["line"].position_class,
                                "raise" if act == "bet" else act, rcfg)
    if prior is None:
        return box
    ids, w = rangemodel.expand_prior(prior, h["hero"], rcfg)
    if ids.size == 0:
        return box
    box.update(_range_view(ids, w, h["opp_cards"]))
    return box


# ------------------------------------------------------------------ configuration and the reference field

REPORT_DEFAULTS: dict[str, Any] = {      # config/step05_report.toml, which documents every key
    "top_hands": 25, "pairs_max_hands": 500_000, "show_field_size": True,
    "rare": {"pre_bins": 20, "flop_bins": 10, "min_pool": 30, "facing": ["vs_open", "vs_3bet", "vs_4bet_plus"],
             "pre_fold_q_max": 0.40, "pre_fold_p_max": 0.50, "pre_raise_q_min": 0.55, "pre_raise_p_max": 0.30,
             "pre_call_q_min": 0.55, "pre_call_p_max": 0.30, "flop_fold_q_max": 0.20, "flop_fold_p_max": 0.50,
             "flop_call_q_min": 0.50, "flop_call_p_max": 0.30, "p_floor": 0.005, "perfect_p": 0.05, "good_p": 0.15},
    "metrics": {"pf_min_behind": 20, "pf_min_ahead": 20, "light_reraises_min": 10, "reraises_min": 5,
                "flop_min_dead": 15, "flop_min_win": 15, "tr_min_dead": 20, "tr_min_win": 20},
    "scorecard": {"min_n": 20, "top_winners": 10},
}


def _merge(base: dict, extra: Mapping) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
    return out


def load_report_config(extra: Optional[str | Path] = None) -> dict:
    """The ``[report]`` table of config/step05_report.toml over the built-in defaults, then the ``[report]`` keys of
    ``extra`` (a TOML file) on top."""
    cfg = copy.deepcopy(REPORT_DEFAULTS)
    for path in ([REPORT_CONFIG] if REPORT_CONFIG.exists() else []) + ([Path(extra)] if extra else []):
        with open(path, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh).get("report", {}))
    return cfg


def screen_config(con: Any = None) -> dict:
    """The ``[step05]`` settings the field's screen ran with (``report_meta`` item ``config_json``, written by
    05_detection/screen.py, so a screen run with --config is read back as it ran), else the ``[step05]`` table of
    config/step05_screening.toml; {} when neither can be read."""
    if con is not None and warehouse.table_exists(con, "report_meta"):
        try:
            row = warehouse.query_df(con, "SELECT text FROM report_meta WHERE item = 'config_json' LIMIT 1")
            if len(row):
                return json.loads(row["text"].iat[0])
        except Exception:   # noqa: BLE001 - an older or damaged report_meta: fall back to the file
            pass
    try:
        with open(SCREEN_CONFIG, "rb") as fh:
            return tomllib.load(fh).get("step05", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_id_map(path: Optional[str | Path]) -> dict[str, str]:
    """``--id-map``: a CSV with the columns ``case_id`` (a player id of this dataset) and ``ref_id`` (the same
    player in the field F); {} without a file."""
    if not path:
        return {}
    t = pd.read_csv(path, dtype=str)
    if not {"case_id", "ref_id"} <= set(t.columns):
        raise SystemExit(f"{path}: an id map needs the columns case_id and ref_id")
    t = t.dropna(subset=["case_id", "ref_id"])
    return dict(zip(t["case_id"].str.strip(), t["ref_id"].str.strip()))


def _window_all(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of the window "all" (one scope) of a table written by screen.py."""
    if "time_window" in df.columns:
        df = df[df["time_window"] == "all"]
    if "scope" in df.columns and df["scope"].nunique() > 1:
        scopes = set(df["scope"])
        df = df[df["scope"] == ("flop_hu" if "flop_hu" in scopes else sorted(scopes)[0])]
    return df.reset_index(drop=True)


def _pair_rows(con: Any, ref: str) -> pd.DataFrame:
    """His rows (as A) of the pair_outliers table, window "all"."""
    cols = warehouse.table_columns(con, "pair_outliers")
    a, b = ("A", "B") if "A" in cols else ("player_id", "opponent_id")
    t = _window_all(warehouse.query_df(con, f'SELECT * FROM pair_outliers WHERE "{a}" = :r', r=ref))
    return t.rename(columns={a: "A", b: "B"})


def load_field(con: Any, name: str, ref: str, rc: Mapping, scfg: Mapping) -> dict[str, Any]:
    """What the report reads from the reference field F (``con`` is open on it; ``ref`` is the suspect's id there).

    ``scores``: ``player_scores``, window "all"; ``pairs``: his rows (as A) of ``pair_outliers``, or of the pair table
    computed here when F has no such table and at most ``pairs_max_hands`` hands; ``pair_cfg``: the pair settings and
    the victim rule (``[step05.pair_outliers]``); ``obs``: the observer metrics with their field ranks
    (``player_scores``' ``obs_*`` columns, or computed here); ``pair_scores``: the single-victim pairs against him (the
    main opponent's games against the rest); ``pools``: the counts the rare decisions are scored against, ``ref`` left
    out. ``notes``: one line per missing piece, by section."""
    notes: dict[str, list[str]] = {"scores": [], "pairs": [], "obs": []}
    po = dict(scfg.get("pair_outliers") or {})
    f: dict[str, Any] = {"name": name, "notes": notes, "pair_cfg": pairs.PairConfig.from_toml(po),
                         "obs_cfg": potsize.from_config(scfg.get("observer")), "scores": None, "pairs": None,
                         "obs": None, "pair_scores": None}
    if warehouse.table_exists(con, "player_scores"):
        f["scores"] = _window_all(warehouse.query_df(con, "SELECT * FROM player_scores"))
        f["scores"]["player_id"] = f["scores"]["player_id"].astype(str)
    else:
        notes["scores"].append(f"{name} has no player_scores table: run 05_detection/screen.py --dataset {name}.")
    if warehouse.table_exists(con, "pair_scores"):
        f["pair_scores"] = _window_all(warehouse.query_df(con, "SELECT * FROM pair_scores WHERE opponent_id = :r",
                                                          r=ref))
    if warehouse.table_exists(con, "pair_outliers"):
        f["pairs"] = _pair_rows(con, ref)
    else:
        n = int(warehouse.query_df(con, "SELECT count(*) AS n FROM hands")["n"].iloc[0])
        cap = int(rc["pairs_max_hands"])
        if n > cap:
            notes["pairs"].append(f"{name} has no pair_outliers table and more hands than the {cap:,} this "
                                  f"report pairs itself (pairs_max_hands): run 05_detection/screen.py --dataset {name}.")
        else:
            try:
                rows, dec = pairs.load_inputs(con, showdown_only=bool(po.get("showdown_only", False)),
                                              big_pot=f["pair_cfg"].big_pot)
                t = pairs.pair_table(rows, dec, f["pair_cfg"])
                f["pairs"] = t[t["A"].astype(str) == ref].reset_index(drop=True)
                notes["pairs"].append(f"{name} has no pair_outliers table: the pairs were computed here "
                                      f"({n:,} hands).")
            except Exception as exc:  # noqa: BLE001 - a field the pair queries cannot read leaves the section out
                notes["pairs"].append(f"The pairs of {name} could not be computed here: {exc}")
    obs_cols = [c for c in f["scores"].columns if c.startswith("obs_")] if f["scores"] is not None else []
    if obs_cols:
        f["obs"] = f["scores"].set_index("player_id")[obs_cols]
    else:
        oc = f["obs_cfg"]
        try:
            m = potsize.player_metrics(con, big_pot=oc["big_pot"], pot_edges=oc["pot_edges"],
                                       flop_edges=oc["flop_edges"], stackoff_share=oc["stackoff_share"])
            f["obs"] = potsize.field_ranks(m, min_hands=oc["min_hands"], min_cases=oc["min_cases"])
            f["obs"].index = f["obs"].index.astype(str)
            notes["obs"].append(f"{name}'s player_scores has no obs_* columns: the observer metrics were computed "
                                f"here, on every hand of {name}.")
        except Exception as exc:  # noqa: BLE001
            notes["obs"].append(f"No observer metrics for {name}: {exc}")
    f["pools"] = rare_pools(con, ref, rc["rare"])
    return f


# ------------------------------------------------------------------ rare decisions (preflop and flop)

RARE_RULES = {
    "pf_fold": "Preflop fold while behind the real hand",
    "pf_raise": "Preflop raise while ahead of the real hand",
    "pf_call": "Preflop call while ahead of the real hand",
    "fl_fold": "Flop fold with little equity against the real hand",
    "fl_call": "Flop call while ahead of the real hand",
}
CANDIDATE_COLUMNS = ["hand_id", "action_index", "street", "situation", "action", "req", "to_call_bb", "hole_cards",
                     "q", "q_random", "r", "opp_id", "opp_hole_cards"]
RARE_COLUMNS = CANDIDATE_COLUMNS + ["rbin", "p_pool", "n_pool", "rule", "rule_name", "score", "tier"]

# the pool: every decision with one live opponent but the suspect's; preflop by situation x band of r x action
_PRE_POOL_SQL = """
SELECT d.situation, LEAST({b} - 1, floor(rp.eq_vs_range * {b})) AS rbin, d.action, count(*) AS n
FROM decisions d
JOIN action_range_equity_pre rp ON rp.hand_id = d.hand_id AND rp.action_index = d.action_index
WHERE d.street = 'preflop' AND d.n_live = 2 AND rp.eq_vs_range IS NOT NULL AND d.player_id <> :me
GROUP BY 1, 2, 3
"""
# ... the flop facing a bet by band of r: decisions, folds, continues (call or raise)
_FLOP_POOL_SQL = """
SELECT LEAST({b} - 1, floor(re.eq_vs_range * {b})) AS rbin, count(*) AS n,
       count(*) FILTER (WHERE d.action = 'fold') AS n_fold,
       count(*) FILTER (WHERE d.action IN ('call', 'raise')) AS n_cont
FROM decisions d
JOIN action_range_equity re ON re.hand_id = d.hand_id AND re.action_index = d.action_index
WHERE d.street = 'flop' AND d.n_live = 2 AND d.to_call_bb > 0 AND re.eq_vs_range IS NOT NULL AND d.player_id <> :me
GROUP BY 1
"""
# his preflop and flop decisions with one live opponent and both hands known
_CANDIDATE_SQL = """
SELECT d.hand_id, d.action_index, d.street, d.situation, d.action, d.pot_odds AS req, d.to_call_bb, d.hole_cards,
       e.eq_vs_real AS q, e.eq_vs_random AS q_random, {r} AS r,
       d.live_opponents[1].player_id AS opp_id, o.hole_cards AS opp_hole_cards
FROM decisions d
JOIN action_equity e ON e.hand_id = d.hand_id AND e.action_index = d.action_index AND e.eq_real_known
LEFT JOIN hand_players o ON o.hand_id = d.hand_id AND o.player_id = d.live_opponents[1].player_id
{joins}
WHERE d.player_id = :p AND d.n_live = 2 AND d.street IN ('preflop', 'flop')
ORDER BY d.hand_id, d.action_index
"""


def rare_pools(con: Any, exclude: str, rc: Mapping) -> dict[str, Optional[pd.DataFrame]]:
    """The pool's decision counts, every player of the warehouse but ``exclude``: ``pre`` per (situation, band of r,
    action), ``flop`` per band of r facing a bet (decisions, folds, continues); None without the range tables."""
    out: dict[str, Optional[pd.DataFrame]] = {"pre": None, "flop": None}
    if warehouse.table_exists(con, "action_range_equity_pre"):
        out["pre"] = warehouse.query_df(con, _PRE_POOL_SQL.format(b=int(rc["pre_bins"])), me=str(exclude))
    if warehouse.table_exists(con, "action_range_equity"):
        out["flop"] = warehouse.query_df(con, _FLOP_POOL_SQL.format(b=int(rc["flop_bins"])), me=str(exclude))
    return out


def rare_candidates(con: Any, player: str) -> pd.DataFrame:
    """His preflop and flop decisions with one live opponent and both hands known: q, r (preflop: against the
    field's range for the opponent's line, ``action_range_equity_pre``; flop: ``action_range_equity``), pot odds."""
    if not warehouse.table_exists(con, "action_equity"):
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    joins, pre, post = [], "CAST(NULL AS DOUBLE)", "CAST(NULL AS DOUBLE)"
    if warehouse.table_exists(con, "action_range_equity_pre"):
        joins.append("LEFT JOIN action_range_equity_pre rp ON rp.hand_id = d.hand_id "
                     "AND rp.action_index = d.action_index")
        pre = "rp.eq_vs_range"
    if warehouse.table_exists(con, "action_range_equity"):
        joins.append("LEFT JOIN action_range_equity re ON re.hand_id = d.hand_id AND re.action_index = d.action_index")
        post = "re.eq_vs_range"
    r = f"CASE WHEN d.street = 'preflop' THEN {pre} ELSE {post} END"
    return warehouse.query_df(con, _CANDIDATE_SQL.format(r=r, joins="\n".join(joins)), p=str(player))


def select_rare(cands: pd.DataFrame, pools: Mapping[str, Optional[pd.DataFrame]], rc: Mapping,
                top_n: int) -> pd.DataFrame:
    """The rare decisions among ``cands``: the pool's rate of his action at his strength (``p_pool``, NaN when the
    band holds fewer than ``min_pool`` decisions; ``n_pool`` = the band's size), the rules of ``[report.rare]``, the
    score -ln(max(p_pool, p_floor)) and the tag (Perfect / Good / Average). The best decision per hand and street,
    then the top ``top_n`` preflop and the top ``top_n`` flop decisions, the rarest first.

    Rules (q = equity against the real hand, req = pot odds): preflop fold facing a raise with q below
    ``pre_fold_q_max`` and below req, the pool folding less often than ``pre_fold_p_max``; preflop raise (opens
    included) with q >= ``pre_raise_q_min``, the pool raising less often than ``pre_raise_p_max``; preflop call facing
    a raise with q >= ``pre_call_q_min``, the pool calling less often than ``pre_call_p_max``; flop fold facing a bet
    with q below ``flop_fold_q_max``, the pool folding less often than ``flop_fold_p_max``; flop call facing a bet with
    q >= ``flop_call_q_min``, the pool continuing less often than ``flop_call_p_max``."""
    c = cands[cands["r"].notna() & cands["q"].notna()].copy().reset_index(drop=True)
    for col in ("q", "r", "req", "to_call_bb"):
        c[col] = pd.to_numeric(c[col], errors="coerce").astype(float)
    pre, fl = (c["street"] == "preflop").to_numpy(), (c["street"] == "flop").to_numpy()
    bp, bf = int(rc["pre_bins"]), int(rc["flop_bins"])
    c["rbin"] = np.where(pre, np.minimum(bp - 1, np.floor(c["r"] * bp)), np.minimum(bf - 1, np.floor(c["r"] * bf)))
    c["p_pool"], c["n_pool"] = np.nan, 0.0
    min_pool = float(rc["min_pool"])
    pp, fp = pools.get("pre"), pools.get("flop")
    if pp is not None and len(pp) and pre.any():
        cell = pp.groupby(["situation", "rbin"], dropna=False)["n"].sum().rename("n_cell").reset_index()
        x = (c.loc[pre, ["situation", "rbin", "action"]].merge(cell, on=["situation", "rbin"], how="left")
             .merge(pp.rename(columns={"n": "n_act"}), on=["situation", "rbin", "action"], how="left"))
        n_cell, n_act = x["n_cell"].fillna(0).to_numpy(float), x["n_act"].fillna(0).to_numpy(float)
        c.loc[pre, "n_pool"] = n_cell
        c.loc[pre, "p_pool"] = np.where(n_cell >= min_pool, n_act / np.maximum(n_cell, 1.0), np.nan)
    if fp is not None and len(fp) and fl.any():
        t = fp.assign(p_fold=fp["n_fold"] / fp["n"], p_cont=fp["n_cont"] / fp["n"])
        x = c.loc[fl, ["rbin", "action"]].merge(t, on="rbin", how="left")
        n_cell = x["n"].fillna(0).to_numpy(float)
        p = np.where(x["action"].to_numpy() == "fold", x["p_fold"].to_numpy(float), x["p_cont"].to_numpy(float))
        c.loc[fl, "n_pool"] = n_cell
        c.loc[fl, "p_pool"] = np.where(n_cell >= min_pool, p, np.nan)
    q, req, p, act = c["q"], c["req"], c["p_pool"], c["action"]
    facing = c["situation"].isin(list(rc["facing"]))
    bet = c["to_call_bb"].fillna(0) > 0
    pre_s, fl_s = pd.Series(pre, index=c.index), pd.Series(fl, index=c.index)
    rules = {
        "pf_fold": pre_s & facing & (act == "fold") & (q < float(rc["pre_fold_q_max"])) & (q < req)
                   & (p < float(rc["pre_fold_p_max"])),
        "pf_raise": pre_s & (act == "raise") & (q >= float(rc["pre_raise_q_min"])) & (p < float(rc["pre_raise_p_max"])),
        "pf_call": pre_s & facing & (act == "call") & (q >= float(rc["pre_call_q_min"]))
                   & (p < float(rc["pre_call_p_max"])),
        "fl_fold": fl_s & bet & (act == "fold") & (q < float(rc["flop_fold_q_max"])) & (p < float(rc["flop_fold_p_max"])),
        "fl_call": fl_s & bet & (act == "call") & (q >= float(rc["flop_call_q_min"]))
                   & (p < float(rc["flop_call_p_max"])),
    }
    c["rule"] = ""
    for name, hit in rules.items():
        c.loc[hit.fillna(False).astype(bool) & (c["rule"] == ""), "rule"] = name
    s = c[c["rule"] != ""].copy()
    s["rule_name"] = s["rule"].map(RARE_RULES)
    s["score"] = -np.log(s["p_pool"].clip(lower=float(rc["p_floor"])))
    s["tier"] = np.select([s["p_pool"] < float(rc["perfect_p"]), s["p_pool"] < float(rc["good_p"])],
                          ["Perfect", "Good"], default="Average")
    s = s.sort_values(["score", "p_pool", "n_pool", "hand_id", "action_index"],
                      ascending=[False, True, False, True, True], kind="mergesort")
    s = s.drop_duplicates(["hand_id", "street"])
    out = pd.concat([s[s["street"] == "preflop"].head(top_n), s[s["street"] == "flop"].head(top_n)])
    return out.reindex(columns=RARE_COLUMNS).reset_index(drop=True)


def _class_of(cards: Any) -> str:
    """'AhKd' -> 'AKo'; '' when the cards are unknown."""
    return hand_class(cards) if isinstance(cards, str) and len(cards) == 4 else ""


def rare_rows(sel: pd.DataFrame, hands: Mapping[str, Any], dec: pd.DataFrame, uid: int, uid_of: Mapping[str, int]
              ) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """The rare decisions as rows of the suspicious-hands page (category S, tagged by how rarely the pool makes the
    decision) and, per (street, hand), the decision row its range box reads: the package's fields plus the four
    reference numbers, the pool's rate and the opponent's id."""
    at = {(h, int(s)): i for i, (h, s) in enumerate(zip(dec["hand_uid"], dec["seq"]))}
    rows, keys = [], {}
    for s in sel.to_dict("records"):
        gid, street = s["hand_id"], s["street"]
        hand, i = hands.get(gid), at.get((gid, int(s["action_index"])))
        if hand is None or i is None:
            continue
        d = dec.iloc[i].to_dict()
        seat = hand.seat_of(uid)
        ou = uid_of.get(str(s["opp_id"]))
        oseat = hand.seat_of(ou) if ou is not None else None
        board = hand.board[:BOARD_LEN[street]]
        if street == "preflop":
            mine, theirs = _class_of(s["hole_cards"]), _class_of(s["opp_hole_cards"])
        else:
            mine = CATEGORY_NAMES[evaluate(board + seat.hole)[0]] if seat and seat.hole and len(board) >= 3 else ""
            theirs = (CATEGORY_NAMES[evaluate(board + oseat.hole)[0]]
                      if oseat and oseat.hole and len(board) >= 3 else "")
        q, r, req, p = float(s["q"]), float(s["r"]), _num(s["req"]), float(s["p_pool"])
        bet = _num(d.get("bet_faced")) or 0.0
        pot_pre = max((_num(d.get("pot_before")) or 0.0) - bet, 1.0)
        price = f"price {req:.0%}" if req is not None else "nothing to call"
        rows.append({
            "hand_uid": gid, "start_utc": pd.to_datetime(hand.start_ts, unit="s"), "category": "S",
            "category_name": f"{s['rule_name']}: the pool does this "
                             + ("under 0.5%" if p < 0.005 else f"{p:.1%}") + " of the time at this strength",
            "decision_label": f"{s['tier']} (pool {p:.1%})", "street": street,
            "key_action": f"{s['action']} at q {q:.0%} vs the real hand, r {r:.0%} vs the line, {price}",
            "bet_to_pot": round(bet / pot_pre, 2) if bet else np.nan, "decision_time_s": d.get("think_time"),
            "snap_decision": False, "equity_at_decision": round(q, 3),
            "player_cards": to_pretty(seat.hole) if seat else "", "opponent_uid": str(s["opp_id"]),
            "opponent_cards": to_pretty(oseat.hole) if oseat else "", "board": to_pretty(hand.board),
            "player_hand": mine, "opponent_hand": theirs,
            "pot_bb": round(hand.contested_pot() / hand.blind, 1) if hand.blind else 0.0,
            "net_win": hand.settlement.get(uid, 0.0), "eq_vs_all_hands": _num(s["q_random"]), "eq_vs_range": r,
            "eq_vs_real": q, "pot_odds": req, "rule": s["rule"], "p_pool": p, "n_pool": int(s["n_pool"]),
            "score": float(s["score"]),
        })
        d.update(eq_vs_all_hands=_num(s["q_random"]), eq_vs_range=r, eq_vs_real=q, pot_odds=req, p_pool=p,
                 n_pool=int(s["n_pool"]), opponent_id=str(s["opp_id"]))
        keys[(street, gid)] = d
    return pd.DataFrame(rows), keys


def rare_note(d: Mapping, street: str, rc: Mapping, with_n: bool = True) -> dict:
    """Badge and text of a rare decision's range box: how often the pool makes the decision, then the range's
    verdict (equity against the predicted range set against the pot odds)."""
    bins = int(rc["pre_bins"] if street == "preflop" else rc["flop_bins"])
    where = (f"in his situation and {100 / bins:g}% band of equity against the opponent's line" if street == "preflop"
             else f"facing a bet in his {100 / bins:g}% band of equity against the opponent's line")
    v = verdict(d)
    return {"tier": "rare", "severity": 0.0,
            "comment": f"The pool makes this decision {d['p_pool']:.1%} of the time ("
                       + (f"{int(d['n_pool']):,} decisions " if with_n else "")
                       + f"{where}). {v[:1].upper() + v[1:]}."}


def rare_intro(street: str, rc: Mapping, field: str, top_n: int, n: int, reasons: list[str]) -> str:
    """The text above a tab's rare decisions: the rules, the pool and the order (or why there are none)."""
    pct = lambda v: f"{100 * float(v):g}%"   # noqa: E731
    if street == "preflop":
        rules = (f"folds facing a raise while behind the real hand (q under {pct(rc['pre_fold_q_max'])} and under the "
                 f"pot odds) where the pool folds less than {pct(rc['pre_fold_p_max'])} of the time; raises while "
                 f"ahead (q at least {pct(rc['pre_raise_q_min'])}) where the pool raises less than "
                 f"{pct(rc['pre_raise_p_max'])}; calls of a raise while ahead (q at least {pct(rc['pre_call_q_min'])}) "
                 f"where the pool calls less than {pct(rc['pre_call_p_max'])}. The pool's rate is read in his "
                 f"situation and {pct(1 / int(rc['pre_bins']))} band of equity against the opponent's line (r)")
    else:
        rules = (f"folds facing a bet with q under {pct(rc['flop_fold_q_max'])} where the pool folds less than "
                 f"{pct(rc['flop_fold_p_max'])} of the time; calls of a bet while ahead (q at least "
                 f"{pct(rc['flop_call_q_min'])}) where the pool continues less than {pct(rc['flop_call_p_max'])}. The "
                 f"pool's rate is read facing a bet in his {pct(1 / int(rc['flop_bins']))} band of r")
    text = (f"<b>Rare {street} decisions ({n}).</b> His decisions that the opponent's real cards made right and that "
            f"the pool rarely makes at the same strength against the same line: {html.escape(rules)}. Pool: every "
            f"other player of {html.escape(field)}, decisions with one live opponent; a band counts with at least "
            f"{int(rc['min_pool'])} decisions. Ranked by -ln(the pool's rate); one decision per hand, the top {top_n}. "
            f"Tags: Perfect = the pool does it less than {pct(rc['perfect_p'])} of the time, Good = less than "
            f"{pct(rc['good_p'])}, Average = otherwise. The box under each hand shows the opponent's predicted range "
            "before the decision and where his real hand sits in it.")
    if reasons:
        text += " " + " ".join(html.escape(x) for x in reasons)
    return text


# ------------------------------------------------------------------ report sections from the field

METRIC_ROWS = (
    # street, player_scores column, metric, what it measures ({...}: thresholds of the screen), value format, sample
    ("Preflop", "pf_z", "pf_z", "Folds when behind the opponent's real hand and raises when ahead, against the field "
     "at the same strength against his line (CMH inside situation x {pf_band} band of r)", "z", "pf"),
    ("Preflop", "pf_aggr_z", "Aggression z", "Raises when the opponent's real hand is worse", "z", "pf"),
    ("Preflop", "pf_fold_cmh_z", "Fold z", "Folds when the opponent's real hand is better", "z", "pf"),
    ("Preflop", "pf_fold_gap", "Fold gap (raw)", "Fold rate facing a raise when behind (q < {behind}) minus when ahead "
     "(q >= {ahead})", "pp", "pf"),
    ("Preflop", "pf_aggr_gap", "Aggression gap (raw)", "Raise rate when ahead minus when behind", "pp", "pf"),
    ("Preflop", "pf_rr_light_succ", "Light re-raise success", "Re-raises with a hand weak against the line "
     "(r < {light}) that the opponent folds to at once", "rate", "light"),
    ("Preflop", "pf_3bet_succ", "3-bet success", "The opponent folds to his 3-bet at once (needs no cards)", "rate",
     "3bet"),
    ("Preflop", "pf_4bet_succ", "4-bet success", "The opponent folds to his 4-bet at once (needs no cards)", "rate",
     "4bet"),
    ("Preflop", "pf_5bet_succ", "5-bet success", "The opponent folds to his 5-bet at once (needs no cards)", "rate",
     "5bet"),
    ("Flop", "fl_e_disc", "Flop e_disc", "Folds when behind and continues when ahead beyond the field at the same "
     "range strength (behind = q < {fl_dead})", "e", "fl"),
    ("Flop", "z_b_flop", "z_b_flop", "The same as a z-statistic (method b on the flop)", "z", "fl"),
    ("Turn + river", "e_disc", "e_disc", "The same on the turn and the river (dead = q < {tr_dead}); volume-free", "e",
     "tr"),
    ("Turn + river", "x_dead", "Excess folds while dead", "Fold rate when dead (q < {tr_dead}) minus the field's at the "
     "same range decile", "pps", "tr"),
    ("Turn + river", "x_win", "Excess continues while winning", "Continue rate when winning (q >= {tr_win}) minus the "
     "field's at the same range decile", "pps", "tr"),
    ("Turn + river", "z_b", "z_b", "z-test of the two excesses (method b, the article's core test)", "z", "tr"),
)
SAMPLE_RULES = {
    # sample -> (count column, [report.metrics] key of its minimum) pairs; the screen's flag when the counts are
    # missing; what the count counts (named in "Players ranked")
    "pf": ((("pf_nd", "pf_min_behind"), ("pf_nw", "pf_min_ahead")), "pf_eligible", ""),
    "fl": ((("fl_nd", "flop_min_dead"), ("fl_nw", "flop_min_win")), "fl_eligible", ""),
    "tr": ((("nd", "tr_min_dead"), ("nw", "tr_min_win")), "rb_eligible", ""),
    "light": ((("pf_rr_light_n", "light_reraises_min"),), None, "light re-raises"),
    "3bet": ((("pf_3bet_n", "reraises_min"),), None, "3-bets"),
    "4bet": ((("pf_4bet_n", "reraises_min"),), None, "4-bets"),
    "5bet": ((("pf_5bet_n", "reraises_min"),), None, "5-bets"),
}
GAPS = {"pf_fold_gap": ("pf_fold_behind", "pf_fold_ahead"), "pf_aggr_gap": ("pf_aggr_ahead", "pf_aggr_behind")}
METRIC_COLUMNS = ["Street", "Metric", "What it measures", "Players ranked", "Value", "Percentile", "Rank",
                  "Best other player"]


def _fmt_value(v: Any, kind: str, n: Any = None) -> str:
    if v is None or pd.isna(v):
        return "–"
    if kind == "z":
        return f"{v:.2f}"
    if kind == "e":
        return f"{v:.3f}"
    if kind == "pp":
        return f"{100 * v:.1f} pp"
    if kind == "pps":
        return f"{100 * v:+.1f} pp"
    return f"{100 * v:.1f}%" + (f" ({int(n)})" if n is not None and not pd.isna(n) else "")


def _what_values(scfg: Mapping) -> dict[str, str]:
    """The thresholds the metric texts quote, from config/step05_screening.toml."""
    pre, fl, mb = scfg.get("preflop", {}), scfg.get("flop", {}), scfg.get("method_b", {})
    pct = lambda v: f"{100 * float(v):g}%"   # noqa: E731
    return {"pf_band": pct(1 / int(pre.get("range_bins", 20))), "behind": pct(pre.get("behind_q", 0.40)),
            "ahead": pct(pre.get("ahead_q", 0.55)), "light": pct(pre.get("light_r", 0.45)),
            "fl_dead": pct(fl.get("dead_q", 0.20)), "tr_dead": pct(mb.get("dead_q", 0.02)),
            "tr_win": pct(mb.get("win_q", 0.5))}


def _midrank(vals: pd.Series, x: float) -> float:
    """Mid-rank percentile of x among vals (0-100): the share below it, ties counting half."""
    return 100.0 * (float((vals < x).sum()) + 0.5 * float((vals == x).sum())) / len(vals)


def metrics_table(scores: Optional[pd.DataFrame], ref: str, mc: Mapping, scfg: Mapping, with_n: bool = True
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """"Metrics by street" against F's ``player_scores`` (window "all"). For each row of :data:`METRIC_ROWS` the
    players with the row's minimum sample (``[report.metrics]``) are ranked: Players ranked = how many (him included
    when he qualifies), Percentile = the share of them he beats (mid-rank: ties count half), Rank = 1 + the number
    above him, Best other player = the best value among the others. Returns the table as shown, the same in raw
    numbers (metrics_summary.csv) and notes. ``with_n`` False drops "Players ranked" (the field's size)."""
    notes: list[str] = []
    if scores is None or scores.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame(), notes
    s = scores.drop_duplicates("player_id").set_index("player_id")
    if ref not in s.index:
        notes.append(f"Player {ref} has no row in the field's player_scores: the table shows the field only.")
    what = _what_values(scfg)
    shown, raw = [], []
    for street, col, label, text, kind, rule in METRIC_ROWS:
        counts, flag, unit = SAMPLE_RULES[rule]
        n_col = counts[0][0] if unit else None
        if col in s.columns:
            v = pd.to_numeric(s[col], errors="coerce").astype(float)
        elif col in GAPS and set(GAPS[col]) <= set(s.columns):
            v = (pd.to_numeric(s[GAPS[col][0]], errors="coerce") - pd.to_numeric(s[GAPS[col][1]], errors="coerce"))
        else:
            v = None
            notes.append(f"{label}: the field's player_scores has no column {col} (written by a newer "
                         "05_detection/screen.py).")
        if all(c in s.columns for c, _ in counts):
            ok = pd.Series(True, index=s.index)
            for c, key in counts:
                ok &= pd.to_numeric(s[c], errors="coerce").fillna(0) >= float(mc[key])
        elif flag and flag in s.columns:
            ok = s[flag].fillna(False).astype(bool)
        else:
            ok = pd.Series(True, index=s.index)
        vals = v[ok & v.notna()] if v is not None else pd.Series(dtype=float)
        mine = _num(v.get(ref)) if v is not None else None
        ranked = ref in vals.index
        pct = _midrank(vals, vals[ref]) if ranked else None
        rank = int((vals > vals[ref]).sum()) + 1 if ranked else None
        others = vals.drop(ref, errors="ignore")
        best_id = others.idxmax() if len(others) else None
        best = float(others[best_id]) if best_id is not None else None
        n_me = s.at[ref, n_col] if n_col and n_col in s.columns and ref in s.index else None
        n_best = s.at[best_id, n_col] if n_col and n_col in s.columns and best_id is not None else None
        minimum = int(mc[counts[0][1]])
        value = _fmt_value(mine, kind, n_me) + ("" if ranked or mine is None else " (sample below the minimum)")
        players = "–" if v is None else f"{len(vals):,}" + (f" (>= {minimum} {unit})" if unit else "")
        shown.append({"Street": street, "Metric": label, "What it measures": text.format(**what),
                      "Players ranked": players,
                      "Value": value, "Percentile": "–" if pct is None else f"{pct:.1f}",
                      "Rank": "–" if rank is None else f"#{rank}", "Best other player": _fmt_value(best, kind, n_best)})
        raw.append({"street": street, "metric": label, "column": col,
                    "players_ranked": None if v is None else len(vals), "value": mine,
                    "sample": _num(n_me), "ranked": ranked, "percentile": pct, "rank": rank, "best_other": best,
                    "best_other_sample": _num(n_best), "best_other_id": best_id})
    shown, raw = pd.DataFrame(shown, columns=METRIC_COLUMNS), pd.DataFrame(raw)
    if not with_n:   # a rank next to a percentile gives the field's size back: keep the percentile only
        shown = shown.drop(columns=["Players ranked", "Rank"])
        raw = raw.drop(columns=["players_ranked", "rank"], errors="ignore")
    return shown, raw, notes


PART_LABELS = {"wr": "win rate", "sd": "showdowns", "call": "calls", "bet": "bets", "allin": "all-ins",
               "pre": "preflop", "fold": "folds"}


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """The first of ``names`` that is a column of ``df``; NaN when none is."""
    return next((df[n] for n in names if n in df.columns), pd.Series(np.nan, index=df.index))


def victims_tables(fp: Optional[pd.DataFrame], ref: str, pcfg: pairs.PairConfig, back: Mapping[str, str],
                   with_n: bool = True, hide_unmapped: Optional[bool] = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """The opponents he beats beyond his own level (``sudetect.pairs.victims`` on his rows of F's pair table): the
    raw rows (pairs_victims.csv: ``opponent`` = this dataset's id when the id map knows him, else F's; ``label`` =
    "pair #k of N (top x%)"), the parts view, the pot-size view, and his most suspicious ranked pair (None when he has
    no ranked pair). ``with_n`` False: labels say "top x% of the field's pairs" and the raw rows drop ``n_ranked``."""
    empty = pd.DataFrame()
    if fp is None or fp.empty:
        return empty, empty, empty, None
    t = fp.assign(A=fp["A"].astype(str), B=fp["B"].astype(str))
    mine = t[t["A"] == ref]
    v = pairs.victims(mine, ref, pcfg) if not mine.empty else mine
    outside = "an opponent outside this dataset"   # a published report never prints another field's own ids
    hide = (not with_n) if hide_unmapped is None else hide_unmapped
    v = v.assign(opponent=[back.get(b, outside if hide else b) for b in v["B"]],
                 label=[pairs.pair_label(r, with_n) for r in v.to_dict("records")])
    v = v[["opponent", "label"] + [c for c in v.columns if c not in ("opponent", "label")]]
    best = None
    ranked = mine[mine["field_rank"].notna()].sort_values("field_rank")
    if len(ranked):
        r0 = ranked.iloc[0].to_dict()
        best = {**r0, "opponent": back.get(r0["B"], outside if hide else r0["B"]),
                "label": pairs.pair_label(r0, with_n)}
    if not with_n:   # rank and share together give the field's size back
        v = v.drop(columns=["n_ranked", "field_rank", "top_pct"], errors="ignore")
    if v.empty:
        return v, empty, empty, best
    z = lambda x: "–" if pd.isna(x) else f"{x:.2f}"   # noqa: E731
    sd_n = _col(v, "sd_n")
    parts = pd.DataFrame({
        "Opponent": v["opponent"], "Hands": v["hands"].map(lambda x: f"{int(x):,}"),
        "bb/100": v["bb100"].map(lambda x: "–" if pd.isna(x) else f"{x:+.1f}"),
        "Showdowns won": ["–" if pd.isna(w) or pd.isna(n) or not n else f"{w:.0%} of {int(n)}"
                          for w, n in zip(v["sd_won"], sd_n)],
        **{f"z {PART_LABELS[p]}": _col(v, f"{p}_z").map(z) for p in pairs.PARTS},
        "Score": v["score"].map(z), "Pair": v["label"]})

    def rate(x, n):
        return "–" if pd.isna(x) else f"{x:.0%} ({int(n)})"

    big = f"{pcfg.big_pot:g}bb"
    pot = pd.DataFrame({
        "Opponent": v["opponent"],
        f"W$SD, pots >= {big}": [rate(x, n) for x, n in zip(_col(v, "wsd_big"), _col(v, "wsd_big_n"))],
        "z W$SD": _col(v, "wsd_big_z").map(z),
        "W$SD growth": _col(v, "growth").map(lambda x: "–" if pd.isna(x) else f"{100 * x:+.0f} pp"),
        "z growth": _col(v, "growth_z").map(z),
        "River bets won": [rate(x, n) for x, n in zip(_col(v, "rbet_win"), _col(v, "rbet_win_n", "rbet_n"))],
        "z river bets": _col(v, "rbet_win_z", "rbet_z").map(z),
        "River calls won": [rate(x, n) for x, n in zip(_col(v, "rcall_win"), _col(v, "rcall_win_n", "rcall_n"))],
        "z river calls": _col(v, "rcall_win_z", "rcall_z").map(z),
        "His stack-offs": [rate(x, n) for x, n in zip(_col(v, "stackoff"), _col(v, "stackoff_n"))],
        "z stack-offs": _col(v, "stackoff_z").map(z),
        f"Net in pots >= {big}": ["–" if pd.isna(x) else f"{x:+.1f} bb ({int(n)})"
                                  for x, n in zip(_col(v, "net_big"), _col(v, "net_big_n"))],
        "z net": _col(v, "net_big_z").map(z)})
    return v, parts, pot, best


def victims_intro(pcfg: pairs.PairConfig, best: Optional[dict], field: str, with_n: bool = True) -> str:
    """How the pairs are scored and what makes a victim."""
    n = int(best["n_ranked"]) if best else 0
    cut = max(pcfg.victim_min_rank, math.ceil(pcfg.victim_rank_share * n)) if n else pcfg.victim_min_rank
    return (f"Every ordered pair of {field} (he against one opponent, in the hands the two of them played out to the "
            "end) is scored on its parts: his win rate and showdowns against that opponent, his calls, bets, all-ins "
            "and preflop calls and raises that were ahead of the opponent's real hand, and his folds that were behind "
            "it. Each part compares what he does against that opponent with what he does against everyone else and "
            "with what everyone else does against that opponent (z of the leave-pair-out residual, a hand counted once); "
            "the score is the sum of the parts' z over the square root of the sum of their correlations (so an honest "
            "pair scores around 0 with a spread of about 1), and pairs with at least two parts are ranked. "
            + (f"A victim: score at least {pcfg.victim_min_score:g} and a rank inside the top {cut:,}"
               + (f" of the {n:,} ranked pairs." if n else ".") if with_n else
               f"A victim: score at least {pcfg.victim_min_score:g} and a rank inside the top "
               f"{100 * pcfg.victim_rank_share:g}% of the ranked pairs (at least the top {pcfg.victim_min_rank}).")
            + f" Pot-size columns: the same pair seen only through money and showdowns (pots of {pcfg.big_pot:g}bb or "
            "more); the rates with a z against the pool of all pairs (stack-offs turned, so fewer gives a higher z), "
            "the growth and the net with a z against zero.")


OBS_ROWS = (
    # column, metric, what it measures ({...}: the observer settings), value format, count column
    ("obs_bb100", "bb/100", "Net per 100 hands", "bb", "obs_hands"),
    ("obs_big_net100", "Big-pot net per 100 hands", "Net in pots of {e3}bb or more, per 100 hands", "bb", "obs_hands"),
    ("obs_wtsd", "WTSD", "Went to showdown, of the hands that saw the flop (lower = more suspicious)", "rate",
     "obs_wtsd_n"),
    ("obs_wtsd_srp", "WTSD, single-raised pots", "The same, pot at the flop under {f1}bb", "rate", "obs_wtsd_srp_n"),
    ("obs_wtsd_3bp", "WTSD, 3-bet pots", "The same, pot at the flop {f1}-{f2}bb", "rate", "obs_wtsd_3bp_n"),
    ("obs_wtsd_4bp", "WTSD, 4-bet pots", "The same, pot at the flop {f2}bb or more", "rate", "obs_wtsd_4bp_n"),
    ("obs_wsd", "W$SD", "Showdowns won with two players at the showdown (a split counts 1/2)", "rate", "obs_wsd_n"),
    ("obs_wsd_b1", "W$SD, small pots", "The same, final pot under {e1}bb", "rate", "obs_wsd_b1_n"),
    ("obs_wsd_b2", "W$SD, medium pots", "The same, final pot {e1}-{e2}bb", "rate", "obs_wsd_b2_n"),
    ("obs_wsd_b3", "W$SD, large pots", "The same, final pot {e2}-{e3}bb", "rate", "obs_wsd_b3_n"),
    ("obs_wsd_b4", "W$SD, the biggest pots", "The same, final pot {e3}bb or more", "rate", "obs_wsd_b4_n"),
    ("obs_wsd_growth", "W$SD growth", "W$SD in the biggest pots minus in the small ones", "pps", "obs_growth_n"),
    ("obs_rbet_win", "River bets won", "He bet or raised the river, the pot was over {big}bb and the hand reached a "
     "showdown: share won", "rate", "obs_rbet_win_n"),
    ("obs_rcall_win", "River calls won", "He called a river bet heads-up, the pot after the call over {big}bb: share "
     "won", "rate", "obs_rcall_win_n"),
    ("obs_rcall_freq", "River call frequency", "Those calls over the river bets he faced that way", "rate",
     "obs_rcall_freq_n"),
    ("obs_stackoff", "River stack-offs", "He saw the river and over {big}bb went in on it: share of those hands in "
     "which he lost {so} or more of his stack (lower = more suspicious)", "rate", "obs_stackoff_n"),
)
OBS_COLUMNS = ["Metric", "What it measures", "Value", "Cases", "Suspicion percentile", "Rank", "Players ranked"]


def observer_table(obs: Optional[pd.DataFrame], ref: str, oc: Mapping, with_n: bool = True
                   ) -> tuple[pd.DataFrame, str, list[str]]:
    """The observer metrics (``sudetect/potsize.py``: money and showdowns, no hidden cards) of his row with their
    field ranks: the percentile is turned so that higher is always more suspicious (``<metric>_pct``); a rank needs
    the ranks' minimum hands and cases. Returns the table, a line with his net by pot size, and notes."""
    notes: list[str] = []
    if obs is None or obs.empty:
        return pd.DataFrame(columns=OBS_COLUMNS), "", notes
    if ref not in obs.index:
        notes.append(f"Player {ref} has no observer metrics in the field.")
        return pd.DataFrame(columns=OBS_COLUMNS), "", notes
    row = obs.loc[ref]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    e1, e2, e3 = (f"{float(x):g}" for x in oc["pot_edges"])
    f1, f2 = (f"{float(x):g}" for x in oc["flop_edges"])
    fill = {"e1": e1, "e2": e2, "e3": e3, "f1": f1, "f2": f2, "big": f"{float(oc['big_pot']):g}",
            "so": f"{float(oc['stackoff_share']):.0%}"}
    rows = []
    for col, label, text, kind, ncol in OBS_ROWS:
        v, n = _num(row.get(col)), _num(row.get(ncol))
        pct, rank, of = _num(row.get(f"{col}_pct")), _num(row.get(f"{col}_rank")), _num(row.get(f"{col}_of"))
        if v is None:
            value = "–"
        elif kind == "bb":
            value = f"{v:+.1f}"
        elif kind == "pps":
            value = f"{100 * v:+.1f} pp"
        else:
            value = f"{100 * v:.1f}%"
        rows.append({"Metric": label, "What it measures": text.format(**fill), "Value": value,
                     "Cases": "–" if n is None else f"{int(n):,}",
                     "Suspicion percentile": "–" if pct is None else f"{pct:.1f}",
                     "Rank": "–" if rank is None else f"#{int(rank)}", "Players ranked": "–" if of is None else f"{int(of):,}"})
    nets = [_num(row.get(f"obs_net_b{b}")) for b in range(1, 5)]
    line = ("Net by final pot: " + ", ".join(
        f"{lab} {'–' if x is None else f'{x:+,.1f} bb'}" for lab, x in
        zip((f"under {e1}bb", f"{e1}-{e2}bb", f"{e2}-{e3}bb", f"{e3}bb or more"), nets)) + ".")
    notes.append(f"Ranks among the field's players with at least {int(oc['min_hands']):,} hands and, for a rate, "
                 f"{int(oc['min_cases'])} cases (config/step05_screening.toml [step05.observer]).")
    shown = pd.DataFrame(rows, columns=OBS_COLUMNS)
    return (shown if with_n else shown.drop(columns=["Players ranked", "Rank"])), line, notes


SCORECARD = (("ofr_pair", "Oracle-fold ratio: postflop, pair or better"), ("ofr_turn", "Oracle-fold ratio: turn"),
             ("ofr_river", "Oracle-fold ratio: river"), ("fold_to_value", "Fold to value bets"),
             ("fold_to_bluff", "Fold to bluffs"), ("disc_gap", "Discrimination gap"), ("bluff_index", "Bluff success"))


def _signal_values(s: pd.DataFrame, name: str, min_n: int) -> pd.Series:
    """One article signal for every player of F, NaN below ``min_n`` qualifying decisions."""
    if name == "disc_gap":
        if not {"disc_gap", "fold_to_value__n", "fold_to_bluff__n"} <= set(s.columns):
            return pd.to_numeric(s["disc_gap"], errors="coerce") if "disc_gap" in s.columns else pd.Series(
                np.nan, index=s.index)
        ok = (s["fold_to_value__n"] >= min_n) & (s["fold_to_bluff__n"] >= min_n)
        return pd.to_numeric(s["disc_gap"], errors="coerce").where(ok)
    if name not in s.columns:
        return pd.Series(np.nan, index=s.index)
    v = pd.to_numeric(s[name], errors="coerce")
    return v.where(s[f"{name}__n"] >= min_n) if f"{name}__n" in s.columns else v


def _rest_value(orow: Mapping, prow: Optional[Mapping], name: str) -> Optional[float]:
    """A signal of the opponent from his counts minus those of his decisions against the suspect (``prow``)."""
    def kn(nm: str) -> tuple[Optional[float], Optional[float]]:
        k, n = _num(orow.get(f"{nm}__k")), _num(orow.get(f"{nm}__n"))
        if k is None or n is None:
            return None, None
        if prow is not None:
            k, n = k - (_num(prow.get(f"{nm}__k")) or 0.0), n - (_num(prow.get(f"{nm}__n")) or 0.0)
        return k, n

    if name == "disc_gap":
        (kv, nv), (kb, nb) = kn("fold_to_value"), kn("fold_to_bluff")
        return kv / nv - kb / nb if nv and nb else None
    k, n = kn(name)
    if n is None:
        return _num(orow.get(name)) if prow is None else None
    return k / n if n > 0 else None


def _ordinal(pct: Optional[float]) -> str:
    if pct is None:
        return "–"
    n = int(round(pct))
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def scorecard_table(scores: Optional[pd.DataFrame], pair_rows: Optional[pd.DataFrame], ref: str,
                    opp_ref: Optional[str], sc: Mapping, opp_name: Optional[str] = None
                    ) -> tuple[pd.DataFrame, list[str]]:
    """The article's scorecard against F: for each of its seven signals his value, his percentile and the median among
    F's players with at least ``min_n`` qualifying decisions, the mean of F's ``top_winners`` biggest winners (net bb,
    him left out), and the main opponent's value in his games against everyone else (his counts in F minus those of
    his decisions against the suspect, from F's ``pair_scores``)."""
    notes: list[str] = []
    cols = ["Signal", "Suspect", "Field percentile", "Field median", f"Top {int(sc['top_winners'])} winners (mean)",
            "Main opponent vs the rest"]
    if scores is None or scores.empty:
        return pd.DataFrame(columns=cols), notes
    s = scores.drop_duplicates("player_id").set_index("player_id")
    min_n = int(sc["min_n"])
    winners: list = []
    if {"bb100", "hands"} <= set(s.columns):
        net = (pd.to_numeric(s["bb100"], errors="coerce") * pd.to_numeric(s["hands"], errors="coerce") / 100)
        winners = list(net.drop(ref, errors="ignore").dropna().sort_values(ascending=False).head(
            int(sc["top_winners"])).index)
    orow = s.loc[opp_ref].to_dict() if opp_ref is not None and opp_ref in s.index else None
    prow = None
    if orow is not None and pair_rows is not None and not pair_rows.empty:
        m = pair_rows[(pair_rows["player_id"].astype(str) == opp_ref) & (pair_rows["opponent_id"].astype(str) == ref)]
        prow = m.iloc[0].to_dict() if len(m) else None
    rows = []
    for name, label in SCORECARD:
        v = _signal_values(s, name, min_n).dropna()
        mine = _num(s.at[ref, name]) if ref in s.index and name in s.columns else None
        pct = _midrank(v, v[ref]) if ref in v.index else None
        top = v.reindex(winners).dropna()
        opp = _rest_value(orow, prow, name) if orow is not None else None

        def fmt(x: Optional[float]) -> str:
            if x is None or pd.isna(x):
                return "–"
            return f"{100 * x:.1f} pts" if name == "disc_gap" else f"{100 * x:.1f}%"

        rows.append({"Signal": label, "Suspect": fmt(mine), "Field percentile": _ordinal(pct),
                     "Field median": fmt(float(v.median()) if len(v) else None),
                     cols[4]: fmt(float(top.mean()) if len(top) else None), cols[5]: fmt(opp)})
    notes.append(f"Percentile and median among the field's players with at least {min_n} qualifying decisions on "
                 "the signal; low is suspicious on folds to bluffs.")
    name = opp_name or opp_ref
    if opp_ref is None:
        notes.append("No main opponent: he never played a hand down to one live opponent.")
    elif orow is None:
        notes.append(f"The main opponent ({name}) has no row in the field's player_scores.")
    elif prow is None:
        notes.append(f"Main opponent {name}: all his games (the field's pair_scores has no row of his decisions "
                     "against the suspect to take out).")
    else:
        notes.append(f"Main opponent {name}: his games against everyone but the suspect.")
    return pd.DataFrame(rows, columns=cols), notes


def street_tails(srow: Optional[Mapping], scfg: Mapping) -> tuple[Optional[str], dict]:
    """The street-tails rule on his row of F's ``player_scores``: the verdict line and the numbers for summary.json;
    (None, {}) when the screen that wrote the table predates the rule."""
    if srow is None or "street_hits" not in srow:
        return None, {}
    st = (scfg.get("escalation") or {}).get("street_tails") or {}
    esc_min, rev_min = int(st.get("escalate_min_hits", 2)), int(st.get("review_min_hits", 1))

    def hit(k: str) -> bool:
        x = srow.get(k)
        return x is not None and not pd.isna(x) and bool(x)

    def val(k: str, d: int = 2) -> str:
        x = _num(srow.get(k))
        return "–" if x is None else f"{x:.{d}f}"

    n = int(_num(srow.get("street_hits")) or 0)
    esc_col = "esc_street_tails" if "esc_street_tails" in srow else next(
        (k for k in srow if str(k).startswith("esc_street_tails")), None)
    esc = hit(esc_col) if esc_col else n >= esc_min
    review = hit("review_street_tails") if "review_street_tails" in srow else n >= rev_min
    word = "**ESCALATE for manual review**" if esc else ("**REVIEW**" if review else "below the rule")
    parts = [f"preflop {'HIT' if hit('pf_hit') else 'no'} (pf_zr {val('pf_zr')})",
             f"flop {'HIT' if hit('fl_hit') else 'no'} (z_b_flop {val('z_b_flop')}, flop e_disc {val('fl_e_disc', 3)})",
             f"turn + river {'HIT' if hit('tr_hit') else 'no'} (z_b {val('z_b')}, e_disc {val('e_disc', 3)})"]
    text = (f"**Street-tails rule** (the top tail on several streets at once; escalate at {esc_min} or more of the "
            f"three streets, review at {rev_min}): " + "; ".join(parts) + f". {n} of 3 streets: {word}.")
    return text, {"street_hits": n, "escalate": bool(esc), "review": bool(review),
                  **{k: hit(k) for k in ("pf_hit", "fl_hit", "tr_hit")}}


def pairs_box_html(field: str, intro: str, parts: pd.DataFrame, pot: pd.DataFrame, lines: list[str]) -> str:
    """The pairs summary at the top of suspicious_hands.html: the victims table (parts, then pot sizes)."""
    def table(df: pd.DataFrame) -> str:
        return '<div style="overflow-x:auto">' + df.to_html(index=False, classes="sum", border=0) + "</div>"

    body = [f"<b>Opponents he beats beyond his own level ({html.escape(field)}).</b> {html.escape(intro)}"]
    if not parts.empty:
        body += [table(parts), table(pot)]
    body += [f'<div class="muted">{html.escape(x)}</div>' for x in lines]
    return '<div class="box">' + "".join(body) + "</div>"


# ------------------------------------------------------------------ the second adapter run, the main opponent


def renumber(run: Mapping[str, Any], uid_of: Mapping[str, int]) -> tuple[dict, pd.DataFrame, dict[str, int]]:
    """The hands and decision rows of a second adapter run, in the player numbers of the first run.

    The adapter numbers the players of the hands it reads, so two runs over different hand sets give a non-numeric
    id two numbers; a player the first run never met gets a number after its largest (his own when the ids are
    numeric and it is free). Returns the hands, the decision rows and the merged map (id -> number)."""
    merged = dict(uid_of)
    used = set(merged.values())
    nxt = max(used, default=0) + 1
    old_to_new: dict[int, int] = {}
    for pid, u in sorted(run["uid_of"].items(), key=lambda kv: kv[1]):
        if pid not in merged:
            if str(pid).isdigit() and int(pid) not in used:
                merged[pid] = int(pid)
            else:
                while nxt in used:
                    nxt += 1
                merged[pid] = nxt
            used.add(merged[pid])
        old_to_new[u] = merged[pid]
    if all(k == v for k, v in old_to_new.items()):
        return dict(run["hands"]), run["decisions"], merged

    def m(u):
        return old_to_new.get(u, u)

    hands = {}
    for gid, h in run["hands"].items():
        hands[gid] = dataclasses.replace(
            h, seats={k: dataclasses.replace(s, uid=m(s.uid)) for k, s in h.seats.items()},
            actions={st: [dataclasses.replace(a, uid=m(a.uid)) for a in acts] for st, acts in h.actions.items()},
            settlement={m(u): w for u, w in h.settlement.items()}, committed={m(u): c for u, c in h.committed.items()})
    dec = run["decisions"].copy()
    for c in ("uid", "bettor_uid"):
        if c in dec.columns:
            dec[c] = [u if u is None or pd.isna(u) else m(int(u)) for u in dec[c]]
    return hands, dec, merged


def main_opponent(con: Any, player: str) -> Optional[str]:
    """The opponent he met in the most hands with one live opponent left."""
    t = warehouse.query_df(con, """
        SELECT d.live_opponents[1].player_id AS opp, count(DISTINCT d.hand_id) AS n
        FROM decisions d
        WHERE d.player_id = :p AND d.n_live = 2
        GROUP BY 1 ORDER BY n DESC, opp LIMIT 1""", p=str(player))
    return None if t.empty or t["opp"].iloc[0] is None else str(t["opp"].iloc[0])


# ------------------------------------------------------------------ main


def analyse(dataset: str, player: str, opponent: Optional[str] = None, scope: str | bool = "flop_hu",
            times_path: Optional[str] = None, range_config: Optional[str] = None, range_hands: int = 3,
            out_dir: Optional[str] = None, log=print, field_dataset: Optional[str] = None,
            id_map: Optional[str | Path | Mapping[str, str]] = None, top_hands: Optional[int] = None,
            report_config: Optional[str] = None) -> dict:
    ad = adapter()
    rc = load_report_config(report_config)
    scfg = screen_config()
    with_n = bool(rc.get("show_field_size", True))   # false: a report published from a private field
    top_n = int(rc["top_hands"] if top_hands is None else top_hands)
    ids = {str(k): str(v) for k, v in id_map.items()} if isinstance(id_map, Mapping) else read_id_map(id_map)
    back: dict[str, str] = {}
    for case_id, ref_id in ids.items():              # F's id -> this dataset's; the first one listed wins
        back.setdefault(ref_id, case_id)
    fname = field_dataset or dataset
    if fname != dataset and not warehouse.db_path(fname).exists():
        raise SystemExit(f"--field-dataset {fname}: no warehouse at {warehouse.db_path(fname)}")
    fshown = fname if with_n else "the reference field"   # the name pages and summary.json may print
    if ids and str(player) not in ids:
        raise SystemExit(f"--id-map has no row for player {player}: add it (case_id,ref_id) or drop --id-map")
    ref = ids.get(str(player), str(player))
    times_file = Path(times_path) if times_path else warehouse.data_dir() / dataset / "input" / "decision_times.csv"
    times = ad.read_times(times_file)
    con = warehouse.connect(dataset, read_only=True)
    try:
        result = ad.analyse_player(con, player, opponent, scope, times=times)
        if not result["meta"].get("hands_analysed"):
            raise SystemExit(f"{dataset}: player {player} has no hands in scope")
        uid = result["player"]
        fcon = con if fname == dataset else warehouse.connect(fname, read_only=True)
        try:
            scfg = screen_config(fcon) or scfg
            field = load_field(fcon, fname, ref, rc, scfg)
        finally:
            if fcon is not con:
                fcon.close()
        multiway = pd.DataFrame()
        uid_of, all_hands, other_dec = dict(result["uid_of"]), dict(result["hands"]), pd.DataFrame()
        if ad.scope_name(scope) != "all":          # the package on every hand: the flags the scope leaves out
            every = ad.analyse_player(con, player, opponent, "all", times=times)
            # the second run numbers the players of its own hands: bring its hands into the first run's numbers
            ev_hands, ev_dec, uid_of = renumber(every, uid_of)
            if not every["suspicious_hands"].empty:
                hu = _hand_ids(result["suspicious_hands"])
                multiway = every["suspicious_hands"][~every["suspicious_hands"]["hand_uid"].isin(hu)].reset_index(drop=True)
            all_hands = {**ev_hands, **result["hands"]}
            other_dec = ev_dec[~ev_dec["hand_uid"].isin(set(result["hands"]))]
        result["uid_of"] = uid_of
        # the adapter's own numbering of the players in the analysed hands: renumbering every player of the dataset
        # gives other numbers when player ids are not numeric (hashed ids, simulator names)
        name_of = {u: p for p, u in uid_of.items()}
        dec = result["decisions"].drop(columns=["eq_vs_real"], errors="ignore")
        dec = dec.merge(range_frame(con), on=["hand_uid", "seq"], how="left")
        result["decisions"] = dec

        sus = result["suspicious_hands"]
        keys = key_decisions(result)
        for c in RANGE_COLUMNS[:-1]:
            sus[c] = [_num(keys.get(g, {}).get(c)) for g in sus["hand_uid"]] if not sus.empty else []
        sus["range_verdict"] = [verdict(keys[g]) if g in keys else "" for g in sus["hand_uid"]] if not sus.empty else []
        extra, extra_keys = range_only_rows(result, name_of, range_hands)
        if not extra.empty:
            for c in RANGE_COLUMNS[:-1]:
                extra[c] = [_num(extra_keys[g].get(c)) for g in extra["hand_uid"]]
            extra["range_verdict"] = [verdict(extra_keys[g]) for g in extra["hand_uid"]]
        keys.update(extra_keys)

        # rare decisions: preflop at any table of the scope's kind, flop in the analysed hands not already listed
        cands = rare_candidates(con, player)
        pre_hands = ({g for g, h in all_hands.items() if h.n_players == 2} if ad.scope_name(scope) == "hu_tables"
                     else set(all_hands))
        listed = _hand_ids(sus) | _hand_ids(extra) | _hand_ids(multiway)
        keep = ((cands["street"].eq("preflop") & cands["hand_id"].isin(pre_hands))
                | (cands["street"].eq("flop") & cands["hand_id"].isin(set(result["hands"]))
                   & ~cands["hand_id"].isin(listed)))
        sel = select_rare(cands[keep], field["pools"], rc["rare"], top_n)
        all_dec = pd.concat([dec, other_dec], ignore_index=True) if not other_dec.empty else dec
        rare, rare_keys = rare_rows(sel, all_hands, all_dec, uid, uid_of)
        reasons: dict[str, list[str]] = {"preflop": [], "flop": []}
        for street, table, pool in (("preflop", "action_range_equity_pre", "pre"), ("flop", "action_range_equity", "flop")):
            step = "04_predicted_range/preflop.py" if street == "preflop" else "04_predicted_range/run.py"
            if not warehouse.table_exists(con, "action_equity"):
                reasons[street].append(f"{dataset} has no action_equity: run 02_equity/run.py --dataset {dataset}.")
            if not warehouse.table_exists(con, table):
                reasons[street].append(f"{dataset} has no {table}: run {step} --dataset {dataset}.")
            if field["pools"][pool] is None and (fname != dataset or warehouse.table_exists(con, table)):
                reasons[street].append(f"No pool: {fshown} has no {table} (run {step} --dataset {fshown}).")

        boxes: dict[str, dict] = {}
        rare_boxes: dict[str, dict[str, dict]] = {"preflop": {}, "flop": {}}
        tables = cfg = examples = None
        if warehouse.table_exists(con, "rng_lik_post"):
            run04 = _module("step04_run", ROOT / "04_predicted_range" / "run.py")
            examples = _module("step04_examples", ROOT / "04_predicted_range" / "examples.py")
            cfg = run04.load_config(range_config or DEFAULT_RANGE_CONFIG)
            tables, _ = run04.load_tables(con)
        if tables is not None and warehouse.table_exists(con, "action_range_equity"):
            for gid, row in keys.items():
                hand = result["hands"][gid]
                live = [u for u in _live_uids(hand, row) if u != uid]
                if len(live) != 1:              # a multiway spot: the range model scores heads-up pots only
                    continue
                boxes[gid] = range_box(con, cfg, tables, examples, gid, row, result["player_id"], name_of[live[0]])
        failed = 0
        for (street, gid), d in rare_keys.items():
            box = None
            if tables is not None:
                make = preflop_box if street == "preflop" else range_box
                try:
                    box = make(con, cfg, tables, examples, gid, d, result["player_id"], d["opponent_id"])
                except Exception:  # noqa: BLE001 - a hand the range model cannot read keeps its row, without a range
                    failed += 1
            if box is None:
                box = {c: _num(d.get(c)) for c in RANGE_COLUMNS[:-1]}
                box.update(range_verdict=verdict(d), opponent=d["opponent_id"])
            rare_boxes[street][gid] = {**box, **rare_note(d, street, rc["rare"], with_n)}
        if failed:
            log(f"{failed} rare decisions without a range box (the range model could not read the hand)")
        main_opp = str(opponent) if opponent else main_opponent(con, player)
    finally:
        con.close()

    # the field's sections of report.md and report.html
    srow = None
    if field["scores"] is not None and ref in set(field["scores"]["player_id"]):
        srow = field["scores"].drop_duplicates("player_id").set_index("player_id").loc[ref].to_dict()
    metrics_shown, metrics_raw, metric_notes = metrics_table(field["scores"], ref, rc["metrics"], scfg, with_n)
    vraw, vparts, vpot, best = victims_tables(field["pairs"], ref, field["pair_cfg"], back, with_n,
                                              hide_unmapped=(not with_n) and fname != dataset)
    obs_shown, obs_line, obs_notes = observer_table(field["obs"], ref, field["obs_cfg"], with_n)
    opp_ref = (ids.get(main_opp) if ids else main_opp) if main_opp else None   # never a guess across id spaces
    card, card_notes = scorecard_table(field["scores"], field["pair_scores"], ref, opp_ref, rc["scorecard"], main_opp)
    tails_text, tails = street_tails(srow, scfg)
    if ref != str(player):   # every note names him by this dataset's id, never by the field's
        metric_notes = [x.replace(ref, str(player)) for x in metric_notes]
        obs_notes = [x.replace(ref, str(player)) for x in obs_notes]
        card_notes = [x.replace(ref, str(player)) for x in card_notes]
        field["notes"] = {k: [x.replace(ref, str(player)) for x in v] for k, v in field["notes"].items()}
    if fshown != fname:
        field["notes"] = {k: [x.replace(fname, fshown) for x in v] for k, v in field["notes"].items()}
    labels = dict(zip(vraw["opponent"].astype(str), vraw["label"])) if not vraw.empty else {}
    notes = field["notes"]
    vlist = ", ".join(f"{o} ({lab})" for o, lab in labels.items())
    pair_line = (f"**Pairs**: he beats {len(labels)} opponent{'s' if len(labels) != 1 else ''} beyond his own level "
                 f"and beyond what the rest of {fshown} does against them: {vlist}." if labels else
                 f"**Pairs**: no opponent passes the victim rule; his most suspicious pair: {best['opponent']} "
                 f"({best['label']}, score {best['score']:.2f})." if best else
                 "**Pairs**: none of his pairs is ranked." if field["pairs"] is not None else
                 "**Pairs**: not available (see below).")
    intro = victims_intro(field["pair_cfg"], best, fshown, with_n)
    sections = [
        {"title": "Verdict",
         "blocks": [f"**Article rule**: {_escalation_text(result['escalation'])}",
                    tails_text or "**Street-tails rule**: not in the field's player_scores (it comes from "
                                  "05_detection/screen.py).", pair_line],
         "notes": notes["scores"]},
        {"title": "Metrics by street",
         "blocks": [f"His value on each street's tests against the players of {fshown} with the minimum sample "
                    "(config/step05_report.toml [report.metrics]). Percentile = the share of them he beats (ties "
                    "count half, himself included); " + ("Rank = his place among them; " if with_n else "")
                    + "Best other player = the best value of the others.", metrics_shown],
         "notes": notes["scores"] + metric_notes},
        {"title": "Opponents he beat",
         "blocks": [intro] + ([vparts, f"The same pairs by pot size (pots of {field['pair_cfg'].big_pot:g}bb or "
                                       "more):", vpot] if not vparts.empty else [pair_line]),
         "notes": notes["pairs"]},
        {"title": "Article scorecard",
         "blocks": [f"The article's seven signals (turn and river decisions, equity against the opponent's real "
                    f"hand) against {fshown}.", card],
         "notes": notes["scores"] + card_notes},
        {"title": "Observer metrics",
         "blocks": ["Money and showdowns only: what someone who reads the hand histories sees without the hidden "
                    "cards. The percentile is turned so that higher is always more suspicious.", obs_shown]
                   + ([obs_line] if obs_line else []),
         "notes": notes["obs"] + obs_notes},
    ]
    for s in sections:                              # a missing table leaves its note, not an empty table
        s["blocks"] = [b for b in s["blocks"] if not (isinstance(b, pd.DataFrame) and b.empty)]

    out = Path(out_dir) if out_dir else warehouse.data_dir() / dataset / "report"
    write_report(result, out, sections=sections)    # the package's report, his rows only, and the field's sections
    full = _stack([sus, extra]) if not extra.empty else sus
    result["suspicious_hands"] = full
    full.to_csv(out / "suspicious_hands.csv", index=False)
    dec.to_csv(out / "decisions.csv", index=False)
    # the ranked page: every flagged hand stays; the Equity review orders and colours them, it removes none
    ranked = full.assign(_tier="norange", _sev=0.0, _comment="", _pos=range(len(full)))
    if not full.empty and "eq_vs_range" in full.columns:
        fold = full["category"].isin(["A", "D"]) | full["key_action"].astype(str).str.contains("folded")
        bluff = full["category"].eq("C")
        gap = (full["eq_vs_range"] - full["pot_odds"]).where(fold, full["pot_odds"] - full["eq_vs_range"])
        # severity: how far the betting pointed the other way, counted only when the real cards made the decision right
        real_ok = ((fold & (full["eq_vs_real"] < full["pot_odds"])) | (~fold & (full["eq_vs_real"] >= full["pot_odds"])))
        sev = gap.clip(lower=0).where(real_ok & ~bluff, 0.0).fillna(0.0)
        keyrows = key_decisions(result)
        notes_ranked = []
        for i, row in full.iterrows():
            gid, kd = row["hand_uid"], keyrows.get(row["hand_uid"])
            note = equity_note(row, bool(fold[i]), float(sev[i]),
                               bool(kd is not None and faced_raise(result["hands"][gid], kd)))
            notes_ranked.append(note)
            boxes.setdefault(gid, {}).update(note)
        ranked = ranked.assign(_tier=[n["tier"] for n in notes_ranked], _sev=[n["severity"] for n in notes_ranked],
                               _comment=[n["comment"] for n in notes_ranked])
    if not multiway.empty:
        for g in multiway["hand_uid"]:
            boxes[g] = {"tier": "norange", "severity": 0.0, "comment": MULTIWAY_NOTE}
        ranked = _stack([ranked, multiway.assign(_tier="norange", _sev=0.0, _comment=MULTIWAY_NOTE,
                                                 _pos=range(len(full), len(full) + len(multiway)))])
    ranked["_t"] = ranked["_tier"].map(TIER_ORDER.index)
    ranked = ranked.sort_values(["_t", "_sev", "_pos"], ascending=[True, False, True]).reset_index(drop=True)
    (out / "suspicious_hands_all.html").write_text(suspicious_hands_html(result, ranges=boxes), encoding="utf-8")
    # suspicious_hands.html: tab Postflop = the ranked page, then the rare flop decisions; tab Preflop = the rare
    # preflop decisions; the pairs summary on top and "pair #k of N" next to every victim's id
    shown = ranked.drop(columns=[c for c in ranked.columns if c.startswith("_")])
    rare_fl = rare[rare["street"] == "flop"].reset_index(drop=True) if not rare.empty else rare
    rare_pf = rare[rare["street"] == "preflop"].reset_index(drop=True) if not rare.empty else rare
    post_rows = _stack([shown, rare_fl]) if not rare_fl.empty else shown
    pair_lines = notes["pairs"] + ([] if labels else [pair_line.replace("**", "")])
    top = pairs_box_html(fshown, intro, vparts, vpot, pair_lines)
    if not ranked.empty:
        top += overview_html(ranked, len(multiway))
    top += field_note(dataset, field_dataset, player, with_n)   # his folds, whether or not a hand is listed
    first = (f"First the {len(shown)} hands of the package's rules A-D, the range rule R and the multiway flags, in "
             "the order of the Equity review; then " if len(shown) else "No suspicious hands found. ")
    post_intro = (f"<b>Postflop.</b> {first}"
                  + rare_intro("flop", rc["rare"], fshown, top_n, len(rare_fl), reasons["flop"])
                  + (" The flop decisions of hands already listed above are not repeated." if len(shown) else ""))
    page = tabbed_hands_html({**result, "hands": all_hands, "suspicious_hands": shown}, [
        {"key": "postflop", "label": "Postflop", "rows": post_rows, "boxes": {**boxes, **rare_boxes["flop"]},
         "intro": post_intro, "anchor": "h", "title": "NO."},
        {"key": "preflop", "label": "Preflop", "rows": rare_pf, "boxes": rare_boxes["preflop"],
         "intro": rare_intro("preflop", rc["rare"], fshown, top_n, len(rare_pf), reasons["preflop"]),
         "anchor": "pf", "title": "PREFLOP NO."}],
        counted=shown, outside_scope=len(multiway), labels=labels, top=top)
    (out / "suspicious_hands.html").write_text(page, encoding="utf-8")
    cols = ["hand_uid", "category", "decision_label", "street", "key_action", "bet_to_pot", "pot_bb", "player_hand",
            "opponent_hand", "eq_vs_all_hands", "eq_vs_range", "eq_vs_real", "pot_odds"]
    ranked.assign(rank=range(1, len(ranked) + 1), equity_review=ranked["_tier"].map(TIER_NAMES),
                  severity=ranked["_sev"].round(4), comment=ranked["_comment"]).reindex(
        columns=["rank"] + cols + ["equity_review", "severity", "comment"]).to_csv(out / "suspicious_hands_ranked.csv", index=False)
    shown_rare = {(s, g) for s, g in rare_keys}
    if not with_n:   # a report published from a private field carries none of the field's ids or counts either
        vraw = vraw.drop(columns=["A", "B"], errors="ignore")
        metrics_raw = metrics_raw.drop(columns=["best_other_id"], errors="ignore")
        sel = sel.drop(columns=["n_pool"], errors="ignore")
    sel[np.array([(s, g) in shown_rare for s, g in zip(sel["street"], sel["hand_id"])], dtype=bool)].to_csv(
        out / "rare_decisions.csv", index=False)
    vraw.to_csv(out / "pairs_victims.csv", index=False)
    metrics_raw.to_csv(out / "metrics_summary.csv", index=False)
    tiers = ", ".join(f"{TIER_NAMES[t]} {int((ranked['_tier'] == t).sum())}" for t in TIER_ORDER
                      if (ranked["_tier"] == t).any())
    log(f"ranked page: {tiers or 'nothing ranked'} ({len(ranked)} hands)")
    log(f"rare decisions: preflop {len(rare_pf)}, flop {len(rare_fl)} (pool: {fname}); "
        f"victims: {', '.join(labels) or 'none'}")
    for stale in ("suspicious_hands_removed.html", "suspicious_hands_removed.csv"):
        (out / stale).unlink(missing_ok=True)

    counts = full["category"].value_counts().to_dict() if not full.empty else {}
    lines = ["", "## Predicted range of the opponent", "",
             "Equity of the key decision against all hands (step 02), against the opponent's predicted range "
             "(step 04) and against his real cards, with the pot odds. Category R: hands that only the range rule "
             f"finds (fold with range equity >= pot odds + {FOLD_MARGIN:.2f} while drawing dead, or a call with range "
             f"equity <= pot odds - {CALL_MARGIN:.2f} while ahead); they are not part of the escalation rule.", ""]
    if not full.empty:
        view = full[["hand_uid", "category", "decision_label", "street"] + RANGE_COLUMNS].copy()
        for c in RANGE_COLUMNS[:-1]:
            view[c] = view[c].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.0%}")
        lines.append(view.to_markdown(index=False))
    else:
        lines.append("No suspicious hands found.")
    with open(out / "report.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary.update(player_id=result["player_id"], dataset=dataset, by_category=counts,
                   range_only_hands=int(counts.get("R", 0)), decision_times_restored=len(times) > 0,
                   range_boxes=len([b for b in boxes.values() if b.get("top_classes")]),
                   field_dataset=fshown, field_player_id=ref if with_n else None,
                   rare_decisions={"preflop": int(len(rare_pf)), "flop": int(len(rare_fl)),
                                   "range_boxes": int(sum(bool(b.get("top_classes"))
                                                          for v in rare_boxes.values() for b in v.values()))},
                   victims=[{"opponent": o, "label": lab} for o, lab in labels.items()],
                   street_tails=tails, report_notes=[x for v in notes.values() for x in v])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    e = result["escalation"]
    log(f"{result['meta']['hands_analysed']} hands; suspicious by category {counts}; "
        f"escalation: {e['suspicious']} of {e['hands']} in {e['window_days']} days -> "
        f"{'ESCALATE' if e['escalate'] else 'below thresholds'}; report: {out}")
    return {"result": result, "summary": summary, "out": out, "rare": rare, "victims": vraw,
            "metrics": metrics_raw, "field": field}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 05: per-suspect analysis with the predicted range")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--player", default=str(DEFAULT_PLAYER), help=f"player id (default: {DEFAULT_PLAYER})")
    ap.add_argument("--opponent", help="only hands where this player also sat (single-victim exposure)")
    ap.add_argument("--scope", choices=("flop_hu", "hu_tables", "all"), default="flop_hu",
                    help="flop_hu (default): two-player tables and heads-up pots on the flop at larger tables; "
                         "hu_tables: two-player tables only; all: every hand, multiway pots included")
    ap.add_argument("--all-tables", action="store_true", help="the same as --scope all")
    ap.add_argument("--times", help="decision_times.csv (default: $SU_DATA_DIR/<dataset>/input/decision_times.csv)")
    ap.add_argument("--range-config", help="step 04 config the ranges were scored with (default: config/step04_range.toml)")
    ap.add_argument("--field-dataset",
                    help="the reference field F (default: the dataset itself): the pool of the rare decisions, the "
                         "street metrics, scorecard and observer ranks of its player_scores, its pair_outliers; also "
                         "the field comparison of the real-card share at the top of the page (only when given)")
    ap.add_argument("--id-map", help="CSV with the columns case_id,ref_id: this dataset's player ids in the field F")
    ap.add_argument("--top-hands", type=int, help="rare decisions per tab, preflop and flop (default: [report] "
                                                  "top_hands of config/step05_report.toml, 25)")
    ap.add_argument("--report-config", help="TOML file whose [report] keys override config/step05_report.toml")
    ap.add_argument("--range-hands", type=int, default=3, help="hands added by the range rule (category R)")
    ap.add_argument("--out", help="output folder (default: $SU_DATA_DIR/<dataset>/report)")
    a = ap.parse_args(argv)
    analyse(a.dataset, a.player, a.opponent, "all" if a.all_tables else a.scope, a.times, a.range_config,
            a.range_hands, a.out, field_dataset=a.field_dataset, id_map=a.id_map, top_hands=a.top_hands,
            report_config=a.report_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
