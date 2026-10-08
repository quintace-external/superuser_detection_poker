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

The thresholds and definitions are those of the package (`superuser/`). Outputs go to ``$SU_DATA_DIR/<dataset>/report/``:
``report.md``, ``suspicious_hands.html`` (with a predicted-range box per hand), ``suspicious_hands.csv``,
``decisions.csv``, ``stats.csv``, ``signals.csv`` and ``summary.json``.

Usage::

    python 05_detection/run.py --dataset article_sample                 # the sample's subject
    python 05_detection/run.py --dataset <name> --player <id> [--opponent <id>] [--all-tables]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudetect import rangemodel, warehouse  # noqa: E402
from sudetect.cards import CLASS_LABELS  # noqa: E402
from superuser.cards import CATEGORY_NAMES, evaluate, to_pretty  # noqa: E402
from superuser.config import DEFAULT_PLAYER  # noqa: E402
from superuser.html_report import suspicious_hands_html  # noqa: E402
from superuser.parser import BOARD_LEN  # noqa: E402
from superuser.report import write_report  # noqa: E402
from superuser.suspicious import _category  # noqa: E402

RANGE_COLUMNS = ["eq_vs_all_hands", "eq_vs_range", "eq_vs_real", "pot_odds", "range_verdict"]
FOLD_MARGIN, CALL_MARGIN = 0.24, 0.20          # the range rule of category R
DEAD_BELOW, AHEAD_FROM = 0.02, 0.5
DEFAULT_RANGE_CONFIG = ROOT / "config" / "step04_range.toml"


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


def field_note(dataset: str, field_dataset: str | None, player: str | None = None) -> str:
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
            f"for the median player ({len(ref)} players).</div>")


TIER_ORDER = ("blatant", "strong", "thin", "context", "raise", "agrees", "mistake", "norange")
TIER_NAMES = {"blatant": "BLATANT", "strong": "STRONG", "thin": "THIN", "context": "BLUFF", "raise": "RAISE / OVERBET",
              "agrees": "RANGE AGREES", "mistake": "PLAIN MISTAKE", "norange": "NO RANGE"}
RULE_NAMES = {"A": "River fold when beaten", "B": "River bluff-catch", "C": "Successful river bluff",
              "D": "Early fold, near dead", "R": "Range rule"}
MULTIWAY_NOTE = "Multiway table: the predicted range covers heads-up decisions only."


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
    mw = (f" The last {n_mw} come from multiway tables, outside the heads-up hands the range model covers." if n_mw else "")
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
    """The decision row the package used for each suspicious hand (first qualifying row, priority A > B > C > D)."""
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
    if res is None:
        return box
    mass = rangemodel.range_classes(res.ids, res.w)
    order = np.argsort(-mass, kind="stable")[:10]
    grid, labels = rangemodel.class_grid(mass)
    p = np.asarray(res.w, dtype=float) / float(np.sum(res.w))
    # effective size 1 / sum(p^2): a range of n equal combos gives n; light tails count for little
    box.update(n_combos=int(res.ids.size), n_effective=int(round(1.0 / float(np.sum(p * p)))),
               grid=grid.tolist(), grid_labels=labels,
               top_classes=[(CLASS_LABELS[i], float(mass[i])) for i in order if mass[i] > 0])
    return box


# ------------------------------------------------------------------ main


def analyse(dataset: str, player: str, opponent: Optional[str] = None, heads_up_only: bool = True,
            times_path: Optional[str] = None, range_config: Optional[str] = None, range_hands: int = 3,
            out_dir: Optional[str] = None, log=print, field_dataset: Optional[str] = None) -> dict:
    ad = adapter()
    times_file = Path(times_path) if times_path else warehouse.data_dir() / dataset / "input" / "decision_times.csv"
    times = ad.read_times(times_file)
    con = warehouse.connect(dataset, read_only=True)
    try:
        result = ad.analyse_player(con, player, opponent, heads_up_only, times=times)
        if not result["meta"].get("hands_analysed"):
            raise SystemExit(f"{dataset}: player {player} has no hands in scope")
        uid = result["player"]
        multiway, mw_hands = pd.DataFrame(), {}
        if heads_up_only:                           # the package on every table: the flags the heads-up scope leaves out
            every = ad.analyse_player(con, player, opponent, False, times=times)
            if not every["suspicious_hands"].empty:
                hu = set(result["suspicious_hands"]["hand_uid"]) if not result["suspicious_hands"].empty else set()
                multiway = every["suspicious_hands"][~every["suspicious_hands"]["hand_uid"].isin(hu)].reset_index(drop=True)
                mw_hands = {g: every["hands"][g] for g in multiway["hand_uid"]}
        name_of = {u: p for p, u in ad.player_uids(warehouse.query_df(
            con, "SELECT DISTINCT player_id FROM hand_players").player_id).items()}
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

        boxes: dict[str, dict] = {}
        if warehouse.table_exists(con, "action_range_equity") and warehouse.table_exists(con, "rng_lik_post"):
            run04 = _module("step04_run", ROOT / "04_predicted_range" / "run.py")
            examples = _module("step04_examples", ROOT / "04_predicted_range" / "examples.py")
            cfg = run04.load_config(range_config or DEFAULT_RANGE_CONFIG)
            tables, _ = run04.load_tables(con)
            for gid, row in keys.items():
                hand = result["hands"][gid]
                opp = next((s.uid for s in hand.seats.values() if s.uid != uid), None)
                if opp is None or len(hand.seats) != 2:
                    continue
                boxes[gid] = range_box(con, cfg, tables, examples, gid, row, result["player_id"], name_of[opp])
    finally:
        con.close()

    out = Path(out_dir) if out_dir else warehouse.data_dir() / dataset / "report"
    write_report(result, out)                       # the package's report, his rows only
    full = pd.concat([sus, extra], ignore_index=True) if not extra.empty else sus
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
        notes = []
        for i, row in full.iterrows():
            gid, dec = row["hand_uid"], keyrows.get(row["hand_uid"])
            note = equity_note(row, bool(fold[i]), float(sev[i]),
                               bool(dec is not None and faced_raise(result["hands"][gid], dec)))
            notes.append(note)
            boxes.setdefault(gid, {}).update(note)
        ranked = ranked.assign(_tier=[n["tier"] for n in notes], _sev=[n["severity"] for n in notes],
                               _comment=[n["comment"] for n in notes])
    if not multiway.empty:
        for g in multiway["hand_uid"]:
            boxes[g] = {"tier": "norange", "severity": 0.0, "comment": MULTIWAY_NOTE}
        ranked = pd.concat([ranked, multiway.assign(_tier="norange", _sev=0.0, _comment=MULTIWAY_NOTE,
                                                    _pos=range(len(full), len(full) + len(multiway)))], ignore_index=True)
    ranked["_t"] = ranked["_tier"].map(TIER_ORDER.index)
    ranked = ranked.sort_values(["_t", "_sev", "_pos"], ascending=[True, False, True]).reset_index(drop=True)
    (out / "suspicious_hands_all.html").write_text(suspicious_hands_html(result, ranges=boxes), encoding="utf-8")
    page = suspicious_hands_html({**result, "hands": {**mw_hands, **result["hands"]},
                                  "suspicious_hands": ranked.drop(columns=[c for c in ranked.columns if c.startswith("_")])},
                                 ranges=boxes, outside_scope=len(multiway))
    if not ranked.empty:
        page = page.replace("</h1>", "</h1>" + overview_html(ranked, len(multiway)) + field_note(dataset, field_dataset, player), 1)
    (out / "suspicious_hands.html").write_text(page, encoding="utf-8")
    cols = ["hand_uid", "category", "decision_label", "street", "key_action", "bet_to_pot", "pot_bb", "player_hand",
            "opponent_hand", "eq_vs_all_hands", "eq_vs_range", "eq_vs_real", "pot_odds"]
    ranked.assign(rank=range(1, len(ranked) + 1), equity_review=ranked["_tier"].map(TIER_NAMES),
                  severity=ranked["_sev"].round(4), comment=ranked["_comment"]).reindex(
        columns=["rank"] + cols + ["equity_review", "severity", "comment"]).to_csv(out / "suspicious_hands_ranked.csv", index=False)
    log("ranked page: " + ", ".join(f"{TIER_NAMES[t]} {int((ranked['_tier'] == t).sum())}" for t in TIER_ORDER
                                    if (ranked["_tier"] == t).any()) + f" ({len(ranked)} hands)")
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
    with open(out / "report.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary.update(player_id=result["player_id"], dataset=dataset, by_category=counts,
                   range_only_hands=int(counts.get("R", 0)), decision_times_restored=len(times) > 0,
                   range_boxes=len([b for b in boxes.values() if b.get("top_classes")]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    e = result["escalation"]
    log(f"{result['meta']['hands_analysed']} hands; suspicious by category {counts}; "
        f"escalation: {e['suspicious']} of {e['hands']} in {e['window_days']} days -> "
        f"{'ESCALATE' if e['escalate'] else 'below thresholds'}; report: {out}")
    return {"result": result, "summary": summary, "out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 05: per-suspect analysis with the predicted range")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--player", default=str(DEFAULT_PLAYER), help=f"player id (default: {DEFAULT_PLAYER})")
    ap.add_argument("--opponent", help="only hands where this player also sat (single-victim exposure)")
    ap.add_argument("--all-tables", action="store_true", help="include multiway tables (default: heads-up only)")
    ap.add_argument("--times", help="decision_times.csv (default: $SU_DATA_DIR/<dataset>/input/decision_times.csv)")
    ap.add_argument("--range-config", help="step 04 config the ranges were scored with (default: config/step04_range.toml)")
    ap.add_argument("--field-dataset", help="dataset the range tables were fitted on, for the field comparison at the top of the page")
    ap.add_argument("--range-hands", type=int, default=3, help="hands added by the range rule (category R)")
    ap.add_argument("--out", help="output folder (default: $SU_DATA_DIR/<dataset>/report)")
    a = ap.parse_args(argv)
    analyse(a.dataset, a.player, a.opponent, not a.all_tables, a.times, a.range_config, a.range_hands, a.out,
            field_dataset=a.field_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
