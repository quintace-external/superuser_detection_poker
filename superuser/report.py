"""Write the analysis as Markdown, HTML, CSV and JSON."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd

from .html_report import CSS, suspicious_hands_html

STAT_LABELS = {
    "hands": ("Hands", "n"), "net_win": ("Net win (chips)", "chips"), "bb_100": ("bb/100", "bb"),
    "vpip": ("VPIP", "%"), "pfr": ("PFR", "%"), "three_bet": ("3-bet", "%"), "fold_to_3bet": ("Fold to 3-bet", "%"),
    "flop_cbet": ("Flop c-bet", "%"), "fold_to_cbet": ("Fold to c-bet", "%"),
    "aggression_factor": ("Aggression factor", "x"), "aggression_frequency": ("Aggression frequency", "%"),
    "wtsd": ("WTSD", "%"), "w_sd": ("W$SD", "%"), "fold_to_river_bet": ("Fold to river bet", "%"),
    "river_call_win_rate": ("River call win rate", "%"), "median_think_time": ("Median action time", "s"),
}
SIGNAL_LABELS = {
    "perfect_fold_ratio_postflop": "Perfect fold ratio, postflop",
    "perfect_fold_ratio_turn": "Perfect fold ratio, turn",
    "perfect_fold_ratio_river": "Perfect fold ratio, river",
    "pair_plus_folds_behind": "Pair-or-better postflop folds while behind",
    "bluff_index": "Bluff index (bluff success rate)",
    "fold_vs_value_bet": "Fold when facing a value bet",
    "fold_vs_bluff": "Fold when facing a bluff",
    "discrimination_gap": "Discrimination gap",
    "river_decision_accuracy": "River decision accuracy",
}
SUSPICIOUS_COLUMNS = ["hand_uid", "start_utc", "decision_label", "category_name", "player_cards",
                      "opponent_cards", "board", "player_hand", "opponent_hand", "key_action", "decision_time_s", "snap_decision", "pot_bb"]


def _fmt(v, kind):
    if v is None:
        return "–"
    return {"n": f"{v:,.0f}", "chips": f"{v:+,.0f}", "bb": f"{v:+.1f}", "%": f"{100 * v:.1f}%",
            "x": f"{v:.2f}", "s": f"{v:.0f} s"}[kind]


def stats_table(stats: dict) -> pd.DataFrame:
    return pd.DataFrame([{"stat": lab, "value": _fmt(stats.get(k), kind)} for k, (lab, kind) in STAT_LABELS.items()])


def signals_table(signals: dict) -> pd.DataFrame:
    rows = []
    for k, lab in SIGNAL_LABELS.items():
        s = signals[k]
        if k == "discrimination_gap":
            val = f"{100 * s['value']:.1f} pts" if s["value"] is not None else "–"
        else:
            val = f"{100 * s['value']:.1f}% ({s['k']:,} / {s['n']:,})" if s["value"] is not None else "–"
        rows.append({"signal": lab, "value": val})
    return pd.DataFrame(rows)


def _escalation_text(e: dict) -> str:
    if not e["hands"]:
        return "No hands in scope."
    verdict = "**ESCALATE for manual review**" if e["escalate"] else "Below escalation thresholds"
    return (f"Last {e['window_days']} days ({e['window_start_utc']} to {e['window_end_utc']} UTC): "
            f"{e['suspicious']} suspicious of {e['hands']:,} hands ({e['ratio']:.1%}). "
            f"Rule: ratio > 5% and more than 5 suspicious hands. {verdict}.")


def to_markdown(result: dict, top_n: int = 25) -> str:
    sus = result["suspicious_hands"]
    scope = f"hands against {result['opponent']}" if result["opponent"] else "all submitted hands"
    m = result["meta"]
    lines = [f"# Superuser analysis: player {result['player']}", "",
             f"Scope: {scope}. Hands analysed: {m['hands_analysed']:,} "
             f"(rows submitted {m['rows']:,}; unparseable {m['unparseable']:,}; player not in hand {m['not_in_hand']:,}; "
             f"filtered out {m['filtered']:,}).", "", "## Player statistics", "",
             stats_table(result["stats"]).to_markdown(index=False), "", "## Hidden-information signals", "",
             signals_table(result["signals"]).to_markdown(index=False), "", "## Escalation rule", "",
             _escalation_text(result["escalation"]), "", "## Suspicious hands", ""]
    if sus.empty:
        lines.append("No suspicious hands found.")
    else:
        counts = sus["decision_label"].value_counts()
        lines += [f"{len(sus):,} suspicious hands ({len(sus) / max(m['hands_analysed'], 1):.1%} of hands analysed). "
                  + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".", "",
                  f"Top {min(top_n, len(sus))} (Perfect first, then Good, Average; larger pots first):", "",
                  sus[SUSPICIOUS_COLUMNS].head(top_n).to_markdown(index=False)]
    return "\n".join(lines) + "\n"


def to_html(result: dict, top_n: int = 25) -> str:
    """report.md as one HTML page; every flagged hand in full is in suspicious_hands.html."""
    sus = result["suspicious_hands"]
    scope = f"hands against {result['opponent']}" if result["opponent"] else "all submitted hands"
    m, uid = result["meta"], result["player"]

    def table(df: pd.DataFrame) -> str:
        return '<div style="overflow-x:auto">' + df.to_html(index=False, classes="sum", border=0) + "</div>"

    h2 = '<h2 style="margin-top:22px">{}</h2>'
    parts = [f"<h1>Superuser analysis: player {uid}</h1>",
             f"<p>Scope: {html.escape(scope)}. Hands analysed: {m['hands_analysed']:,} "
             f"(rows submitted {m['rows']:,}; unparseable {m['unparseable']:,}; player not in hand {m['not_in_hand']:,}; "
             f"filtered out {m['filtered']:,}).</p>",
             h2.format("Player statistics"), table(stats_table(result["stats"])),
             h2.format("Hidden-information signals"), table(signals_table(result["signals"]))]
    text = html.escape(_escalation_text(result["escalation"]).replace("**", ""))
    text = text.replace("ESCALATE for manual review", "<b>ESCALATE for manual review</b>")
    parts += [h2.format("Escalation rule"), f'<div class="box">{text}</div>', h2.format("Suspicious hands")]
    if sus.empty:
        parts.append("<p>No suspicious hands found.</p>")
    else:
        share = f"{len(sus):,} suspicious hands ({len(sus) / max(m['hands_analysed'], 1):.1%} of hands analysed)"
        counts = ", ".join(f"{k}: {v}" for k, v in sus["decision_label"].value_counts().items())
        parts += [f"<p>{html.escape(share + '. ' + counts + '.')} "
                  '<a href="suspicious_hands.html">Every flagged hand in full</a>.</p>',
                  f"<p>Top {min(top_n, len(sus))} (Perfect first, then Good, Average; larger pots first):</p>",
                  table(sus[SUSPICIOUS_COLUMNS].head(top_n))]
    return ('<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
            f'initial-scale=1"><title>Superuser analysis: player {uid}</title><style>{CSS}</style></head><body>'
            + "".join(parts) + "</body></html>")


OUTPUT_FILES = ["report.md", "report.html", "suspicious_hands.html", "stats.csv", "signals.csv",
                "suspicious_hands.csv", "summary.json"]


def _clear(out: Path) -> None:
    """Empty the output folder (keeps .gitkeep) so every run starts clean."""
    for p in out.iterdir():
        if p.name == ".gitkeep":
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def write_report(result: dict, out_dir: str | Path, top_n: int = 25) -> Path:
    """Write the outputs to `out_dir`, replacing whatever the previous run left there."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    (out / "report.md").write_text(to_markdown(result, top_n), encoding="utf-8")
    (out / "report.html").write_text(to_html(result, top_n), encoding="utf-8")
    (out / "suspicious_hands.html").write_text(suspicious_hands_html(result), encoding="utf-8")
    stats_table(result["stats"]).to_csv(out / "stats.csv", index=False)
    signals_table(result["signals"]).to_csv(out / "signals.csv", index=False)
    result["suspicious_hands"].to_csv(out / "suspicious_hands.csv", index=False)
    summary = {k: result[k] for k in ("player", "opponent", "meta", "stats", "signals", "escalation")}
    summary["suspicious_hand_count"] = int(len(result["suspicious_hands"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out
