"""Hand-by-hand HTML view of the suspicious hands (hand-history layout with a decision box per hand)."""
from __future__ import annotations

import html
import json

import pandas as pd

from .cards import SUIT_SYMBOLS, RANKS
from .parser import BOARD_LEN, STREETS, Hand

CSS = """
:root{--bg:#fff;--fg:#1f1f1f;--mut:#6b6b6b;--line:#e3e3e3;--me:#c62828;--card:#fafafa}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1b;--fg:#e6e6e6;--mut:#9a9a9a;--line:#333;--me:#ff6b6b;--card:#232323}}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:24px;line-height:1.5}
h1{font-size:22px;font-weight:500}h2{font-size:17px;font-weight:500;margin:0 0 8px}
.box,.decision{border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:14px;margin:12px 0}
.hand{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:18px 0;background:var(--card);overflow-x:auto}
.ln{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}.st{margin-top:10px}
table.act{border-collapse:collapse;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;margin:2px 0}
table.act td{padding:1px 10px 1px 0;white-space:nowrap}.num{text-align:right}.tm{color:var(--mut)}
table.sum{border-collapse:collapse;font-size:14px;margin:6px 0}table.sum td,table.sum th{border-bottom:1px solid var(--line);padding:3px 10px;text-align:left}
.me,.me td{color:var(--me);font-weight:600}.me .r{color:inherit}.r{color:#d32f2f}
.tag{font-size:12px;padding:2px 8px;border-radius:10px;margin-left:6px;vertical-align:middle}
.qPerfect{background:#c62828;color:#fff}.qGood{background:#f9a825;color:#222}.qAverage{background:#9e9e9e;color:#fff}
.snap{background:#6a1b9a;color:#fff}.cat{background:#e2ecfd;color:#1c3f8a}.muted{color:var(--mut)}
"""


def _cards(cards) -> str:
    out = []
    for c in cards:
        r, s = c >> 2, c & 3
        cls = "r" if s in (0, 2) else "b"
        out.append(f'<span class="{cls}">{SUIT_SYMBOLS[s]}{"10" if r == 8 else RANKS[r]}</span>')
    return " ".join(out)


def _fmt(v) -> str:
    return f"{v:,.0f}"


RANGE_CSS = """
.qRange{background:#1565c0;color:#fff}
.rng{--sev:#9e9e9e;border:1px solid var(--line);border-left:6px solid var(--sev);border-radius:8px;padding:12px 14px;margin:12px 0;font-size:14px}
.rng.sev-blatant{--sev:#c62828;background:rgba(198,40,40,.07)}
.rng.sev-strong{--sev:#ef6c00;background:rgba(239,108,0,.06)}
.rng.sev-thin{--sev:#f9a825}
.rng.sev-raise{--sev:#546e7a}
.rng .title{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--sev);margin-bottom:6px}
.rng .title span{font-weight:400;letter-spacing:0;text-transform:none;color:var(--mut)}
.rng .head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px;line-height:1.45}
.rng .badge{background:var(--sev);color:#fff;font-weight:700;font-size:12px;padding:2px 10px;border-radius:10px;letter-spacing:.05em;white-space:nowrap}
.rng .bars{display:grid;grid-template-columns:160px minmax(120px,1fr) 52px;gap:6px 12px;align-items:center;max-width:640px}
.rng .lab{color:var(--mut);font-size:13px}.rng .val{font-family:ui-monospace,Menlo,Consolas,monospace;text-align:right}
.rng .track{position:relative;height:12px;background:var(--line);border-radius:6px}
.rng .fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:#1565c0}
.rng .fill.real{background:var(--sev)}
.rng .po{position:absolute;top:-5px;bottom:-5px;width:3px;margin-left:-1px;background:var(--fg);border-radius:1px}
.rng .meta{color:var(--mut);font-size:13px;margin-top:8px}
.rng .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.rng .chip{border:1px solid var(--line);border-radius:12px;padding:1px 8px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.rng details{margin-top:8px}.rng summary{cursor:pointer;color:var(--mut);font-size:13px}
.rng table.grid{border-collapse:collapse;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;margin-top:6px}
.rng pre.rtxt{font:11px/1.35 ui-monospace,Menlo,Consolas,monospace;background:var(--line);padding:6px 8px;border-radius:6px;overflow-x:auto;white-space:pre;margin:6px 0;max-width:100%}
.rng table.grid td{width:30px;height:18px;text-align:center;border:1px solid var(--line);padding:0}
"""

SEVERITY_LABELS = {"blatant": "BLATANT", "strong": "STRONG", "thin": "THIN", "context": "BLUFF",
                   "raise": "RAISE / OVERBET", "agrees": "RANGE AGREES", "mistake": "PLAIN MISTAKE", "norange": "NO RANGE"}


def _pct(v) -> str:
    return "–" if v is None or v != v else f"{v:.0%}"


def _bar(label: str, v, pot_odds, cls: str = "fill") -> str:
    w = 0 if v is None or v != v else max(0.0, min(1.0, v)) * 100
    po = "" if pot_odds is None or pot_odds != pot_odds else f'<span class="po" style="left:{pot_odds * 100:.1f}%" title="pot odds"></span>'
    return (f'<span class="lab">{label}</span><span class="track"><span class="{cls}" style="width:{w:.1f}%"></span>{po}</span>'
            f'<span class="val">{_pct(v)}</span>')


def _range_box(r: dict) -> str:
    """The key decision measured three ways, the verdict of the predicted range, and the opponent's range."""
    tier = r.get("tier") or "agrees"
    po = r.get("pot_odds")
    out = [f'<div class="rng sev-{tier}"><div class="title">Equity review <span>· the predicted range against the pot odds; computed from the betting, separate from the hand-set A–D rules above</span></div><div class="head"><span class="badge">{SEVERITY_LABELS.get(tier, tier.upper())}</span>'
           f'<span>{html.escape(str(r.get("comment") or r.get("range_verdict") or ""))}</span></div>',
           ]
    if all(r.get(k) is None for k in ("eq_vs_all_hands", "eq_vs_range", "eq_vs_real")):
        out.append("</div>")                        # no equities to show (multiway table): verdict only
        return "".join(out)
    out += ['<div class="bars">' + _bar("equity vs all hands", r.get("eq_vs_all_hands"), po)
           + _bar("vs the predicted range", r.get("eq_vs_range"), po)
           + _bar("vs the real cards", r.get("eq_vs_real"), po, "fill real") + "</div>"]
    size = (f'about {r["n_effective"]} effective combos of {r.get("n_combos", "")}'
            if r.get("n_effective") is not None else f'{r.get("n_combos", "")} combos')
    out.append(f'<div class="meta">Black mark: pot odds {_pct(po)}. Opponent {html.escape(str(r.get("opponent", "")))}: {size}; '
               f'heaviest classes of his predicted range:</div>')
    top = r.get("top_classes") or []
    if top:
        out.append('<div class="chips">' + "".join(f'<span class="chip">{html.escape(c)} {w:.1%}</span>' for c, w in top) + "</div>")
    grid, labels = r.get("grid"), r.get("grid_labels")
    if grid is not None and labels is not None:
        peak = max(max(row) for row in grid) or 1.0
        rows = []
        for i in range(13):
            cells = []
            for j in range(13):
                a = grid[i][j] / peak
                cells.append(f'<td style="background:rgba(21,101,192,{a:.2f})" title="{labels[i][j]} {grid[i][j]:.2%}">'
                             f'{labels[i][j]}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        ranks = [labels[k][k][0] for k in range(13)]
        tsv = "\n".join(["range %\t" + "\t".join(ranks)]
                        + [ranks[i] + "\t" + "\t".join(f"{grid[i][j] * 100:.2f}" for j in range(13)) for i in range(13)])
        cells = sorted(((grid[i][j], labels[i][j]) for i in range(13) for j in range(13) if grid[i][j] > 0), reverse=True)
        listing = "\n".join(f"{c}\t{w * 100:.2f}" for w, c in cells)
        out.append('<details><summary>Predicted range, 13 × 13 grid</summary><table class="grid">' + "".join(rows)
                   + "</table></details>")
        out.append('<details><summary>Predicted range as text (weights)</summary>'
                   '<div class="meta">Weights in % of the range. Grid: rows and columns A … 2, pairs on the diagonal, '
                   'suited hands above it, offsuit below. Then every class with weight, heaviest first.</div>'
                   f'<pre class="rtxt">{html.escape(tsv)}</pre><pre class="rtxt">{html.escape(listing)}</pre></details>')
        data = {"hand": r.get("hand_uid"), "opponent": r.get("opponent"), "equity_review": SEVERITY_LABELS.get(tier, tier),
                "severity": r.get("severity"), "eq_vs_all_hands": r.get("eq_vs_all_hands"), "eq_vs_range": r.get("eq_vs_range"),
                "eq_vs_real": r.get("eq_vs_real"), "pot_odds": po, "n_combos": r.get("n_combos"),
                "n_effective": r.get("n_effective"),
                "range_weights": {c: round(w, 6) for w, c in cells}}
        out.append('<script type="application/json" class="range-data">'
                   + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    out.append("</div>")
    return "".join(out)


def _hand_block(n: int, hand: Hand, row: pd.Series, uid: int, rng: dict | None = None) -> str:
    label = row["decision_label"]
    tags = f'<span class="tag q{label.split()[0]}">{html.escape(label)}</span>'
    if row["snap_decision"]:
        tags += '<span class="tag snap">⚡ Snap decision</span>'
    tags += f'<span class="tag cat">{html.escape(row["category_name"])}</span>'
    t0 = pd.to_datetime(hand.start_ts, unit="s")
    t1 = pd.to_datetime(hand.end_ts, unit="s") if hand.end_ts and hand.end_ts > hand.start_ts else None
    lines = [f'<div class="hand" id="h{n}"><h2>NO.{n} {tags}</h2>',
             f'<div class="ln"><b>table:</b> {html.escape(hand.game_type)} &nbsp; hand:{html.escape(hand.hand_uid)}'
             f' &nbsp; table no:{hand.table_no} &nbsp; blind:{_fmt(hand.blind)} &nbsp; ante:{_fmt(hand.ante)} &nbsp; players:{hand.n_players}</div>',
             f'<div class="ln"><b>time:</b> start {t0:%Y-%m-%d %H:%M:%S}'
             + (f' &nbsp; end {t1:%Y-%m-%d %H:%M:%S}' if t1 is not None else "") + " (UTC)</div>"]
    for st in STREETS:
        board = hand.board[:BOARD_LEN[st]] if len(hand.board) >= BOARD_LEN[st] else []
        if st != "preflop" and not board and not hand.actions[st]:
            continue
        lines.append(f'<div class="st"><b><i>***{st}***</i></b> {_cards(board) if st != "preflop" else ""}</div>')
        rows = []
        for a in hand.actions[st]:
            s = hand.seats.get(a.seat_no)
            me = ' class="me"' if a.uid == uid else ""
            rows.append(f"<tr{me}><td>seat:{a.seat_no}</td><td>player:{a.uid}</td><td>stack:{_fmt(s.stake) if s else ''}</td>"
                        f"<td>{html.escape(s.label) if s else ''}</td><td>{_cards(s.hole) if s else ''}</td>"
                        f'<td>{a.name}</td><td class="num">{_fmt(a.amount)}</td><td class="num tm">{a.time:.0f}s</td></tr>')
        if rows:
            lines.append('<table class="act">' + "".join(rows) + "</table>")
        pot = hand.street_end_pot.get(st, 0)
        if pot:
            lines.append(f'<div class="ln">pot:{_fmt(pot)} / {pot / hand.blind:.1f}bb</div>' if hand.blind else f'<div class="ln">pot:{_fmt(pot)}</div>')
    lines.append('<div class="st"><b><i>***settlement***</i></b></div>')
    for u, w in hand.settlement.items():
        me = " me" if u == uid else ""
        lines.append(f'<div class="ln{me}">player:{u} &nbsp;|&nbsp; net:{w:+,.0f}</div>')
    lines.append(
        f'<div class="decision"><div><b>{html.escape(label)}:</b> {html.escape(row["player_hand"] or "no made hand")} '
        f'against {html.escape(row["opponent_hand"] or "no made hand")}, equity at the decision {row["equity_at_decision"]:.0%}.</div>'
        f'<div><b>Key action ({row["street"]}):</b> {html.escape(row["key_action"])} &nbsp; '
        f'<b>decision time:</b> {row["decision_time_s"]:.0f}s</div>'
        f'<div><b>Pot:</b> {row["pot_bb"]:.1f} bb &nbsp; <b>Player net:</b> {row["net_win"]:+,.0f}</div></div>'
        + (_range_box({**rng, "hand_uid": row["hand_uid"]}) if rng else "") + "</div>")
    return "\n".join(lines)


def suspicious_hands_html(result: dict, ranges: dict | None = None, outside_scope: int = 0) -> str:
    """ranges (optional): {hand_uid: dict} with eq_vs_all_hands, eq_vs_range, eq_vs_real, pot_odds, range_verdict,
    opponent, n_combos, top_classes [(class, share)], grid / grid_labels (13 x 13). Without it the page is unchanged.
    outside_scope: how many of the listed hands come from outside the analysed hands (multiway tables), counted apart."""
    uid, sus, hands = result["player"], result["suspicious_hands"], result["hands"]
    m, e = result["meta"], result["escalation"]
    scope = f"hands against {result['opponent']}" if result["opponent"] else "all submitted hands"
    counts = sus["decision_label"].value_counts().to_dict() if not sus.empty else {}
    n_in = len(sus) - outside_scope
    summary = (f"<p>Player <b>{uid}</b>, {scope}. {m['hands_analysed']:,} hands analysed, "
               f"<b>{n_in:,} suspicious</b> ({n_in / max(m['hands_analysed'], 1):.1%})"
               + (f", plus {outside_scope} flagged on multiway tables" if outside_scope else "") + ".</p>")
    if e.get("hands"):
        verdict = "<b>ESCALATE for manual review</b>" if e["escalate"] else "below escalation thresholds"
        summary += (f"<p>Last {e['window_days']} days: {e['suspicious']} suspicious of {e['hands']:,} hands "
                    f"({e['ratio']:.1%}): {verdict}.</p>")
    legend = ('<div class="box"><b>Decision labels</b><br>'
              "<b>Fold</b>: Perfect = pair or better folded with ≤5% equity · Good = ≤5% equity, or ≤20% with a made hand · Average = otherwise.<br>"
              "<b>Bluff-catch</b>: Perfect = high card or one pair · Good = two pair · Average = trips or better.<br>"
              "<b>Bluff</b>: Perfect = high card made a pair-or-better fold · Good = opponent folded a made hand · Average = otherwise.<br>"
              "<b>⚡ Snap decision</b>: a Perfect decision made in ≤2 s with a pot of ≥30 bb. "
              f'<span class="muted">Label counts: {", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"}. '
              "Rows for the analysed player are in red; amounts are as stored in the replay; times are UTC.</span></div>")
    if ranges is not None:
        n_r = int((sus["category"] == "R").sum()) if not sus.empty else 0
        summary += (f"<p>{n_r} of the hands below are found only by the predicted-range rule (category R) and are "
                    "not counted above or in the escalation rule.</p>" if n_r else "")
        legend = legend[:-6] + ("<br><b>Equity review</b> (box under each hand): equity against the opponent's range as "
                                "predicted from his actions, next to equity against all hands and against his real "
                                "cards. <b>R</b> = a fold with range equity at least 24 points above the pot odds while "
                                "drawing dead, or a call at least 20 points below the pot odds while ahead.</div>")
    blocks = [_hand_block(i + 1, hands[r["hand_uid"]], r, uid, (ranges or {}).get(r["hand_uid"]))
              for i, (_, r) in enumerate(sus.iterrows())]
    body = "\n".join(blocks) if blocks else "<p>No suspicious hands found.</p>"
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Suspicious hands: player {uid}</title><style>{CSS}{RANGE_CSS if ranges is not None else ''}</style></head><body>"
            f"<h1>Suspicious hands: player {uid}</h1>{summary}{legend}{body}</body></html>")
