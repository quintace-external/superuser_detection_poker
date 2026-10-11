"""Hand-by-hand HTML view of the suspicious hands (hand-history layout with a decision box per hand).

Players are printed by the id the data gives them: ``result["uid_of"]`` (warehouse id -> the module's integer id,
set by ``05_detection/adapter.py``) maps the module's numbers back; without it the module's ids are printed.
``tabbed_hands_html`` puts several lists of hands on one page with a tab selector (a few lines of inline script; the
first tab shows without it) and can name, next to an opponent's id, the pair he forms with the player
(``labels``: id -> text, given for the player's victims only).
"""
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


def _secs(v) -> str:
    """Seconds as '3s'; a dash when the data has no decision times."""
    try:
        return "–" if v is None or v != v else f"{float(v):.0f}s"
    except (TypeError, ValueError):
        return "–"


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
.rng.sev-rare{--sev:#6a1b9a}
"""

TAB_CSS = """
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 6px;border-bottom:1px solid var(--line)}
.tabs button{font:inherit;font-size:14px;color:var(--mut);background:none;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;padding:6px 14px;cursor:pointer}
.tabs button[aria-selected="true"]{color:var(--fg);background:var(--card);font-weight:600}
body[data-tab="postflop"] [data-tab="preflop"],body[data-tab="preflop"] [data-tab="postflop"]{display:none}
.tag.pair{background:#ede7f6;color:#4527a0}
"""
TAB_JS = ("document.querySelectorAll('.tabs button').forEach(function(b){b.addEventListener('click',function(){"
          "document.body.setAttribute('data-tab',b.getAttribute('data-show'));"
          "document.querySelectorAll('.tabs button').forEach(function(x){x.setAttribute('aria-selected',String(x===b))})"
          "})})")

SEVERITY_LABELS = {"blatant": "BLATANT", "strong": "STRONG", "thin": "THIN", "context": "BLUFF",
                   "raise": "RAISE / OVERBET", "agrees": "RANGE AGREES", "mistake": "PLAIN MISTAKE", "norange": "NO RANGE",
                   "rare": "RARE IN THE POOL"}


def _pct(v) -> str:
    return "–" if v is None or v != v else f"{v:.0%}"


def _bar(label: str, v, pot_odds, cls: str = "fill") -> str:
    w = 0 if v is None or v != v else max(0.0, min(1.0, v)) * 100
    po = "" if pot_odds is None or pot_odds != pot_odds else f'<span class="po" style="left:{pot_odds * 100:.1f}%" title="pot odds"></span>'
    return (f'<span class="lab">{label}</span><span class="track"><span class="{cls}" style="width:{w:.1f}%"></span>{po}</span>'
            f'<span class="val">{_pct(v)}</span>')


def _actual_text(r: dict) -> str:
    """Where the opponent's real hand sits in his predicted range: "AQo: 2.5% of the range (#4 of 169)"."""
    cls, share, rank = r.get("actual_class"), r.get("actual_share"), r.get("actual_rank")
    if not share:
        return f"Opponent's actual hand, {cls}: not in the predicted range."
    return f"Opponent's actual hand, {cls}: {share:.1%} of the range (#{rank} of 169)."


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
    if r.get("actual_class"):
        out.append(f'<div class="meta">{html.escape(_actual_text(r))}</div>')
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
                "n_effective": r.get("n_effective"), "actual_class": r.get("actual_class"),
                "actual_share": r.get("actual_share"), "actual_rank": r.get("actual_rank"),
                "range_weights": {c: round(w, 6) for w, c in cells}}
        out.append('<script type="application/json" class="range-data">'
                   + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    out.append("</div>")
    return "".join(out)


def _names(result: dict) -> dict:
    """The module's integer id -> the id the data gives the player (``result["uid_of"]``); empty without it."""
    return {u: p for p, u in (result.get("uid_of") or {}).items()}


def _pair_tag(labels: dict | None, pid) -> str:
    """The pair label next to an opponent's id: only for the ids in ``labels`` (the player's victims)."""
    text = (labels or {}).get(str(pid))
    return f' <span class="tag pair">{html.escape(text)}</span>' if text else ""


def _hand_block(n: int, hand: Hand, row: pd.Series, uid: int, rng: dict | None = None, names: dict | None = None,
                labels: dict | None = None, tab: str | None = None, anchor: str | None = None,
                title: str = "NO.") -> str:
    """One hand: tags, the actions street by street, the settlement, the decision box and the range box.

    names: the module's id -> the printed id; labels: printed id -> pair text (the player's victims only); tab: the
    page tab of the block (``data-tab``); anchor: the element id (default ``h{n}``); title: the text before n."""
    def shown(u):
        return names.get(u, u) if names else u

    label = row["decision_label"]
    tags = f'<span class="tag q{label.split()[0]}">{html.escape(label)}</span>'
    if row["snap_decision"]:
        tags += '<span class="tag snap">⚡ Snap decision</span>'
    tags += f'<span class="tag cat">{html.escape(row["category_name"])}</span>'
    t0 = pd.to_datetime(hand.start_ts, unit="s")
    t1 = pd.to_datetime(hand.end_ts, unit="s") if hand.end_ts and hand.end_ts > hand.start_ts else None
    attrs = f' id="{anchor or f"h{n}"}"' + (f' data-tab="{tab}"' if tab else "")
    lines = [f'<div class="hand"{attrs}><h2>{title}{n} {tags}</h2>',
             f'<div class="ln"><b>table:</b> {html.escape(hand.game_type)} &nbsp; hand:{html.escape(hand.hand_uid)}'
             f' &nbsp; table no:{html.escape(str(hand.table_no))} &nbsp; blind:{_fmt(hand.blind)} &nbsp; ante:{_fmt(hand.ante)} &nbsp; players:{hand.n_players}</div>',
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
            rows.append(f"<tr{me}><td>seat:{a.seat_no}</td><td>player:{html.escape(str(shown(a.uid)))}</td>"
                        f"<td>stack:{_fmt(s.stake) if s else ''}</td>"
                        f"<td>{html.escape(s.label) if s else ''}</td><td>{_cards(s.hole) if s else ''}</td>"
                        f'<td>{a.name}</td><td class="num">{_fmt(a.amount)}</td><td class="num tm">{_secs(a.time)}</td></tr>')
        if rows:
            lines.append('<table class="act">' + "".join(rows) + "</table>")
        pot = hand.street_end_pot.get(st, 0)
        if pot:
            lines.append(f'<div class="ln">pot:{_fmt(pot)} / {pot / hand.blind:.1f}bb</div>' if hand.blind else f'<div class="ln">pot:{_fmt(pot)}</div>')
    lines.append('<div class="st"><b><i>***settlement***</i></b></div>')
    for u, w in hand.settlement.items():
        me = " me" if u == uid else ""
        lines.append(f'<div class="ln{me}">player:{html.escape(str(shown(u)))}{_pair_tag(labels, shown(u))} '
                     f'&nbsp;|&nbsp; net:{w:+,.0f}</div>')
    opp = row.get("opponent_uid")
    known = opp is not None and str(opp) != "" and (isinstance(opp, str) or not pd.isna(opp))
    opp_line = f'<div><b>Opponent:</b> {html.escape(str(opp))}{_pair_tag(labels, opp)}</div>' if known else ""
    lines.append(
        f'<div class="decision"><div><b>{html.escape(label)}:</b> {html.escape(row["player_hand"] or "no made hand")} '
        f'against {html.escape(row["opponent_hand"] or "no made hand")}, equity at the decision {row["equity_at_decision"]:.0%}.</div>'
        + opp_line +
        f'<div><b>Key action ({row["street"]}):</b> {html.escape(row["key_action"])} &nbsp; '
        f'<b>decision time:</b> {_secs(row["decision_time_s"])}</div>'
        f'<div><b>Pot:</b> {row["pot_bb"]:.1f} bb &nbsp; <b>Player net:</b> {row["net_win"]:+,.0f}</div></div>'
        + (_range_box({**rng, "hand_uid": row["hand_uid"]}) if rng else "") + "</div>")
    return "\n".join(lines)


def _summary_legend(result: dict, sus: pd.DataFrame, with_ranges: bool, outside_scope: int) -> tuple[str, str]:
    """The summary lines (hands, suspicious share, escalation) and the legend box of a suspicious-hands page."""
    m, e = result["meta"], result["escalation"]
    uid = html.escape(str(_names(result).get(result["player"], result["player"])))
    scope = f"hands against {html.escape(str(result['opponent']))}" if result["opponent"] else "all submitted hands"
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
    if with_ranges:
        n_r = int((sus["category"] == "R").sum()) if not sus.empty else 0
        summary += (f"<p>{n_r} of the hands below are found only by the predicted-range rule (category R) and are "
                    "not counted above or in the escalation rule.</p>" if n_r else "")
        legend = legend[:-6] + ("<br><b>Equity review</b> (box under each hand): equity against the opponent's range as "
                                "predicted from his actions, next to equity against all hands and against his real "
                                "cards. <b>R</b> = a fold with range equity at least 24 points above the pot odds while "
                                "drawing dead, or a call at least 20 points below the pot odds while ahead.</div>")
    return summary, legend


def suspicious_hands_html(result: dict, ranges: dict | None = None, outside_scope: int = 0) -> str:
    """ranges (optional): {hand_uid: dict} with eq_vs_all_hands, eq_vs_range, eq_vs_real, pot_odds, range_verdict,
    opponent, n_combos, top_classes [(class, share)], grid / grid_labels (13 x 13). Without it the page is unchanged.
    outside_scope: how many of the listed hands come from outside the analysed hands (multiway tables), counted apart."""
    uid, sus, hands = result["player"], result["suspicious_hands"], result["hands"]
    names = _names(result)
    shown = html.escape(str(names.get(uid, uid)))
    summary, legend = _summary_legend(result, sus, ranges is not None, outside_scope)
    blocks = [_hand_block(i + 1, hands[r["hand_uid"]], r, uid, (ranges or {}).get(r["hand_uid"]), names)
              for i, (_, r) in enumerate(sus.iterrows())]
    body = "\n".join(blocks) if blocks else "<p>No suspicious hands found.</p>"
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Suspicious hands: player {shown}</title><style>{CSS}{RANGE_CSS if ranges is not None else ''}</style></head><body>"
            f"<h1>Suspicious hands: player {shown}</h1>{summary}{legend}{body}</body></html>")


def tabbed_hands_html(result: dict, tabs: list[dict], counted: pd.DataFrame, outside_scope: int = 0,
                      labels: dict | None = None, top: str = "") -> str:
    """Several lists of hands on one page behind a tab selector; the first tab is the default and shows without the
    script, "All" shows every list.

    tabs: [{"key", "label", "rows" (DataFrame in the suspicious-hands format), "boxes" ({hand_uid: range box}),
    "intro" (HTML above the list), "title" (text before each block's number), "anchor" (prefix of the block ids)}].
    counted: the hands of the package's rules (A-D, R and the multiway flags) for the summary line and the label
    counts. top: HTML under the title. labels: printed id -> pair text, given for the player's victims only."""
    uid, hands = result["player"], result["hands"]
    names = _names(result)
    shown = html.escape(str(names.get(uid, uid)))
    summary, legend = _summary_legend(result, counted, True, outside_scope)
    total = sum(len(t["rows"]) for t in tabs)
    buttons = [(t["key"], f'{t["label"]} ({len(t["rows"]):,})') for t in tabs] + [("all", f"All ({total:,})")]
    bar = ('<div class="tabs" role="tablist">'
           + "".join(f'<button type="button" role="tab" data-show="{k}" aria-selected="{str(i == 0).lower()}">'
                     f"{html.escape(text)}</button>" for i, (k, text) in enumerate(buttons)) + "</div>")
    body = []
    for t in tabs:
        if t.get("intro"):
            body.append(f'<div class="box" data-tab="{t["key"]}">{t["intro"]}</div>')
        for i, (_, r) in enumerate(t["rows"].iterrows()):
            body.append(_hand_block(i + 1, hands[r["hand_uid"]], r, uid, t["boxes"].get(r["hand_uid"]), names, labels,
                                    t["key"], f'{t.get("anchor", "h")}{i + 1}', t.get("title", "NO.")))
        if t["rows"].empty:
            body.append(f'<p data-tab="{t["key"]}">No hands in this list.</p>')
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Suspicious hands: player {shown}</title><style>{CSS}{RANGE_CSS}{TAB_CSS}</style></head>"
            f'<body data-tab="{tabs[0]["key"]}"><h1>Suspicious hands: player {shown}</h1>{top}{summary}{legend}{bar}'
            + "\n".join(body) + f"<script>{TAB_JS}</script></body></html>")
