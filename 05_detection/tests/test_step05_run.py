"""Step 05 on the article-case sample: the package's verdicts plus the predicted-range columns and boxes, and the
report's field sections (the sample is its own field: pairs and observer metrics computed on the fly, no screen), also
with the package's flags hidden; then the report of a player without a suspicious hand (the sample's hands as a mined
history keeps them)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mod(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_detection_on_sample_with_ranges(tmp_path, monkeypatch, capsys):
    from superuser import DEFAULT_INPUT, read_hands

    monkeypatch.setenv("SU_DATA_DIR", str(tmp_path))
    inp = tmp_path / "article_sample" / "input"
    _mod("step01_from_json", ROOT / "01_action_table" / "from_replay_json.py").convert(
        read_hands(DEFAULT_INPUT).to_dict("records"), inp, log=lambda _m: None)
    _mod("step01_run", ROOT / "01_action_table" / "run.py").load(
        "article_sample", [str(inp / "hands.txt")], source="operator", log=lambda _m: None)
    assert _mod("step02_run", ROOT / "02_equity" / "run.py").main(["--dataset", "article_sample", "--procs", "1"]) in (0, None)
    _mod("step04_run", ROOT / "04_predicted_range" / "run.py").main(
        ["--dataset", "article_sample", "--steps", "fit,enrich,leakage", "--procs", "1"])
    _mod("step04_preflop", ROOT / "04_predicted_range" / "preflop.py").main(["--dataset", "article_sample"])
    out = _mod("step05_run", ROOT / "05_detection" / "run.py").analyse("article_sample", "1001", log=lambda _m: None)

    sus = out["result"]["suspicious_hands"]
    assert sus.loc[sus.category != "R", "category"].value_counts().to_dict() == {"A": 31, "B": 15, "C": 7, "D": 21}
    assert {"eq_vs_all_hands", "eq_vs_range", "eq_vs_real", "pot_odds", "range_verdict"} <= set(sus.columns)
    assert sus["eq_vs_range"].notna().mean() > 0.9 and len(sus[sus.category == "R"]) <= 3
    assert out["summary"]["escalation"]["escalate"] and out["summary"]["decision_times_restored"]
    dec = out["result"]["decisions"]
    assert {"eq_vs_range", "pot_odds"} <= set(dec.columns) and dec["think_time"].notna().all()
    page = (out["out"] / "suspicious_hands_all.html").read_text(encoding="utf-8")
    assert page.count('class="rng sev-') >= 0.9 * len(sus) and page.count('class="grid"') >= 0.9 * len(sus)
    main = (out["out"] / "suspicious_hands.html").read_text(encoding="utf-8")
    assert main.count('class="hand"') >= page.count('class="hand"') > 0      # ranked page: every flag, multiway too
    assert main.count("Equity review") >= main.count('class="rng sev-')
    # every flagged hand is on the ranked page (tab Postflop, first), in the order of the Equity review tiers; the
    # rare preflop and flop decisions follow in their tabs
    ranked = pd.read_csv(out["out"] / "suspicious_hands_ranked.csv")
    rare = pd.read_csv(out["out"] / "rare_decisions.csv")
    assert len(ranked) >= len(sus) and main.count('class="hand"') == len(ranked) + len(rare)
    order = ["BLATANT", "STRONG", "THIN", "BLUFF", "RAISE / OVERBET", "RANGE AGREES", "PLAIN MISTAKE", "NO RANGE"]
    assert ranked["equity_review"].map(order.index).is_monotonic_increasing
    assert set(ranked["hand_uid"]) >= set(sus["hand_uid"])
    # one page, three tabs, Postflop shown by default; the preflop decisions carry the opponent's preflop range
    assert '<body data-tab="postflop">' in main and 'data-show="preflop"' in main and 'data-show="all"' in main
    blocks = main.split('<div class="hand" ')[1:]
    post = [b for b in blocks if 'data-tab="postflop"' in b[:80]]
    pre = [b for b in blocks if 'data-tab="preflop"' in b[:80]]
    assert len(post) == len(ranked) + int((rare["street"] == "flop").sum()) and len(pre) == (rare["street"] == "preflop").sum() > 0
    assert all('class="grid"' in b and "actual hand" in b for b in pre)
    assert set(rare["hand_id"][rare["street"] == "flop"]).isdisjoint(ranked["hand_uid"])
    # the pair label sits next to the suspect's victims only (here the one opponent, when the pair passes the rule)
    tagged = set(re.findall(r'(?:player:|<b>Opponent:</b> )([^ <&]+) <span class="tag pair">', main))
    assert tagged == {v["opponent"] for v in out["summary"]["victims"]} <= {"1002"}
    report = (out["out"] / "report.html").read_text(encoding="utf-8")
    for title in ("Verdict", "Metrics by street", "Opponents he beat", "Article scorecard", "Observer metrics",
                  "Decision times"):
        assert f">{title}</h2>" in report
    assert all((out["out"] / f).exists() for f in ("pairs_victims.csv", "metrics_summary.csv", "decision_times.csv"))
    times = pd.read_csv(out["out"] / "decision_times.csv").set_index("action")
    assert times.loc["all", "n"] > 0 and times.loc["all", "min_s"] <= times.loc["all", "median_s"] <= times.loc["all", "max_s"]

    # published from a private field: no field size, no field ids, a neutral field name; the id map is strict
    private = tmp_path / "private.toml"
    private.write_text("[report]\nshow_field_size = false\n", encoding="utf-8")
    ids = tmp_path / "ids.csv"
    ids.write_text("case_id,ref_id\n1001,1001\n1002,1002\n", encoding="utf-8")
    pub = _mod("step05_run", ROOT / "05_detection" / "run.py").analyse(
        "article_sample", "1001", log=lambda _m: None, field_dataset="article_sample", id_map=ids,
        report_config=str(private), out_dir=str(tmp_path / "published"))
    assert pub["summary"]["field_dataset"] == "the reference field" and pub["summary"]["field_player_id"] is None
    text = "".join((pub["out"] / f).read_text(encoding="utf-8") for f in (
        "report.md", "report.html", "suspicious_hands.html", "pairs_victims.csv", "metrics_summary.csv",
        "rare_decisions.csv"))
    for leak in ("Players ranked", "n_ranked", "field_rank", "top_pct", "players_ranked", "best_other_id", "n_pool",
                 "pair #"):
        assert leak not in text, leak
    assert "| Rank" not in (pub["out"] / "report.md").read_text(encoding="utf-8")
    ids.write_text("case_id,ref_id\n1002,1002\n", encoding="utf-8")                   # no row for the suspect
    try:
        _mod("step05_run", ROOT / "05_detection" / "run.py").analyse(
            "article_sample", "1001", log=lambda _m: None, field_dataset="article_sample", id_map=ids,
            out_dir=str(tmp_path / "strict"))
        raise AssertionError("an id map without the suspect must stop the report")
    except SystemExit as exc:
        assert "no row for player 1001" in str(exc)

    # the package flags nothing: the range rule's hands (R), the rare decisions and the flags of the run on every
    # hand meet an empty frame and keep their types (pandas 2.2 warns about such a concatenation, pandas 3 turns the
    # typed columns into object); the box on his folds against the field stays at the top of the page
    run05 = _mod("step05_run", ROOT / "05_detection" / "run.py")
    ad = run05.adapter()
    find, every = ad.find_suspicious, ad.analyse_player
    blind: set = set()                                     # the scopes whose run sees no flag of the package

    def analyse_player(con, player, opponent=None, scope="flop_hu", times=None):
        ad.find_suspicious = (lambda h, d, u: find(h, d.iloc[0:0], u)) if ad.scope_name(scope) in blind else find
        return every(con, player, opponent, scope, times=times)

    monkeypatch.setattr(ad, "find_suspicious", find)       # both are put back after the test
    monkeypatch.setattr(ad, "analyse_player", analyse_player)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "The behavior of DataFrame concatenation", FutureWarning)
        for name, scopes, range_hands in (("r", {"flop_hu", "all"}, 3), ("rare", {"flop_hu", "all"}, 0),
                                          ("multiway", {"flop_hu"}, 0)):
            blind.clear()
            blind.update(scopes)
            res = run05.analyse("article_sample", "1001", range_hands=range_hands, field_dataset="article_sample",
                                out_dir=str(tmp_path / f"blind_{name}"), log=lambda _m: None)
            full, summary = res["result"]["suspicious_hands"], res["summary"]
            ranked = pd.read_csv(res["out"] / "suspicious_hands_ranked.csv")
            page = (res["out"] / "suspicious_hands.html").read_text(encoding="utf-8")
            md = (res["out"] / "report.md").read_text(encoding="utf-8")
            n_rare = summary["rare_decisions"]["preflop"] + summary["rare_decisions"]["flop"]
            assert summary["suspicious_hand_count"] == 0 and "Real cards beyond the range." in page
            assert page.count('class="hand"') == len(ranked) + n_rare
            if name == "r":
                assert summary["by_category"] == {"R": len(full)} and 0 < len(full) <= 3 and len(ranked) == len(full)
                assert pd.api.types.is_datetime64_any_dtype(full["start_utc"])
                assert all(pd.api.types.is_float_dtype(full[c]) for c in ("pot_bb", "bet_to_pot", "eq_vs_range"))
                assert summary["range_boxes"] == len(full)
            else:
                assert summary["by_category"] == {} and full.empty and md.endswith("No suspicious hands found.\n")
            if name == "rare":
                assert ranked.empty and summary["rare_decisions"]["flop"] > 0
                assert '<b>Postflop.</b> No suspicious hands found. <b>Rare flop decisions' in page
            if name == "multiway":                     # every flag of the run on every hand, without a range
                assert len(ranked) == 74 and set(ranked["equity_review"]) == {"NO RANGE"}


def test_report_without_a_suspicious_hand(tmp_path, monkeypatch):
    """Hole cards only at the showdowns, as a mined history keeps them: the sample's first 40 hands that end without a
    showdown, without their Dealt to lines. The player has hands but no suspicious hand; the report is written all the
    same, with the default scope and with --scope all."""
    monkeypatch.setenv("SU_DATA_DIR", str(tmp_path))
    blocks = (ROOT / "data" / "sample_hands.txt").read_text(encoding="utf-8").split("\n\n")
    hands = [b for b in blocks if b.strip() and "*** SHOW DOWN ***" not in b][:40]
    inp = tmp_path / "mined_sample" / "input" / "hands.txt"
    inp.parent.mkdir(parents=True)
    inp.write_text("\n\n".join("\n".join(x for x in b.split("\n") if not x.startswith("Dealt to ")) for b in hands)
                   + "\n", encoding="utf-8")
    _mod("step01_run", ROOT / "01_action_table" / "run.py").load("mined_sample", [str(inp)], source="mined",
                                                               log=lambda _m: None)
    assert _mod("step02_run", ROOT / "02_equity" / "run.py").main(["--dataset", "mined_sample", "--procs", "1"]) in (0, None)
    run05 = _mod("step05_run", ROOT / "05_detection" / "run.py")
    for scope in ([], ["--scope", "all"]):
        out = tmp_path / "report" / ("all" if scope else "default")
        assert run05.main(["--dataset", "mined_sample", "--player", "1001", "--out", str(out)] + scope) == 0
        assert all((out / f).exists() for f in ("report.md", "report.html", "suspicious_hands.html", "summary.json"))
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["meta"]["hands_analysed"] == 40 and summary["suspicious_hand_count"] == 0
        md = (out / "report.md").read_text(encoding="utf-8")   # the package's section and the range section
        assert "## Suspicious hands\n\nNo suspicious hands found." in md and md.endswith("No suspicious hands found.\n")
        assert "<p>No suspicious hands found.</p>" in (out / "report.html").read_text(encoding="utf-8")
        page = (out / "suspicious_hands.html").read_text(encoding="utf-8")
        assert "No suspicious hands found." in page and 'class="hand"' not in page
        # the empty tables keep their header: a script reads them back and filters them like any other
        sus = pd.read_csv(out / "suspicious_hands.csv")
        assert sus.empty and {"hand_uid", "category", "decision_label", "pot_bb", "eq_vs_range"} <= set(sus.columns)
        assert pd.read_csv(out / "suspicious_hands_ranked.csv").empty
