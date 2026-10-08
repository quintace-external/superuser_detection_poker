"""Step 05 on the article-case sample: the package's verdicts plus the predicted-range columns and boxes."""

from __future__ import annotations

import importlib.util
import sys
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
        ["--dataset", "article_sample", "--steps", "fit,enrich", "--procs", "1"])
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
    # every flagged hand is on the ranked page, in the order of the Equity review tiers
    ranked = pd.read_csv(out["out"] / "suspicious_hands_ranked.csv")
    assert len(ranked) >= len(sus) and main.count('class="hand"') == len(ranked)
    order = ["BLATANT", "STRONG", "THIN", "BLUFF", "RAISE / OVERBET", "RANGE AGREES", "PLAIN MISTAKE", "NO RANGE"]
    assert ranked["equity_review"].map(order.index).is_monotonic_increasing
    assert set(ranked["hand_uid"]) >= set(sus["hand_uid"])
