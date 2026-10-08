# Superuser detection

Detection of superusers in online no-limit hold'em: players who see their opponents' hole cards while they play,
through a compromised client, a leaked server feed or an insider account. Such a player folds when he is beaten,
calls when the opponent bluffs and bluffs when the opponent is weak, far more often than his visible information
allows. This repository measures that from hand histories in which the operator knows every hole card.

Each step reads the tables of the previous ones from a DuckDB warehouse and writes its own. Steps 01 to 04 build the
action table, the equities and the opponent's predicted range. Step 05 has two entry points:

- **population screening** (`05_detection/screen.py`): every player of a window, compared with comparable players,
  stages 1 to 5 (win-rate outliers by hand-count band and style cluster, the hidden-information signals, the hand
  tests with a risk score, escalation), plus the single-victim pair test and the onset changepoint;
- **the per-suspect report** (`05_detection/run.py`): one player in depth, every flagged hand ranked by the
  opponent's predicted range, with the classic statistics, nine hidden-information signals and an escalation verdict.

A simulator (step 00) generates a population with planted superusers and an answer key, which is how the screening
is evaluated without real population data.

## Data it needs

- Hand histories of no-limit hold'em with every dealt player's hole cards, folded hands included (the operator's
  view). Two input formats are supported (step 01): PokerStars-style text, and JSON hand records.
- Hand start times (for the time windows and the 15-day escalation window); decision times if available (for the
  snap-decision flag).
- For the predicted range, a population of hands of the same game to fit the field tables on (step 04). Ideally
  full-information hands; observer hand histories (cards only at the showdown) work with a known bias.
- For the screening, the population itself: every player of the window, not only the suspects.

Requirements: Python 3.11 or newer, numpy, pandas, pyarrow, duckdb, tabulate; matplotlib for the figures of the
step-04 examples. Optional: `pyrust_poker` (a faster equity backend). Tests: pytest; phevaluator and pyrust_poker for
the cross-checks of the evaluator and the equity backends (without them those tests are skipped). `pip install -e .`
installs the code and the `superuser-report` command.

## Pipeline

Every warehouse lives in `$SU_DATA_DIR/<dataset>.duckdb`. `<field>` stands for a full-information heads-up
population (every hole card known), loaded by step 01 the same way; it is not part of this repository.

```
export SU_DATA_DIR=/path/to/data

# 00 simulator (optional): a synthetic population with planted superusers and its answer key
python 00_simulator/simulate.py --dataset sim_small
python 01_action_table/run.py --dataset sim_small --input "$SU_DATA_DIR/sim_small/hands/*.txt" --source sim

# 01 action table: JSON hand records -> text + decision times -> hands, hand_players, actions
python 01_action_table/from_replay_json.py --input data/sample_hands.csv --out $SU_DATA_DIR/article_sample/input
python 01_action_table/run.py --dataset article_sample --input $SU_DATA_DIR/article_sample/input/hands.txt --source operator

# 02 equity: equity vs all hands and vs the real cards for every postflop decision
python 02_equity/run.py --dataset article_sample

# 03 player statistics: 16 classic statistics + 9 hidden-information signals per player
python 03_player_stats/run.py --dataset article_sample

# 04 predicted range: fit the field tables on a population, then equity vs the opponent's predicted range
python 04_predicted_range/run.py --dataset <field> --steps fit
python 04_predicted_range/run.py --dataset article_sample --fit-from <field> --steps fit,enrich,leakage

# 05 screening: stages 1-5 over every player of a population (here the simulator's, fitted on itself)
python 04_predicted_range/run.py --dataset sim_small --steps fit,enrich,leakage
python 05_detection/screen.py --dataset sim_small
python 05_detection/eval_planted.py --dataset sim_small      # simulator only: the screen against the answer key

# 05 per-suspect report, ranked by the Equity review; --field-dataset adds the field comparison
# (it needs steps 02 and 04 --steps enrich,leakage run on <field> as well)
python 05_detection/run.py --dataset article_sample --field-dataset <field>
```

Without a population, step 04 can fit on the dataset itself when all its hands are full-information
(`python 04_predicted_range/run.py --dataset article_sample --steps fit,enrich,leakage`, then step 05 without
`--field-dataset`). The "field" of that fit is the two players themselves, so the ranges are rough and there is no
field comparison. On the bundled case it ranks 6 BLATANT, 8 STRONG, 12 THIN, 7 BLUFF, 10 RAISE / OVERBET and 31 RANGE AGREES; the three BLATANT hands of the published page (h0758, h0736, h0227) are BLATANT there too.

| step | output | detail |
|---|---|---|
| 00 | `$SU_DATA_DIR/<dataset>/hands/*.txt`, `ground_truth.json`, `population.json` | `00_simulator/README.md` |
| 01 | `hands`, `hand_players`, `actions` (hand, player-in-hand and action level) | `01_action_table/README.md` |
| 02 | `action_equity` (`eq_vs_random` = vs all hands, `eq_vs_real`), `hand_equity` | `02_equity/README.md` |
| 03 | `player_stats` | `03_player_stats/README.md` |
| 04 | field tables `rng_*`, `action_range_equity` (`eq_vs_range`) | `04_predicted_range/README.md` |
| 05 screening | `player_scores`, `pair_scores`, `window_scores`, `flagged_hands`, `escalations`, `report_meta` | `05_detection/README.md` |
| 05 report | `$SU_DATA_DIR/<dataset>/report/`: report.html, report.md, suspicious_hands.html, CSVs, summary.json | `05_detection/README.md` |

## Result on the bundled sample

`data/sample_hands.csv` holds the article's case: the suspect's (1001) 718 heads-up hands against the victim (1002)
at $25/50, in this repository's JSON hand-record format (`01_action_table/README.md`, input variant 2: one
operator's export mapped onto neutral names). Player, table and hand ids are replaced and amounts are in dollars. The
timestamps are synthetic: the hands keep their order, and the 250 hands of the last 15 days stay in that window, so
the escalation rule gives the same result; the dates, the gaps between hands, the sessions and the times of day are
random. `data/sample_hands.txt` is the same case in the text format, with `data/sample_hands_decision_times.csv`
holding the decision times. Step 01 loads all 718 hands and every invariant passes.

- The forensic package on its own (`python -m superuser`, flop equity sampled as in the article) reproduces the
  article's scorecard: fold vs value bets 86.2% against fold vs bluffs 16.3% (a discrimination gap of 69.9 points),
  pair-or-better folds while behind 90.3% (56 of 62), river decision accuracy 85.2%, bluff index 40.0% (the
  article: 39.5%). It flags 76 hands and escalates: 29 suspicious of the 250 hands of the last 15 days.
- Step 05, with the exact equities of step 02, flags 74 hands (A 31, B 15, C 7, D 21) and escalates (27 of 250);
  fold vs bluffs 15.3%, gap 71.0 points. The article counts 757 hands and 72 flagged hands; the bundled sample is the
  suspect's 718 heads-up hands against the victim, so the counts differ slightly.
- With range tables fitted on a full-information heads-up population (one operator's heads-up hands, kept outside
  the repository), every flagged hand stays on the page and the Equity review ranks them: 3 blatant, 4 strong and 13
  thin range contradictions first, then 7 bluffs, 10 folds to a raise or an overbet and 37 hands where the range
  agrees with his decision (74 in all).
- On his 97 turn and river responses to a bet, knowing the opponents' real cards explains his folds better than for
  93% of field players measured on the same number of responses (30.0% added to the fit for him, 20.8% for the median
  player, 297 players). The victim, measured the same way on his 136 responses, is above 53% of them: an ordinary
  player.

`05_detection/example/` holds the outputs for this case: `report.html` (statistics, signals, escalation, the top
flagged hands), `suspicious_hands.html` (every flagged hand with the opponent's predicted range as a grid, as text
and as JSON) and the CSVs.

The article's population figures (the suspect's win-rate percentile among 193 heads-up players, his percentiles on
the signals, the share of fold-rate variation the predicted range explains) come from the operator's population; they
cannot be reproduced from this repository. The screening code that produces such figures runs on any population, and
on the simulator in the tests.

## Population screening

`05_detection/screen.py` scores every player of a dataset, in three time windows (`all`, `last30`, `last15`):

1. **Stage 1, win-rate outliers among comparable players.** Five statistics (bb/100, W$SD, low WTSD, fold to river
   bets, river calls won) as percentiles inside the player's window x hand-count band (250-499, 500-999,
   1,000-1,499, 1,500-2,499, 2,500+) x style cluster (VPIP x PFR tertiles, or k-means); flagged in the top 10% of
   the composite, or above the 95th percentile on 3 of the 5.
2. **Stage 2, the hidden-information signals** of the Stage-1 survivors: the oracle-fold ratio on the turn and the
   river and the bluff index, top 10%.
3. **Stage 3, the hand tests**: categories A-D under two rule sets, the article's (real cards and pot odds) and a
   range-conditioned one (the predicted range must also say the decision was wrong on what could be seen).
4. **Stage 4, a risk score per flagged hand**: category weight x hand-strength factor x ln(1 + pot in bb) x snap
   factor, scaled so that the player's riskiest hand scores 100, and a version based on the field's probability of
   the action, comparable across players.
5. **Stage 5, escalation over the last 15 days**: the article's fixed rule (more than 5% and more than 5 flagged
   hands) and a population-calibrated rule (the flagged share above the population's 99th percentile in the same
   hand-count band).

Next to the stages: the range-adjusted test (`z_b`, folds while dead and calls while winning against what the range
predicts), the win-rate percentile over random runs of the same length (720 hands), weekly series with a changepoint
(onset), and ordered pairs with an opponent-controlled residual tested with Benjamini-Hochberg (single victim).
On the simulator (`sim_demo`: 300 players, about 400,000 hands, seven planted superusers; the evaluation in
`05_detection/results/eval_sim_demo.md`), the range-adjusted screen flags the full-table plants that use the
information on 100% and 50% of their decisions (z_b 11.5 and 5.3; not the one at 25%) and the single-victim plants at
92% and 50%; the 20% single-victim plant stays below the player-level bar, and the pair test ranks its pair first of
308 (q < 0.0001). The onset plant shows as a changepoint (z 4.9) in the week it starts. Among the 285 honest players
the screen flags none; win rate alone (Stage 1) flags 26 of them, and 12 of 29 strong honest regulars, none of whom
the decision signals flag. These are simulator numbers, not detection rates for real data.

On the bundled case the screen has no population (two players): by the article's fixed rule both players escalate,
by the range-conditioned rule neither does. One case is what the per-suspect report is for.

## Implementation scenarios

| exposure | what the cheat sees | how to run |
|---|---|---|
| Full table | every opponent's cards at every hand | `screen.py`: Stages 1-2 and the `z_b` screen over everyone; then `run.py --player <id>` for the flagged players |
| Single victim | one opponent's cards (a compromised account, a partner) | `screen.py`: `pair_scores` (the pair test, BH); then `run.py --player <id> --opponent <victim>`, so the signals are not diluted by the other opponents |
| Onset after normal play | nothing at first, every card later | `screen.py`: the `last30` and `last15` windows, the weekly series and the changepoint; the escalation rule looks at the last 15 days |

## Thresholds

Every cutoff is a starting point, to be tuned on reviewed cases of your own population, stakes and review capacity.
The screening's values are in `config/step05_screening.toml`; the per-suspect package's:

| rule | value |
|---|---|
| Behind / ahead | equity against the real cards below / at least 50% |
| A, river fold when beaten | river fold facing a bet with equity 0 |
| B, river bluff-catch | river call or raise facing a bet with equity ≥ 0.999 |
| C, successful river bluff | river bet with equity < 0.5, every opponent folds |
| D, early fold near dead | flop or turn fold with equity < 0.10 |
| Labels (folds) | Perfect: pair or better, equity ≤ 5%; Good: ≤ 5%, or ≤ 20% with a made hand; Average: otherwise |
| Snap decision | Perfect label, ≤ 2 s, pot ≥ 30 bb |
| Escalation | last 15 days: flagged / hands > 5% and more than 5 flagged hands |
| R, range rule (step 05) | fold: eq_vs_range - pot odds ≥ 0.24 and eq_vs_real < 0.02; call: pot odds - eq_vs_range ≥ 0.20 and eq_vs_real ≥ 0.5 |

The escalation rule is a fixed count. A sound player folds beaten hands and calls bluffs often enough over 15 days
to pass 5% and 5 hands: on the bundled case the victim, run as the player, escalates too (24 of 250). On a real
platform escalate a player when he sits in the top tail of comparable players on several signals at once, for
example the top 5% on 3 of the 9, over 15- and 30-day windows, compared only with players measured on as many
hands; `screen.py` implements the population-calibrated rule.

## Predicted range: what the tables are fitted on

The bundled results use range tables fitted on a full-information heads-up population (every player's hole cards
known; the hands stay outside the repository). Refitting on about three times as many hands moved the equity against
the range by 0.2 points at the median (90% of the case's decisions within 1.3 points), and every flagged hand kept its
tier. Fitted on observer hand histories instead (hole cards only at the showdown), the tables see showdown hands
only: the ranges come out stronger than the true ranges of each betting line and the "range says call" verdicts
conservative. `04_predicted_range/variants/observer_fit.toml` is the setting for that case; only `--fit-from`
changes.

## Pictures (ELI5)

HTML pages to open in a browser: `eli5.html` (how a superuser gets caught), `eli5_technical.html` (the data
pipeline map), and one per step: `01_action_table/eli5.html`, `02_equity/eli5.html`, `03_player_stats/eli5.html`,
`04_predicted_range/eli5.html`, `05_detection/eli5.html`.

## Repository layout

```
00_simulator/        simulate.py: the synthetic population with planted superusers (settings in config/sim.toml)
01_action_table/     run.py (text loader), from_replay_json.py (JSON hand records), sql/, tests/
02_equity/           run.py, bench.py, examples.py, sql/, tests/
03_player_stats/     run.py
04_predicted_range/  run.py, examples.py, variants/ (observer_fit, cap240), sql/, tests/
05_detection/        screen.py (population screening, sql/), eval_planted.py and eval_quick.py (simulator answer
                     key), results/, run.py and adapter.py (per-suspect report), tests/, example/ (the case's outputs)
config/              defaults.toml (step 01), step02_equity.toml, step04_range.toml, step05_screening.toml, sim.toml
data/                sample_hands.csv (the article's case, JSON hand records), sample_hands.txt (the same, text
                     format), sample_hands_decision_times.csv
sudetect/            library: cards, evaluator, equity, handclass, hhparse, replay, positions, situations,
                     warehouse, rangemodel, metrics, screening; sim/ (rules engine, hand writer, player policies,
                     population, calibration); README_dev.md
superuser/           the per-suspect forensic package: parser, decisions, stats, signals, suspicious,
                     pipeline, report, html_report, cli (python -m superuser)
tests/               library, simulator and package tests
```

## Tests

```
python -m pytest -q
```

Tests that need a simulated warehouse (the parity of the screening with the per-suspect package) are skipped until
`sim_small` is built with steps 00 to 04.

## Licence

MIT, see `LICENSE`.
