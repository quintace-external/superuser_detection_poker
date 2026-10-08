# Step 05: detection

Step 05 has two entry points on the tables of steps 01 to 04:

- `screen.py`, **population screening**: every player of a dataset, stages 1 to 5, the single-victim pair test and
  the onset changepoint (see "Population screening" below and in the root README);
- `run.py`, **the per-suspect report**: the forensic package (`superuser/` at the repository root) for one player,
  with the opponent's predicted range from step 04 on every decision it reports. The package's methodology,
  thresholds and definitions are described first, then the range extension.

```
python 05_detection/run.py --dataset article_sample                       # the sample's subject, player 1001
python 05_detection/run.py --dataset <name> --player <id>                 # one player, all his heads-up hands
python 05_detection/run.py --dataset <name> --player <id> --opponent <id> # only hands where both sat
python 05_detection/run.py --dataset <name> --player <id> --all-tables    # multiway tables too
python 05_detection/run.py --dataset article_sample --field-dataset <name> # compare with the field the range was fitted on
```

`adapter.py` builds the package's two inputs from the warehouse: decision rows (equity against the real cards of
every live opponent from step 02's `action_equity`, the bet faced, the bettor's equity, bluff success) and hand
objects (seats, hole cards, board, actions, net results). Amounts are chips at 100 per big blind. Decision times
come from `$SU_DATA_DIR/<dataset>/input/decision_times.csv` (written by `01_action_table/from_replay_json.py`;
`--times` points elsewhere), otherwise from `actions.decision_s`, otherwise they are empty and no decision is a snap
decision. The package can also be run on its own on the hand records: `python -m superuser --input <file>
--player <id>`. On the bundled sample it flags 76 hands, two more than step 05: it samples the flop equity, while
step 05 uses the exact equities of step 02, and two flop folds sit near the 10% line of rule D.

## Methodology of the forensic package

Stages, in the order the code runs them:

1. **Scope.** Hands of the player (optionally only against one opponent, only heads-up, above a blind level, in one
   currency); unusable hands are counted, not analysed.
2. **Decision rows.** One row per action with the actor's equity against the real hole cards of every player still
   in the hand (in step 05 the exact equities of step 02; the package run on its own samples the flop), the bet
   faced and the bettor's equity at his bet.
3. **Player statistics and hidden-information signals** (16 + 9, defined in `03_player_stats/README.md`).
4. **Suspicious hands**, categories A to D with decision labels, a key action and a snap-decision flag.
5. **Escalation rule**: over the last 15 days of submitted hands, suspicious hands / hands > 5% and more than 5
   suspicious hands.

| Item | Definition |
|---|---|
| Equity | Share of the pot won against every player still in the hand, from their real hole cards |
| Behind | Equity below 50% at the moment of the action |
| Perfect fold ratio | Folds made while behind ÷ all folds (postflop, turn, river) |
| Bluff | Postflop bet/raise made while behind; success = every later action by others on that street is a fold |
| Value bet / bluff faced | The last aggressor's equity at the time of their bet ≥ 50% / < 50% |
| Discrimination gap | Fold rate vs value bets − fold rate vs bluffs |
| River decision accuracy | Facing a river bet: fold when behind or continue when ahead |

**Suspicious-hand categories** (first qualifying decision per hand, priority A > B > C > D)

| Category | Rule |
|---|---|
| A, River fold when beaten | Folds the river facing a bet with 0% equity |
| B, River bluff-catch | Calls or raises a river bet while ahead |
| C, Successful river bluff | Bets the river while behind and every opponent folds |
| D, Early fold, near dead | Folds the flop or turn with under 10% equity |

**Labels.** Fold: Perfect = pair or better folded with ≤5% equity; Good = ≤5% equity, or ≤20% with a made hand;
Average = otherwise. Bluff-catch: Perfect = high card or one pair; Good = two pair; Average = trips or better.
Bluff: Perfect = high card made a pair-or-better fold; Good = opponent folded a made hand; Average = otherwise.

**Snap decision**: a Perfect decision made in ≤2 s with a pot of ≥30 bb (flag only).

Suspicious hands are sorted Perfect first, then Good, then Average, and by pot size within a label.

## Predicted-range extension

The categories use the real cards only. A strong player also folds hands that are behind, so the package asks
"was the fold correct in hindsight"; the predicted range asks "was the fold reasonable from what could be seen".
Step 04 predicts the opponent's range from his preflop line and his postflop actions, fitted on the field, and
computes the hero's equity against it (`eq_vs_range`). Step 05 puts four numbers next to each decision:

| column | source | meaning |
|---|---|---|
| `eq_vs_all_hands` | step 02 `eq_vs_random` | equity against one random hand |
| `eq_vs_range` | step 04 `action_range_equity` | equity against the opponent's predicted range |
| `eq_vs_real` | step 02 `eq_vs_real` | equity against the opponent's real cards |
| `pot_odds` | step 01 `actions.pot_odds` | equity a call needs |

`decisions.csv` carries them for every decision row; `suspicious_hands.csv` for the key decision of every
suspicious hand, with `range_verdict`: for a fold, "range says call" when `eq_vs_range - pot_odds >= 0.24`; for a
call or raise facing a bet, "range says fold" when `pot_odds - eq_vs_range >= 0.20` ("thin" below those margins).
Up to three hands that the package does not flag are added as category **R** when the range rule fires together
with the real cards: a fold with `eq_vs_range - pot_odds >= 0.24` and `eq_vs_real < 0.02`, or a call or raise with
`pot_odds - eq_vs_range >= 0.20` and `eq_vs_real >= 0.5`. R hands are listed separately and never enter the
escalation rule. In `suspicious_hands.html` every hand with a range gets an Equity review box with the three equities, the
pot odds, the ten heaviest classes of the opponent's range and the 13 x 13 class grid.

The bundled results use range tables fitted on a full-information heads-up population. Tables fitted on observer
data (see `04_predicted_range/README.md`, "Observer fit") lean toward hands that reached a showdown, so `eq_vs_range`
comes out too low and the "range says call" verdicts conservative.

The range reads the opponent's bet sizes (step 04 `[range] sizing = true`, "Bet size" in
`04_predicted_range/README.md`), so a river overbet keeps the hands that overbet, not every hand that bets.

## Outputs

`$SU_DATA_DIR/<dataset>/report/` (emptied and rewritten on every run):

| file | contents |
|---|---|
| `report.html` | the package report as one page: statistics, signals, escalation, the top flagged hands |
| `report.md` | statistics, signals, escalation verdict, the top suspicious hands, and the predicted-range table |
| `suspicious_hands.html` | every flagged hand, ranked by the Equity review (see "The ranked page"), in hand-history layout with the package's decision box and an Equity review box; the opponent's predicted range as a grid, as text and as JSON |
| `suspicious_hands_all.html` | the same hands in the package's own order |
| `suspicious_hands_ranked.csv` | the ranking: tier, severity, the equities, the pot odds and the verdict per hand |
| `suspicious_hands.csv` | all suspicious hands (A to D, then R) with every detail and the range columns |
| `decisions.csv` | every decision row with the package's fields and the four reference numbers |
| `stats.csv`, `signals.csv` | the two summary tables |
| `summary.json` | everything above in one file, plus counts by category and the escalation numbers |

## Result on the article sample

`data/sample_hands.csv` holds the article's case: the suspect's (1001) 718 heads-up hands against the victim (1002)
at $25/50, in this repository's JSON hand-record format (`01_action_table/README.md`, input variant 2: one
operator's export mapped onto neutral names). Player, table and hand ids are replaced and amounts are in dollars.
The timestamps are synthetic:
the hands keep their order, and the 250 hands of the last 15 days stay in that window, so the escalation rule gives
the same result; the dates, the gaps between hands, the sessions and the times of day are random.
`data/sample_hands.txt` is the same case in the text format, with `data/sample_hands_decision_times.csv` holding the
decision times. Through steps 01 to 05 (commands in the root README), on the 718 hands:

- 74 suspicious hands: A 31, B 15, C 7, D 21 (38 Perfect folds).
- Escalation: 27 suspicious of 250 hands in the last 15 days: escalate.
- Signals: perfect fold ratio 86.3% postflop, 92.9% on the turn, 91.2% on the river; pair-or-better folds behind
  90.3% (56 of 62); fold vs value bets 86.3% against fold vs bluffs 15.3%, a discrimination gap of 71.0 points;
  bluff index 41.1%; river decision accuracy 85.2%. The package on its own, with the flop equity sampled as in the
  article, reproduces the article's figures: 86.2% against 16.3%, a gap of 69.9 points, bluff index 40.0% (the
  article: 39.5%).
- With range tables fitted on a full-information heads-up population (one operator's heads-up hands, kept
  outside the repository), the ranked page holds all 74 flagged hands. First come 3 blatant, 4 strong and 13 thin range contradictions. The three
  blatant hands are flop and turn folds: h0758 (65% against the range at pot odds of 25%; the opponent held a pair),
  h0736 (54% at 17%; three of a kind) and h0227 (65% at 30%; a straight). Then 7 bluffs, 10 folds to a raise or an
  overbet and 37 hands where the range agrees with his decision (24 of them A and D folds, 18 labelled Perfect fold).

  | package rule | flagged | BLATANT | STRONG | THIN | BLUFF | RAISE / OVERBET | RANGE AGREES |
  |---|---|---|---|---|---|---|---|
  | A, river fold when beaten | 31 | 0 | 3 | 5 | 0 | 8 | 15 |
  | B, river bluff-catch | 15 | 0 | 0 | 2 | 0 | 0 | 13 |
  | C, successful river bluff | 7 | 0 | 0 | 0 | 7 | 0 | 0 |
  | D, early fold, near dead | 21 | 3 | 1 | 6 | 0 | 2 | 9 |

- Step 04's leakage split for the suspect alone: on his 97 turn and river responses to a bet the real cards add
  30.0% to the fit of his folds beyond the predicted range. The share is inflated on small samples, so the page
  compares it with field players measured on the same 97 responses (`--field-dataset`): median 20.8%, 90th
  percentile 29.1%; he is above 93% of 297 players. The victim, measured the same way on his 136 responses, is
  above 53% of them.
- Refitting the range tables on about three times as many hands left every flagged hand in its tier.

## The ranked page

`suspicious_hands.html` is the page for the expert. Every hand the package flags stays on it; the Equity review
removes none. Each hand carries two verdicts:

- the tag at the top (A-D) is the package's hand-set rule, read from the opponent's real cards: a fold against a
  better hand, a call against a worse one, and so on;
- the **Equity review** box under the hand is computed from the betting: the three equities (vs all hands, vs the
  predicted range, vs the real cards), the pot odds, a plain-words verdict and the opponent's predicted range: a
  13 x 13 grid, the same weights as text in a toggle of their own ("Predicted range as text", a grid and a list in %
  of the range), and a JSON block (`script.range-data`) for scripts.

The Equity review orders and colours the hands:

1. Severity: for a fold, equity vs the range minus the pot odds; for a call, the pot odds minus equity vs the range;
   counted only when the real cards made the decision right. It says how far the betting pointed the other way.
2. BLATANT (severity >= 0.30, red), STRONG (0.15-0.30, orange) and THIN (below 0.15, yellow) come first, the largest
   severity first: the betting said one thing, the opponent's real cards another, and he followed the real cards.
3. BLUFF (grey): successful river bluffs (C). The range cannot show whether he knew the opponent was weak.
4. RAISE / OVERBET (slate): folds facing a bet or raise of 150% of the pot or more. The range model reads the bet
   size but not the betting line, so its verdict there is weak evidence.
5. RANGE AGREES: the range says the same as his decision, so an honest player does the same; the package flags the
   hand because the opponent's real hand made the decision right. PLAIN MISTAKE: the range and the real cards both
   point the other way.
6. NO RANGE: hands without a predicted range (the range model covers heads-up decisions; with `--all-tables` the
   hands flagged on multiway tables land here).

The top of the page explains why rules A and D also flag ordinary folds: they read only the real cards, whatever
the bet size or the pot odds. The rules find the candidates; the Equity review puts the strongest first.

`suspicious_hands_all.html` shows the heads-up hands in the package's own order, and `suspicious_hands_ranked.csv` lists
the ranking with the equities, the tier and the verdict. `example/` holds the outputs for the bundled case: the
ranked page, `report.html`, `report.md`, the CSVs and `summary.json`.

## Thresholds

| rule | value |
|---|---|
| Category A, river fold when beaten | equity 0 |
| Category B, river bluff-catch | equity ≥ 0.999 |
| Category C, river bluff | equity < 0.5 and every opponent folds |
| Category D, early fold | flop or turn, equity < 0.10 |
| Snap decision | Perfect label, ≤ 2 s, pot ≥ 30 bb |
| Escalation | last 15 days, suspicious / hands > 5% and suspicious > 5 |
| Category R (this step) | fold: range margin ≥ 0.24 and real equity < 0.02; call: range margin ≥ 0.20 and real equity ≥ 0.5 |

The escalation rule is fixed. A good regular also folds beaten hands and calls bluffs, so it can fire for honest
players: on the bundled case the victim, run as the player, escalates too (24 of 250). On a real platform escalate
a player when he sits in the top tail of comparable players on several signals at once (for example the top 5% on
3 of the 9, over 15- and 30-day windows, compared only with players measured on as many hands); `screen.py`
implements the population-calibrated rule (Stage 5).

## Population screening

`screen.py` reads the decisions and hands of every player through `sql/scored_decisions.sql` and
`sql/player_hands.sql`, computes stages 1 to 5 in the windows `all`, `last30` and `last15`, and writes into the
dataset's warehouse:

| table | contents |
|---|---|
| `player_scores` | one row per player x window: volume and style, the scorecard, the range-adjusted test (`z_b`), the Stage 1-2 flags, the win-rate runs percentile and the changepoint (`all` rows), the Stage 5 verdicts |
| `pair_scores` | ordered pairs (i -> j) that met heads-up: the scorecard of i against j, the split test, the opponent-controlled residual, its robust z and the Benjamini-Hochberg q-value |
| `window_scores` | weekly series per player (decisions while dead and while winning, their excess over the range, `z_b`) |
| `flagged_hands` | Stages 3-4: the flagged hands under both rule sets, with labels and the risk scores |
| `escalations` | Stage 5: the fixed and the population-calibrated verdict per player and rule set |
| `report_meta` | the config hash, the population sizes and the counts tested |

Every setting is in `config/step05_screening.toml`, `[step05]`: the windows, the bands and clusters, the Stage-1
rule, the hand-test margins, the risk weights, the escalation rules, the pair test and the onset series. The
implementation of the stages (`sudetect/screening.py`) is a replaceable module: `[step05.impl] module`.

`eval_planted.py` scores the screen against the simulator's answer key (`results/eval_<dataset>.md`) and
`eval_quick.py` prints the planted players' rows; both are evaluation code and read the answer key, the screen
never does. `tests/test_screen_formulas.py` checks the formulas on hand-made rows and
`tests/test_screen_parity.py` checks, on `sim_small`, that the screen's article rule set gives the same signals,
flagged hands, labels and escalation as the per-suspect package.

## Data notes (hand records)

- `act` is one of check, fold, call, bet, raise, allin, walk; the integer codes 1 to 7, in that order, are accepted
  too. Anything else is read as an unknown action, and step 01 skips such a hand.
- Bet and raise amounts are street totals; call amounts are chips added. An all-in counts as aggressive only
  when it exceeds the current highest street total.
- A card is {`rank` 0-12 for 2 ... A, `suit` 0-3}; a missing key counts as 0 (a 2, the first suit).
- Hands with invalid cards (rank above 12) are dropped and counted as unparseable.
- `think_s` is the time the player took, in seconds; 0 to 1 s values may be pre-selected actions.
