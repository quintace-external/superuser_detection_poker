# Superuser detection

Detection of superusers in online no-limit hold'em: players who see their opponents' hole cards while they play,
through a compromised client, a leaked server feed or an insider account. Such a player folds when he is beaten,
calls when the opponent bluffs and bluffs when the opponent is weak, far more often than his visible information
allows. This repository measures that from hand histories in which the operator knows every hole card.

Each step reads the tables of the previous ones from a DuckDB warehouse and writes its own. Steps 01 to 04 build the
action table, the equities and the opponent's predicted range. Step 05 has two entry points:

- **population screening** (`05_detection/screen.py`): every player of a window, compared with comparable players,
  stages 1 to 5 (win-rate outliers by hand-count band and style cluster, the hidden-information signals, the hand
  tests with a risk score, escalation), plus the single-victim pair test and the onset changepoint. It also writes
  the table `pair_outliers` (every ordered pair "A against B" of the dataset, scored and ranked against all the other
  pairs), the observer metrics (the `obs_*` columns of `player_scores`: money and showdowns, no hidden cards) and the
  escalation rule set `street_tails` (the top tail of the preflop, flop and turn + river tests on two of the three
  streets at once);
- **the per-suspect report** (`05_detection/run.py`): one player in depth, every flagged hand ranked by the
  opponent's predicted range, with the classic statistics, nine hidden-information signals and an escalation verdict.
  Against a reference field (`--field-dataset`, by default the dataset itself) it adds the street-tails verdict, a
  "Metrics by street" table (his value, percentile and rank on the tests of each street), the opponents he beats
  beyond his own level (the victims section, from `pair_outliers`), the article's scorecard and the observer metrics.
  Its page of hands has three tabs. Postflop, the default, holds the hands flagged by the article's rules A-D (and
  the range rule R) in the order of the Equity review, then the rare flop decisions; Preflop holds the rare preflop
  decisions with the opponent's preflop range; All holds both. A rare decision is one that the opponent's real cards
  made right and that the other players of the field seldom make at the same strength against the same line.

A simulator (step 00) generates a population with planted superusers and an answer key, which is how the screening
is evaluated without real population data.

## Data it needs

- Hand histories of no-limit hold'em with every dealt player's hole cards, folded hands included (the operator's
  view). Two input formats are supported (step 01): PokerStars-style text, and JSON hand records. The text layout
  line by line, what may vary, what is dropped and why: `docs/HAND_HISTORY_FORMAT.md`; `data/sample_hands.txt` is
  a complete file in that layout.
- Hand start times (for the time windows and the 15-day escalation window); decision times if available (for the
  snap-decision flag).
- For the predicted range, a population of hands of the same game to fit the field tables on (step 04). Ideally
  full-information hands; observer hand histories (cards only at the showdown) work with a known bias.
- For the screening, the population itself: every player of the window, not only the suspects.
- **Without full information the results are distorted.** Hand histories with hole cards only at the showdowns (hands
  recorded by an observer at the tables) load and run through every step, but every test that needs the cards behind
  a fold goes blind or biased, and the report still prints those sections: on such data its verdict, hand flags,
  scorecard and signals are distorted. Use full-information hands for a verdict, or adapt the method to hands without
  showdowns (`docs/HAND_HISTORY_FORMAT.md`, `docs/THIRD_PARTY_MINING.md`).

Requirements: Python 3.11 or newer, numpy, pandas, pyarrow, duckdb, tabulate; matplotlib for the figures of the
step-04 examples. Optional: `pyrust_poker` (a faster equity backend). Tests: pytest; phevaluator and pyrust_poker for
the cross-checks of the evaluator and the equity backends (without them those tests are skipped). `pip install -e .`
installs the code and the `superuser-report` command.

## Your own hands: where they go, what to run, where the result is

1. **One folder of hand histories.** Step 01 reads every `.txt` file under the folder, sub-folders included:
   PokerStars-style text, with a `Dealt to <player> [Xx Yy]` line for every seat when every hole card is known, which
   makes the hand full-information. `docs/HAND_HISTORY_FORMAT.md` shows the layout line by line, what may vary, what
   is dropped and why, and how to check your files; `data/sample_hands.txt` is a file in that layout. JSON hand
   records go through `01_action_table/from_replay_json.py` first, which writes the text and the decision times
   (`01_action_table/README.md`, input variant 2).
2. **The steps**, from the repository root; `mine` is the name of the dataset:

   ```
   # bash
   export SU_DATA_DIR=/path/to/warehouses              # mine.duckdb and its folders land here
   python 01_action_table/run.py --dataset mine --input /path/to/hands --source operator
   python 02_equity/run.py --dataset mine
   python 04_predicted_range/run.py --dataset mine --steps fit,enrich,leakage
   python 04_predicted_range/preflop.py --dataset mine
   python 05_detection/screen.py --dataset mine
   ```

   ```
   # Windows PowerShell
   $env:SU_DATA_DIR = "D:\warehouses"
   python 01_action_table/run.py --dataset mine --input D:\hands --source operator
   python 02_equity/run.py --dataset mine
   python 04_predicted_range/run.py --dataset mine --steps "fit,enrich,leakage"
   python 04_predicted_range/preflop.py --dataset mine
   python 05_detection/screen.py --dataset mine
   ```

   Step 04 fits the range tables on the dataset itself, which needs every hand full-information; for hands collected
   by an observer at the tables (cards only at the showdown) use the mined route of `docs/HAND_HISTORY_FORMAT.md`
   ("Try it") and `docs/THIRD_PARTY_MINING.md`. Step 03
   (`player_stats`) is optional: neither the screen nor the report reads it.
3. **The suspects.** In `player_scores`, rows `time_window = 'all'`: `esc_street_tails` (escalated by the street-tails
   rule), `street_hits` (how many of the three streets hit), `victims_n` and `best_pair_*` (his most suspicious pair);
   in `pair_outliers`, the pairs at the top of `field_rank`. The queries are in `docs/FROM_ZERO.md`, sections 9 and 11.
   On hands with cards only at the showdowns the escalation rules rest on tests that go blind, and a victim label
   needs many hands per pair, so both can come out empty: read
   [When nothing passes the bars](docs/THIRD_PARTY_MINING.md#when-nothing-passes-the-bars) in
   `docs/THIRD_PARTY_MINING.md` before you take anyone from the top of a list.
4. **One report per suspect**, each in a folder of its own (`--out`; the report empties the folder before it writes):

   ```
   # bash: the suspects' player ids
   for P in PLAYER_1 PLAYER_2; do
     python 05_detection/run.py --dataset mine --player "$P" --out "$SU_DATA_DIR/mine/reports/$P"
   done
   ```

   ```
   # Windows PowerShell
   foreach ($p in "PLAYER_1", "PLAYER_2") {
     python 05_detection/run.py --dataset mine --player $p --out "$env:SU_DATA_DIR\mine\reports\$p"
   }
   ```

   Each folder holds `report.html` (start there), `suspicious_hands.html` (the hands: Postflop, Preflop, All) and the
   CSVs listed in `05_detection/README.md`, "Outputs". Without `--out` the report goes to
   `$SU_DATA_DIR/<dataset>/report/`, emptied on every run. For a report made against a private field and shown
   outside it, `show_field_size = false` in `config/step05_report.toml` leaves the field's size and the field's own
   player ids out of what the report prints (`05_detection/README.md`, "The report against a field").

`docs/FROM_ZERO.md` walks the same path step by step: what each step writes, picking suspects with SQL, every pair of
the field, hashed ids, a reference field and what to do when something is missing. `docs/THIRD_PARTY_MINING.md` is the
path for hands collected by an observer at the tables.

The bundled sample stays the default input: `python -m superuser` (the forensic package on its own, one command) and
`from_replay_json.py` read `data/sample_hands.csv`, and the report's default `--player` is the sample's suspect, 1001.
The whole pipeline on it in one go:

```
# bash
export SU_DATA_DIR=/path/to/warehouses
python 01_action_table/from_replay_json.py --out "$SU_DATA_DIR/article_sample/input" &&
python 01_action_table/run.py --dataset article_sample --input "$SU_DATA_DIR/article_sample/input/hands.txt" --source operator &&
python 02_equity/run.py --dataset article_sample &&
python 04_predicted_range/run.py --dataset article_sample --steps fit,enrich,leakage &&
python 04_predicted_range/preflop.py --dataset article_sample &&
python 05_detection/screen.py --dataset article_sample &&
python 05_detection/run.py --dataset article_sample
```

```
# Windows PowerShell: the lines run one after another
$env:SU_DATA_DIR = "D:\warehouses"
python 01_action_table/from_replay_json.py --out "$env:SU_DATA_DIR\article_sample\input"
python 01_action_table/run.py --dataset article_sample --input "$env:SU_DATA_DIR\article_sample\input\hands.txt" --source operator
python 02_equity/run.py --dataset article_sample
python 04_predicted_range/run.py --dataset article_sample --steps "fit,enrich,leakage"
python 04_predicted_range/preflop.py --dataset article_sample
python 05_detection/screen.py --dataset article_sample
python 05_detection/run.py --dataset article_sample
```

The report lands in `$SU_DATA_DIR/article_sample/report/`. Fitted on the sample itself the ranges are rough, and
every comparison with the field compares the suspect with one other player (see "Pipeline").

## Your local database

Steps 01 to 04 and the screen write into one DuckDB file per dataset, `$SU_DATA_DIR/<dataset>.duckdb` (`work/` in the
repository when `SU_DATA_DIR` is not set); the per-suspect report reads it and writes files of its own. The tables:

| step | tables |
|---|---|
| 01 | `hands`, `hand_players`, `actions`, `parse_funnel`; views `decisions`, `hu_decisions` |
| 02 | `action_equity`, `hand_equity`, `equity_settings`; views `actions_eq`, `decisions_eq`, `hu_decisions_eq`, `hand_players_eq` |
| 03 | `player_stats` |
| 04 | the field tables `rng_lik_post`, `rng_lik_size`, `rng_prior_pre`, `rng_field`, `rng_player_tilt`, `rng_meta`; `action_range_equity`, `rng_leak_matrix`; `preflop.py`: `action_range_equity_pre` |
| 05 | `player_scores`, `pair_scores`, `pair_outliers`, `window_scores`, `flagged_hands`, `escalations`, `street_flags`, `report_meta`, `scored_decisions` |

Next to the file, `$SU_DATA_DIR/<dataset>/` holds the converted input, step 02's staged parts and the default report
folder, and `$SU_DATA_DIR/<dataset>_step04/` step 04's scored chunks and `summary.json`.

Open it read-only: `duckdb -readonly "$SU_DATA_DIR/mine.duckdb"` (a client at least as new as the `duckdb` Python
package that wrote the file) or `duckdb.connect(path, read_only=True)` in Python, and close such a session before a
step writes to the same file. Queries to start from: the suspects and their pairs in `docs/FROM_ZERO.md` (sections 9
and 11), the observer metrics in `docs/THIRD_PARTY_MINING.md`; every step's README documents its columns. An AI
assistant that can run SQL on the file can draft a report from it: check its numbers against the tables before anyone
acts on them.

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

# 02 equity: equity vs all hands and vs the real cards for every postflop decision, and for every preflop
# decision with one live opponent
python 02_equity/run.py --dataset article_sample

# 03 player statistics: 16 classic statistics + 9 hidden-information signals per player
python 03_player_stats/run.py --dataset article_sample

# 04 predicted range: fit the field tables on a population, then equity vs the opponent's predicted range
python 04_predicted_range/run.py --dataset <field> --steps fit
python 04_predicted_range/run.py --dataset article_sample --fit-from <field> --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset article_sample --fit-from <field>   # preflop: equity vs the line's range

# 05 screening: stages 1-5 over every player of a population (here the simulator's, fitted on itself)
python 02_equity/run.py --dataset sim_small
python 04_predicted_range/run.py --dataset sim_small --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset sim_small
python 05_detection/screen.py --dataset sim_small
python 05_detection/eval_planted.py --dataset sim_small      # simulator only: the screen against the answer key

# 05 per-suspect report, ranked by the Equity review; --field-dataset is the field he is compared with (default:
# the dataset itself; run steps 02, 04 with preflop.py and screen.py on <field> as well, and pass --id-map when
# the two datasets name the same players differently)
python 05_detection/run.py --dataset article_sample --field-dataset <field>
```

Without a population, step 04 can fit on the dataset itself when all its hands are full-information
(`python 04_predicted_range/run.py --dataset article_sample --steps fit,enrich,leakage`, then step 05 without
`--field-dataset`). The "field" of that fit is the two players themselves, so the ranges are rough, and the report's
comparisons with the field compare the suspect with one other player. On the bundled case it ranks 6 BLATANT,
8 STRONG, 12 THIN, 7 BLUFF, 10 RAISE / OVERBET and 31 RANGE AGREES; the three BLATANT hands of the published page
(h0758, h0736, h0227) are BLATANT there too.

| step | output | detail |
|---|---|---|
| 00 | `$SU_DATA_DIR/<dataset>/hands/*.txt`, `ground_truth.json`, `population.json` | `00_simulator/README.md` |
| 01 | `hands`, `hand_players`, `actions` (hand, player-in-hand and action level) | `01_action_table/README.md` |
| 02 | `action_equity` (`eq_vs_random` = vs all hands, `eq_vs_real`), `hand_equity` | `02_equity/README.md` |
| 03 | `player_stats` | `03_player_stats/README.md` |
| 04 | field tables `rng_*`, `action_range_equity` (`eq_vs_range`), `action_range_equity_pre` (preflop) | `04_predicted_range/README.md` |
| 05 screening | `player_scores`, `pair_scores`, `pair_outliers`, `window_scores`, `flagged_hands`, `escalations`, `street_flags`, `report_meta`, `scored_decisions` | `05_detection/README.md` |
| 05 report | `$SU_DATA_DIR/<dataset>/report/` or `--out`: report.html, report.md, suspicious_hands.html (tabs Postflop / Preflop / All), CSVs (among them rare_decisions, pairs_victims, metrics_summary, decision_times), summary.json | `05_detection/README.md` |

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
- The report's comparisons with a field (the street-tails verdict, the metrics by street, the opponents he beat, the
  article's scorecard, the observer metrics) and the rare preflop and flop decisions run on this case too. With the
  sample as its own field they compare the suspect with the victim alone (the observer metrics rank neither of them:
  both have fewer hands than `[step05.observer] min_hands`): they show what each section holds, not where he stands
  in a population.

`05_detection/example/` holds the outputs for this case: `report.html` (statistics, signals, escalation, the top
flagged hands), `suspicious_hands.html` (every flagged hand with the opponent's predicted range as a grid, as text
and as JSON) and the CSVs. Its comparisons with a field were made against a reference field that is a sample of a
larger full-information population; its percentiles are within that sample.

The article's population figures (the suspect's win-rate percentile among 193 heads-up players, his percentiles on
the signals, the share of fold-rate variation the predicted range explains) come from the operator's population; they
cannot be reproduced from this repository. The screening code that produces such figures runs on any population, and
on the simulator in the tests.

## What is measured

The primary metrics, on the decisions of heads-up confrontations (q = equity against the opponent's real cards, r =
equity against his predicted range). The screen writes them to `player_scores` (the pair score to `pair_outliers`);
the report sets the suspect against the field on them in its sections "Metrics by street", "Opponents he beat" and
"Observer metrics".

| street | metric | column | what it measures |
|---|---|---|---|
| Preflop | preflop z | `pf_z`, `pf_zr` | (fold z + aggression z) / √2; `pf_zr` is `pf_z` robust-standardised on the population |
| Preflop | fold z | `pf_fold_cmh_z` | folds to a raise more often when behind the real hand (q < 40%) than when ahead (q ≥ 55%), inside the same situation and 5% band of r |
| Preflop | aggression z | `pf_aggr_z` | raises more often when ahead of the real hand than when behind, the same way, on every preflop decision |
| Preflop | raw gaps | `pf_fold_gap`, `pf_aggr_gap` | the same two differences as plain rates, without r; honest players have positive gaps too, so the z values carry the test |
| Preflop | light re-raise success | `pf_rr_light_succ` | the share of his re-raises with a hand weak against the opponent's line (r < 45%) that the opponent folds to at once |
| Preflop | 3-, 4- and 5-bet success | `pf_3bet_succ`, `pf_4bet_succ`, `pf_5bet_succ` | the share of his re-raises the opponent folds to at once; needs no hidden cards |
| Flop | flop excess discrimination | `fl_e_disc` | folds when behind (q < 20%) and continues when winning (q ≥ 50%) beyond the field at the same range decile |
| Flop | flop z | `z_b_flop` | the range-adjusted test on flop responses to a bet |
| Turn + river | excess discrimination | `e_disc` | the same on the turn and the river, behind = dead (q < 2%); an effect size, it does not grow with the sample |
| Turn + river | excess folds while dead | `x_dead` | fold rate when dead minus the field's at the same street and range decile |
| Turn + river | excess continues while winning | `x_win` | continue rate when winning minus the field's |
| Turn + river | range-adjusted test | `z_b` | z-test of the two excesses, centred on the population |
| Pairs | pair score | `score` in `pair_outliers` | what A does to one opponent B beyond what A does to everyone else and everyone else does to B ("Pairs" below) |
| Money and showdowns | observer metrics | `obs_*` | win rate, net and showdowns won by pot size, WTSD, river bets and calls won, river stack-offs; no hidden cards (`docs/THIRD_PARTY_MINING.md`) |

The article's metrics are all kept: Stages 1 to 5 of the screen, the nine hidden-information signals and the 16
classic statistics (`03_player_stats/README.md`), the flags A-D with their labels and the snap-decision flag, the
15-day escalation rule, the risk score per flagged hand, the win-rate percentile over random runs of the same length,
and the real-card share (how much the opponents' real cards add to the fit of his turn and river folds beyond the
predicted range, step 04's leakage split).

## Population screening

`05_detection/screen.py` scores every player of a dataset, in three time windows (`all`, `last30`, `last15`):

1. **Stage 1, win-rate outliers among comparable players.** Five statistics (bb/100, W$SD, low WTSD, fold to river
   bets, river calls won) as percentiles inside the player's window x table size x hand-count band (250-499, 500-999,
   1,000-1,499, 1,500-2,499, 2,500+) x style cluster (VPIP x PFR tertiles, or k-means); flagged in the top 10% of
   the composite, or above the 95th percentile on 3 of the 5.
2. **Stage 2, the hidden-information signals** of the Stage-1 survivors: the oracle-fold ratio on the turn and the
   river and the bluff index, top 10%.
3. **Stage 3, the hand tests**: categories A-D under two rule sets, the article's (real cards and pot odds) and a
   range-conditioned one (the predicted range must also say the decision was wrong on what could be seen).
4. **Stage 4, a risk score per flagged hand**: category weight x hand-strength factor x ln(1 + pot in bb) x snap
   factor, scaled so that the player's riskiest hand scores 100, and a version based on the field's probability of
   the action, comparable across players.
5. **Stage 5, escalation**: over the last 15 days, the article's fixed rule (more than 5% and more than 5 flagged
   hands) and a population-calibrated rule (the flagged share above the population's 99th percentile in the same
   hand-count band); over the whole dataset, the rule set `street_tails` ("Escalation" below).

Next to the stages: the range-adjusted test (`z_b`, folds while dead and calls while winning against what the range
predicts), the win-rate percentile over random runs of the same length (720 hands), weekly series with a changepoint
(onset), ordered pairs with an opponent-controlled residual tested with Benjamini-Hochberg (single victim), every
ordered pair of the dataset scored and ranked (`pair_outliers`, "Pairs" below) and the observer metrics (`obs_*`).
On the simulator (`sim_demo`: 300 players, about 400,000 hands, seven planted superusers; the evaluation in
`05_detection/results/eval_sim_demo.md`), the range-adjusted screen flags the full-table plants that use the
information on 100% and 50% of their decisions (z_b 11.5 and 5.3; not the one at 25%) and the single-victim plants at
92% and 50%; the 20% single-victim plant stays below the player-level bar, and the pair test ranks its pair first of
308 (q < 0.0001). The onset plant shows as a changepoint (z 4.9) in the week it starts. Among the 285 honest players
the screen flags none; win rate alone (Stage 1) flags 26 of them, and 12 of 29 strong honest regulars, none of whom
the decision signals flag. These are simulator numbers, not detection rates for real data.

On the bundled case the screen has no population (two players): by the article's fixed rule both players escalate,
by the range-conditioned rule neither does. The street-tails rule runs there too, but its tests then compare the two
players with each other. One case is what the per-suspect report is for.

### Escalation: three rules side by side

| rule | a player escalates when | window | where |
|---|---|---|---|
| the article's fixed rule | flagged hands / hands > 5% and more than 5 flagged hands | the last 15 days of his heads-up hands | `escalations`, rule `article_fixed`; `player_scores.esc_<rule set>_article_fixed` |
| population-calibrated | the flagged share is above the population's 99th percentile in his hand-count band (median + z x 1.4826 MAD; a band of fewer than 20 players is compared with every player), and more than 5 flagged hands | the same | rule `population_calibrated`; `esc_<rule set>_population_calibrated` |
| `street_tails` | he sits in the top tail on at least two of the three streets at once: preflop `pf_zr` ≥ 4.0; flop `z_b_flop` ≥ 3.0 and `fl_e_disc` ≥ 0.30; turn + river `z_b` ≥ 3.0 and `e_disc` ≥ 0.30, each street with its own minimum sample ("Thresholds"); one street puts him up for review | the whole dataset (window `all`) | rule set `street_tails`, rule `several_streets`; `pf_hit`, `fl_hit`, `tr_hit`, `street_hits`, `esc_street_tails`, `review_street_tails` |

The first two rules run on each hand-test rule set (`article` and `range_conditioned`) separately. At a
fixed effect a z grows with the sample, and an effect size alone is noisy on a thin one, so the flop and the turn +
river hits need both. The values are in `config/step05_screening.toml`, `[step05.escalation]` and
`[step05.escalation.street_tails]`. On the simulator (`sim_pf`: 200 players, about 200,000 hands, six planted
superusers), `street_tails` escalates all six planted players, three on all three streets and three on two, and none
of the 194 honest players; one honest player reaches review. The fixed rule also fires for sound honest players
("Thresholds"), and the calibrated rule needs a population of comparable players. These are simulator numbers, not
detection rates for real data.

### Pairs: when the cheat targets a few opponents

A superuser who sees the cards of only a few opponents plays honestly against everyone else, and his overall numbers
stay ordinary ([eli5_selective_cheating.html](eli5_selective_cheating.html)). The screen therefore scores every ordered
pair "A against B" of the dataset (window `all`) in the table `pair_outliers` and ranks the pairs against each other
(`sudetect/pairs.py`). Pair hands are the hands that end with at most two players in, at any table size: A and B are
the two who stayed in longest (both to the showdown, or the winner and the last player to fold), each with at least
one voluntary action. The score has seven parts, from A's side:

- `wr`: his net per pair hand (counted from 30 hands; every other part from 7 cases);
- `sd`: his showdowns won against B;
- `call`: his calls of B's bets on the flop, turn or river that were ahead of B's real hand, against the pool at the
  same street and decile of equity against B's predicted range;
- `bet`: his bets and raises on the flop, turn or river that were ahead, the same way;
- `allin`: his equity against B's real hand when the money went in (preflop included), against the pool at the same
  street and range band;
- `pre`: his preflop calls and raises facing B's raise that were ahead, against the pool at the same situation and 5%
  band of equity against B's line;
- `fold`: his folds facing B's bet that were behind B's real hand, against the pool at the same strength
  (full-information data only).

Each part is a leave-pair-out residual, S(A vs B) - S(A vs everyone else) - S(everyone else vs B) + S(all), over its
standard error, so an opponent who loses to everybody, or a player who beats everybody, does not make a pair stand
out by itself. Three corrections keep the z values honest: a hand counts once (the decisions of one hand share its
cards, so the variance of a decision part carries a design effect); the score is the sum of the parts' z over the
square root of the sum of their correlations, estimated over every pair of the dataset, because the parts read the
same hands; and the pairs are scored again, with the pairs that reached the victim score in the previous pass left
out of the "everyone else" terms, until that set stops changing (a few passes), so one strong pair does not shift the
others of its two players. Pairs with at
least two parts are ranked (`field_rank` of `n_ranked`). A victim of A is a pair of his with a score of at least 3
and a rank inside the top 1% of the ranked pairs or the top 10, whichever is larger. `player_scores` carries the
summary per player (`victims_n`, `best_pair_score`, `best_pair_rank`, `best_pair_opponent`); the per-suspect report
lists his victims with every part and the same pairs seen through pot sizes, and tags a victim's id with "pair #k of
N (top x%)" in every hand.

On the simulator the planted single-victim pairs come first: the three of `sim_demo` are pairs 1, 2 and 3 of 2,532
ranked pairs, the one of `sim_pf` pair 1 of 2,340. A cheat who sees the whole table beats everybody, so his pairs
seldom stand out (the best pair of a plant who sees every opponent from the start ranked between 9th and 385th; the
onset plant, who sees everyone in the last quarter of the days only, reached rank 3 to 4): pairs are for the selective
kind, the street tests for the full-table kind. Honest pairs can get the victim label too (3 of the 194 honest
players of `sim_pf` have one, none of the 293 of `sim_demo`; the planted pairs score 7.0 to 9.2 on `sim_demo` and 7.1
on `sim_pf`, the best honest pairs 2.2 to 3.8), so a label is a reason to look, not a verdict. `pair_outliers` covers
the whole dataset, and a cheat may run for a while only: as a review practice, load the hands of a period into a
dataset of its own and run the same steps on it (`docs/THIRD_PARTY_MINING.md`).

### Tables of any size, preflop and flop

The screen reads heads-up confrontations at tables of any size: on the flop, turn and river the hands that **exactly
two players saw the flop of** (a six-max pot two players take to the flop counts, a three-way flop does not), and
preflop every decision with **one live opponent** (`[step05] scope = "flop_hu"`; `"hu_tables"` keeps the article's
two-player tables). On top of the turn-and-river core it adds a preflop fold test and a preflop aggression test (does
the player fold more when behind the opponent's real hand, and raise more when ahead, than his strength against the
opponent's line explains), re-raise success (3-bets, 4-bets and 5-bets the opponent folded to at once, light ones
apart), a flop version of the range-adjusted test, and six hand rules for review (`street_flags`). The per-suspect
report (`05_detection/run.py`) reads the same scope (`--scope`). These are new and not calibrated on real data: test
them on your own population first. What adds noise, and the simulator check, are in
`05_detection/README.md` and `05_detection/results/eval_preflop_flop.md`.

**Use at your own discretion.** This repository is a reference implementation, not a ready-made detection product.
Every scope, rule and threshold in it was set on one reviewed case and a simulator, and none is calibrated for any
operator, game, stake or table size. Before acting on any output, adapt and test the steps on hands of your own that
you have already reviewed, and keep a human reviewer between a flag and any decision about a player.

## Implementation scenarios

| exposure | what the cheat sees | how to run |
|---|---|---|
| Full table | every opponent's cards at every hand | `screen.py`: Stages 1-2 and the `z_b` screen over everyone, and `street_tails`; then `run.py --player <id>` for the flagged players |
| Single victim | one opponent's cards (his account or machine is compromised) | `screen.py`: `pair_outliers` ranks every pair of the dataset, and `player_scores` counts each player's victims and names his most suspicious pair (`victims_n`, `best_pair_*`); `pair_scores` keeps the pair test with BH. Then `run.py --player <id>`: its victims section lists the opponents he beats and tags them in every hand; `--opponent <victim>` keeps the signals to their hands, so they are not diluted by the other opponents |
| Onset after normal play | nothing at first, every card later | `screen.py`: the `last30` and `last15` windows, the weekly series and the changepoint; the escalation rule looks at the last 15 days |

## Thresholds

Every cutoff is a starting point, to be tuned on reviewed cases of your own population, stakes and review capacity.
The screening's values are in `config/step05_screening.toml`, the report's in `config/step05_report.toml`; the main
ones:

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
| Preflop and flop (step 05) | behind / ahead preflop: q < 40% / q ≥ 55%; flop behind: q < 20%; hand rules PF1-PF4, FL1-FL2 in `05_detection/README.md` |
| Street tails (step 05) | preflop `pf_zr` ≥ 4.0 (players with 20 responses to a raise behind and 20 ahead); flop `z_b_flop` ≥ 3.0 and `fl_e_disc` ≥ 0.30 (15 flop responses behind, 15 winning); turn + river `z_b` ≥ 3.0 and `e_disc` ≥ 0.30 (20 dead, 20 winning); escalate on 2 of the 3 streets, review on 1 |
| Victim (step 05) | pair score ≥ 3 and a rank inside the top 1% of the ranked pairs or the top 10, whichever is larger; a pair is ranked with 2 or more parts (win rate from 30 hands, every other part from 7 cases) |
| Rare decisions (report) | the real cards made the decision right and the pool (every other player of the field, one live opponent) seldom makes it: preflop fold to a raise with q < 40% and below the pot odds where the pool folds < 50% of the time; preflop raise with q ≥ 55% where it raises < 30%; preflop call of a raise with q ≥ 55% where it calls < 30%; flop fold to a bet with q < 20% where it folds < 50%; flop call with q ≥ 50% where it continues < 30%. Pool bands: situation x 5% of r preflop, 10% of r on the flop, 30 decisions or more; tags Perfect under 5%, Good under 15% |

The escalation rule is a fixed count. A sound player folds beaten hands and calls bluffs often enough over 15 days
to pass 5% and 5 hands: on the bundled case the victim, run as the player, escalates too (24 of 250). On a real
platform escalate a player when he sits in the top tail of comparable players on several signals at once, for
example the top 5% on 3 of the 9, over 15- and 30-day windows, compared only with players measured on as many
hands; `screen.py` implements the population-calibrated rule and the rule set `street_tails`, the top tail on several
streets at once.

## Predicted range: what the tables are fitted on

The bundled results use range tables fitted on a full-information heads-up population (every player's hole cards
known; the hands stay outside the repository). Refitting on about three times as many hands moved the equity against
the range by 0.2 points at the median (90% of the case's decisions within 1.3 points), and every flagged hand kept its
tier. Fitted on observer hand histories instead (hole cards only at the showdown), the tables see showdown hands
only: the ranges come out stronger than the true ranges of each betting line and the "range says call" verdicts
conservative. `04_predicted_range/variants/observer_fit.toml` is the setting for that case; only `--fit-from`
changes.

## Pictures (ELI5)

HTML pages to open in a browser: `eli5.html` (how a superuser gets caught), `eli5_selective_cheating.html` (the cheat
who sees only a few players, and how the pairs find him), `eli5_technical.html` (the data pipeline map), and one per
step: `01_action_table/eli5.html`, `02_equity/eli5.html`, `03_player_stats/eli5.html`,
`04_predicted_range/eli5.html`, `05_detection/eli5.html`.

## Repository layout

```
00_simulator/        simulate.py: the synthetic population with planted superusers (settings in config/sim.toml)
01_action_table/     run.py (text loader), from_replay_json.py (JSON hand records), sql/, tests/
02_equity/           run.py, bench.py, examples.py, sql/, tests/
03_player_stats/     run.py
04_predicted_range/  run.py, preflop.py, examples.py, variants/ (observer_fit, cap240), sql/, tests/
05_detection/        screen.py (population screening, sql/), eval_planted.py and eval_quick.py (simulator answer
                     key), results/, run.py and adapter.py (per-suspect report), variants/ (observer_300, for mined
                     hands), tests/, example/ (the case's outputs)
config/              defaults.toml (step 01), step02_equity.toml, step04_range.toml, step05_screening.toml,
                     step05_report.toml (the per-suspect report), sim.toml
data/                sample_hands.csv (the article's case, JSON hand records), sample_hands.txt (the same, text
                     format), sample_hands_decision_times.csv
docs/                HAND_HISTORY_FORMAT.md (the input text, line by line), FROM_ZERO.md (from an empty data
                     directory to the reports), THIRD_PARTY_MINING.md (hands collected by an observer at the tables)
sudetect/            library: cards, evaluator, equity, handclass, hhparse, replay, positions, situations,
                     warehouse, rangemodel, metrics, screening, streets (the preflop and flop tests), pairs (the pair
                     outliers), potsize (the observer metrics); sim/ (rules engine, hand writer, player policies,
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
