# Step 05: detection

Step 05 has two entry points on the tables of steps 01 to 04:

- `screen.py`, **population screening**: every player of a dataset, stages 1 to 5, the single-victim pair test, the
  onset changepoint, the pair outliers (`pair_outliers`), the observer metrics (`obs_*`) and the escalation rule set
  `street_tails` (see "Population screening" below and in the root README);
- `run.py`, **the per-suspect report**: the forensic package (`superuser/` at the repository root) for one player,
  with the opponent's predicted range from step 04 on every decision it reports, and the sections that set him against
  a reference field (see "The report against a field"). The package's methodology, thresholds and definitions are
  described first, then the range extension.

```
python 05_detection/run.py --dataset article_sample                       # the sample's subject, player 1001
python 05_detection/run.py --dataset <name> --player <id>                 # one player, all his heads-up pots
python 05_detection/run.py --dataset <name> --player <id> --opponent <id> # only hands where both sat
python 05_detection/run.py --dataset <name> --player <id> --scope hu_tables  # two-player tables only (the article)
python 05_detection/run.py --dataset <name> --player <id> --scope all     # multiway pots too (or --all-tables)
python 05_detection/run.py --dataset <name> --player <id> --out <folder>  # a folder of his own, emptied first
python 05_detection/run.py --dataset <case> --player <id> --field-dataset <field> --id-map ids.csv
                                                                          # against a larger field
```

| flag | default | what it does |
|---|---|---|
| `--dataset` | (required) | the warehouse whose hands are analysed |
| `--player` | `1001`, the sample's suspect | his player id, as stored in `hand_players` |
| `--opponent` | | only the hands where this player also sat (single-victim exposure) |
| `--scope` | `flop_hu` | `hu_tables`: two-player tables only; `all`: multiway pots too (`--all-tables` is the same) |
| `--times` | `$SU_DATA_DIR/<dataset>/input/decision_times.csv` | the decision times |
| `--range-config` | `config/step04_range.toml` | the step-04 settings the ranges were scored with |
| `--range-hands` | 3 | hands added by the range rule (category R) |
| `--field-dataset` | the dataset itself | the reference field F ("The report against a field") |
| `--id-map` | | a CSV with the columns `case_id` and `ref_id`: this dataset's player ids in F |
| `--top-hands` | `[report] top_hands`, 25 | rare decisions per tab, preflop and flop |
| `--report-config` | | a TOML file whose `[report]` keys override `config/step05_report.toml` |
| `--out` | `$SU_DATA_DIR/<dataset>/report` | the output folder, emptied before every run |

Scope (`--scope`, default `flop_hu`, the same as `screen.py`): every hand dealt to two players and, at larger
tables, every hand the player took to the flop with exactly one other player. A six-max pot two players take to the
flop is analysed like a heads-up hand; a pot three or more players see the flop of is not (the categories read
equity against every live opponent), and its flags are listed at the end of the page without a range. On the
bundled case (two-player tables) every scope gives the same 74 hands.

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

`$SU_DATA_DIR/<dataset>/report/`, or the folder of `--out` (emptied and rewritten on every run, sub-folders
included: give every player a folder of his own):

| file | contents |
|---|---|
| `report.html` | the report as one page: the verdict (the article's rule, the street-tails rule, the pairs), metrics by street, the opponents he beat, the article scorecard and the observer metrics against F, then the package's statistics, decision times, signals, escalation and the top flagged hands |
| `report.md` | the same sections, and the predicted-range table |
| `suspicious_hands.html` | one page with the tabs Postflop (the default), Preflop and All: the victims table at the top, every flagged hand ranked by the Equity review, the rare decisions (see "The ranked page"); the opponent's predicted range as a grid, as text and as JSON |
| `suspicious_hands_all.html` | the flagged hands in the package's own order |
| `suspicious_hands_ranked.csv` | the ranking: tier, severity, the equities, the pot odds and the verdict per hand |
| `suspicious_hands.csv` | all suspicious hands (A to D, then R) with every detail and the range columns |
| `rare_decisions.csv` | the rare decisions on the page: `hand_id`, `action_index`, `street`, `situation`, `action`, the pot odds `req`, `to_call_bb`, his `hole_cards`, `q`, `q_random`, `r`, the opponent `opp_id` and his `opp_hole_cards`, the band `rbin`, the pool's rate `p_pool` and size `n_pool`, `rule`, `rule_name`, `score`, `tier` |
| `pairs_victims.csv` | his victims in F: `opponent` (this dataset's id when the id map knows him, F's otherwise), `label` ("pair #k of N (top x%)"), then the columns of `pair_outliers`; only the header when he has none |
| `metrics_summary.csv` | "Metrics by street" in raw numbers: `street`, `metric`, `column`, `players_ranked`, `value`, `sample`, `ranked`, `percentile`, `rank`, `best_other`, `best_other_sample`, `best_other_id` |
| `decisions.csv` | every decision row with the package's fields and the four reference numbers |
| `decision_times.csv` | his decision times by action type (`action`, `n`, `mean_s`, `median_s`, `min_s`, `max_s`), then over all his actions (`all`); only the header when the data has no times |
| `stats.csv`, `signals.csv` | the two summary tables |
| `summary.json` | the package's results (scope counts, statistics, signals, escalation), the counts by category, the field's name and his id there, the street-tails hits, the victims with their labels, the rare-decision counts and the notes on missing pieces |

On hands with hole cards only at the showdowns the report prints the same files and sections, computed on the hands
where cards were shown: its verdict, hand flags, scorecard and signals are then distorted
(`docs/HAND_HISTORY_FORMAT.md`, "Without full information the results are distorted"); "Observer metrics" and the
pairs' win-rate, showdown and pot-size columns are the parts that stay meaningful (`docs/THIRD_PARTY_MINING.md`).

## The report against a field

The report sets the player against a reference field F: `--field-dataset`, by default the dataset itself. When the
dataset is an extract of a larger field whose player ids differ, `--id-map` translates them (a CSV with the columns
`case_id`, an id of this dataset, and `ref_id`, the same player in F); every lookup in F goes through it. F needs
steps 01 and 02, step 04 with `preflop.py`, and `screen.py`; `docs/FROM_ZERO.md` has the commands. From F the report
reads:

| section | from F | when F lacks it |
|---|---|---|
| Verdict, the street-tails line | his row of `player_scores` (window `all`): the hits, the values, `esc_street_tails` and `review_street_tails` | the line says the rule is not in F's `player_scores` |
| Metrics by street | `player_scores`, window `all` | a note: run `screen.py` on F |
| Opponents he beat | his rows (as A) of `pair_outliers` | the report pairs F itself when F has at most `pairs_max_hands` hands, otherwise a note |
| Article scorecard | `player_scores`; `pair_scores` for the main opponent's games against everyone else | a note |
| Observer metrics | the `obs_*` columns of `player_scores` | computed by the report on every hand of F, with a note |
| Rare decisions | the decisions of every other player of F with one live opponent, with `action_range_equity_pre` (preflop) and `action_range_equity` (flop) | the tab names the step to run |
| Real cards beyond the range (top of the page; only when `--field-dataset` is given) | F's step-04 chunks (`<F>_step04/out/`), against his own share in `<dataset>_step04/summary.json` | no box |

**Verdict.** Three lines: the article's 15-day rule on his hands; the street-tails rule from his row of F, each
street HIT or not with its values (`pf_zr`; `z_b_flop` and `fl_e_disc`; `z_b` and `e_disc`), ESCALATE at
`escalate_min_hits` streets and REVIEW at `review_min_hits`; and the pairs: his victims with their labels, or his most
suspicious ranked pair when none passes the victim rule.

**Metrics by street.** Fifteen rows: `pf_z`, the aggression z, the fold z, the raw fold and aggression gaps, light
re-raise success and 3-, 4- and 5-bet success (preflop); `fl_e_disc` and `z_b_flop` (flop); `e_disc`, the excess
folds while dead, the excess continues while winning and `z_b` (turn + river). Each row ranks the players of F with
the row's minimum sample (`[report.metrics]`): Players ranked (how many, him included when he qualifies), Value (his;
"sample below the minimum" when he does not qualify), Percentile (the share of them he beats, ties counting half),
Rank (1 + the number above him) and Best other player (the best value among the others).

**Opponents he beat.** His victims in F (`sudetect.pairs.victims`: score at least `victim_min_score` and a rank
inside the top `victim_rank_share` of the ranked pairs or the top `victim_min_rank`, whichever is larger), the most
suspicious first: hands, bb/100, showdowns won, the z of each of the seven parts, the score and the label "pair #k of
N (top x%)"; then the same pairs by pot size (showdowns won in pots of `big_pot` bb or more and their growth over
smaller pots, river bets won, river calls won, his stack-offs, net in big pots, each with its z). The table also sits
at the top of `suspicious_hands.html`, and the label sits next to a victim's id in every hand, and next to no other
opponent's.

**Article scorecard.** The article's seven signals (the oracle-fold ratio of pair-or-better postflop folds, of the
turn and of the river; fold to value bets; fold to bluffs; the discrimination gap; bluff success): his value, his
percentile and the median among F's players with at least `min_n` qualifying decisions, the mean of F's `top_winners`
biggest winners by net bb (him left out), and the main opponent's value in his games against everyone else (his
counts in F minus those of his decisions against the suspect, from F's `pair_scores`). The main opponent is
`--opponent`, or the player he met most often with one live opponent left.

**Observer metrics.** His `obs_*` values ("Observer metrics" below) with the cases, the suspicion percentile (turned
so that higher is always more suspicious), the rank and the players ranked, among F's players with at least
`[step05.observer] min_hands` hands and, for a rate, `min_cases` cases; then his net by final pot.

**Rare decisions.** His preflop and flop decisions with one live opponent and both hands known that the opponent's
real cards made right and that the pool, every other player of F with one live opponent, seldom makes at the same
strength against the same line (`[report.rare]`):

| rule | his decision | the pool makes it, in his band, less often than |
|---|---|---|
| `pf_fold` | preflop fold facing a raise, q below 40% and below the pot odds | 50% |
| `pf_raise` | preflop raise (opens included), q at least 55% | 30% |
| `pf_call` | preflop call facing a raise, q at least 55% | 30% |
| `fl_fold` | flop fold facing a bet, q below 20% | 50% |
| `fl_call` | flop call facing a bet, q at least 50% (the pool's rate: calls and raises) | 30% |

The band is his situation and 5% band of r preflop, his 10% band of r facing a bet on the flop; a band counts with at
least `min_pool` decisions. Score = -ln(max(p, `p_floor`)); the best decision per hand and street; the top
`top_hands` of each street, the rarest first; tags Perfect (the pool does it less than 5% of the time), Good (less
than 15%) and Average. Preflop decisions come from all his hands (two-player tables only under `--scope hu_tables`),
flop decisions from the analysed hands not already listed in the Postflop tab. Each gets a box, RARE IN THE POOL,
with the pool's rate, the range's verdict and the opponent's predicted range before the decision: preflop the field's
range for his last preflop action (the range `04_predicted_range/preflop.py` measures r against; none when he has not
acted yet), on the flop the range of step 04.

**Configuration**, `config/step05_report.toml`; `--report-config my.toml` overrides any `[report]` key it repeats:

| key | default | meaning |
|---|---|---|
| `[report] top_hands` | 25 | rare decisions per tab (`--top-hands`) |
| `pairs_max_hands` | 500,000 | F without `pair_outliers`: the report pairs F itself only up to this many hands |
| `show_field_size` | true | false: no field size and none of F's own ids (below) |
| `[report.rare]` | the table above | `pre_bins` 20, `flop_bins` 10, `min_pool` 30, `facing` (the situations that face a raise), the `*_q_max` / `*_q_min` and `*_p_max` pairs, `p_floor` 0.005, `perfect_p` 0.05, `good_p` 0.15 |
| `[report.metrics]` | 20 decisions (15 on the flop); 10 light re-raises; 5 re-raises of each size | the minimum samples of "Metrics by street": `pf_min_behind`, `pf_min_ahead`, `flop_min_dead`, `flop_min_win`, `tr_min_dead`, `tr_min_win`, `light_reraises_min`, `reraises_min` |
| `[report.scorecard]` | `min_n` 20, `top_winners` 10 | the scorecard's minimum sample and reference group |

The report reads the screen's settings (the pair outliers and the victim rule, the observer metrics, the street-tails
rule and the thresholds its texts quote) back from F's `report_meta` (item `config_json`: the settings the screen ran
with, a `--config` file included); a field screened by an older `screen.py` falls back to
`config/step05_screening.toml`.

`show_field_size = false` is for a report made against a private field and shown outside it. The report then leaves
the field's size and the field's own ids out of the places it prints them, for example: no "Players ranked" columns
and no ranks next to the percentiles, pair labels that say "top x% of the field's pairs" instead of "pair #k of N", no
`n_ranked` or pair ranks in `pairs_victims.csv`, no pool counts in the rare decisions, F called "the reference field",
and no `A` and `B` in `pairs_victims.csv`, `best_other_id` in `metrics_summary.csv` or `field_player_id` in
`summary.json`. It covers what the report prints by design, not every number a reader could work back from: read the
page and the files before they go out. The bundled example (`example/`) was made this way: its reference field is a
sample of a larger full-information population, and its percentiles are within that sample.

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
- The sections against a field (the street-tails verdict, metrics by street, the opponents he beat, the article
  scorecard, the observer metrics) and the rare preflop and flop decisions run on this case too. With the sample as
  its own field they compare the suspect with the victim alone (the observer metrics rank neither of them: both have
  fewer hands than `[step05.observer] min_hands`): they show what each section holds, not where he stands in a
  population.

## The ranked page

`suspicious_hands.html` is the page for the expert. At the top sit the opponents he beats beyond his own level (the
victims table, its parts and its pot sizes) and the boxes that explain the order. Below them a selector switches
between three tabs: **Postflop**, the default, which shows without the page's script; **Preflop**; and **All**, both
lists.

The Postflop tab holds every hand the package flags; the Equity review removes none. Each hand carries two verdicts:

- the tag at the top (A-D) is the package's hand-set rule, read from the opponent's real cards: a fold against a
  better hand, a call against a worse one, and so on;
- the **Equity review** box under the hand is computed from the betting: the three equities (vs all hands, vs the
  predicted range, vs the real cards), the pot odds, a plain-words verdict and the opponent's predicted range: a
  13 x 13 grid, the same weights as text in a toggle of their own ("Predicted range as text", a grid and a list in %
  of the range), and a JSON block (`script.range-data`) for scripts. The box also says where the opponent's real hand
  sits in the range: its class, its share of the range and its rank among the 169 classes.

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
6. NO RANGE: hands without a predicted range (the range model covers heads-up pots; the hands flagged outside the
   scope, multiway pots, land here).

After them the Postflop tab lists the rare flop decisions, and the Preflop tab the rare preflop decisions ("The
report against a field"), each with its RARE IN THE POOL box. A victim's id carries "pair #k of N (top x%)" in every
hand: in the settlement lines and in the decision box.

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
| Rare decisions (report) | the pool makes the decision less often than 50% (folds) or 30% (raises and calls) of the time in his band; Perfect under 5%, Good under 15% ("The report against a field") |
| Victim (screen and report) | pair score ≥ 3 and a rank inside the top 1% of the ranked pairs or the top 10, whichever is larger |
| Street tails (screen) | escalate on 2 of the 3 streets, review on 1: preflop `pf_zr` ≥ 4.0; flop `z_b_flop` ≥ 3.0 and `fl_e_disc` ≥ 0.30; turn + river `z_b` ≥ 3.0 and `e_disc` ≥ 0.30 ("The rule set street_tails") |

The escalation rule is fixed. A good regular also folds beaten hands and calls bluffs, so it can fire for honest
players: on the bundled case the victim, run as the player, escalates too (24 of 250). On a real platform escalate
a player when he sits in the top tail of comparable players on several signals at once (for example the top 5% on
3 of the 9, over 15- and 30-day windows, compared only with players measured on as many hands); `screen.py`
implements the population-calibrated rule (Stage 5) and the rule set `street_tails`.

## Population screening

`screen.py` reads the decisions and hands of every player through `sql/scored_decisions.sql` and
`sql/player_hands.sql`, and the pair hands and the observer counts through the queries of `sudetect/pairs.py` and
`sudetect/potsize.py`. It computes stages 1 to 5 in the windows `all`, `last30` and `last15` and writes into the
dataset's warehouse:

| table | contents |
|---|---|
| `player_scores` | one row per player x window: volume, style and main table size, the scorecard, the range-adjusted test (`z_b`), the preflop and flop tests (`pf_*`, `z_b_flop`, `fl_e_disc`), the Stage 1-2 flags, the street hits (`pf_hit`, `fl_hit`, `tr_hit`, `street_hits`), the win-rate runs percentile and the changepoint (`all` rows), the Stage 5 verdicts (`esc_*`, `review_street_tails`; `all` rows), the pair summary (`victims_n`, `best_pair_*`; `all` rows) and the observer metrics with their field ranks (`obs_*`; `all` rows) |
| `pair_scores` | ordered pairs (i -> j) that met heads-up: the scorecard of i against j, the split test, the opponent-controlled residual, its robust z and the Benjamini-Hochberg q-value |
| `pair_outliers` | every ordered pair "A against B" of the whole dataset (window `all`): the parts of its score, the score, its rank among the ranked pairs and the pot-size view ("Pair outliers" below) |
| `window_scores` | weekly series per player (decisions while dead and while winning, their excess over the range, `z_b`) |
| `flagged_hands` | Stages 3-4: the flagged hands under both rule sets, with labels and the risk scores |
| `escalations` | Stage 5: the fixed and the population-calibrated verdict per player and rule set, and one row per player of the rule set `street_tails` |
| `street_flags` | the preflop and flop hand rules PF1-PF4, FL1-FL2 (reported, not escalated) |
| `report_meta` | the config hash, the population sizes and the counts tested, among them `pairs_ranked` (the N of "pair #k of N"), `pairs_victims`, `players_observer_ranked`, `players_street_tails` and `escalations_several_streets`; `config_json`, the settings the screen ran with, which the report reads back |
| `scored_decisions` | the postflop decision rows the metrics were computed on: ids, street, action, q, r, the pot odds `req`, the range decile `rbin`, the field's fold rate `p_all` |

Every setting is in `config/step05_screening.toml`, `[step05]`: the windows, the bands and clusters, the Stage-1
rule, the hand-test margins, the risk weights, the escalation rules (`[step05.escalation]`,
`[step05.escalation.street_tails]`), the pair test, the pair outliers (`[step05.pair_outliers]`), the observer
metrics (`[step05.observer]`) and the onset series. `screen.py --config my.toml` overrides any key the file repeats
under `[step05]`; `--dry-run` computes and prints without writing. The implementation of the stages
(`sudetect/screening.py`) is a replaceable module: `[step05.impl] module`.

`eval_planted.py` scores the screen against the simulator's answer key (`results/eval_<dataset>.md`) and
`eval_quick.py` prints the planted players' rows; both are evaluation code and read the answer key, the screen
never does. `tests/test_screen_formulas.py` checks the formulas on hand-made rows and
`tests/test_screen_parity.py` checks, on `sim_small`, that the screen's article rule set gives the same signals,
flagged hands, labels and escalation as the per-suspect package. `tests/test_screen_new.py` checks the newer outputs
(the tables and column types the screen writes, the pair hands, the pair summary, the street-tails hits and rows,
`fl_e_disc` and the raw preflop gaps) and `tests/test_report_new.py` the report's (the rare-decision selection, the
tabbed page and its pair labels, the metrics table, the victims through an id map, the street-tails line, the
decision times, the opponent's real hand in his range, `show_field_size`).

### Pair outliers

`sudetect/pairs.py` scores every ordered pair "A against B" of the dataset. Pair hands are the hands in which the two
stayed in longest (both to the showdown, or the winner and the last player to fold), each with at least one voluntary
action; a hand with three or more players still in at the end is left out, and so are the hands the screen leaves out
(non-standard hands, hands with an error flag, and larger tables under `scope = "hu_tables"`). Each hand gives one row
from each side. The decision parts read the decisions with one live opponent and both hands known, with r from step 04
(`action_range_equity` after the flop, `action_range_equity_pre` preflop); without step 02 or step 04 only `wr` and
`sd` score.

| part | from A's side |
|---|---|
| `wr` | net per pair hand, bb |
| `sd` | showdowns won against B, a split = 1/2 |
| `call` | calls facing B's bet on the flop, turn or river that were ahead of B's real hand, against the pool's rate at the same street and decile of r |
| `bet` | bets and raises on the flop, turn or river that were ahead, the same way |
| `allin` | equity against B's real hand when the money went in (his all-in, or his call of one; preflop included), against the pool at the same street and band of r |
| `pre` | preflop calls and raises facing B's raise that were ahead, against the pool at the same situation and 5% band of r |
| `fold` | folds facing B's bet that were behind B's real hand, against the pool at the same strength; full-information data only |

Each part is a leave-pair-out residual, S(A vs B) - S(A vs everyone else) - S(everyone else vs B) + S(all), over its
standard error (the two "everyone else" means are pulled toward the overall mean by `prior` pseudo-cases). Three
corrections keep the z values honest:

- a hand counts once: the variance of each decision part (`call`, `bet`, `allin`, `pre`, `fold`) is multiplied by its
  design effect, the sum over hands of the squared per-hand sum of the centred cases over the sum of the case
  variances (at least 1);
- the parts read the same hands, so `score` = the sum of the parts' z over the square root of the sum of their
  correlations over the parts present, the correlations estimated over every pair of the dataset;
- the pairs are scored again with the pairs that reached `victim_min_score` in the previous pass, in either
  direction, left out of the "everyone else" terms of both players, until that set stops changing (at most
  `MAX_PASSES` = 6 passes), so one strong pair does not shift the rest.

Columns of `pair_outliers`:

| columns | meaning |
|---|---|
| `A`, `B`, `time_window` (`all`), `scope` | the pair: A's side against B |
| `hands`, `bb100`, `sd_won` | pair hands; A's net per 100 of them, bb; A's share of the showdowns against B won |
| `k`, `score` | the parts with a z; the score above |
| `field_rank`, `n_ranked`, `top_pct` | the rank among the pairs with at least two parts (1 = the most suspicious; empty below two parts), how many pairs are ranked, and 100 x `field_rank` / `n_ranked` |
| `<part>_n`, `<part>_mean`, `<part>_resid`, `<part>_z` | for each part: the cases, A's mean against B (net per hand, the showdown share, or the excess over the pool), the residual and its z (empty below `min_hands_wr` hands for `wr` and `min_cases` cases for the others) |
| `wsd_big`, `wsd_big_n`, `wsd_big_z` | A's showdowns won against B in pots of `big_pot` bb or more, two players at the showdown |
| `growth`, `growth_z` | `wsd_big` minus his showdown share against B in smaller pots |
| `rbet_win`, `rbet_win_n`, `rbet_win_z` | he bet or raised the river, the pot was over `big_pot` and the hand reached a two-player showdown: share won |
| `rcall_win`, `rcall_win_n`, `rcall_win_z` | he called a river bet with one opponent left, the pot after the call over `big_pot`, and the hand reached a two-player showdown: share won |
| `stackoff`, `stackoff_n`, `stackoff_z` | he saw the river and more than `big_pot` went in on it: share of those hands in which he lost 90% or more of his effective starting stack (the z is turned: fewer stack-offs give a higher z) |
| `net_big`, `net_big_n`, `net_big_z` | his net in pots of `big_pot` bb or more, bb; z = the sum over the root of the sum of squares |

The pot-size columns are descriptive and not part of the score; each z needs `min_cases` cases (the growth: in each
pot class), and a rate's z is binomial against the rate of all pairs. A victim of A is a pair of his with at least two
parts, a score of at least `victim_min_score` and a rank inside the top `victim_rank_share` of the ranked pairs or the
top `victim_min_rank`, whichever is larger. `showdown_only = true` keeps the decision parts to the hands that reached
a showdown: what an observer of the same games would see, for evaluation.

| key, `[step05.pair_outliers]` | default |
|---|---|
| `min_hands_wr`, `min_cases` | 30 hands, 7 cases |
| `prior` | 20 |
| `score_parts` | `wr`, `sd`, `call`, `bet`, `allin`, `pre`, `fold` |
| `victim_min_score`, `victim_rank_share`, `victim_min_rank` | 3.0, 0.01, 10 |
| `big_pot` | 30 bb |
| `showdown_only` | false |

### Newer columns of `player_scores`

| columns | rows | meaning |
|---|---|---|
| `pf_fold_gap`, `pf_aggr_gap` | every window | the raw preflop gaps: fold rate facing a raise when behind the real hand minus when ahead; raise rate when ahead minus when behind; not adjusted for r |
| `fl_e_disc` | every window | the flop's excess discrimination: folds when behind (q < `[step05.flop] dead_q`) minus folds when winning (q ≥ `win_q`), beyond the field's gap at the same range decile, on `fl_nd` and `fl_nw` responses |
| `pf_hit`, `fl_hit`, `tr_hit`, `street_hits` | every window | the street-tails hits of that window's tests, and how many |
| `esc_street_tails`, `review_street_tails` | `all`, empty otherwise | `street_hits` ≥ `escalate_min_hits` / ≥ `review_min_hits` |
| `victims_n` | `all`, empty otherwise | his victims among his pairs as A; 0 without a ranked pair |
| `best_pair_score`, `best_pair_rank`, `best_pair_opponent` | `all` | his most suspicious ranked pair as A; empty without one |
| `obs_*` | `all` | the observer metrics below |

### Observer metrics

`sudetect/potsize.py` counts what money and showdowns show, without the hidden cards, over every standard hand of the
dataset without a parsing error, whatever the scope. Pot lines from `[step05.observer]`: `big_pot_bb` 30, final pot
bands at 30, 100 and 200 bb, pot at the flop at 15 and 40 bb, a stack-off at 90% of the effective starting stack (the
smaller of his stack and the largest other stack).

| column | definition | suspicious when |
|---|---|---|
| `obs_hands` | hands played | |
| `obs_bb100` | net per 100 hands, bb | high |
| `obs_net_b1` ... `obs_net_b4` | net bb in final pots under 30bb, 30-100bb, 100-200bb, 200bb or more (the final contested pot) | descriptive |
| `obs_big_net100` | net in pots of 200bb or more, per 100 hands | high |
| `obs_wtsd` | went to showdown, of the hands that saw the flop | low |
| `obs_wtsd_srp`, `obs_wtsd_3bp`, `obs_wtsd_4bp` | the same by the pot at the flop: under 15bb, 15-40bb, 40bb or more | low |
| `obs_wsd` | showdowns won with two players at the showdown, a split = 1/2 | high |
| `obs_wsd_b1` | the same in final pots under 30bb | shown, not ranked: he reaches the small showdowns with hands he can check down |
| `obs_wsd_b2` ... `obs_wsd_b4` | the same in final pots of 30-100bb, 100-200bb, 200bb or more | high |
| `obs_wsd_growth` | `obs_wsd_b4` minus `obs_wsd_b1` | high |
| `obs_rbet_win` | he bet or raised the river, the pot was over 30bb and the hand reached a two-player showdown: share won | high |
| `obs_rcall_win` | he called a river bet with one opponent left, the pot after the call over 30bb, and the hand reached a two-player showdown: share won | high |
| `obs_rcall_freq` | those calls over the river bets he faced that way | low: he folds the bets he would lose, so a low rate next to a high `obs_rcall_win` is the pattern |
| `obs_stackoff` | he saw the river and more than 30bb went in on it: share of those hands in which he lost 90% or more of his effective starting stack | low |

Every rate has its count `<column>_n` (`obs_growth_n` for the growth: the smaller of the two counts). The field ranks
add `<column>_pct` (a mid-rank percentile from 0 to 100, turned so that higher is always more suspicious),
`<column>_rank` (1 = the most suspicious) and `<column>_of` (the players ranked), for every column above but
`obs_hands`, `obs_net_b1` ... `obs_net_b4` and `obs_wsd_b1`, among the players with at least `min_hands` (1,000) hands
and, for a rate, `min_cases` (20) cases. What these columns catch on the simulator, and how to use them on hands
collected by an observer, is in `docs/THIRD_PARTY_MINING.md`.

### The rule set street_tails

Escalation on the top tail of several streets at once, from the player-level test of each street
(`sudetect/screening.py`, `street_tails`; `[step05.escalation.street_tails]`):

| hit | condition |
|---|---|
| `pf_hit` | `pf_zr` ≥ `pf_zr_min` (4.0), among the preflop test's eligible players (`pf_eligible`: `[step05.preflop] min_behind` and `min_ahead` responses to a raise, 20 each) |
| `fl_hit` | `z_b_flop` ≥ `fl_z_b_min` (3.0) and `fl_e_disc` ≥ `fl_e_disc_min` (0.30), with `fl_nd` ≥ `fl_min_dead` (15) and `fl_nw` ≥ `fl_min_win` (15) |
| `tr_hit` | `z_b` ≥ `tr_z_b_min` (3.0) and `e_disc` ≥ `tr_e_disc_min` (0.30), with `nd` ≥ `tr_min_dead` (20) and `nw` ≥ `tr_min_win` (20) |

At a fixed effect a z grows with the sample, and an effect size alone is noisy on a thin one, so the flop and the turn
+ river hits need both. A player escalates with `street_hits` ≥ `escalate_min_hits` (2) and is up for review with
`street_hits` ≥ `review_min_hits` (1). `escalations` gets one row per player of the `all` window: `rule_set =
'street_tails'`, `rule = 'several_streets'`, `window_start` and `window_end` his first and last hand, `hands`,
`flagged` = the streets hit, `ratio` = their share of the three, `band`, an empty `threshold_ratio`,
`threshold_count` = `escalate_min_hits` - 1 (every rule set escalates with `flagged` > `threshold_count`) and
`escalate`. On the simulator (`sim_pf`: 200 players, about 200,000 hands) the rule escalates all six planted
superusers, three on all three streets and three on two, and none of the 194 honest players; one honest player
reaches review. These are simulator numbers, not detection rates for real data.

## Tables of any size, preflop and flop

**Scope.** `screen.py` reads heads-up confrontations at tables of any size (`[step05] scope = "flop_hu"`, the
default):

- flop, turn and river: every decision of a hand that **exactly two players saw the flop of**, whatever the number of
  players dealt in. A six-max pot two players take to the flop is scored like a heads-up hand; a pot three or more
  players see the flop of is left out (a superuser who sees every card plays multiway pots too, but equity against
  several real hands and several predicted ranges is not what these tests measure);
- preflop: every decision with **exactly one live opponent** (everyone else has folded), whether or not a flop
  follows: the button against the big blind after the rest of the table folded, a 3-bet pot between two players.

`scope = "hu_tables"` keeps the article's scope (two-player tables only). On two-player tables both scopes give the
same postflop rows, so the bundled case's figures do not move. Stage 1 ranks win rates inside the table size a
player played most first (`table_class_groups`: heads-up, 3-6, 7+), and Stage 5 divides by his heads-up hands.
Step 04 scores the predicted range of heads-up pots at every table size (`[enrich] tables = "all"`).

**Preflop** (`sudetect/streets.py`, `[step05.preflop]`). Step 02 scores every preflop decision with one live opponent
(`include_preflop`, `preflop_max_opp = 1`): `q` against the opponent's real cards (Monte Carlo, 4,096 deals). Step 04's
`preflop.py` adds `r`, the equity against the field's range for the opponent's preflop line so far (opened, 3-bet,
4-bet, limped, from his position; `rng_prior_pre`), and `req` is the price as on later streets.

| column | what it measures |
|---|---|
| `pf_fold_behind`, `pf_fold_ahead` | fold rate facing a raise when behind the real hand (q < 40%) and when ahead (q ≥ 55%) |
| `pf_fold_gap` | `pf_fold_behind` - `pf_fold_ahead`, not adjusted for r: honest players have positive gaps too |
| `pf_fold_cmh_z` | fold more when behind than when ahead, inside the same situation (vs open, vs 3-bet, ...) and the same 5% band of r (CMH z) |
| `pf_aggr_behind`, `pf_aggr_ahead`, `pf_aggr_z` | the same for raising, on every preflop decision: raise more when ahead (sign: positive = more aggression when ahead) |
| `pf_aggr_gap` | `pf_aggr_ahead` - `pf_aggr_behind`, not adjusted for r |
| `pf_z`, `pf_zr` | (fold z + aggression z) / √2; `pf_zr` = `pf_z` robust-standardised on the population (median, 1.4826 MAD) once 20 players are eligible |
| `pf_3bet_n` / `_succ`, `pf_4bet_*`, `pf_5bet_*` | re-raises and the share the opponent folded to at once ("4-bet bluff success") |
| `pf_rr_light_n`, `_succ`, `_ahead` | light re-raises (r < 45%, weak against the line): how many, how often the opponent folded at once, how often the real hand behind them was ahead |

**Flop** (`[step05.flop]`). The range-adjusted test of the turn and the river, run on flop responses with a flop-sized
"behind" class (q < 20%: with two cards to come few hands are dead): `z_b_flop`, `fl_cmh_z`, robust-standardised
`z_b_flop_r`, and the flop's excess discrimination `fl_e_disc` on the same classes. The core test `z_b` stays on the
turn and the river.

**Hand rules** (table `street_flags`, ranked by gap = |q − r|, tiers BLATANT ≥ 0.30, STRONG ≥ 0.15, THIN):

| rule | decision |
|---|---|
| PF1 | preflop fold facing a raise; the range said continue (r − price ≥ 0.10); the real hand was ahead of his (q < price, q < 40%) |
| PF2 | preflop call; the range said fold (r < price); his hand was ahead of the real one (q ≥ 55%) |
| PF3 | preflop re-raise with a hand weak against the line (r < 45%) that was ahead of the real one (q ≥ 55%) |
| PF4 | preflop flat call with a hand strong against the line (r ≥ 60%) that was behind the real one (q < 40%) |
| FL1 | flop fold facing a bet with q < 20% where the range said continue (r − price ≥ 0.20) |
| FL2 | flop call or raise facing a bet with q ≥ 50% where the range said fold (price − r ≥ 0.15) |

They are reported for review and are not counted in Stage 5: their false-positive rate is not calibrated on real data.

**Test them on your own data first.** What adds noise:

- a single preflop fold proves nothing: an honest player folds a hand the range says to defend, and the opponent
  happens to hold a better one, on 1.3% of his responses to a raise on the simulator (PF1). Only the rate,
  against comparable players, carries signal;
- the raw preflop z is not centred at 0 for honest players (simulator median 1.1, sd 1.2): a 5% band
  of r still mixes stronger and weaker own hands. Screen on `pf_zr`, and only inside a population of comparable
  players;
- re-raise frequencies and their success without cards move with style and with the opponent: on the bundled case
  the victim's 3-bet success is as high as the suspect's, because each of the two often folds to the other's 3-bets
  (the sample holds no other opponent). They say whom to look at, not who cheats;
- the light re-raise rules (PF3) and the passive rule (PF4) do not separate on the simulator, whose planted players
  never re-raise light: they are untested, not proven useless;
- a superuser's opponent gets distorted numbers too: the bundled victim's preflop z is negative;
- the flop z is wider than the turn and river one (sd 1.6 on honest simulated players): screen on
  `z_b_flop_r`;
- preflop ranges are read from the hands that saw a flop, keyed by the last preflop line, so a 4-bet range comes from
  4-bets that were called; thin cells fall back to coarser levels (`prior_level` in `action_range_equity_pre`).

On the bundled case (two-player tables, so the scope changes nothing there): the suspect's `pf_z` is 7.1 (fold when
behind 69% against 9% when ahead; raise when ahead 76% against 23% when behind), the victim's −1.1; 19 light
re-raises of the suspect, 95% folded to at once (the victim: 8, 50%); `z_b_flop` 5.6 against 0.0. On the simulator
with six-max tables (`sim_pf`, 200,000 hands, 30% six-max: every planted superuser caught by `z_b` in the new scope, the six-max-only one missed in the old; no honest player at 4 on any test) see `results/eval_preflop_flop.md`.

## Data notes (hand records)

- `act` is one of check, fold, call, bet, raise, allin, walk; the integer codes 1 to 7, in that order, are accepted
  too. Anything else is read as an unknown action, and step 01 skips such a hand.
- Bet and raise amounts are street totals; call amounts are chips added. An all-in counts as aggressive only
  when it exceeds the current highest street total.
- A card is {`rank` 0-12 for 2 ... A, `suit` 0-3}; a missing key counts as 0 (a 2, the first suit).
- Hands with invalid cards (rank above 12) are dropped and counted as unparseable.
- `think_s` is the time the player took, in seconds; 0 to 1 s values may be pre-selected actions.
