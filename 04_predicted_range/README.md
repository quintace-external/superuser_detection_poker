# Step 04: predicted range

For every heads-up postflop decision this step estimates which hands the opponent can hold, given his preflop line
and the postflop actions he has taken so far, and computes the hero's equity against that weighted range. The
equity against the predicted range is the reference an honest player can act on. Step 05 shows it next to the
equity against all hands and against the opponent's real cards: a player who folds hands that are well ahead of
the predicted range, exactly when they are behind the real cards, acts on information he should not have.

## Inputs and outputs

Reads the step 01 tables `actions`, `hand_players` and `hands` and the views `decisions` and `hu_decisions`. When
step 02's `action_equity` exists, the real-card equities computed here are compared with it.

| table | grain | content |
|---|---|---|
| `rng_lik_post` | level x spot x hand class | field counts of fold, check, call, bet and raise; levels K6 to K1 |
| `rng_lik_size` | level x spot x hand class x faced size tier | the same counts with every bet and raise split into five size tiers; levels S6 to S1, F1, F0 (see "Bet size") |
| `rng_prior_pre` | level x preflop key x 169 class | how many players who saw a flop held each starting hand; levels P4 to P0 |
| `rng_field` | table-size class | field VPIP, PFR and postflop aggression of the regular players |
| `rng_player_tilt` | player x table-size class | the player's VPIP, PFR, postflop actions and aggression |
| `rng_meta` | key | which dataset the field tables were fitted on |
| `action_range_equity` | decision | the hero's equity against the predicted range and the range's size and shape |
| `rng_leak_matrix` | street x range bin x real class | responses to a bet and folds, input of the leakage split |
| `action_range_opp` | step x 169 class | the narrowing example's class weights after each step |

The first five are the fit; `action_range_equity` is the step's product.

## How to run

```
export SU_DATA_DIR=/path/to/data          # folder of the <dataset>.duckdb warehouses
V=04_predicted_range/variants

# fit the field tables on a dataset (needs every actor's cards, see "Fit tables")
python 04_predicted_range/run.py --dataset <full-information dataset> --steps fit

# score the article sample with tables fitted on another dataset (the bundled results: a full-information population)
python 04_predicted_range/run.py --dataset article_sample --fit-from <full-information dataset> --steps fit,enrich,leakage

# observer hand histories (cards only at the showdown) need the observer setting
python 04_predicted_range/run.py --dataset <observer dataset> --steps fit --config $V/observer_fit.toml
```

Stages: `fit` (field tables, or a copy of another dataset's tables with `--fit-from`, plus the dataset's own player
tilts), `enrich` (score the decisions into `action_range_equity`), `leakage` (range bins against the real-card class
of the responses to a bet), `examples` (the narrowing example of one hand of the dataset named in `[examples]` of
the configuration; it draws nothing unless that dataset exists in `$SU_DATA_DIR`). Settings live in `config/step04_range.toml`; `--config FILE` overrides any key.
`--stage-tag TAG` scores a variant of the settings into `<dataset>_step04_<tag>/` without touching the warehouse
tables. `variants/cap240.toml` holds the constants of a narrower setting of the range model (a 240-combo cap, a relative
prune, a class floor, live-combo weights); the default keeps the whole posterior over the opponent's live combos.

## The model

The model is `sudetect/rangemodel.py`. For one decision of the hero and one live opponent it builds a weight for
each of the opponent's 1,326 possible holdings in six steps. The constants are in `config/step04_range.toml`.

### 1. Prior

The prior is the distribution over the 169 starting-hand classes of the hands that players in the same preflop
spot took to the flop, read from `rng_prior_pre`. The key is the number of players dealt in (`2`, `3-6`, `7+`), the
situation of the opponent's last preflop betting line (`unopened`, `vs_limp`, `vs_open`, `vs_open_call`,
`vs_3bet`, `vs_4bet_plus`), his position class and the action of that line (`check`, `call`, `raise`).

| level | key |
|---|---|
| P4 | table size, situation, position class, action |
| P3 | situation, position class, action |
| P2 | situation, position class |
| P1 | position class, action |
| P1m | position class, call and raise merged (built at lookup time) |
| P0 | every player who saw a flop |

The lookup uses the first level whose cell holds at least 30 hands. P4, P3 and P2 are skipped for a player who never
acted before the flop. The table is built with the key the lookup uses, so an opener who called a 3-bet is counted
under (`vs_3bet`, `call`) and a 4-bettor whose raise was called under (`vs_3bet`, `raise`). Only players still in the
hand when the flop was dealt are counted, so folded hands never enter the prior.

The table size is in the key because heads-up and six-handed opening ranges differ by about a factor of two.

### 2. Looseness tilt

A player who plays more hands than the field gets a weaker range:

```
r      = clip(stat / field_stat, 0.5, 2)        stat = PFR if he raised before the flop, VPIP otherwise
pi1(c) ~ pi0(c) * r ^ (1.4 * (0.5 - s_c))
```

`s_c` is the strength percentile of class `c` by equity against a random hand (1 for AA, 0 for 32o). With `r = 2`
the weight of AA falls to `2^-0.7 = 0.62` times its prior and the weight of 32o rises to 1.62 times. The tilt needs at
least 15 postflop actions of the player in the same table-size class and is skipped when `|r - 1| < 0.001`.

### 3. Combos

Every class with a positive weight expands to its live combos, those that share no card with the hero's hand or the
board. Each live combo `h` of class `c` gets

```
w(h) ~ pi1(c) / |c|          |c| = 6 for a pair, 4 for a suited class, 12 for an offsuit class
```

normalised over the live combos (`combo_weight = "class_size"`). This is the probability of the combo once the dead
cards are known: every combo of a class is equally likely before the deal, and the blocked ones are removed. The
narrow setting divides by the class's live combos (`combo_weight = "live_combos"`), so a class the board
half blocks keeps its whole weight on the combos it has left. A class that no player in the prior cell took to the
flop has weight 0 and expands to nothing.

### 4. Narrowing

Each postflop action `a` of the opponent before the decision updates every combo `h`:

```
w(h) <- w(h) * (P(a | street, situation, pose, pot type, made(h), draw(h)) + 1e-4)
```

`made(h)` and `draw(h)` are the combo's made-hand class (12 values) and draw class (5) on the board of that action
(`sudetect.handclass.classes_for_all_combos`). The situation, pose (`ip` / `oop`) and pot type are read from the
opponent's row of `actions`, so they come from the same functions that keyed the field table when it was fitted.
The probabilities come from `rng_lik_post`:

| level | key |
|---|---|
| K6 | street, situation, pose, pot type, made hand, draw |
| K5 | street, situation, pose, made hand, draw |
| K4 | street, situation, made hand, draw |
| K3 | street, situation, made hand |
| K2 | street, situation, pose, pot type |
| K1 | street, situation |

The first level whose cell holds at least 25 actions gives `P(a) = (N_a + 1e-4) / (N + 5e-4)`; when none does,
`P = 0.2`. K2 and K1 carry no hand class, so an update that backs off that far changes nothing but the aggression
tilt.

### Bet size

With `[range] sizing = true` (the default, also in `variants/observer_fit.toml`) the narrowing reads the size of
every bet and raise. Before it, a 33% bet and a 125% overbet updated the range alike. A bet or raise gets a tier
from its effective size `size_pot = (amount_eff - to_call) / (pot_before + to_call)` in `actions`; an all-in gets
one from its effective size too:

| tier | size, share of the pot |
|---|---|
| t1 | below 0.40 |
| t2 | 0.40 to 0.60 |
| t3 | 0.60 to 0.90 |
| t4 | 0.90 to 1.10 |
| t5 | 1.10 and more (overbets) |

`sql/fit_lik_size.sql` fits `rng_lik_size` on the same rows as `rng_lik_post`. It has 13 outcomes (fold, check,
call, bet_t1 to bet_t5, raise_t1 to raise_t5), and every key also holds the tier of the bet the actor faced
(`faced_size_pot`, `none` when he faced no bet). Levels S6 to S1 are K6 to K1 with that faced tier added. F1
(street, faced tier) and F0 (street) hold the field's mix of sizes. A bet updates every combo with
`P(bet in its tier | spot, faced tier, made(h), draw(h))`. A check or a call uses its own probability under the
faced-tier key.

Back-off: the model reads the sized cell at the level the plain likelihood came from, with the faced tier added,
when that cell holds at least 25 actions. When it holds fewer, fold, check and call keep their plain probability.
A bet gets its plain probability times the tier mix of the bets at the first level from there on that holds at
least 25 bets; then F1, F0, and 1/5 per tier if none does. A raise works the same way. The 13 probabilities of a
cell still sum to 1. A bet of unknown size uses the plain probability. `sizing = false` is the size-blind model
exactly (`tests/test_sizing.py`).

Effect on the article's case (1,641 heads-up decisions) with tables fitted on a full-information population: the
mean effective range is 696 combos (flop 791, turn 654, river 524), and the opponent's real hand lies inside the
predicted range in every decision.

Check on full-information data (`sim_demo`, 750,635 decisions, every hole card known; the honest field is every
player outside the answer key):

| | sizing = false | sizing = true |
|---|---|---|
| real combo with a non-zero weight | 100% | 100% |
| mean effective combos, flop / turn / river | 664 / 500 / 389 | 663 / 495 / 384 |
| calibration error (mean eq_vs_real per eq_vs_range decile vs the diagonal) | 0.0066 | 0.0063 |
| mean log-likelihood of the real combo | -6.376 | -6.369 |
| honest field, fold-rate AUC by eq_vs_range, turn and river responses | 0.728 | 0.719 |

In the simulator the sizes say little about the hand: ranges narrow by about 1%, and calibration stays the same.
The honest field's fold order loses 0.008 of AUC. A sized cell splits the same 25 actions into 13 outcomes, so its
probabilities are noisier than those of a 5-outcome cell.

### 5. Aggression tilt

On the opponent's bets and raises the probability is multiplied by `(af ^ b_g) ^ omega`:

```
af    = clip(agg / field_agg, 0.5, 2)            agg = his share of bets and raises among his postflop
omega = clip((n - 8) / (50 - 8), 0, 1)                 checks, calls, bets and raises (n of them)
```

`agg` is smoothed toward the field below 50 actions: `agg = omega * agg_raw + (1 - omega) * field_agg`. The exponent
`b_g` depends on the combo's made-hand group, so the bets of an aggressive player keep more air in his range:

| group | made-hand classes | b_g |
|---|---|---|
| monster | full_house_plus | 0.0 |
| strong | flush, straight, trips_set, two_pair, overpair | 0.05 |
| top_pair | top_pair_good, top_pair_weak | 0.15 |
| middle_pair | middle_pair | 0.45 |
| weak_pair | weak_pair, underpair | 0.8 |
| air | nothing | 1.0 |

### 6. Pruning

With the default settings nothing is pruned: `cap = 1326`, `prune_frac = 0` and `class_floor = 0` keep every live
combo with a positive weight, and the weights are normalised to sum to 1 after the expansion and after every
update. The range is the whole posterior of the model.

The narrow setting prunes after step 3 and after every update: classes below 0.0008 of the prior are not
expanded, combos lighter than 2e-4 times the heaviest are dropped and the 240 heaviest combos are kept (ties keep
the lower combo id). Those constants are in `variants/cap240.toml`. The next section explains why the default
keeps the whole posterior.

### Equity against the range

Heads-up, the hero's equity against the range is exact on every street (`sudetect.equity.equity_vs_range_batch`
with `exact=True`: every opponent combo and every board completion is enumerated). Multiway rows are off by
default; with `max_live_opp` 2 or 3 every opponent gets his own range, built with the hero's cards and the board as
the only dead cards, and `equity_vs_ranges_mc` deals one combo per opponent from his range, rejects draws where two
opponents share a card, completes the board and splits the pot among the best hands (4,096 samples, seeded per row,
standard error stored).

## Fit tables

The fit reads only rows whose actor's cards are known. On full-information data (the operator's view) that is every
decision, folded hands included. On observer data, where cards appear only at a showdown, it is the showdown hands
alone: the hands that fold are missing and strong hands are over-represented. `run.py` refuses to fit on a dataset
whose share of `cards_visibility = 'full'` hands is below `[fit] min_full_share` (1.0 by default); such a dataset
takes another dataset's tables with `--fit-from`, and its own `rng_player_tilt` is still computed from its own hands
(VPIP, PFR and aggression need no hole cards). Each table is one SQL statement in `sql/` with `GROUPING SETS`;
`grouping(col)` tells which columns a row rolls up, and a rolled-up column is NULL.

Field baselines (`rng_field`) are the means over the players with at least 300 hands in a table-size class whose
VPIP and aggression lie within 3 standard deviations of the mean.

### Observer fit

`variants/observer_fit.toml` sets `min_full_share = 0` so the tables can be fitted on observer hand histories, where
hole cards are known only for hands that reached a showdown. The prior and the player tilts are keyed by table size,
so a heads-up decision draws its prior from the heads-up cells; the likelihood table pools all table sizes, because
the fit has no table-size filter.

These tables describe how hands that reached a showdown were played, not every hand that took the line. The ranges
they predict are too strong: equity against them is lower than against the true range of the line, which makes a
fold look more reasonable than it is and weakens the range verdicts of step 05. The bundled results therefore use a
full-information fit; the commands are the same, with `--fit-from <that dataset>` and the default config.

## Output columns

`action_range_equity`, one row per scored decision (by default the postflop rows of `hu_decisions` whose actor's
cards are known):

| column | type | meaning |
|---|---|---|
| `hand_id`, `action_index` | VARCHAR, SMALLINT | the decision row of `actions` |
| `n_live_opp` | SMALLINT | live opponents, each with his own range |
| `eq_vs_range` | DOUBLE | share of the pot the hero wins on average against the range(s), ties count half |
| `eq_range_method` | VARCHAR | `exact` or `mc` |
| `eq_range_se` | DOUBLE | standard error of `eq_vs_range`, 0 when exact |
| `n_combos_opp` | INTEGER | combos with a positive weight in the opponent's range (multiway: the smallest range); with the default every live combo of a class with a prior, at most 1,081 on the flop |
| `range_top_class` | VARCHAR | heaviest of the 169 classes in that range |
| `range_strong_share` | DOUBLE | share of the range holding two pair or better on the current board |
| `range_air_share` | DOUBLE | share of the range holding no pair |

`rng_lik_post`: `level`, `street`, `situation`, `pose`, `pot_type`, `made_hand`, `draw` (NULL where the level does
not use the column), `c_fold`, `c_check`, `c_call`, `c_bet`, `c_raise`, `n`.

`rng_lik_size`: `level`, `street`, `situation`, `pose`, `pot_type`, `made_hand`, `draw`, `faced_tier` (`none`, `t1`
to `t5`; NULL where the level does not use the column), `c_fold`, `c_check`, `c_call`, `c_bet_t1` to `c_bet_t5`,
`c_raise_t1` to `c_raise_t5`, `n`.

`rng_prior_pre`: `level`, `size_class`, `preflop_situation`, `position_class`, `action`, `hand_class`, `n` (one row
per class with at least one hand).

`rng_field`: `size_class`, `n_eligible`, `n_clean`, `field_vpip`, `field_pfr`, `field_agg`.

`rng_player_tilt`: `player_id`, `size_class` (`2`, `3-6`, `7+` or `all`), `hands`, `n` (postflop checks, calls,
bets and raises), `vpip`, `pfr`, `agg_raw`, `agg` (smoothed).

`rng_leak_matrix`: `street`, `range_bin` (`least(9, floor(10 * eq_vs_range))`), `real_class` (`dead` when the
equity against the real hand is below 0.02, `win` from 0.5, `mid` otherwise), `n`, `folds`.

`action_range_opp`: `hand_id`, `step`, `action_index`, `step_label`, `opp_player_id`, `hero_player_id`,
`n_combos`, `hand_class`, `weight`.

The scoring chunks under `$SU_DATA_DIR/<dataset>_step04/out/` also hold the hindsight columns `eq_vs_real` (equity
against the opponent's real cards), `real_in_range` and `real_combo_weight`, which only the run summary and the
leakage split read.

## Results on the article sample

The article's case (`data/sample_hands.txt`, loaded by step 01 as `article_sample`) has 1,641 heads-up postflop
decisions with known cards, all scored in about 6 s with tables fitted on a full-information heads-up population (kept outside the repository; loading at about 2,700 hands a second end to end, fitting in about 4 s). The leakage stage counts 233 heads-up responses to a
turn or river bet in these hands, 97 of the suspect's and 136 of his opponents', and reports the split for all of them and for each player (`split_by_player`). For the suspect alone the range bins explain 61 and the real-card class adds 26 (30.0%). On so few responses the share is inflated by noise, so step 05 compares it with field players measured on the same number of responses: median 20.8%, 90th percentile 29.1%; he is above 93% of 297 players, his main opponent (on 136 responses) above 63%. Pooled over the whole field, where the noise vanishes, the share is 2.1%. Refitting the tables on about three times as many hands moved equity against the range by 0.6 points on average (median 0.2, 90th percentile 1.3); the largest moves among the flagged hands, up to 45 points, are folds to overbets of two to four times the pot, which step 05 ranks as RAISE / OVERBET.

## Speed

On an 8-core laptop (AMD Ryzen 7 8845H, 16 logical CPUs) running other jobs, a pool of 12 worker processes scored
750,635 decisions in 511 s, 1,470 decisions per second. Per decision a worker spent 0.6 ms building the range and
6.0 ms on the equity; the exact flop enumeration (1,081 board completions) is most of that.

## Limitations

- Fitted on observer data (see "Observer fit"), the tables are biased toward hands that reached a showdown; the
  bundled results use a full-information fit.
- Bet sizes are read in five tiers (see "Bet size"); within a tier a 1.2-pot and a 3-pot overbet read alike. With an
  observer fit the sized cells miss the bluffs that were folded to, like the plain ones.
- The prior is not smoothed. A class that no player in the prior cell took to the flop gets no combos; on a small
  dataset this can remove the opponent's real hand from the range.
- Each opponent's range is built alone; in multiway pots ranges are not conditioned on each other.
- Player statistics are computed on the whole dataset, including hands played after the scored decision.

## Files

| file | content |
|---|---|
| `run.py` | the four stages, staging, the process pool, the summary |
| `examples.py` | the narrowing example (`action_range_opp`) and the range reconstruction step 05 uses for its HTML |
| `sql/fit_lik_post.sql`, `sql/fit_lik_size.sql`, `sql/fit_prior_pre.sql` | the field tables (`fit_lik_size.sql`: the likelihood by bet-size tier) |
| `sql/tilt_raw.sql`, `sql/fit_field.sql`, `sql/player_tilt.sql` | tilt inputs and field baselines |
| `sql/stage_*.sql` | the decisions, their live opponents and the opponents' postflop actions |
| `sql/schema.sql` | the output tables filled from Python |
| `variants/observer_fit.toml` | the setting for observer hand histories (cards only at the showdown) |
| `variants/cap240.toml` | the narrow 240-combo setting of the range model (size-blind, `sizing = false`) |
| `tests/` | model invariants, two golden decisions (default and narrow-setting constants), the size tiers and their back-off with a frozen river overbet (`test_sizing.py`), the step end to end on a small warehouse of randomly dealt hands |
