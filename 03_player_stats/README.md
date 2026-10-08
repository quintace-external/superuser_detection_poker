# Step 03: player statistics

`run.py` writes the table `player_stats`: one row per player with the 16 classic statistics and the 9
hidden-information signals of the per-suspect forensic package (`superuser/stats.py`, `superuser/signals.py`),
computed from the step 01 and step 02 tables through `05_detection/adapter.py`. A signal is stored as its value
(a share) and its sample size `<signal>_n`.

```
python 03_player_stats/run.py --dataset article_sample [--heads-up-only] [--min-hands 1]
```

Decision times come from `$SU_DATA_DIR/<dataset>/input/decision_times.csv` when step 01 loaded operator replay rows
(`--times` points elsewhere); without them `median_think_time` is empty. Money is in chips at 100 per big blind.

Step 04 computes VPIP, PFR and postflop aggression for its range tilts with its own SQL (`sql/tilt_raw.sql`), keyed
by table size; those are not read from this table.

## Classic statistics (16)

| column | definition |
|---|---|
| `hands` | hands the player was dealt into |
| `net_win` | sum of the player's net results |
| `bb_100` | net result in big blinds per 100 hands |
| `vpip` | share of hands with a voluntary preflop call, bet, raise or all-in |
| `pfr` | share of hands with a preflop raise |
| `three_bet` | re-raises at the first decision facing exactly one raise (the open) / such decisions |
| `fold_to_3bet` | folds after making the open raise and facing a re-raise / such spots |
| `flop_cbet` | flop bets by the preflop aggressor when no one bet before him / such spots |
| `fold_to_cbet` | folds to the preflop aggressor's first flop bet / such spots |
| `aggression_factor` | postflop bets and raises / postflop calls |
| `aggression_frequency` | postflop bets and raises / (bets and raises + calls + folds) |
| `wtsd` | hands that reached a showdown / hands in which the player saw the flop |
| `w_sd` | showdowns won (net result > 0) / showdowns |
| `fold_to_river_bet` | river folds facing a bet / river decisions facing a bet |
| `river_call_win_rate` | river calls that won at the showdown / river calls |
| `median_think_time` | median decision time in seconds over all the player's actions |

## Hidden-information signals (9)

Equity is the share of the pot won against every player still in the hand, from their real hole cards (step 02's
`eq_vs_real`). Behind = equity below 50% at the moment of the action.

| column | definition |
|---|---|
| `perfect_fold_ratio_postflop` | folds made while behind / all folds, postflop |
| `perfect_fold_ratio_turn` | the same on the turn |
| `perfect_fold_ratio_river` | the same on the river |
| `pair_plus_folds_behind` | pair-or-better postflop folds made while behind / pair-or-better postflop folds |
| `bluff_index` | successful bluffs / bluffs; a bluff is a postflop bet or raise made while behind, successful when every later action by the others on that street is a fold |
| `fold_vs_value_bet` | fold rate facing a bet whose bettor had at least 50% equity when betting |
| `fold_vs_bluff` | fold rate facing a bet whose bettor had less than 50% equity when betting |
| `discrimination_gap` | fold rate vs value bets - fold rate vs bluffs |
| `river_decision_accuracy` | facing a river bet: fold when behind or continue when ahead, / such decisions |

A player who sees the hole cards folds almost only when behind, calls almost only when ahead and folds far more
against value bets than against bluffs. Step 05 turns these into suspicious hands and an escalation verdict for one
player at a time; this table puts every player side by side, so an operator can rank a population on the signals
(for example above the 90th to 95th percentile on several of them) before choosing whom to analyse.
