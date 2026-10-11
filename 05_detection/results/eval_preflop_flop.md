# Tables of any size, preflop and flop: simulator check

Dataset `sim_pf` (`config/sim.toml`): 200 players (193 honest, 6 planted superusers, 1 victim), 40 days,
about 200,000 hands; 70% of table-hands at heads-up tables, 30% at six-max tables. The planted players use the
information preflop too (`00_simulator/simulate.py --dataset sim_pf --preflop-info`): `SU_FULL_100` / `SU_FULL_050` see
every opponent on 100% / 50% of their decisions, `SU_VICTIM_92` sees one victim, `SU_ONSET` starts on day 30,
`SU_RING_100` / `SU_RING_050` play six-max tables only.

```
python 00_simulator/simulate.py --dataset sim_pf --preflop-info
python 01_action_table/run.py --dataset sim_pf --input "$SU_DATA_DIR/sim_pf/hands/*.txt" --source sim
python 02_equity/run.py --dataset sim_pf
python 04_predicted_range/run.py --dataset sim_pf --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset sim_pf
python 05_detection/screen.py --dataset sim_pf                         # scope flop_hu (the default)
python 05_detection/screen.py --dataset sim_pf --config old_scope.toml  # a file with [step05] scope = "hu_tables"
```

## Planted players, window `all`

Old = the article's scope (two-player tables); new = heads-up confrontations at any table (two players on the flop;
preflop, one live opponent). Cells: value (rank among all 200 players); "too few" = below the test's minimum of
decisions. Caught = at least 4 on one of the three tests.

| plant | hands (old) | z_b | pf_zr | z_b_flop_r | caught | hands (new) | z_b | pf_zr | z_b_flop_r | caught |
|---|---|---|---|---|---|---|---|---|---|---|
| SU_FULL_100 | 1,663 | 10.26 (#2) | 5.95 (#2) | 10.45 (#2) | yes | 2,469 | 10.15 (#2) | 7.06 (#2) | 9.33 (#2) | yes |
| SU_FULL_050 | 2,425 | 8.20 (#3) | 4.22 (#3) | 6.94 (#3) | yes | 4,309 | 8.30 (#3) | 5.02 (#3) | 6.76 (#3) | yes |
| SU_VICTIM_92 | 2,929 | 12.79 (#1) | 7.23 (#1) | 13.59 (#1) | yes | 3,460 | 12.73 (#1) | 7.77 (#1) | 11.38 (#1) | yes |
| SU_ONSET | 1,851 | 4.61 (#5) | 3.21 (#4) | 5.26 (#4) | yes | 3,716 | 4.90 (#6) | 3.94 (#4) | 4.55 (#5) | yes |
| SU_RING_100 | 217 | 3.14 (too few) | 1.56 (#13) | 3.82 (too few) | **no** | 2,803 | 7.36 (#4) | 3.71 (#5) | 6.14 (#4) | yes |
| SU_RING_050 | 355 | 5.35 (#4) | -0.02 (#101) | 2.10 (#8) | yes | 2,766 | 5.40 (#5) | 1.03 (#35) | 3.18 (#6) | yes |

The six-max-only plant at 100% has a few hundred hands at two-player tables and is missed there; once six-max
heads-up pots count, every plant is caught by `z_b`, and the six plants hold ranks 1 to 6 of all 200 players on
`z_b` and on `z_b_flop_r`. The planted players' preflop rule (call when the real equity beats the price, fold otherwise,
raise only with 80% or more) folds when behind but never re-raises light, so their preflop aggression stays near 0
and `pf_zr` reaches the bar for three of them (two more sit above every honest player, below 4).

## Honest players

| test | eligible (old) | median | sd | max | ≥ 4 | plants ≥ 4 | eligible (new) | median | sd | max | ≥ 4 | plants ≥ 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z_b | 184 | -0.09 | 1.10 | 2.95 | 0 | 5 of 6 | 189 | -0.01 | 1.14 | 2.93 | 0 | 6 of 6 |
| pf_zr | 189 | -0.02 | 0.97 | 3.07 | 0 | 3 of 6 | 193 | -0.01 | 1.09 | 3.51 | 0 | 3 of 6 |
| z_b_flop_r | 183 | -0.09 | 1.14 | 3.33 | 0 | 4 of 6 | 192 | -0.08 | 1.02 | 2.45 | 0 | 5 of 6 |

Before standardisation the preflop z of honest players sits at a median of 1.12 (sd 1.20,
max 5.01) and the flop z has an sd of 1.62: both are screened on their
population-standardised versions (`pf_zr`, `z_b_flop_r`).

## Hand rules: flags per 1,000 responses

Responses to a raise preflop (PF rules; honest 152,376, planted 6,308) and to a
bet on the flop (FL rules; honest 78,041, planted 3,168).

| rule | honest | planted | ratio |
|---|---|---|---|
| PF1 | 13.1 | 39.5 | 3.0 |
| PF2 | 0.1 | 1.1 | 22.2 |
| PF3 | 1.9 | 1.7 | 0.9 |
| PF4 | 1.5 | 0.5 | 0.3 |
| FL1 | 0.7 | 22.4 | 30.7 |
| FL2 | 0.1 | 0.6 | 4.2 |

PF1 fires on about 1.3% of honest responses to a raise: a single PF1 hand is not
evidence. PF3 and PF4 cannot separate here because the planted players never re-raise light or flat strong hands; FL1
separates strongly. These are simulator numbers, not detection rates for real data.
