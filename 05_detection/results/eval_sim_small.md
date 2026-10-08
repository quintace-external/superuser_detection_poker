# Planted-player evaluation: sim_small

Step 05 tables of the last run (config hash `12388f3a68ccf660`, data up to 2001-01-20). Bars: z_b screen 4 with >= 20 dead and >= 20 winning decisions; the same bar 4 for E's CMH z and the changepoint z; runs percentile >= 95; BH at q = 0.01. Recall and false-flag counts hold on the simulator with planted players only; they are not detection rates for real data.

## Acceptance checks

The acceptance is defined on sim_demo; on sim_small the checks are informational (fewer players and plants).

| check | status | numbers |
|---|---|---|
| SU_FULL_100 flagged by the player-level screen (z_b >= 4, all window) | PASS | z_b 7.32, dead 31 / winning 43 decisions |
| SU_VICTIM_92 flagged by the player-level screen (z_b >= 4, all window) | PASS | z_b 6.49, dead 21 / winning 42 decisions |
| SU_VICTIM_20 not flagged at player level, its pair flagged after BH | N/A | plant not in this dataset |
| SU_ONSET flagged in last15 and not (or weaker) in all | FAIL | z_b last15 1.84 on 10 dead / 8 winning decisions (eligible: False, needs 20/20); z_b all 1.63; changepoint z n/a at week None (planted from 2001-01-16) |
| strong honest regulars flagged by win rate but not by the decision signals | FAIL | 6 strong honest regulars; flagged by win rate (Stage 1 or runs percentile >= 95): 0; of these flagged by the z_b screen in any window: 0, by the article's Stage 2: 0 |
| honest field z_b centred near 0 with sd near 1 (all window) | PASS | n 28, median -0.25 (|median| <= 0.25), sd 1.07 (inside 0.8 to 1.25), p99.9 1.77, max 1.79 |

## Planted players, all window (flagged? value)

| plant | win rate (Stage 1) | runs pct | article (Stage 2) | method (b) z_b | E (CMH z) | pair test (BH) | changepoint z | last15 z_b |
|---|---|---|---|---|---|---|---|---|
| SU_FULL_100 | yes | 95.2 | yes | yes 7.32 | yes 5.57 | 0 pairs pass | no 0.61 | yes 7.00 |
| SU_VICTIM_92 | yes | 100.0 | yes | yes 6.49 | yes 4.47 | no z 2.59, q 0.0145 | no n/a | no 6.10 (not eligible) |
| SU_ONSET | no | n/a | no | no 1.63 | no 1.36 | 0 pairs pass | no n/a | no 1.84 (not eligible) |

## Honest field

| window | eligible | z_b median | sd | p90 | p99 | p99.9 | max | flagged: Stage 1 / Stage 2 / z_b / E |
|---|---|---|---|---|---|---|---|---|
| all | 28 | -0.25 | 1.07 | 0.82 | 1.59 | 1.77 | 1.79 | 0 / 0 / 0 / 0 |
| last30 | 28 | -0.25 | 1.07 | 0.82 | 1.59 | 1.77 | 1.79 | 1 / 0 / 0 / 0 |
| last15 | 18 | -0.25 | 1.12 | 1.20 | 1.75 | 1.81 | 1.82 | 2 / 0 / 0 / 0 |

Runs percentile >= 95: 0 honest players. Changepoint z: median 0.12, p99.9 1.72, 0 at or above the bar. Pair test: 1 honest eligible pairs, 0 pass BH.

Strong honest regulars (control): 6; flagged by win rate 0 (Stage 1 0, runs 0); of those, z_b screen 0, Stage 2 0.
