# Planted-player evaluation: sim_demo

Step 05 tables of the last run (config hash `12388f3a68ccf660`, data up to 2001-03-01). Bars: z_b screen 4 with >= 20 dead and >= 20 winning decisions; the same bar 4 for E's CMH z and the changepoint z; runs percentile >= 95; BH at q = 0.01. Recall and false-flag counts hold on the simulator with planted players only; they are not detection rates for real data.

## Acceptance checks

| check | status | numbers |
|---|---|---|
| SU_FULL_100 flagged by the player-level screen (z_b >= 4, all window) | PASS | z_b 11.53, dead 66 / winning 148 decisions |
| SU_VICTIM_92 flagged by the player-level screen (z_b >= 4, all window) | PASS | z_b 12.66, dead 89 / winning 165 decisions |
| SU_VICTIM_20 not flagged at player level, its pair flagged after BH (q = 0.01) | PASS | player z_b all 2.80, last30 1.66, last15 -0.07 (screen flags: 0); pair test z 11.20, q 0.00000, rank 1 of 308 |
| SU_ONSET flagged in last15 and not (or weaker) in all | FAIL | z_b last15 5.39 on 14 dead / 20 winning decisions (eligible: False, needs 20/20); z_b all 2.08; changepoint z 4.93 at week 2001-02-16 (planted from 2001-02-15) |
| strong honest regulars flagged by win rate but not by the decision signals | PASS | 29 strong honest regulars; flagged by win rate (Stage 1 or runs percentile >= 95): 12; of these flagged by the z_b screen in any window: 0, by the article's Stage 2: 2 |
| honest field z_b centred near 0 with sd near 1 (all window) | PASS | n 285, median 0.03 (|median| <= 0.25), sd 1.14 (inside 0.8 to 1.25), p99.9 3.28, max 3.42 |

## Planted players, all window (flagged? value)

| plant | win rate (Stage 1) | runs pct | article (Stage 2) | method (b) z_b | E (CMH z) | pair test (BH) | changepoint z | last15 z_b |
|---|---|---|---|---|---|---|---|---|
| SU_FULL_100 | yes | 98.2 | yes | yes 11.53 | yes 10.84 | 1 pairs pass | no -0.12 | no 5.16 (not eligible) |
| SU_FULL_050 | yes | 92.1 | yes | yes 5.27 | yes 4.12 | 0 pairs pass | no 0.87 | no 2.13 (not eligible) |
| SU_FULL_025 | yes | 73.1 | no | no 1.96 | no 1.88 | 0 pairs pass | no 1.34 | no 1.38 (not eligible) |
| SU_VICTIM_92 | yes | 99.3 | yes | yes 12.66 | yes 11.19 | yes z 3.81, q 0.0044 | no -0.35 | yes 5.32 |
| SU_VICTIM_50 | yes | 99.6 | yes | yes 5.19 | yes 4.99 | yes z 8.55, q 0.0000 | no 0.05 | no 1.69 |
| SU_VICTIM_20 | yes | 51.3 | no | no 2.80 | no 2.47 | yes z 11.20, q 0.0000 | no 0.53 | no -0.07 |
| SU_ONSET | no | 41.2 | no | no 2.08 | no 2.31 | 0 pairs pass | yes 4.93 | no 5.39 (not eligible) |

## Honest field

| window | eligible | z_b median | sd | p90 | p99 | p99.9 | max | flagged: Stage 1 / Stage 2 / z_b / E |
|---|---|---|---|---|---|---|---|---|
| all | 285 | 0.03 | 1.14 | 1.58 | 2.69 | 3.28 | 3.42 | 26 / 5 / 0 / 0 |
| last30 | 243 | 0.06 | 1.14 | 1.55 | 2.28 | 2.56 | 2.57 | 22 / 4 / 0 / 0 |
| last15 | 163 | -0.03 | 1.11 | 1.45 | 2.29 | 2.54 | 2.54 | 17 / 3 / 0 / 0 |

Runs percentile >= 95: 10 honest players. Changepoint z: median 0.82, p99.9 3.49, 0 at or above the bar. Pair test: 302 honest eligible pairs, 1 pass BH.

Strong honest regulars (control): 29; flagged by win rate 12 (Stage 1 10, runs 4); of those, z_b screen 0, Stage 2 2.
