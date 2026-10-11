# Superuser analysis: player 1001

Scope: all submitted hands. Hands analysed: 718 (rows submitted 718; unparseable 0; player not in hand 0; filtered out 0).

## Verdict

**Article rule**: Last 15 days (2001-02-21 03:09:25 to 2001-03-08 03:09:25 UTC): 27 suspicious of 250 hands (10.8%). Rule: ratio > 5% and more than 5 suspicious hands. **ESCALATE for manual review**.

**Street-tails rule** (the top tail on several streets at once; escalate at 2 or more of the three streets, review at 1): preflop HIT (pf_zr 6.31); flop HIT (z_b_flop 4.84, flop e_disc 0.697); turn + river HIT (z_b 3.43, e_disc 0.466). 3 of 3 streets: **ESCALATE for manual review**.

**Pairs**: he beats 1 opponent beyond his own level and beyond what the rest of the reference field does against them: 1002 (top 0.03% of the field's pairs).

## Metrics by street

His value on each street's tests against the players of the reference field with the minimum sample (config/step05_report.toml [report.metrics]). Percentile = the share of them he beats (ties count half, himself included); Best other player = the best value of the others. The reference field is a sample of a larger full-information population; percentiles are within that sample.

| Street       | Metric                         | What it measures                                                                                                                                                | Value      | Percentile   | Best other player   |
|:-------------|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:-------------|:--------------------|
| Preflop      | pf_z                           | Folds when behind the opponent's real hand and raises when ahead, against the field at the same strength against his line (CMH inside situation x 5% band of r) | 7.11       | 99.9         | 2.81                |
| Preflop      | Aggression z                   | Raises when the opponent's real hand is worse                                                                                                                   | 6.37       | 99.9         | 3.00                |
| Preflop      | Fold z                         | Folds when the opponent's real hand is better                                                                                                                   | 3.69       | 99.9         | 2.95                |
| Preflop      | Fold gap (raw)                 | Fold rate facing a raise when behind (q < 40%) minus when ahead (q >= 55%)                                                                                      | 60.6 pp    | 99.0         | 69.3 pp             |
| Preflop      | Aggression gap (raw)           | Raise rate when ahead minus when behind                                                                                                                         | 52.4 pp    | 99.3         | 59.2 pp             |
| Preflop      | Light re-raise success         | Re-raises with a hand weak against the line (r < 45%) that the opponent folds to at once                                                                        | 94.7% (19) | 99.8         | 93.3% (15)          |
| Preflop      | 3-bet success                  | The opponent folds to his 3-bet at once (needs no cards)                                                                                                        | 65.3% (75) | 82.9         | 100.0% (5)          |
| Preflop      | 4-bet success                  | The opponent folds to his 4-bet at once (needs no cards)                                                                                                        | 71.4% (7)  | 92.1         | 100.0% (8)          |
| Preflop      | 5-bet success                  | The opponent folds to his 5-bet at once (needs no cards)                                                                                                        | –          | –            | 66.7% (6)           |
| Flop         | Flop e_disc                    | Folds when behind and continues when ahead beyond the field at the same range strength (behind = q < 20%)                                                       | 0.697      | 99.9         | 0.319               |
| Flop         | z_b_flop                       | The same as a z-statistic (method b on the flop)                                                                                                                | 4.84       | 99.4         | 5.19                |
| Turn + river | e_disc                         | The same on the turn and the river (dead = q < 2%); volume-free                                                                                                 | 0.466      | 99.9         | 0.357               |
| Turn + river | Excess folds while dead        | Fold rate when dead (q < 2%) minus the field's at the same range decile                                                                                         | +24.0 pp   | 99.4         | +24.7 pp            |
| Turn + river | Excess continues while winning | Continue rate when winning (q >= 50%) minus the field's at the same range decile                                                                                | +7.3 pp    | 78.3         | +25.8 pp            |
| Turn + river | z_b                            | z-test of the two excesses (method b, the article's core test)                                                                                                  | 3.43       | 99.1         | 7.60                |

## Opponents he beat

Every ordered pair of the reference field (he against one opponent, in the hands the two of them played out to the end) is scored on its parts: his win rate and showdowns against that opponent, his calls, bets, all-ins and preflop calls and raises that were ahead of the opponent's real hand, and his folds that were behind it. Each part compares what he does against that opponent with what he does against everyone else and with what everyone else does against that opponent (z of the leave-pair-out residual, a hand counted once); the score is the sum of the parts' z over the square root of the sum of their correlations (so an honest pair scores around 0 with a spread of about 1), and pairs with at least two parts are ranked. A victim: score at least 3 and a rank inside the top 1% of the ranked pairs (at least the top 10). Pot-size columns: the same pair seen only through money and showdowns (pots of 30bb or more); the rates with a z against the pool of all pairs (stack-offs turned, so fewer gives a higher z), the growth and the net with a z against zero.

|   Opponent |   Hands |   bb/100 | Showdowns won   |   z win rate |   z showdowns |   z calls |   z bets | z all-ins   |   z preflop |   z folds |   Score | Pair                           |
|-----------:|--------:|---------:|:----------------|-------------:|--------------:|----------:|---------:|:------------|------------:|----------:|--------:|:-------------------------------|
|       1002 |     387 |    240.8 | 69% of 69       |         0.99 |           2.3 |      2.76 |     2.81 | –           |        0.44 |      2.27 |    3.58 | top 0.03% of the field's pairs |

The same pairs by pot size (pots of 30bb or more):

|   Opponent | W$SD, pots >= 30bb   |   z W$SD | W$SD growth   |   z growth | River bets won   | z river bets   | River calls won   | z river calls   | His stack-offs   | z stack-offs   | Net in pots >= 30bb   |   z net |
|-----------:|:---------------------|---------:|:--------------|-----------:|:-----------------|:---------------|:------------------|:----------------|:-----------------|:---------------|:----------------------|--------:|
|       1002 | 96% (12)             |     3.18 | +33 pp        |       2.06 | 88% (4)          | –              | 100% (5)          | –               | 0% (6)           | –              | +491.1 bb (26)        |    2.59 |

## Article scorecard

The article's seven signals (turn and river decisions, equity against the opponent's real hand) against the reference field. The reference field is a sample of a larger full-information population; percentiles are within that sample.

| Signal                                      | Suspect   | Field percentile   | Field median   | Top 10 winners (mean)   | Main opponent vs the rest   |
|:--------------------------------------------|:----------|:-------------------|:---------------|:------------------------|:----------------------------|
| Oracle-fold ratio: postflop, pair or better | 94.4%     | 100th              | 64.3%          | 62.9%                   | 61.1%                       |
| Oracle-fold ratio: turn                     | 92.9%     | 100th              | 69.1%          | 68.3%                   | 65.9%                       |
| Oracle-fold ratio: river                    | 91.2%     | 97th               | 77.5%          | 75.6%                   | 74.0%                       |
| Fold to value bets                          | 86.3%     | 100th              | 50.0%          | 54.8%                   | 57.1%                       |
| Fold to bluffs                              | 15.1%     | 10th               | 25.3%          | 26.6%                   | 28.8%                       |
| Discrimination gap                          | 71.2 pts  | 100th              | 25.3 pts       | 28.2 pts                | 28.3 pts                    |
| Bluff success                               | 41.1%     | 97th               | 24.6%          | 27.8%                   | 29.0%                       |

*Percentile and median among the field's players with at least 20 qualifying decisions on the signal; low is suspicious on folds to bluffs.*

*Main opponent 1002: his games against everyone but the suspect.*

## Observer metrics

Money and showdowns only: what someone who reads the hand histories sees without the hidden cards. The percentile is turned so that higher is always more suspicious.

| Metric                    | What it measures                                                                                                                       | Value    |   Cases | Suspicion percentile   |
|:--------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:---------|--------:|:-----------------------|
| bb/100                    | Net per 100 hands                                                                                                                      | +125.6   |     720 | –                      |
| Big-pot net per 100 hands | Net in pots of 200bb or more, per 100 hands                                                                                            | +29.9    |     720 | –                      |
| WTSD                      | Went to showdown, of the hands that saw the flop (lower = more suspicious)                                                             | 23.0%    |     305 | –                      |
| WTSD, single-raised pots  | The same, pot at the flop under 15bb                                                                                                   | 23.2%    |     271 | –                      |
| WTSD, 3-bet pots          | The same, pot at the flop 15-40bb                                                                                                      | 18.8%    |      32 | –                      |
| WTSD, 4-bet pots          | The same, pot at the flop 40bb or more                                                                                                 | 50.0%    |       2 | –                      |
| W$SD                      | Showdowns won with two players at the showdown (a split counts 1/2)                                                                    | 69.3%    |      70 | –                      |
| W$SD, small pots          | The same, final pot under 30bb                                                                                                         | 63.8%    |      58 | –                      |
| W$SD, medium pots         | The same, final pot 30-100bb                                                                                                           | 95.0%    |      10 | –                      |
| W$SD, large pots          | The same, final pot 100-200bb                                                                                                          | –        |       0 | –                      |
| W$SD, the biggest pots    | The same, final pot 200bb or more                                                                                                      | 100.0%   |       2 | –                      |
| W$SD growth               | W$SD in the biggest pots minus in the small ones                                                                                       | +36.2 pp |       2 | –                      |
| River bets won            | He bet or raised the river, the pot was over 30bb and the hand reached a showdown: share won                                           | 87.5%    |       4 | –                      |
| River calls won           | He called a river bet heads-up, the pot after the call over 30bb: share won                                                            | 100.0%   |       5 | –                      |
| River call frequency      | Those calls over the river bets he faced that way                                                                                      | 25.0%    |      20 | –                      |
| River stack-offs          | He saw the river and over 30bb went in on it: share of those hands in which he lost 90% or more of his stack (lower = more suspicious) | 0.0%     |       6 | –                      |

Net by final pot: under 30bb +413.5 bb, 30-100bb +276.0 bb, 100-200bb +0.0 bb, 200bb or more +215.1 bb.

*Ranks among the field's players with at least 1,000 hands and, for a rate, 20 cases (config/step05_screening.toml [step05.observer]).*

## Player statistics

| stat                 | value   |
|:---------------------|:--------|
| Hands                | 718     |
| Net win (chips)      | +90,242 |
| bb/100               | +125.7  |
| VPIP                 | 66.3%   |
| PFR                  | 48.6%   |
| 3-bet                | 24.5%   |
| Fold to 3-bet        | 65.8%   |
| Flop c-bet           | 76.6%   |
| Fold to c-bet        | 36.9%   |
| Aggression factor    | 4.19    |
| Aggression frequency | 63.4%   |
| WTSD                 | 22.7%   |
| W$SD                 | 65.2%   |
| Fold to river bet    | 63.0%   |
| River call win rate  | 69.2%   |
| Median action time   | 3 s     |
| Mean action time     | 3.9 s   |
| Fastest action       | 0.0 s   |
| Slowest action       | 17.0 s  |

## Decision times

Seconds per action, by action type.

| Action   |   Actions |   Mean (s) |   Median (s) |   Min (s) |   Max (s) |
|:---------|----------:|-----------:|-------------:|----------:|----------:|
| check    |       343 |        2.7 |            2 |         0 |        15 |
| fold     |       307 |        3.5 |            3 |         0 |        17 |
| call     |       200 |        4   |            3 |         1 |        15 |
| bet      |       263 |        4.6 |            4 |         1 |        16 |
| raise    |       374 |        4.6 |            4 |         1 |        15 |
| all      |      1487 |        3.9 |            3 |         0 |        17 |

## Hidden-information signals

| signal                                     | value           |
|:-------------------------------------------|:----------------|
| Perfect fold ratio, postflop               | 86.3% (82 / 95) |
| Perfect fold ratio, turn                   | 92.9% (26 / 28) |
| Perfect fold ratio, river                  | 91.2% (31 / 34) |
| Pair-or-better postflop folds while behind | 90.3% (56 / 62) |
| Bluff index (bluff success rate)           | 41.1% (30 / 73) |
| Fold when facing a value bet               | 86.3% (82 / 95) |
| Fold when facing a bluff                   | 15.3% (13 / 85) |
| Discrimination gap                         | 71.0 pts        |
| River decision accuracy                    | 85.2% (46 / 54) |

## Escalation rule

Last 15 days (2001-02-21 03:09:25 to 2001-03-08 03:09:25 UTC): 27 suspicious of 250 hands (10.8%). Rule: ratio > 5% and more than 5 suspicious hands. **ESCALATE for manual review**.

## Suspicious hands

74 flagged hands (10.3% of hands analysed).

Top 25:

| hand_uid   | start_utc           | player_cards   | opponent_cards   | board            | player_hand     | opponent_hand   | key_action                                      |   decision_time_s |   pot_bb |
|:-----------|:--------------------|:---------------|:-----------------|:-----------------|:----------------|:----------------|:------------------------------------------------|------------------:|---------:|
| h0280      | 2001-01-28 17:58:30 | ♥K ♥Q          | ♦10 ♣9           | ♠5 ♥7 ♥10 ♣3 ♠10 | Pair            | Three of a Kind | faced 8,308 into ~6,080 (137% pot), folded      |                 2 |     60.8 |
| h0356      | 2001-02-08 02:50:09 | ♠J ♦J          | ♥J ♥8            | ♠7 ♦5 ♥10 ♣6 ♥4  | Pair            | Straight        | faced 7,434 into ~6,080 (122% pot), folded      |                 4 |     60.8 |
| h0053      | 2001-01-04 09:15:54 | ♣K ♣8          | ♥4 ♥2            | ♠5 ♥8 ♦3 ♦9 ♦J   | Pair            | High Card       | faced 1,950 into ~1,300 (150% pot), call 1,950  |                12 |     52   |
| h0198      | 2001-01-18 21:51:00 | ♦A ♣8          | ♦K ♦4            | ♠7 ♦3 ♥4 ♠10 ♠A  | Pair            | Pair            | faced 1,630 into ~1,086 (150% pot), call 1,164  |                11 |     38.8 |
| h0422      | 2001-02-14 07:58:08 | ♠A ♣A          | ♠K ♠9            | ♥4 ♠8 ♦7 ♦9 ♦K   | Pair            | Two Pair        | faced 2,850 into ~3,800 (75% pot), folded       |                 2 |     38   |
| h0051      | 2001-01-04 04:07:21 | ♦6 ♣6          | ♥Q ♥10           | ♣2 ♦9 ♣8 ♥6 ♠7   | Three of a Kind | Straight        | faced 2,440 into ~3,252 (75% pot), folded       |                 2 |     32.5 |
| h0338      | 2001-02-05 21:19:32 | ♥Q ♦8          | ♦J ♣9            | ♠6 ♠Q ♥8 ♥3 ♥10  | Two Pair        | Straight        | faced 8,590 into ~2,820 (305% pot), folded      |                 2 |     28.2 |
| h0497      | 2001-02-18 06:19:14 | ♣Q ♣7          | ♦A ♦J            | ♦Q ♦3 ♥10 ♦5 ♥8  | Pair            | Flush           | faced 8,250 into ~2,120 (389% pot), folded      |                 7 |     22.2 |
| h0321      | 2001-02-02 17:25:13 | ♠A ♣2          | ♠10 ♣8           | ♠9 ♣A ♥6 ♥3 ♠7   | Pair            | Straight        | faced 3,100 into ~1,550 (200% pot), folded      |                14 |     18.6 |
| h0521      | 2001-02-20 17:00:18 | ♦Q ♥6          | ♦J ♠6            | ♦4 ♦A ♠Q ♣7 ♦3   | Pair            | High Card       | faced 466 into ~620 (75% pot), call 466         |                 3 |     15.5 |
| h0716      | 2001-03-03 23:16:07 | ♦9 ♣9          | ♦K ♣Q            | ♣J ♥4 ♠Q ♣4      | Two Pair        | Two Pair        | faced 1,140 into ~1,520 (75% pot), folded       |                 2 |     15.2 |
| h0340      | 2001-02-05 22:36:07 | ♠K ♣9          | ♠J ♥10           | ♠6 ♣7 ♥5 ♣K ♠Q   | Pair            | High Card       | faced 390 into ~520 (75% pot), call 390         |                 2 |     13   |
| h0438      | 2001-02-15 22:45:08 | ♠Q ♠2          | ♠K ♥3            | ♣A ♣K ♠8 ♥10 ♠6  | High Card       | Pair            | bet 976 into ~1,300 (75% pot), opponent folded  |                 3 |     13   |
| h0714      | 2001-03-03 19:35:17 | ♣A ♥5          | ♥J ♠6            | ♥8 ♦7 ♥3 ♣6 ♠K   | High Card       | Pair            | bet 650 into ~1,300 (50% pot), opponent folded  |                 3 |     13   |
| h0727      | 2001-03-04 15:59:05 | ♥K ♥7          | ♦6 ♣3            | ♣2 ♦4 ♣4 ♠5 ♣7   | Two Pair        | Straight        | faced 946 into ~1,260 (75% pot), folded         |                 2 |     12.6 |
| h0365      | 2001-02-08 22:27:13 | ♥K ♥5          | ♥Q ♥4            | ♦9 ♠5 ♣4 ♦2 ♣Q   | Pair            | Two Pair        | faced 930 into ~1,240 (75% pot), folded         |                 2 |     12.4 |
| h0337      | 2001-02-05 19:30:41 | ♠10 ♠6         | ♥J ♠9            | ♥5 ♣J ♣7 ♥Q ♠A   | High Card       | Pair            | raise 856 into ~776 (110% pot), opponent folded |                 3 |      9.3 |
| h0413      | 2001-02-13 14:48:31 | ♠K ♦8          | ♣J ♥6            | ♠6 ♣8 ♠Q ♠7 ♦6   | Two Pair        | Three of a Kind | faced 1,260 into ~840 (150% pot), folded        |                 2 |      8.4 |
| h0549      | 2001-02-22 02:45:50 | ♥A ♣J          | ♠Q ♣6            | ♥J ♥K ♣9 ♥10 ♠9  | Two Pair        | Straight        | faced 630 into ~840 (75% pot), folded           |                 2 |      8.4 |
| h0717      | 2001-03-04 00:28:47 | ♥7 ♦6          | ♠J ♥10           | ♣10 ♠8 ♥4 ♣4 ♥J  | Pair            | Two Pair        | faced 630 into ~840 (75% pot), folded           |                 4 |      8.4 |
| h0735      | 2001-03-05 05:27:40 | ♦K ♣3          | ♣Q ♣8            | ♥7 ♦Q ♦3 ♣5 ♥2   | Pair            | Pair            | faced 1,260 into ~840 (150% pot), folded        |                 2 |      8.4 |
| h0743      | 2001-03-05 23:53:26 | ♠J ♥5          | ♦A ♠3            | ♥K ♣K ♦5 ♥A ♠4   | Two Pair        | Two Pair        | faced 630 into ~840 (75% pot), folded           |                 4 |      8.4 |
| h0755      | 2001-03-06 15:01:42 | ♣A ♥K          | ♣Q ♦10           | ♦Q ♥Q ♣10        | Pair            | Full House      | faced 420 into ~630 (67% pot), folded           |                 5 |      8.4 |
| h0773      | 2001-03-07 16:48:13 | ♦Q ♦4          | ♣10 ♥9           | ♣K ♣4 ♥7 ♠10 ♠A  | Pair            | Pair            | faced 210 into ~840 (25% pot), folded           |                 3 |      8.4 |
| h0180      | 2001-01-16 18:18:59 | ♣10 ♣5         | ♦Q ♦8            | ♥3 ♠3 ♣Q ♠9 ♠J   | Pair            | Two Pair        | faced 716 into ~720 (99% pot), folded           |                 1 |      8.2 |

## Predicted range of the opponent

Equity of the key decision against all hands (step 02), against the opponent's predicted range (step 04) and against his real cards, with the pot odds.

| hand_uid   | street   | eq_vs_all_hands   | eq_vs_range   | eq_vs_real   | pot_odds   |
|:-----------|:---------|:------------------|:--------------|:-------------|:-----------|
| h0280      | river    | 38%               | 15%           | 0%           | 37%        |
| h0356      | river    | 55%               | 14%           | 0%           | 35%        |
| h0053      | river    | 60%               | 23%           | 100%         | 38%        |
| h0198      | river    | 80%               | 40%           | 100%         | 30%        |
| h0422      | river    | 80%               | 35%           | 0%           | 30%        |
| h0051      | river    | 66%               | 53%           | 0%           | 30%        |
| h0338      | river    | 91%               | 50%           | 0%           | 43%        |
| h0497      | river    | 80%               | 55%           | 0%           | 44%        |
| h0321      | river    | 81%               | 40%           | 0%           | 38%        |
| h0521      | river    | 70%               | 30%           | 100%         | 30%        |
| h0716      | turn     | 63%               | 36%           | 5%           | 30%        |
| h0340      | river    | 83%               | 76%           | 100%         | 30%        |
| h0438      | river    | 33%               | 17%           | 0%           | –          |
| h0714      | river    | 29%               | 19%           | 0%           | –          |
| h0727      | river    | 76%               | 31%           | 0%           | 30%        |
| h0365      | river    | 64%               | 29%           | 0%           | 30%        |
| h0337      | river    | 24%               | 7%            | 0%           | 17%        |
| h0413      | river    | 66%               | 15%           | 0%           | 38%        |
| h0549      | river    | 58%               | 18%           | 0%           | 30%        |
| h0717      | river    | 7%                | 3%            | 0%           | 30%        |
| h0735      | river    | 54%               | 20%           | 0%           | 38%        |
| h0743      | river    | 69%               | 37%           | 0%           | 30%        |
| h0755      | flop     | 69%               | 36%           | 1%           | 17%        |
| h0773      | river    | 48%               | 21%           | 0%           | 17%        |
| h0180      | river    | 23%               | 5%            | 0%           | 30%        |
| h0130      | river    | 63%               | 38%           | 0%           | 17%        |
| h0342      | river    | 57%               | 30%           | 0%           | 30%        |
| h0190      | flop     | 26%               | 15%           | 4%           | 20%        |
| h0192      | river    | 35%               | 11%           | 0%           | 17%        |
| h0222      | turn     | 43%               | 29%           | 5%           | 30%        |
| h0271      | river    | 27%               | 26%           | 0%           | 30%        |
| h0355      | turn     | 14%               | 6%            | 0%           | 30%        |
| h0447      | river    | 55%               | 41%           | 0%           | 30%        |
| h0657      | river    | 11%               | 3%            | 0%           | 17%        |
| h0072      | river    | 62%               | 50%           | 0%           | 30%        |
| h0227      | turn     | 77%               | 65%           | 0%           | 30%        |
| h0276      | turn     | 50%               | 41%           | 5%           | 30%        |
| h0418      | river    | 33%               | 16%           | 0%           | 30%        |
| h0500      | turn     | 54%               | 41%           | 5%           | 33%        |
| h0600      | river    | 33%               | 20%           | 0%           | 30%        |
| h0616      | turn     | 15%               | 10%           | 0%           | 25%        |
| h0736      | flop     | 60%               | 54%           | 2%           | 17%        |
| h0196      | river    | 20%               | 14%           | 0%           | 17%        |
| h0725      | river    | 35%               | 30%           | 0%           | 30%        |
| h0749      | river    | 51%               | 52%           | 0%           | 38%        |
| h0254      | river    | 96%               | 79%           | 100%         | 38%        |
| h0522      | river    | 96%               | 85%           | 100%         | 30%        |
| h0038      | river    | 89%               | 88%           | 100%         | 17%        |
| h0299      | river    | 82%               | 73%           | 100%         | 17%        |
| h0705      | river    | 11%               | 9%            | 0%           | –          |
| h0232      | river    | 88%               | 67%           | 100%         | 30%        |
| h0721      | river    | 7%                | 3%            | 0%           | 47%        |
| h0197      | turn     | 54%               | 25%           | 7%           | 30%        |
| h0322      | turn     | 43%               | 28%           | 9%           | 30%        |
| h0151      | river    | 7%                | 4%            | 0%           | 38%        |
| h0268      | turn     | 38%               | 30%           | 7%           | 17%        |
| h0586      | river    | 5%                | 6%            | 0%           | –          |
| h0758      | turn     | 73%               | 65%           | 9%           | 25%        |
| h0583      | river    | 10%               | 12%           | 0%           | 30%        |
| h0404      | river    | 97%               | 80%           | 100%         | 35%        |
| h0416      | river    | 95%               | 85%           | 100%         | 38%        |
| h0118      | river    | 2%                | 1%            | 0%           | –          |
| h0183      | river    | 90%               | 80%           | 100%         | 30%        |
| h0552      | river    | 98%               | 96%           | 100%         | 30%        |
| h0318      | river    | 91%               | 85%           | 100%         | 30%        |
| h0366      | river    | 15%               | 13%           | 0%           | –          |
| h0400      | river    | 98%               | 94%           | 100%         | 30%        |
| h0702      | turn     | 16%               | 16%           | 7%           | 38%        |
| h0647      | flop     | 27%               | 20%           | 6%           | 30%        |
| h0745      | flop     | 21%               | 16%           | 6%           | 35%        |
| h0771      | flop     | 31%               | 23%           | 5%           | 30%        |
| h0205      | turn     | 40%               | 31%           | 9%           | 30%        |
| h0546      | turn     | 41%               | 32%           | 7%           | 30%        |
| h0594      | turn     | 16%               | 9%            | 7%           | 25%        |
