# Superuser analysis: player 1001

Scope: all submitted hands. Hands analysed: 718 (rows submitted 718; unparseable 0; player not in hand 0; filtered out 0).

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

74 suspicious hands (10.3% of hands analysed). Perfect fold: 38, Good fold: 7, Average fold: 7, Average call: 6, Good call: 5, Perfect call: 4, Perfect bluff: 3, Good bluff: 2, Average bluff: 2.

Top 25 (Perfect first, then Good, Average; larger pots first):

| hand_uid   | start_utc           | decision_label   | category_name          | player_cards   | opponent_cards   | board            | player_hand     | opponent_hand   | key_action                                      |   decision_time_s | snap_decision   |   pot_bb |
|:-----------|:--------------------|:-----------------|:-----------------------|:---------------|:-----------------|:-----------------|:----------------|:----------------|:------------------------------------------------|------------------:|:----------------|---------:|
| h0280      | 2001-01-28 17:58:30 | Perfect fold     | River fold when beaten | ♥K ♥Q          | ♦10 ♣9           | ♠5 ♥7 ♥10 ♣3 ♠10 | Pair            | Three of a Kind | faced 8,308 into ~6,080 (137% pot), folded      |                 2 | True            |     60.8 |
| h0356      | 2001-02-08 02:50:09 | Perfect fold     | River fold when beaten | ♠J ♦J          | ♥J ♥8            | ♠7 ♦5 ♥10 ♣6 ♥4  | Pair            | Straight        | faced 7,434 into ~6,080 (122% pot), folded      |                 4 | False           |     60.8 |
| h0053      | 2001-01-04 09:15:54 | Perfect call     | River bluff-catch      | ♣K ♣8          | ♥4 ♥2            | ♠5 ♥8 ♦3 ♦9 ♦J   | Pair            | High Card       | faced 1,950 into ~1,300 (150% pot), call 1,950  |                12 | False           |     52   |
| h0198      | 2001-01-18 21:51:00 | Perfect call     | River bluff-catch      | ♦A ♣8          | ♦K ♦4            | ♠7 ♦3 ♥4 ♠10 ♠A  | Pair            | Pair            | faced 1,630 into ~1,086 (150% pot), call 1,164  |                11 | False           |     38.8 |
| h0422      | 2001-02-14 07:58:08 | Perfect fold     | River fold when beaten | ♠A ♣A          | ♠K ♠9            | ♥4 ♠8 ♦7 ♦9 ♦K   | Pair            | Two Pair        | faced 2,850 into ~3,800 (75% pot), folded       |                 2 | True            |     38   |
| h0051      | 2001-01-04 04:07:21 | Perfect fold     | River fold when beaten | ♦6 ♣6          | ♥Q ♥10           | ♣2 ♦9 ♣8 ♥6 ♠7   | Three of a Kind | Straight        | faced 2,440 into ~3,252 (75% pot), folded       |                 2 | True            |     32.5 |
| h0338      | 2001-02-05 21:19:32 | Perfect fold     | River fold when beaten | ♥Q ♦8          | ♦J ♣9            | ♠6 ♠Q ♥8 ♥3 ♥10  | Two Pair        | Straight        | faced 8,590 into ~2,820 (305% pot), folded      |                 2 | False           |     28.2 |
| h0497      | 2001-02-18 06:19:14 | Perfect fold     | River fold when beaten | ♣Q ♣7          | ♦A ♦J            | ♦Q ♦3 ♥10 ♦5 ♥8  | Pair            | Flush           | faced 8,250 into ~2,120 (389% pot), folded      |                 7 | False           |     22.2 |
| h0321      | 2001-02-02 17:25:13 | Perfect fold     | River fold when beaten | ♠A ♣2          | ♠10 ♣8           | ♠9 ♣A ♥6 ♥3 ♠7   | Pair            | Straight        | faced 3,100 into ~1,550 (200% pot), folded      |                14 | False           |     18.6 |
| h0521      | 2001-02-20 17:00:18 | Perfect call     | River bluff-catch      | ♦Q ♥6          | ♦J ♠6            | ♦4 ♦A ♠Q ♣7 ♦3   | Pair            | High Card       | faced 466 into ~620 (75% pot), call 466         |                 3 | False           |     15.5 |
| h0716      | 2001-03-03 23:16:07 | Perfect fold     | Early fold, near dead  | ♦9 ♣9          | ♦K ♣Q            | ♣J ♥4 ♠Q ♣4      | Two Pair        | Two Pair        | faced 1,140 into ~1,520 (75% pot), folded       |                 2 | False           |     15.2 |
| h0340      | 2001-02-05 22:36:07 | Perfect call     | River bluff-catch      | ♠K ♣9          | ♠J ♥10           | ♠6 ♣7 ♥5 ♣K ♠Q   | Pair            | High Card       | faced 390 into ~520 (75% pot), call 390         |                 2 | False           |     13   |
| h0438      | 2001-02-15 22:45:08 | Perfect bluff    | Successful river bluff | ♠Q ♠2          | ♠K ♥3            | ♣A ♣K ♠8 ♥10 ♠6  | High Card       | Pair            | bet 976 into ~1,300 (75% pot), opponent folded  |                 3 | False           |     13   |
| h0714      | 2001-03-03 19:35:17 | Perfect bluff    | Successful river bluff | ♣A ♥5          | ♥J ♠6            | ♥8 ♦7 ♥3 ♣6 ♠K   | High Card       | Pair            | bet 650 into ~1,300 (50% pot), opponent folded  |                 3 | False           |     13   |
| h0727      | 2001-03-04 15:59:05 | Perfect fold     | River fold when beaten | ♥K ♥7          | ♦6 ♣3            | ♣2 ♦4 ♣4 ♠5 ♣7   | Two Pair        | Straight        | faced 946 into ~1,260 (75% pot), folded         |                 2 | False           |     12.6 |
| h0365      | 2001-02-08 22:27:13 | Perfect fold     | River fold when beaten | ♥K ♥5          | ♥Q ♥4            | ♦9 ♠5 ♣4 ♦2 ♣Q   | Pair            | Two Pair        | faced 930 into ~1,240 (75% pot), folded         |                 2 | False           |     12.4 |
| h0337      | 2001-02-05 19:30:41 | Perfect bluff    | Successful river bluff | ♠10 ♠6         | ♥J ♠9            | ♥5 ♣J ♣7 ♥Q ♠A   | High Card       | Pair            | raise 856 into ~776 (110% pot), opponent folded |                 3 | False           |      9.3 |
| h0413      | 2001-02-13 14:48:31 | Perfect fold     | River fold when beaten | ♠K ♦8          | ♣J ♥6            | ♠6 ♣8 ♠Q ♠7 ♦6   | Two Pair        | Three of a Kind | faced 1,260 into ~840 (150% pot), folded        |                 2 | False           |      8.4 |
| h0549      | 2001-02-22 02:45:50 | Perfect fold     | River fold when beaten | ♥A ♣J          | ♠Q ♣6            | ♥J ♥K ♣9 ♥10 ♠9  | Two Pair        | Straight        | faced 630 into ~840 (75% pot), folded           |                 2 | False           |      8.4 |
| h0717      | 2001-03-04 00:28:47 | Perfect fold     | River fold when beaten | ♥7 ♦6          | ♠J ♥10           | ♣10 ♠8 ♥4 ♣4 ♥J  | Pair            | Two Pair        | faced 630 into ~840 (75% pot), folded           |                 4 | False           |      8.4 |
| h0735      | 2001-03-05 05:27:40 | Perfect fold     | River fold when beaten | ♦K ♣3          | ♣Q ♣8            | ♥7 ♦Q ♦3 ♣5 ♥2   | Pair            | Pair            | faced 1,260 into ~840 (150% pot), folded        |                 2 | False           |      8.4 |
| h0743      | 2001-03-05 23:53:26 | Perfect fold     | River fold when beaten | ♠J ♥5          | ♦A ♠3            | ♥K ♣K ♦5 ♥A ♠4   | Two Pair        | Two Pair        | faced 630 into ~840 (75% pot), folded           |                 4 | False           |      8.4 |
| h0755      | 2001-03-06 15:01:42 | Perfect fold     | Early fold, near dead  | ♣A ♥K          | ♣Q ♦10           | ♦Q ♥Q ♣10        | Pair            | Full House      | faced 420 into ~630 (67% pot), folded           |                 5 | False           |      8.4 |
| h0773      | 2001-03-07 16:48:13 | Perfect fold     | River fold when beaten | ♦Q ♦4          | ♣10 ♥9           | ♣K ♣4 ♥7 ♠10 ♠A  | Pair            | Pair            | faced 210 into ~840 (25% pot), folded           |                 3 | False           |      8.4 |
| h0180      | 2001-01-16 18:18:59 | Perfect fold     | River fold when beaten | ♣10 ♣5         | ♦Q ♦8            | ♥3 ♠3 ♣Q ♠9 ♠J   | Pair            | Two Pair        | faced 716 into ~720 (99% pot), folded           |                 1 | False           |      8.2 |

## Predicted range of the opponent

Equity of the key decision against all hands (step 02), against the opponent's predicted range (step 04) and against his real cards, with the pot odds. Category R: hands that only the range rule finds (fold with range equity >= pot odds + 0.24 while drawing dead, or a call with range equity <= pot odds - 0.20 while ahead); they are not part of the escalation rule.

| hand_uid   | category   | decision_label   | street   | eq_vs_all_hands   | eq_vs_range   | eq_vs_real   | pot_odds   | range_verdict                                                |
|:-----------|:-----------|:-----------------|:---------|:------------------|:--------------|:-------------|:-----------|:-------------------------------------------------------------|
| h0280      | A          | Perfect fold     | river    | 38%               | 15%           | 0%           | 37%        | range agrees with the fold                                   |
| h0356      | A          | Perfect fold     | river    | 55%               | 14%           | 0%           | 35%        | range agrees with the fold                                   |
| h0053      | B          | Perfect call     | river    | 60%               | 23%           | 100%         | 38%        | range says fold (thin), continued                            |
| h0198      | B          | Perfect call     | river    | 80%               | 40%           | 100%         | 30%        | range agrees with the call                                   |
| h0422      | A          | Perfect fold     | river    | 80%               | 35%           | 0%           | 30%        | range says call (thin), folded                               |
| h0051      | A          | Perfect fold     | river    | 66%               | 53%           | 0%           | 30%        | range says call (thin), folded                               |
| h0338      | A          | Perfect fold     | river    | 91%               | 50%           | 0%           | 43%        | range says call (thin), folded                               |
| h0497      | A          | Perfect fold     | river    | 80%               | 55%           | 0%           | 44%        | range says call (thin), folded                               |
| h0321      | A          | Perfect fold     | river    | 81%               | 40%           | 0%           | 38%        | range says call (thin), folded                               |
| h0521      | B          | Perfect call     | river    | 70%               | 30%           | 100%         | 30%        | range says fold (thin), continued                            |
| h0716      | D          | Perfect fold     | turn     | 63%               | 36%           | 5%           | 30%        | range says call (thin), folded                               |
| h0340      | B          | Perfect call     | river    | 83%               | 76%           | 100%         | 30%        | range agrees with the call                                   |
| h0438      | C          | Perfect bluff    | river    | 33%               | 17%           | 0%           | –          | bet with 17% equity against the range                        |
| h0714      | C          | Perfect bluff    | river    | 29%               | 19%           | 0%           | –          | bet with 19% equity against the range                        |
| h0727      | A          | Perfect fold     | river    | 76%               | 31%           | 0%           | 30%        | range says call (thin), folded                               |
| h0365      | A          | Perfect fold     | river    | 64%               | 29%           | 0%           | 30%        | range agrees with the fold                                   |
| h0337      | C          | Perfect bluff    | river    | 24%               | 7%            | 0%           | 17%        | range says fold (thin), continued                            |
| h0413      | A          | Perfect fold     | river    | 66%               | 15%           | 0%           | 38%        | range agrees with the fold                                   |
| h0549      | A          | Perfect fold     | river    | 58%               | 18%           | 0%           | 30%        | range agrees with the fold                                   |
| h0717      | A          | Perfect fold     | river    | 7%                | 3%            | 0%           | 30%        | range agrees with the fold                                   |
| h0735      | A          | Perfect fold     | river    | 54%               | 20%           | 0%           | 38%        | range agrees with the fold                                   |
| h0743      | A          | Perfect fold     | river    | 69%               | 37%           | 0%           | 30%        | range says call (thin), folded                               |
| h0755      | D          | Perfect fold     | flop     | 69%               | 36%           | 1%           | 17%        | range says call (thin), folded                               |
| h0773      | A          | Perfect fold     | river    | 48%               | 21%           | 0%           | 17%        | range says call (thin), folded                               |
| h0180      | A          | Perfect fold     | river    | 23%               | 5%            | 0%           | 30%        | range agrees with the fold                                   |
| h0130      | A          | Perfect fold     | river    | 63%               | 38%           | 0%           | 17%        | range says call (thin), folded                               |
| h0342      | A          | Perfect fold     | river    | 57%               | 30%           | 0%           | 30%        | range agrees with the fold                                   |
| h0190      | D          | Perfect fold     | flop     | 26%               | 15%           | 4%           | 20%        | range agrees with the fold                                   |
| h0192      | A          | Perfect fold     | river    | 35%               | 11%           | 0%           | 17%        | range agrees with the fold                                   |
| h0222      | D          | Perfect fold     | turn     | 43%               | 29%           | 5%           | 30%        | range agrees with the fold                                   |
| h0271      | A          | Perfect fold     | river    | 27%               | 26%           | 0%           | 30%        | range agrees with the fold                                   |
| h0355      | D          | Perfect fold     | turn     | 14%               | 6%            | 0%           | 30%        | range agrees with the fold                                   |
| h0447      | A          | Perfect fold     | river    | 55%               | 41%           | 0%           | 30%        | range says call (thin), folded                               |
| h0657      | A          | Perfect fold     | river    | 11%               | 3%            | 0%           | 17%        | range agrees with the fold                                   |
| h0072      | A          | Perfect fold     | river    | 62%               | 50%           | 0%           | 30%        | range says call (thin), folded                               |
| h0227      | D          | Perfect fold     | turn     | 77%               | 65%           | 0%           | 30%        | range says call: range equity +35% over the pot odds, folded |
| h0276      | D          | Perfect fold     | turn     | 50%               | 41%           | 5%           | 30%        | range says call (thin), folded                               |
| h0418      | A          | Perfect fold     | river    | 33%               | 16%           | 0%           | 30%        | range agrees with the fold                                   |
| h0500      | D          | Perfect fold     | turn     | 54%               | 41%           | 5%           | 33%        | range says call (thin), folded                               |
| h0600      | A          | Perfect fold     | river    | 33%               | 20%           | 0%           | 30%        | range agrees with the fold                                   |
| h0616      | D          | Perfect fold     | turn     | 15%               | 10%           | 0%           | 25%        | range agrees with the fold                                   |
| h0736      | D          | Perfect fold     | flop     | 60%               | 54%           | 2%           | 17%        | range says call: range equity +38% over the pot odds, folded |
| h0196      | A          | Perfect fold     | river    | 20%               | 14%           | 0%           | 17%        | range agrees with the fold                                   |
| h0725      | A          | Perfect fold     | river    | 35%               | 30%           | 0%           | 30%        | range agrees with the fold                                   |
| h0749      | A          | Perfect fold     | river    | 51%               | 52%           | 0%           | 38%        | range says call (thin), folded                               |
| h0254      | B          | Good call        | river    | 96%               | 79%           | 100%         | 38%        | range agrees with the call                                   |
| h0522      | B          | Good call        | river    | 96%               | 85%           | 100%         | 30%        | range agrees with the call                                   |
| h0038      | B          | Good call        | river    | 89%               | 88%           | 100%         | 17%        | range agrees with the call                                   |
| h0299      | B          | Good call        | river    | 82%               | 73%           | 100%         | 17%        | range agrees with the call                                   |
| h0705      | C          | Good bluff       | river    | 11%               | 9%            | 0%           | –          | bet with 9% equity against the range                         |
| h0232      | B          | Good call        | river    | 88%               | 67%           | 100%         | 30%        | range agrees with the call                                   |
| h0721      | A          | Good fold        | river    | 7%                | 3%            | 0%           | 47%        | range agrees with the fold                                   |
| h0197      | D          | Good fold        | turn     | 54%               | 25%           | 7%           | 30%        | range agrees with the fold                                   |
| h0322      | D          | Good fold        | turn     | 43%               | 28%           | 9%           | 30%        | range agrees with the fold                                   |
| h0151      | A          | Good fold        | river    | 7%                | 4%            | 0%           | 38%        | range agrees with the fold                                   |
| h0268      | D          | Good fold        | turn     | 38%               | 30%           | 7%           | 17%        | range says call (thin), folded                               |
| h0586      | C          | Good bluff       | river    | 5%                | 6%            | 0%           | –          | bet with 6% equity against the range                         |
| h0758      | D          | Good fold        | turn     | 73%               | 65%           | 9%           | 25%        | range says call: range equity +40% over the pot odds, folded |
| h0583      | A          | Good fold        | river    | 10%               | 12%           | 0%           | 30%        | range agrees with the fold                                   |
| h0404      | B          | Average call     | river    | 97%               | 80%           | 100%         | 35%        | range agrees with the call                                   |
| h0416      | B          | Average call     | river    | 95%               | 85%           | 100%         | 38%        | range agrees with the call                                   |
| h0118      | C          | Average bluff    | river    | 2%                | 1%            | 0%           | –          | bet with 1% equity against the range                         |
| h0183      | B          | Average call     | river    | 90%               | 80%           | 100%         | 30%        | range agrees with the call                                   |
| h0552      | B          | Average call     | river    | 98%               | 96%           | 100%         | 30%        | range agrees with the call                                   |
| h0318      | B          | Average call     | river    | 91%               | 85%           | 100%         | 30%        | range agrees with the call                                   |
| h0366      | C          | Average bluff    | river    | 15%               | 13%           | 0%           | –          | bet with 13% equity against the range                        |
| h0400      | B          | Average call     | river    | 98%               | 94%           | 100%         | 30%        | range agrees with the call                                   |
| h0702      | D          | Average fold     | turn     | 16%               | 16%           | 7%           | 38%        | range agrees with the fold                                   |
| h0647      | D          | Average fold     | flop     | 27%               | 20%           | 6%           | 30%        | range agrees with the fold                                   |
| h0745      | D          | Average fold     | flop     | 21%               | 16%           | 6%           | 35%        | range agrees with the fold                                   |
| h0771      | D          | Average fold     | flop     | 31%               | 23%           | 5%           | 30%        | range agrees with the fold                                   |
| h0205      | D          | Average fold     | turn     | 40%               | 31%           | 9%           | 30%        | range says call (thin), folded                               |
| h0546      | D          | Average fold     | turn     | 41%               | 32%           | 7%           | 30%        | range says call (thin), folded                               |
| h0594      | D          | Average fold     | turn     | 16%               | 9%            | 7%           | 25%        | range agrees with the fold                                   |
