# Mined hand histories

Mined hand histories are records collected by an observer at the tables. Someone else dealt the games, and hole
cards appear only when a player shows them at a showdown. This page says which parts of the pipeline go blind on such
data, which still work, and how to run it. Paths are relative to the repository root.

Every result on this page comes from the simulator (`00_simulator/`), where the planted superusers are known; the
table of honest maxima in "When nothing passes the bars" is plain arithmetic, and the other numbers are settings.
None of the results is a detection rate for real data.

**Without full information the results are distorted.** The pipeline runs every step on such data, and the
per-suspect report prints every section, but the sections that need the cards behind a fold are computed on the hands
that reached a showdown, which are selected by their outcome. The parts that stay meaningful are below; for a verdict,
use full-information hands or adapt the method to hands without showdowns. The file layout:
`docs/HAND_HISTORY_FORMAT.md` (its section "Without full information the results are distorted"; its "Try it"
writes mined hands with the simulator to run the route below on).

## What goes blind

Step 01 marks such a hand `cards_visibility = 'showdown'`. Step 02 computes the equity against the real cards
(`eq_vs_real`) only where both hands are known, and a hand that ends in a fold shows no cards. Every test that needs
the cards behind a fold goes blind or biased:

| test | why |
|---|---|
| method (b): `z_b`, `e_disc`, `x_dead`, `x_win` (turn and river) | it compares folds and continues while behind and while ahead of the real hand; a fold, or a bet nobody called, shows no cards |
| the preflop tests: `pf_z`, `pf_fold_cmh_z`, `pf_aggr_z`, rules PF1-PF4 | the same, on preflop decisions |
| the flop test: `z_b_flop`, `fl_e_disc`, rules FL1-FL2 | the same, on the flop |
| flags A (river fold when beaten) and D (early fold when nearly dead) | the folded hand is never shown |
| the real-card share (step 04's leakage split) | it needs the real hand behind every response to a bet, folds included |

What is built on them goes with them: Stages 2 to 5 of the screen (the hidden-information signals, the hand tests, the
risk score, the escalation rules), the rule set `street_tails` and the nine signals of step 03. The `eq_vs_real` that
is left comes from hands where both players showed, a sample chosen by the outcome: a player who folds never shows
(`02_equity/README.md`, "Limitations"). Do not compare such numbers with full-information ones.

The predicted range of step 04, fitted on such data, sees only showdown hands and comes out too strong
(`04_predicted_range/README.md`, "Observer fit"). Fit it on a full-information field when you have one (see below).

## What still works: money and showdowns

An observer still sees who put how much into which pot, who reached the showdown and who won it. A player who sees
the cards wins an unusual share of the big pots, wins the showdowns he chooses to reach, skips the ones he would lose
and almost never loses his stack on the river. His edge grows with what is at stake, so `sudetect/potsize.py` splits
the numbers by pot size. The screen (`05_detection/screen.py`) writes them into `player_scores`, on the `all` rows,
counted over every hand of the dataset whatever the scope.

| column | definition | suspicious when |
|---|---|---|
| `obs_hands` | hands dealt in | (the sample) |
| `obs_bb100` | net per 100 hands, bb | high |
| `obs_net_b1` ... `obs_net_b4` | net bb in pots under 30bb, 30-100bb, 100-200bb and 200bb or more (the final contested pot, uncalled bets excluded) | descriptive, not ranked |
| `obs_big_net100` | net in pots of 200bb or more, per 100 hands | high |
| `obs_wtsd` | went to showdown, of the hands that saw the flop | **low**: he skips the showdowns he would lose |
| `obs_wtsd_srp`, `obs_wtsd_3bp`, `obs_wtsd_4bp` | the same by the pot at the flop: under 15bb, 15-40bb, 40bb or more (roughly single-raised, 3-bet and 4-bet pots) | low |
| `obs_wsd` | showdowns won, with exactly two players at the showdown; a split counts 1/2 | high |
| `obs_wsd_b1` ... `obs_wsd_b4` | the same by final pot: under 30bb, 30-100bb, 100-200bb, 200bb or more | high (`obs_wsd_b1`, the small pots, is shown but not ranked: a player who sees the cards wins fewer of the small showdowns, which he reaches with hands he can check down, and more of the big ones) |
| `obs_wsd_growth` | `obs_wsd_b4` minus `obs_wsd_b1`: how much better he does when the pot is big | high |
| `obs_rbet_win` | he bet or raised the river, the pot was over 30bb and the hand reached a two-player showdown: share won | high |
| `obs_rcall_win` | he called a river bet heads-up, with the pot after the call over 30bb: share won | high |
| `obs_rcall_freq` | those calls over the river bets he faced that way | **low**: he folds the river bets he would lose, so a low call rate next to a high `obs_rcall_win` is the pattern |
| `obs_stackoff` | he saw the river and more than 30bb went in on it: share of those hands in which he lost 90% or more of his effective starting stack (the smaller of his stack and the biggest other stack) | **low** |

Every rate has its count `<column>_n` (`obs_growth_n` for the growth: the smaller of the two counts). The field ranks
add `<column>_pct` (a mid-rank percentile from 0 to 100, turned so that higher is always more suspicious),
`<column>_rank` (1 = the most suspicious; for WTSD, the river call rate and stack-offs, the lowest value) and `<column>_of` (the players
ranked). A player is ranked on a metric with at least `min_hands` hands and, for a rate, `min_cases` cases: 1,000 and
20 in `[step05.observer]` of `config/step05_screening.toml`, next to the pot lines and the stack-off share. Mined
samples are often thinner. A smaller `min_hands` (the simulator check below uses 300) goes in a file passed with
`--config`; `05_detection/variants/observer_300.toml` is that file:

```toml
# 05_detection/variants/observer_300.toml
[step05.observer]
min_hands = 300
```

Three more parts of the screen need no hidden cards: Stage 1 (win rate, W$SD, low WTSD, fold to river bets and river
calls won, as percentiles among comparable players), the win-rate percentile over random runs of the same length,
and the preflop re-raise counts with their success (`pf_3bet_succ`, `pf_4bet_succ`, `pf_5bet_succ`: the opponent
folded at once). The re-raise numbers move with style and with the opponent: they say whom to look at, not who cheats.

### What they catch on the simulator

Ranks in `sim_demo` (300 players, about 400,000 hands). A player is ranked on a metric with at least 300 hands and 20
cases; 1 = the most suspicious; "–" = too few cases to rank. `u` is the share of the planted player's eligible
postflop decisions played with the information.

| planted player | `obs_bb100` | `obs_wsd_b2` | `obs_wsd_b3` | `obs_wsd_b4` | `obs_wsd_growth` | `obs_rbet_win` | `obs_rcall_win` | `obs_stackoff` | `obs_wtsd` |
|---|---|---|---|---|---|---|---|---|---|
| full table, u = 1.0 | 2 | 2 | 1 | – | – | 1 | 1 | 1 | 1 |
| full table, u = 0.5 | 13 | 6 | 18 | – | – | 5 | – | 8 | 5 |
| single victim, 92% of his heads-up hands against the victim | 1 | 1 | 2 | – | – | 2 | 2 | 1 | 2 |
| single victim, 50% | 3 | 4 | 6 | – | – | 11 | 13 | 91 | 4 |
| single victim, 20% | 16 | 19 | 61 | 23 | 23 | 59 | 39 | 180 | 74 |
| onset: full table from partway through | 19 | 88 | 15 | – | – | 66 | 62 | 104 | 33 |

With the simulator's 100 bb stacks a final pot of 200 bb or more is rare (both players all in), so `obs_wsd_b4` and
`obs_wsd_growth` rank only the players with 20 such showdowns: 134 of the 300 here, one plant among them. Deeper games
fill that band.

In `sim_pf` (200 players, about 200,000 hands, some planted players at six-max tables): the full-table plant at
u = 1.0 ranks first on `obs_wsd_b2`, `obs_rbet_win` and `obs_stackoff` and third-lowest on `obs_wtsd`; the one at
u = 0.5 first on `obs_bb100`, `obs_wsd_b4` (44 players ranked), `obs_wsd_growth` and `obs_rcall_win`; the
single-victim plant at 92% first on `obs_stackoff`, and second on `obs_bb100`, `obs_wsd_b2`, `obs_rbet_win`,
`obs_rcall_win` and `obs_wtsd`.

On `sim_demo` the full-table plant at u = 1.0 is first or second on every column that ranks him, and his WTSD is the
lowest of the field. The weaker the cheat, the more these numbers fade, and the order is not strict (on `sim_pf` the
plant at u = 0.5 tops some columns the plant at u = 1.0 does not). The single-victim plant at 20% ranks 16th on win
rate and between 19th and 180th on the rest, and the onset plant ranks 15th on `obs_wsd_b3` and 19th on win rate,
between 33rd and 104th elsewhere. The pairs keep all three single-victim plants at the top (next section).

## Per pair: where a selective cheat's edge sits

A superuser who sees only a few opponents can look ordinary in every number above
([eli5_selective_cheating.html](../eli5_selective_cheating.html)): a few thousand hands against his victims sit
inside many thousands of honest ones. His edge sits in a few pairs, and `sudetect/pairs.py` measures it there. The
screen writes the table `pair_outliers`: one row per ordered pair "A against B" of the whole dataset (window `all`),
ranked against all the others. On mined data this is the view to start from.

Pair hands: in every hand, the two players who stayed in longest (both to the showdown, or the winner and the last
player to fold), each with at least one voluntary action. Hands with three or more players still in at the end are
left out.

| part | from A's side (minimum sample) | on mined data |
|---|---|---|
| `wr` | net per pair hand, bb (30 hands) | yes |
| `sd` | showdowns won against B, a split = 1/2 (7) | yes |
| `call` | calls facing B's bet on the flop, turn or river that were ahead of B's real hand, against the pool at the same street and decile of equity against B's predicted range (7) | showdown hands only; needs step 04 |
| `bet` | bets and raises on the flop, turn or river that were ahead, the same way (7) | showdown hands only; needs step 04 |
| `allin` | equity against B's real hand when the money went in (preflop included), against the pool at the same street and range band (7) | showdown hands only; needs step 04 |
| `pre` | preflop calls and raises facing B's raise that were ahead, against the pool at the same situation and 5% band of equity against B's line (7) | showdown hands only; needs `04_predicted_range/preflop.py` |
| `fold` | folds facing B's bet that were behind B's real hand, against the pool at the same strength (7) | no: a fold shows no cards |

Each part is a leave-pair-out residual over its standard error,

    S(A vs B) - S(A vs everyone else) - S(everyone else vs B) + S(all)

so an opponent who loses to everybody, or a player who beats everybody, does not make a pair stand out by itself:
what stands out is A doing to B what neither A does to others nor others do to B. `score` is the sum of the parts' z
over the square root of the sum of their correlations (parts that read the same hands, like the win rate and the showdowns, are not counted twice; an honest pair then scores around 0 with a spread of about 1); pairs with at least two parts are ranked (`field_rank`, 1 = the most
suspicious, of `n_ranked`; `top_pct`). A victim of A is a pair of his with a score of at least 3 and a rank inside the
top 1% of the ranked pairs or the top 10, whichever is larger (`[step05.pair_outliers]` in
`config/step05_screening.toml`). Without the predicted range of step 04 only `wr` and `sd` score, so a pair is ranked
once it has 30 hands and 7 showdowns.

The pot-size columns are the observer's view of the same pair. They are descriptive, not part of the score, and each
gets a z once it has 7 cases (the growth: 7 in each pot class): the shares against the rate of all pairs, the growth
and the net against zero.

| columns | A against B | suspicious when |
|---|---|---|
| `wsd_big`, `wsd_big_n`, `wsd_big_z` | showdowns won in big pots, 30bb or more by default (`big_pot`; two players at the showdown, a split = 1/2) | high |
| `growth`, `growth_z` | `wsd_big` minus his showdowns won against B in smaller pots | high |
| `rbet_win`, `rbet_win_n`, `rbet_win_z` | he bet or raised the river, the pot was over 30bb and the hand reached a showdown: share won | high |
| `rcall_win`, `rcall_win_n`, `rcall_win_z` | he called B's river bet, with the pot after the call over 30bb: share won | high |
| `stackoff`, `stackoff_n`, `stackoff_z` | he saw the river and more than 30bb went in on it: share of those hands in which he lost 90% or more of his effective starting stack | low (the z is turned, so fewer stack-offs give a higher z) |
| `net_big`, `net_big_n`, `net_big_z` | net in big pots, bb (z = the sum over the root of the sum of squares) | high |

### What pairs catch on the simulator

| dataset | full information | showdowns only |
|---|---|---|
| `sim_demo`: 300 players, about 400,000 hands | 1, 2 and 3 of 2,532 ranked pairs: all three single-victim plants | 1, 2 and 3 of 2,243 |
| `sim_pf`: 200 players, about 200,000 hands | 1 of 2,340 | 1 of 1,291 |
| `sim_small`: 60 players, about 20,000 hands | 1 of 192 (5 of 186 without the `fold` part) | 17 of 170: few showdowns per pair |

"Showdowns only" is the full-information simulator with the decision parts kept to the hands that reached a showdown
(`showdown_only = true` in `[step05.pair_outliers]`): what an observer of the same games sees, without the `fold`
part. The predicted ranges of that run were fitted on the full-information hands. The same switch shows what an
observer would see of your own full-information data.

- The `fold` part moved the planted pair from 5th to 1st on `sim_small`, and adding it never hurt.
- A cheat who sees the whole table beats everybody, so his pairs usually do not stand out (the best pair of such a
  plant ranked between 9th and 385th; the one exception, a plant who started late, reached 3rd and 4th with full
  information): pairs are for the selective kind, the player-level numbers above for the full-table kind.
- The score sums parts that read the same hands (win rate and showdowns above all), so it is divided by the root of
  the sum of their correlations, and a hand's several streets count once; an honest pair's score then has a spread of
  about 1, and the victim bar of 3 sits about three honest standard deviations out. With full information no honest
  player of `sim_small` (57) or `sim_demo` (293) gets a victim label, and 3 of 194 in `sim_pf`; showdowns only: 0, 1
  and 2. The planted pairs score 7.0 to 9.2 in `sim_demo` and `sim_pf` (3.2 in the small `sim_small`), the best
  honest pair 2.2 to 3.8. A label is a reason to look, not a verdict.

## Fit the range on a full-information field

The parts `call`, `bet`, `allin` and `pre` compare a decision with the pool at the same equity against the opponent's
predicted range (`r`, step 04). Fitted on mined hands, the range tables see only the hands that reached a showdown,
and the ranges come out too strong. When you have full-information hands of the same game and table sizes (every
hole card known, folded hands included), fit the tables there and score the mined dataset with them (`--fit-from`,
`04_predicted_range/README.md`):

```
python 04_predicted_range/run.py --dataset <field> --steps fit
python 04_predicted_range/run.py --dataset mined --fit-from <field> --steps fit,enrich
python 04_predicted_range/preflop.py --dataset mined --fit-from <field>
```

`<field>` is a dataset that step 01 loaded from full-information hands. Without one, `variants/observer_fit.toml`
fits the tables on the mined hands themselves, with the bias above:

```
V=04_predicted_range/variants
python 04_predicted_range/run.py --dataset mined --steps fit,enrich --config $V/observer_fit.toml
python 04_predicted_range/preflop.py --dataset mined --config $V/observer_fit.toml
```

Or leave step 04 out: the pairs then rank on `wr` and `sd`, and the pot-size columns and the observer metrics do not
change.

## Several accounts of one person

When several accounts belong to one person, merge them before the pair table. Otherwise his edge against a victim is
split over several rows, each below the bar, and the hands between his own accounts count as pairs.
`sudetect.pairs.merge_aliases` maps the accounts to one id on both sides of every pair and drops the hands between two
accounts of the same person. `screen.py` has no option for it; call the library after steps 01 to 04, from a Python
session started in the repository root:

```python
import tomllib

from sudetect import pairs, warehouse

with open("config/step05_screening.toml", "rb") as f:
    cfg = pairs.PairConfig.from_toml(tomllib.load(f)["step05"]["pair_outliers"])
con = warehouse.connect("mined", read_only=True)
rows, dec = pairs.load_inputs(con, big_pot=cfg.big_pot)
con.close()
rows, dec = pairs.merge_aliases(rows, dec, {"<account 2>": "<account 1>", "<account 3>": "<account 1>"})
table = pairs.pair_table(rows, dec, cfg)
v = pairs.victims(table, "<account 1>", cfg)
print(v.assign(pair=[pairs.pair_label(r) for r in v.to_dict("records")])[["B", "hands", "bb100", "score", "pair"]])
```

The mapping goes from each account to the id that stands for the person, here one of his accounts. `table` is ranked
like the screen's `pair_outliers`, except that the screen also drops non-standard hands and hands with an error flag
(and, under the scope `hu_tables`, hands dealt to more than two players). The observer metrics in `player_scores`
stay per account.

## The sampling trap

Mined data is rarely complete, and it is easy to collect one player more completely than everyone else: every file he
appears in, and a sample of the rest. His pairs then carry more hands than the field's pairs. A part's z grows with
its sample (for the same edge, four times the hands give about twice the z), so his pairs rank higher for the
coverage alone. Before you trust a field rank:

1. compare the `hands` of his pairs with the `hands` of the field's top pairs in `pair_outliers`;
2. rank at equal coverage: drop the files you added because he is in them, or sample his files at the same rate as
   everyone else's, then build the dataset again and rank again;
3. keep the rank that survives equal coverage. A rank that falls away came from the coverage.

## When nothing passes the bars

On mined data the report's three verdicts (the article's 15-day rule, the street-tails rule, the pairs) usually come
out below their bars, and that is the expected result, not a failed run. The first two rest on tests that need the
cards behind a fold ("What goes blind" above). The victim label needs a pair score of at least 3 and a rank inside
the top 1% of the ranked pairs or the top 10, whichever is larger (`victim_min_score`, `victim_rank_share`,
`victim_min_rank`); scored on the win rate, the showdowns and the decisions of showdown hands, a cheat's pair needs
many hands for that.

**The top pairs.** An honest pair scores around 0 with a spread of about 1, and the largest of n honest scores grows
with n. The 5th and 95th percentiles of the largest of n independent standard normal values, over 100,000 repeats:

| ranked pairs (n) | 5th percentile | 95th percentile |
|---|---|---|
| 100 | 1.89 | 3.28 |
| 500 | 2.51 | 3.71 |
| 2,000 | 2.97 | 4.05 |

A top score inside that range is what honest pairs alone produce, so on its own it is not a finding: three honest
standard deviations are far out for one pair, not for the best of many, and the victim bar of 3 lies inside the range
at every n above. The scores share players and hands, so they are not fully independent: read the table as a guide.

**Stage 1 and the observer ranks.** By default `s1_flag` marks the top 10% (`top_share`) of the Stage-1 composite in
each group of comparable players (window x table size x hand-count band x style cluster), so it flags about one in
ten of the players with enough hands and statistics to be ranked (`min_hands` and `min_stats` of
`[step05.screening]`) by construction: on its own a reason to look, not evidence. The windows are nested (`last15`
inside `last30` inside `all`) and share their hands, so a flag in all three is close to one flag, not three. Each
`obs_*` metric is ranked over its own number of players (`obs_*_of`): below 30 players only rank 1 reaches a
percentile of 95, below 10 nobody does.

**The mirror.** Every part of a pair's score subtracts S(everyone else vs B). If B beats everyone by the same margin,
that removes his edge, and the pairs against him score like any other. If his edge is uneven, as with a cheat who
sees only some opponents' cards, every A he beat by less than the others moves up, and an A who beat him over a short
run can rank near the top. Before you read a top pair "A against B" as A cheating B, look at B (his rows as A, his
`obs_bb100`): if he wins big from other opponents, A may simply be one he did not beat. The score is computed again
with every pair that scored 3 or more left out of the "everyone else" terms, so the mirror stays only for the part of
B's edge in pairs below 3.

**What to do then.** Take the players at the top of more than one list (the top pairs, Stage 1, an observer rank).
Stage 1 rests on the same win rate, W$SD, WTSD and river calls as the observer ranks of those numbers: a player high
on Stage 1 and on one of those ranks is on one list, not two. For each, list the opponents he won the most from, with
each pair's `field_rank`: his rows as A in `pair_outliers`, sorted by `bb100` x `hands` / 100 (his net in their pair
hands). Rank again at equal coverage ([The sampling trap](#the-sampling-trap)), and read the showdown hands his report
shows (step 6 below).

## Step by step

Python 3.11 or newer; `pip install -e .` from the repository root installs the code. Every command runs from the
repository root.

1. **Put the files in one folder.** Step 01 reads every `.txt` file under it, sub-folders included. The text has to
   follow the layout of `docs/HAND_HISTORY_FORMAT.md` (PokerStars-style hand histories); that page also shows how to
   check your files before a full load, and its "Try it" runs the same route on simulated mined hands. Bring
   every player in at the same coverage (see the sampling trap).
2. **Load them.**
   ```
   export SU_DATA_DIR=/path/to/warehouses
   export SU_ID_SALT='<a secret of your own>'
   python 01_action_table/run.py --dataset mined --input /path/to/folder --source mined --hash-ids
   ```
   `--hash-ids` replaces player, table and hand ids with salted HMAC hashes; keep the salt. The load prints the hands
   read and loaded, every dropped hand by reason (table `parse_funnel`) and the invariant checks. `--incremental` adds
   new days later (load every file of a day in the same run). A player's hashed id:
   ```python
   import os
   from sudetect.replay import IdMapper
   print(IdMapper(os.environ["SU_ID_SALT"].encode("utf-8")).player("<screen name>"))
   ```
3. **Equity where both hands were shown:** `python 02_equity/run.py --dataset mined`. Step 03 is optional here: its
   everyday statistics need no hidden cards, its nine signals do.
4. **The predicted range**, from a full-information field or with the observer setting ("Fit the range on a
   full-information field" above), or none.
5. **The screen:** `python 05_detection/screen.py --dataset mined`, with
   `--config 05_detection/variants/observer_300.toml` for thinner samples.
6. **One player in depth:** `python 05_detection/run.py --dataset mined --player <id> --out "$SU_DATA_DIR/mined/reports/<id>"`
   (one folder per player: without `--out` every report replaces the previous one). If step 4 used the observer
   setting, add `--range-config 04_predicted_range/variants/observer_fit.toml`, so the ranges the report draws use the
   settings step 04 scored with. Keep the default `--field-dataset` (the dataset itself): the mined dataset is the
   field his pairs and ranks come from. The report's sections "Opponents he beat" and "Observer metrics" are the parts
   that work on this data; its hand flags and signals read only the hands where both players showed.

| step | output | where |
|---|---|---|
| 01 | `hands`, `hand_players`, `actions`, `parse_funnel` | `$SU_DATA_DIR/mined.duckdb` |
| 02 | `action_equity` (decisions whose actor's cards were shown; `eq_vs_real` where both hands were), `hand_equity` (heads-up hands with both hands shown) | the same file |
| 04 | `rng_*` (the field tables: copied with `--fit-from`, or fitted with the observer setting), `action_range_equity`, `action_range_equity_pre` | the same file |
| 05, screen | `player_scores` (the `obs_*` columns and the pair summary `victims_n`, `best_pair_score`, `best_pair_rank`, on the `all` rows), `pair_outliers`, `report_meta` (`pairs_ranked`: the N of "pair #k of N") and the screen's other tables | the same file |
| 05, report | `report.html` and `report.md` ("Opponents he beat", "Observer metrics"), `pairs_victims.csv`, `suspicious_hands.html` (the victims table at the top, "pair #k of N (top x%)" next to each victim's id) | the `--out` folder (default `$SU_DATA_DIR/mined/report/`) |

To read the tables directly, from a Python session started in the repository root:

```python
from sudetect import warehouse

con = warehouse.connect("mined", read_only=True)
top_pairs = warehouse.query_df(con, """
    SELECT A, B, hands, bb100, sd_won, k, score, field_rank, n_ranked, wsd_big, wsd_big_n, stackoff, stackoff_n
    FROM pair_outliers WHERE field_rank IS NOT NULL ORDER BY field_rank LIMIT 25""")
lowest_wtsd = warehouse.query_df(con, """
    SELECT player_id, obs_hands, obs_wtsd, obs_wtsd_n, obs_wtsd_rank, obs_wtsd_of, obs_bb100, obs_stackoff
    FROM player_scores WHERE time_window = 'all' AND obs_wtsd_rank IS NOT NULL
    ORDER BY obs_wtsd_rank LIMIT 25""")
con.close()
```

**A period instead of the whole history.** `pair_outliers` covers the whole dataset, and a cheat may run for a while
only. Load the files of a date range into a dataset of its own (`--dataset mined_<period>`, at the same coverage for
everyone), run the same steps on it and compare its ranks with those of the whole dataset.

None of these settings is calibrated on real mined data. Test them on hands you have already reviewed, and keep a
human reviewer between a rank and any decision about a player.
