# Step 02: equity references

Step 02 gives every decision on the flop, turn and river two reference numbers: the actor's equity against one
random hand and the actor's equity against the real cards of the live opponents. Heads-up hands also get each
player's preflop equity against the other player's hand. Later steps compare decisions with these numbers.

## Inputs and outputs

Step 02 reads the step-01 tables `hands`, `hand_players` and `actions` through the view `decisions`, and writes:

| object | kind | grain |
|---|---|---|
| `action_equity` | table | one row per decision with the actor's cards known, on the flop, turn and river |
| `hand_equity` | table | two rows per heads-up hand with both hands known (one per player) |
| `equity_settings` | table | the settings of the last run, one row per setting |
| `actions_eq` | view | `actions` LEFT JOIN `action_equity` |
| `decisions_eq` | view | `decisions` LEFT JOIN `action_equity` |
| `hu_decisions_eq` | view | `hu_decisions` LEFT JOIN `action_equity` |
| `hand_players_eq` | view | `hand_players` LEFT JOIN `hand_equity` |

The equity of step 04 (against a predicted range) goes into its own table, `action_range_equity`.

## How to run

```
python 02_equity/run.py --dataset article_sample
python 02_equity/run.py --dataset <name> --procs 16
python 02_equity/run.py --dataset <name> --backend pyrust --fresh
python 02_equity/bench.py --procs 16
python -m pytest -q 02_equity/tests
```

The warehouse is `$SU_DATA_DIR/<dataset>.duckdb` and step 01 must have loaded it. Settings live in
`config/step02_equity.toml`; `--config FILE` overrides any key and the command-line flags override both.

| key | default | meaning |
|---|---|---|
| `backend` | `numpy` | `numpy`, or `pyrust` when the optional package `pyrust_poker` is installed |
| `include_preflop` | false | also score preflop decisions (see "Preflop rows") |
| `seed` | 20011 | base seed of every Monte Carlo estimate |
| `mc_samples` | 4096 | Monte Carlo deals per row where Monte Carlo is used (at most 65,536) |
| `flop_random_exact` | false | enumerate the flop against a random hand instead of sampling it |
| `hand_equity` | true | build `hand_equity` |
| `hand_equity_samples` | 8192 | Monte Carlo deals per heads-up hand in `hand_equity` |
| `batch_rows`, `batch_hands` | 50,000 | rows per staged part (the unit a rerun can reuse) |
| `chunk_rows` | 2000 | rows per task sent to a worker process |
| `procs` | 0 | worker processes; 0 means all logical CPUs, 1 runs everything in one process |
| `stage_dir` | `stage_equity` | parts go to `$SU_DATA_DIR/<dataset>/<stage_dir>/` |
| `identity_sample` | 1000 | heads-up hands recomputed in both directions after a run (0 skips the check) |
| `lock_wait_s` | 300 | seconds to wait while another process holds the warehouse file |

`run.py` exits with code 1 when a check of `sql/checks.sql` fails and with code 2 when the warehouse is missing.

## Output columns

`action_equity`, key `(hand_id, action_index)`:

| column | type | meaning |
|---|---|---|
| `hand_id`, `action_index` | VARCHAR, SMALLINT | the decision row in `actions` |
| `street` | VARCHAR | flop, turn or river (preflop as well with `include_preflop`) |
| `n_live_opp` | SMALLINT | live opponents at the row: players not folded, all-in players included; equals `n_live - 1` |
| `eq_vs_random` | DOUBLE | equity against one random hand ("equity vs all hands") |
| `eq_method` | VARCHAR | `exact` or `mc`, how `eq_vs_random` was computed |
| `eq_se` | DOUBLE | standard error of `eq_vs_random`; 0 when exact |
| `eq_vs_real` | DOUBLE | equity against the real cards of every live opponent; NULL unless all of them are known |
| `eq_real_known` | BOOLEAN | every live opponent's hole cards are known |
| `eq_real_method` | VARCHAR | `exact` or `mc` for `eq_vs_real`; NULL when `eq_vs_real` is NULL |
| `eq_real_se` | DOUBLE | standard error of `eq_vs_real` |
| `ahead_now` | BOOLEAN | the actor's hand beats every live opponent's hand on the current board; NULL unless all are known, and NULL preflop |

`hand_equity`, key `(hand_id, player_id)`:

| column | type | meaning |
|---|---|---|
| `hand_id` | VARCHAR | a hand dealt to exactly two players, both hands known, standard variant, no error flag |
| `player_id`, `opp_player_id` | VARCHAR | the player and the other player |
| `preflop_eq_vs_opp` | DOUBLE | the player's preflop equity against the other player's real hand |
| `eq_method` | VARCHAR | `mc` with the numpy backend, `exact` with pyrust |
| `eq_se` | DOUBLE | standard error, the same on both rows of a hand |

`hand_players_eq` adds `preflop_eq_vs_opp`, `preflop_eq_method` and `preflop_eq_se` to `hand_players`.

## Method

### Equity

Equity is the share of the pot the actor would take on average if the remaining board cards were dealt with no
more betting. A pot won alone counts 1 and a pot split between m players counts 1/m. With b board cards dealt and
known hands for the actor and k opponents, the exact equity is

```
E = (1 / |R|) * sum over completions r in R of share(r),     |R| = C(52 - 2(k + 1) - b, 5 - b)
share(r) = 1/m if the actor is among the m best hands on board + r, else 0
```

which is 990 completions heads-up on the flop, 44 on the turn and 1 on the river. Against one random hand the
average also runs over the C(52 - 2 - b, 2) holdings the actor cannot see (1,081 on the flop, 1,035 on the turn,
990 on the river), each equally likely.

The three references of the method are:

| name | column | assumption about the opponents |
|---|---|---|
| equity vs all hands | `eq_vs_random` | one opponent holding any two unseen cards |
| equity vs the opponent's hand | `eq_vs_real` | every live opponent's real hole cards |
| equity vs the predicted range | `action_range_equity.eq_vs_range` (step 04) | a modelled range per live opponent |

`eq_vs_random` is computed against one random hand even when several opponents are live. It is a measure of the
actor's own hand strength and is comparable across rows only in heads-up pots; compare multiway rows by
`eq_vs_real`.

### Exact and Monte Carlo

| backend | street | `eq_vs_random` | `eq_vs_real` |
|---|---|---|---|
| numpy (default) | flop | Monte Carlo, 4,096 deals (exact with `flop_random_exact`) | exact |
| numpy | turn, river | exact | exact |
| pyrust | flop, turn, river | exact | exact |
| numpy | preflop rows | exact table | Monte Carlo, 4,096 deals |
| numpy | `hand_equity` | | Monte Carlo, 8,192 deals |
| pyrust | `hand_equity` | | exact |

The numpy backend calls `equity_vs_random_batch` and `equity_vs_hand_batch` of `sudetect.equity` for heads-up
rows and `multiway_exact_batch` of `sudetect.equity_backends` when two or more opponents are live; the latter
enumerates every completion for all players at once (903 boards with two opponents on the flop). A Monte Carlo row
depends only on its own cards and the seed, never on the other rows of its batch, so batching and the number of
processes do not change any value, and the same seed gives the same tables.

A pot share X lies in [0, 1], so Var X = E[X^2] - p^2 <= p - p^2. The stored standard error of a Monte Carlo value
is `sqrt(p (1 - p) / n)`, exact when no deal splits the pot and an upper bound otherwise. With n = 4,096 it is at
most 0.0078. Exact values have a standard error of 0. A Monte Carlo
value of exactly 0 or 1 also gets 0.

### Ties, dead cards and multiway pots

Ties split the pot: two players with the same best hand get 1/2 each, three get 1/3. Only the actor's cards, the
board and, for `eq_vs_real`, the live opponents' cards are removed from the deck. Cards of players who folded
earlier in the hand are not removed, even when they are known; neither reference uses them. In a multiway pot
`eq_vs_real` is the actor's share against all live opponents together, so the shares of all live players sum to 1.
An opponent who is all-in is live.

### ahead_now

`ahead_now` compares the made hands as they stand: the actor's best five cards from hole cards plus board against
each live opponent's. It is TRUE when the actor's hand beats every one of them and FALSE otherwise, so a tie with
the best opponent is FALSE. On the river `ahead_now` equals `eq_vs_real = 1`, which `checks.sql` verifies on every
river row.

### Heads-up hands

For every hand dealt to two players with both hands known, `hand_equity` stores player A's preflop equity against
player B's real hand by Monte Carlo (8,192 deals, standard error at most 0.0055) and player B's as 1 - A. The two
rows of a hand therefore sum to exactly 1. After a run, `run.py` recomputes B against A with its own deals for
`identity_sample` hands and prints how far A + B is from 1.

### Pot odds

The pot odds come from step 01 (`actions.pot_odds`): `pot_odds = to_call_eff / W` with
`W = sum_j min(C_j, C_i + to_call_eff)` over every player, folded ones included, where `C` are the chips committed
so far. A call breaks even against the real cards when `eq_vs_real = pot_odds` (no further betting).

### Preflop rows

By default `action_equity` has no preflop rows; preflop equity against the real cards is kept at hand level for
heads-up hands. With `include_preflop = true`, preflop decisions get `eq_vs_random` from the exact table
`PREFLOP_VS_RANDOM` (`eq_method = 'exact'`), `eq_vs_real` by Monte Carlo with `mc_samples` deals
(`eq_real_method = 'mc'`) and `ahead_now` NULL.

## Speed

`bench.py` measures the CPU time per row of each backend in one process, on 3,000 decision rows per street and
3,000 heads-up hands of randomly dealt legal hands. Its last run was on an 8-core laptop (AMD Ryzen 7 8845H, 16 logical CPUs, Python 3.13,
numpy 2.5) with the machine otherwise nearly idle (CPU load 7 % before, 13 % after):

| backend | flop | turn | river | heads-up hand (preflop) |
|---|---|---|---|---|
| numpy | 0.88 ms | 0.12 ms | 0.010 ms | 1.47 ms (8,192 deals) |
| numpy, `flop_random_exact` | 2.84 ms | 0.10 ms | 0.005 ms | |
| pyrust | 4.27 ms | 0.93 ms | 0.57 ms | 4.18 ms (exact) |

The turn and river rows of the first two lines run the same code, so their differences are measurement noise; the
river times are close to the resolution of the process clock (0.02 to 0.03 CPU seconds for 3,000 rows). A row's
cost is almost all `eq_vs_random`: on the flop it samples 4,096 deals (or enumerates 1,081 holdings times 990
runouts with `flop_random_exact`), on the turn it enumerates 1,035 holdings times 44 rivers. In the same run a pool
of 16 processes scored 9,000 mixed rows at 9,363 rows per second. The times grow with the load of the machine.

A full run from an empty stage folder with 16 processes on the same machine, on the simulated dataset `sim_demo`
(generated by `sudetect/sim/`), scored 1,006,782 decision rows in
131 s (7,696 rows per second) and 280,507 heads-up hands in 95 s. The bundled case (718 hands) takes seconds. A
rerun whose inputs, settings and library code have not changed reuses every part.

## How a run is organised

1. `sql/plan_decisions.sql` counts the rows to score per hand; the hands, in `hand_id` order, are cut into batches
   of about `batch_rows` rows without splitting a hand.
2. `sql/inputs_decisions.sql` reads one batch with the live opponents' cards (`UNNEST` of
   `actions.live_opponents` joined to `hand_players`).
3. A fingerprint (SHA-256 of the batch's input rows, the backend, its settings, the code version and a hash of
   the source of `sudetect/equity_backends.py`, `equity.py`, `evaluator.py` and `cards.py`) is compared with the
   one stored in the batch's parquet part. A matching part is kept; otherwise the batch is split into
   chunks of `chunk_rows` rows and scored in a spawn-started process pool, and the part is written to a temporary
   file and renamed.
4. The heads-up hands go through the same loop with `sql/plan_hands.sql` and `sql/inputs_hands.sql`.
5. Parts numbered beyond the last batch are deleted, `sql/schema.sql` recreates both tables, `load_*.sql` fill them
   from the parts in key order, `sql/views.sql` creates the views, and `sql/checks.sql` and `sql/summary.sql` run.

The warehouse file is open only while SQL runs. Every read (the plan, each batch, the checks, the summary) uses its
own read-only connection, closed before any scoring starts, and the tables are rebuilt in one read-write
transaction at the end, so other processes can use the file while the workers compute. When another process holds
the file lock, a connection attempt is retried after 0.25 s, then after waits that double up to 8 s (with a
random jitter of 25 %), until `lock_wait_s` seconds have passed.

An interrupted run leaves its finished parts behind and the next run starts from them. Reloading step 01 with
different hands changes the fingerprints, so the affected parts are recomputed; until step 02 runs again,
`action_equity` describes the old load. The DuckDB-specific pieces (`UNNEST` in a join, `string_agg ... ORDER BY`,
`read_parquet`) carry a comment with the Spark SQL or Postgres form.

## Backends and cross-checks

`sudetect/equity_backends.py` defines the `Backend` protocol (`vs_random`, `vs_hands` on arrays of rows) and three
backends. `NumpyBackend` is the default. `PyrustBackend` calls `pyrust_poker.exact_equity` once per row with one
thread, sends preflop rows against a random hand to the exact table and rows with more than five opponents to the
numpy code (the engine takes at most six hands), and turns an engine panic, which arrives as a `BaseException`,
into NULL. `ReferenceBackend` calls the single-spot functions of `sudetect.equity` and serves the cross-checks.

`cross_check(a, b, n_spots)` deals random spots on the flop, turn and river with one to three opponents and compares
both equities of every spot; two exact values must agree within a tolerance and a Monte Carlo value must lie within
4.5 combined standard errors. `bench.py` ends with it on 200 spots (400 values) per pair of backends. The numpy
backend with an exact flop and the reference backend agree exactly (largest difference 0). The default numpy backend
differs from the reference by at most 0.0177 on its Monte Carlo flop values, 2.26 combined standard errors. Pyrust
and numpy agree exactly against a random hand and against one known hand, and differ by up to 1.3e-8 against two
or three known hands, because `pyrust_poker` divides split pots with integer arithmetic; that pair is compared with
a tolerance of 1e-7.

## Tests

`02_equity/tests` (36 tests, 62 s on the laptop above) builds a warehouse of 420 randomly dealt legal hands (the test generator in
`sudetect/sim/`) in a temporary folder with the step-01 loader: 300 written with every hole card and 120 written the way an observer sees them. The tests check that

- every stored equity equals a direct call of `sudetect.equity` (exact rows within 1e-9; Monte Carlo rows equal to
  the same seeded batch call and within 4.5 standard errors of the exact value), and `ahead_now` equals a
  comparison with `eval_hand`;
- the NULL rules hold: no row without the actor's cards, `eq_vs_real`, `eq_real_method` and `ahead_now` NULL
  exactly when a live opponent's cards are unknown;
- a rerun reuses every part and gives identical tables, and so does a run from an empty stage folder;
- a run stopped after three parts resumes from them, a deleted part is recomputed alone, and the batch size does
  not change any value;
- two processes give the same tables as one, a new seed changes only the Monte Carlo values, preflop rows appear
  on request, and the pyrust backend agrees with numpy;
- the backends agree on 200 random spots, split pots count 1/m, and malformed card text is rejected;
- no connection to the warehouse is open while rows are scored (a task opens the file read-write in the middle of
  a run), a locked file is retried with growing waits and given up after `lock_wait_s`, and an edit of the library
  code makes the next run recompute every part;
- `examples.py` prints, for every example row, stored values equal to the direct library calls.

## Limitations

- On observer data (hole cards only at the showdown) `eq_vs_real` exists only where cards were shown. That sample
  is selected by the outcome: a player who folds never shows, so statistics of `eq_vs_real` on such data are
  biased and are not used for detection.
- `eq_vs_random` ignores how many opponents are live; use it on heads-up rows.
- Cards of folded players stay in the deck for both references.
- On the flop `eq_vs_random` carries Monte Carlo noise of up to 0.0078 per row with the default settings, and
  `hand_equity` up to 0.0055. Averages over many rows are not affected; `flop_random_exact` removes the noise on the
  flop at 3.2 times the CPU time per flop row (2.84 against 0.88 ms in the last `bench.py` run).
- `ahead_now` counts a tie as not ahead.
