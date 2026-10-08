# sudetect developer notes

`sudetect` is the library behind the step folders (`01_action_table/` to `05_detection/`). The step folders hold the
runnable scripts, SQL files and docs; every importable module lives here.

Python 3.11 or newer (the step scripts read their config with `tomllib`); tested on 3.13 with numpy 2.5.

## Module map

| module | contents |
|---|---|
| `cards` | card ids, the 1,326 two-card combos, the 169 preflop classes, `range_weights` |
| `evaluator` | 5, 6 and 7-card hand evaluator (`eval_hands`), strength decoding, `describe_hand` |
| `equity` | equity against a known hand, a random hand or a weighted range; multiway equity; batch versions for action tables |
| `handclass` | made-hand class (12), draw class (5), flop texture (8), turn and river change (6), ten preflop buckets; scalar and vectorised over all 1,326 combos of a board |
| `positions` | positions from the button and the dealt seats; position classes EP, MP, CO, BTN, SB, BB |
| `situations` | preflop situation (6), postflop spot (9), pot type (5), line prefix, in position |
| `hhparse` | parser for PokerStars-style hand histories: `split_hands`, then `parse_hand` returns a `ParsedHand` or a `GateReject` with a reason |
| `replay` | replays a `ParsedHand` chip by chip into rows of the `hands`, `hand_players` and `actions` tables (pyarrow) |
| `warehouse` | the DuckDB adapter: connect, run a `.sql` file with `:name` parameters, insert and read DataFrames |
| `sim.engine` | no-limit hold'em rules engine for random legal test hands: `play_hand(...)` returns a `HandRecord` |
| `sim.hhtext` | writes `HandRecord`s as hand-history text |

Imports go one way. `evaluator` builds on `cards`, `equity` on both, and `handclass` on all three. The test hand
generator settles showdowns with `evaluator`. `replay` uses `hhparse`,
`positions`, `situations` and `handclass`. Three modules need more than numpy: `warehouse` imports duckdb and
pandas and `replay` imports pyarrow. Every other module needs numpy and the standard library
alone.

## Card encoding and units

```
card = rank * 4 + suit
rank:  2  3  4  5  6  7  8  9  T  J  Q  K  A   ->  0 .. 12
suit:  c  d  h  s                               ->  0 .. 3
```

`2c` is 0, `Ah` is 50 and `As` is 51, so `rank = card >> 2` and `suit = card & 3`. Arrays use -1 for a card that
has not been dealt or is not known; a batch of boards is an `(n, 5)` array padded with -1 on the right.
`parse_cards("Ah Kd")` returns `[50, 45]` and `format_cards` goes back.

`COMBOS[i]` is `(higher card, lower card)` with `i = hi * (hi - 1) // 2 + lo`, so the order is `2d2c, 2h2c, 2h2d,
..., AsAh`. A range is a float array of 1,326 weights in this order, and only the ratios between weights matter.
`combo_index(a, b)` maps two cards in either order to their row and returns -1 when either card is negative.
`range_weights({"QQ": 1, "AK": 1, "AhKh": 0.5})` builds a range from class labels and exact combos.

`CLASS_LABELS` lists the 169 classes: pairs `AA` to `22` (indices 0 to 12), suited hands `AKs` to `32s` (13 to
90), offsuit hands `AKo` to `32o` (91 to 168). `COMBO_TO_CLASS` maps a combo to its class and `CLASS_SIZE` holds
6, 4 or 12 combos per class.

Inside the library, money is an integer number of chip units. The test generator's default blinds are 50/100 units,
so one big blind is 100 units. The parser reads printed amounts into integer units, and `replay` converts them to
big blinds, rounded to 1e-6, only when it writes rows. Warehouse tables store amounts in big blinds.

## Evaluator

`eval_hands(cards)` takes an `(n, k)` integer array, k = 5, 6 or 7, and returns one int32 strength per row. Higher
wins and equal strengths split the pot. With 6 or 7 cards the strength is that of the best five cards.

```
strength = category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5
```

`category` runs from 0 (high card) to 8 (straight flush) and `r1` to `r5` are the ranks that break ties, most
important first, so comparing two strengths as integers compares the hands. All five-card hands produce 7,462
different strengths. `decode`, `describe_value` and `describe_hand` turn a strength or a set of cards into ranks or
text: "a flush, King high", "a straight, Ace to Five", "a Royal Flush".

`eval_hands(cards, check=False)` skips the input check (ids in 0..51, distinct within a row) and runs 40 to 50%
faster. It returns meaningless strengths for invalid input instead of raising, so use it only on cards dealt from
one deck.

The tests compare every one of the 2,598,960 five-card hands with an independent evaluator written in the test
file, and millions of random 5, 6 and 7-card hands with phevaluator.

## Equity

Equity is the share of the pot a hand wins on average if the board is dealt to the end with no more betting. A won
pot counts 1, a pot split k ways 1/k.

| function | assumes about the opponent |
|---|---|
| `equity_vs_hand(hero, villain, board)` | the real hole cards |
| `equity_vs_random(hero, board)` | any two cards the hero cannot see, all equally likely |
| `equity_vs_range(hero, board, weights)` | 1,326 weights in `COMBOS` order; combos that share a card with the hero or the board are ignored; `nan` when no weight is left |
| `multiway_equity(hands, board)` | 2 to 10 known hands; returns every player's share, summing to 1 |

The `*_batch` versions take `(n, 2)` hole cards, `(n, 5)` boards padded with -1 and, for ranges, `(n, 1326)`
weights. They group rows by street and return a float64 array.

Exact means every remaining board, and every opponent holding when the opponent's hand is unknown, is enumerated, so
the result has no noise. Monte Carlo (MC) deals `n_samples` random completions from a seeded generator. Its standard
error is at most `0.5 / sqrt(n_samples)`, which is 0.008 at the batch default of 4,096. What `exact=None`, the
default, does:

| function | preflop | flop | turn, river |
|---|---|---|---|
| `equity_vs_hand` | exact | exact | exact |
| `equity_vs_random` | exact (table) | exact | exact |
| `equity_vs_range` | MC, 100,000 samples | exact | exact |
| `multiway_equity` | exact | exact | exact |
| `equity_vs_hand_batch` | MC, 4,096 samples | exact | exact |
| `equity_vs_random_batch` | exact (table) | MC, 4,096 samples | exact |
| `equity_vs_range_batch` | MC, 4,096 samples | MC, 4,096 samples | exact |

`exact=True` turns every MC cell into enumeration and `exact=False` samples wherever sampling is supported. The
table is `PREFLOP_VS_RANDOM`, the exact equity of the 169 classes against a random hand, produced by
`preflop_vs_random_exact`. The batch functions accept at most 65,536 samples. A batch row's MC result depends only
on its own cards and the seed, never on the other rows, and the errors of many rows average out the way
independent errors do.

The tests check the exact paths against brute-force enumeration on 90 textured spots (largest difference 4e-15),
1,000 random spots against pyrust_poker (all differences below 1e-9), multiway spots against exact fractions, and MC
results against exact values within 4.5 standard errors.

## Test hand generator (`sudetect/sim/`)

The generator deals random legal hands for the tests; it plays no strategy beyond a random legal choice.


`play_hand(config, seats, button_seat, agents, hand_id, table_name, start_time, rng, info_grants=None,
evaluate=None, *, deck=None)` plays one hand. Chips are integer units.

`TableConfig(max_seats, sb=50, bb=100, ante=0, rake_pct=0.05, rake_cap=300, no_flop_no_drop=True)`: 2 to 9 seats
numbered from 1, the small blind between 0 and the big blind, rake as a share of the total pot capped at
`rake_cap` units (`None` for no cap). Player names must be unique single words that do not start with `*` or end
with `:`, so the hand-history text stays unambiguous.

The main rules (the `sim.engine` docstring lists them all):

* Antes are posted first, then the blinds. Heads-up, the button posts the small blind, acts first before the flop
  and last after it.
* A stack smaller than its ante or blind posts what it has and is all-in. The full big blind is still the amount
  to call before the flop.
* The minimum raise increment is the larger of the last full raise increment and the big blind. An all-in that
  raises by less does not reopen the betting for players who have already acted; several short all-ins can add up
  to a full raise.
* Raising is not offered when no other player could respond. When betting is over and at most one player has
  chips behind, the rest of the board is dealt without decisions.
* The uncalled part of the largest contribution is returned. Pots follow the contribution levels of the players
  still in the hand; ties split a pot and the odd chips go to the first tied winner left of the button.
* Rake is `floor(rake_pct * total pot)`, capped, taken only when a flop was dealt (unless `no_flop_no_drop` is
  off), and removed from the main pot first.
* At the showdown the river's last aggressor shows first, otherwise the first player still in the hand left of the
  button. Later players show only a hand that wins or ties a pot they can win, and muck otherwise.

An agent is any object with `act(view) -> Decision`. The `DecisionView` holds the agent's cards, the board, the pot,
`to_call`, `current_bet`, `min_raise_to`, `max_raise_to`, `can_raise`, stacks, street contributions, the action
history and `revealed_cards`. A decision is `Decision.fold()`, `check()`, `call()`, `bet(to)` or `raise_to(to)`,
where `to` is the player's street total after the action. An illegal decision raises `ValueError`.

`info_grants` maps a player to the players whose hole cards that player sees (or `"all"`). `evaluate` replaces the default showdown evaluator, `evaluator.eval_hands`, and
`deck` fixes the card order for replays and tests.

The `HandRecord` holds the seats and starting stacks, hole cards, board, every action as an `ActionRecord` (street,
kind, amount, street total, all-in flag, pot, amount to call and stack before the action), uncalled returns, pots
with their eligible players, winners and shares, rake, and per seat the chips contributed, collected, net and final
stack. It also holds the showdown order, the shown and mucked hands, and `descriptions` (seat to `describe_hand`
text). There is always a main pot; it holds 0 chips only when everyone folds to the big blind at a table with no
small blind and no ante.

Simplifications to know about:

* A player whose only win is an uncontested side pot shows. A losing all-in player shows only when first to show,
  so mined-style text hides some all-in hands that a real site would turn face up.
* Rake comes out of the main pot first, so a short all-in player who wins only the main pot can collect 0.
* All odd chips of a split pot go to the first winner left of the button.
* `table_name` is not validated; a newline in it breaks the text.
* Tables without a small blind play correctly, but the step 01 parser rejects their text.

## Hand-history writer

`write_hand(record, full_information)` returns one hand as text; `write_hands` and `write_file` handle many.
Amounts are units divided by 100 with two decimals, so with a big blind of 100 units they read as big blinds. The
header is `Hand #<n>: Hold'em No Limit (<sb>/<bb>) - YYYY/MM/DD HH:MM:SS UTC` followed by
`Table '<name>' <k>-max Seat #<b> is the button`. Full-information mode adds `Dealt to <player> [Xx Yy]` for every
seat, which is the operator's view. Mined style leaves those lines out, so hole cards appear only when shown. The
two modes differ in nothing else. A shown hand carries the `describe_hand` text, e.g.
`P_3d4e5f: shows [Ah Kh] (a flush, Ace high)`.

## Speed

One process on a laptop (Ryzen 7 8845H), Python 3.13, numpy 2.5, otherwise idle machine.

Evaluator, 2,000,000 random hands per call:

| cards | `check=True` | `check=False` |
|---|---|---|
| 5 | 33 M hands/s | 49 M hands/s |
| 6 | 18 M hands/s | 28 M hands/s |
| 7 | 19 M hands/s | 26 M hands/s |

`describe_hand` takes about 21 µs per call.

Batch equity with the defaults, time per row (2,000 preflop and flop rows, 10,000 turn and river rows; the range
holds the 50 strongest classes):

| street | vs hand | vs random | vs range |
|---|---|---|---|
| preflop | 0.47 ms (MC) | table lookup | 1.1 ms (MC) |
| flop | 0.080 ms (exact) | 0.84 ms (MC) | 1.0 ms (MC) |
| turn | 0.005 ms (exact) | 0.093 ms (exact) | 0.18 ms (exact) |
| river | 0.001 ms (exact) | 0.003 ms (exact) | 0.026 ms (exact) |

With `exact=True` a flop row takes 2.1 ms against a random hand and 4.5 ms against that range. Single spots:
`equity_vs_hand("Ah Ad", "Ks Kc")` preflop, exact, 0.23 s; `equity_vs_random` on a flop, exact, 3 ms;
`equity_vs_range` on a flop, exact, 6 ms, and preflop with MC, 0.11 s (exact preflop takes 12 to 16 s);
three known hands preflop, exact, 0.32 s.

Engine, 6,000 hands with 100 bb stacks reset every hand and the default evaluator. These agents do almost no work,
so the numbers measure the rules engine itself.

| agents | 2 players | 6 players | 9 players |
|---|---|---|---|
| `RandomAgent` | 15,600 hands/s | 7,000 hands/s | 5,400 hands/s |
| check or call every time (all hands reach a showdown) | 6,500 hands/s | 3,100 hands/s | 2,100 hands/s |
| fold whenever facing a bet | 18,500 hands/s | 9,700 hands/s | 4,900 hands/s |

A showdown evaluation costs about 14 µs per hand, a small part of these times.

Writer, showdown texts recomputed on every pass: 7,300 hands/s in full-information mode and 8,100 hands/s in mined
style on 6-player `RandomAgent` hands; 5,900 and 6,600 hands/s on 9-player hands that all reach a showdown.

## Tests

Run from the repository root with `python -m pytest`, which puts the root on `sys.path`, so no install step is
needed:

```
python -m pytest -q tests/                    # library tests, about 5 minutes
python -m pytest -q 01_action_table/tests     # step 01, about 20 seconds
```

Test-only dependencies: pytest; phevaluator, which the evaluator tests compare against;
pyrust_poker, without which `test_equity.py` and `test_evaluator_equity_adversarial.py` are skipped; pandas, pyarrow
and duckdb for the warehouse and step 01 tests. The library itself never imports phevaluator or pyrust_poker.

| file | checks |
|---|---|
| `test_cards.py` | card ids, parsing and formatting, combo order, blocked combos, classes, ranges |
| `test_evaluator.py` | all five-card hands (7,462 values, category counts), random hands against phevaluator, edge cases, texts, speed |
| `test_equity.py` | every equity function against pyrust_poker and known values, MC against exact, batch against single spots, batch timings |
| `test_evaluator_equity_adversarial.py` | all five-card hands and textured equity spots against references written in the test file; dtypes, padding, seeds, extreme weights, MC error |
| `test_handclass.py` | hand classes, vectorised against scalar, and against an independent classifier |
| `test_warehouse.py` | the DuckDB adapter |
| `test_superuser_cards.py`, `test_superuser_pipeline.py` | the per-suspect forensic package on the article-case sample |
