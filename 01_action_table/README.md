# Step 01: action table

Step 01 turns hand histories into three tables in a DuckDB warehouse (`$SU_DATA_DIR/<dataset>.duckdb`). Every later
step reads only these tables, so an operator who can fill them from its own data can run the whole pipeline.

## The three tables

They describe the same hands at three levels of detail.

| table | one row per | what it answers |
|---|---|---|
| `hands` | hand | the table, the blinds, the board, the pots per street, rake, showdown, whether every hole card is known |
| `hand_players` | player in a hand | seat and position, hole cards, preflop flags (VPIP, PFR), how far the player went, chips put in and won |
| `actions` | action (posts included) | who did what, on which street, facing what, with which cards, and the price of the action |

Amounts are in big blinds (`*_bb`), so tables with different stakes can be pooled. Cards are text, two characters
per card (`AhKd`). `hands.cards_visibility` is `full` when every dealt player's hole cards are in the text (the
operator's view) and `showdown` otherwise (an observer's view, where cards appear only at the showdown).

Key columns:

| table | columns |
|---|---|
| `hands` | `hand_id`, `source`, `table_id`, `table_seq`, `n_dealt`, `started_at`, `sb_bb`, `ante_bb`, `button_seat`, `board`, `pot_flop_bb` / `pot_turn_bb` / `pot_river_bb` / `final_pot_bb`, `uncalled_bb`, `rake_bb`, `pot_type`, `last_street`, `went_to_showdown`, `cards_visibility`, `error_flag` |
| `hand_players` | `hand_id`, `player_id`, `seat`, `position`, `position_class`, `stack_start_bb`, `hole_cards`, `hand_class` (169 classes), `cards_known`, `cards_source`, `vpip`, `pfr`, `saw_flop` / `saw_turn` / `saw_river`, `folded_street`, `committed_bb`, `won_bb`, `net_bb`, `won_at_showdown` |
| `actions` | `hand_id`, `action_index`, `player_id`, `street`, `is_post`, `action` (fold, check, call, bet, raise), `is_allin`, `is_decision`, `decision_s`, `n_live`, `live_opponents`, `is_ip`, `situation`, `pot_type_now`, `hole_cards`, `made_hand`, `draw`, `board_at_action`, `amount_bb`, `to_call_bb`, `raise_to_eff_bb`, `pot_before_bb`, `faced_size_pot`, `pot_odds` |

`pot_odds` is the equity a call needs: amount to call / (pot before the action + amount to call), side-pot aware.
`net_bb` is won + returned - committed, after rake. The views `decisions` (actions that are decisions) and
`hu_decisions` (decisions at two-player tables) are created after every load. `parse_funnel` counts every hand
read, by file and reason (`loaded`, or the gate reason that dropped it). The full column list with comments is in
`sql/schema.sql`.

## Input variant 1: hand-history text

`run.py` reads PokerStars-style text (`.txt` files, or `.txt` members of `.zip` archives), splits it into hands and
parses each one with `sudetect/hhparse.py`. The grammar and the gate (the checks that drop a hand, each with a named
reason) are documented at the top of that module. A hand that passes the gate is replayed action by action
(`sudetect/replay.py`) into the three tables.

```
python 01_action_table/run.py --dataset <name> --input <folder, glob or file> [--source operator|mined]
python 01_action_table/run.py --dataset <name> --input "<folder>/**/*.zip" --hash-ids   # ids replaced by HMAC
```

## Input variant 2: operator replay rows (JSON)

`from_replay_json.py` reads the hand records the per-suspect forensic package (`superuser/`) works on: one row per
hand with a replay JSON and a result JSON, in CSV, parquet or JSON lines. The columns are `hand_uid`, `start_ts`,
`end_ts` (Unix seconds), `table_no`, `stake_level`, `chip_type`, `hand_replay` and `hand_result`:

- `hand_replay`: `TableSetup` {`game`, `big_blind`, `ante`, `n_players`}; `Seats.seat_list`, one entry per seat
  {`seat`, `player_id`, `stack`, `position`, `hole_cards`}; `Streets` with `preflop` … `river`, each a list of
  {`order`, `seat`, `act`, `amount`, `think_s`}, a `<street>_end` record with the `pots` after the antes and after
  each street (`ante_end`, `preflop_end` ... `river_end`), and `settlement`, one {`seat`, `net`} per player.
  `TableSetup.n_players` is the table size.
- `hand_result`: the `board` and `contributions` {`player_id`, `total_bet`}.
- A card is {`rank` 0–12 for 2 … A, `suit` 0–3}. `act` is one of check, fold, call, bet, raise, allin, walk.

An operator's own export is mapped onto these names by a few lines of code kept with the data, outside this
repository. It parses each row with
`superuser.parser.parse_hand` and writes the hand as text in the dialect above (every seat's hole cards included),
which `run.py` then loads. Both variants therefore produce the same tables.

```
python 01_action_table/from_replay_json.py --input data/sample_hands.csv --out $SU_DATA_DIR/article_sample/input
python 01_action_table/run.py --dataset article_sample --input $SU_DATA_DIR/article_sample/input/hands.txt --source operator
```

The mapping:

- The export has no post records. The big blind is `TableSetup.big_blind`, the small blind half of it unless the room
  record says otherwise (`small_blind` or `sb`); antes come from `TableSetup.ante`. Heads-up, the first player to act
  preflop is the small blind and the button.
- Bet, raise and all-in amounts in the export are street totals, call amounts are chips added. They are written as
  `bets <total>`, `raises <increment> to <total>` and `calls <added>`.
- `collected` amounts are the export's net result plus the chips the player put in, so `hand_players.net_bb` times
  the big blind equals the export's settlement for every player. The rest of the pot is rake.
- The text has no decision times. They are written to `decision_times.csv` (`hand_id, street, order, seconds`,
  `order` counting the street's actions from 1, posts excluded); steps 03 and 05 read it back.

On the bundled case (718 hands): 718 written and 718 loaded, no gate drop, all 20 invariant checks pass, every net
equals the settlement. A hand that fails the converter's own checks is skipped and counted with its reason
(`settlement_below_contribution`, `call_amount_mismatch`, `check_facing_bet`, ...).

## How an operator builds the tables from its own format

Two routes, in order of effort:

1. Write your hands as text in the dialect of `sudetect/sim/hhtext.py` (the header with the blinds, `Table` and
   `Seat` lines, posts, `*** HOLE CARDS ***`, a `Dealt to <player> [Xx Yy]` line per seat, actions per street with
   PokerStars semantics, board lines, `Uncalled bet (x) returned to`, `shows` lines, `collected` lines, the summary)
   and load it with `run.py --source operator`. `from_replay_json.py` is a worked example of such a writer
   (about 300 lines). Every hole card in the text makes the hands full-information, which step 04 needs for fitting.
2. Fill the three tables directly with the columns of `sql/schema.sql`, then run `sql/post_load.sql`,
   `sql/views.sql` and the checks of `sql/invariants.sql`. Route 1 is safer, because the replay derives positions,
   situations, pot odds and hand classes consistently.

## Checks

`sql/invariants.sql` runs after every load (keys unique, action indexes contiguous, pots and stacks accumulate,
no action after a fold, calls match the amount to call, money identity, VPIP matches the actions, ...). `run.py`
exits with code 1 when one fails or a hand could not be replayed (`--allow-violations` reports and exits 0).
Settings (gate switches, tolerances, batch size) are in `config/defaults.toml`, `[step01]`.

## Tests

`python -m pytest -q 01_action_table/tests`: parser and gate cases, adversarial texts, positions, situations, the
loader on randomly dealt legal hands, and the round trip of the bundled sample (JSON rows to text to tables, nets
equal to the settlement).
