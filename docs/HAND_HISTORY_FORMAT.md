# Hand-history format

Step 01 reads hand histories written as plain text in the PokerStars-style layout, the text format that many poker
sites and tracking tools write. This page says what a file has to look like, what may vary, what the loader drops and
why, and how to check your own files. `data/sample_hands.txt`, the bundled sample, is a complete file in this layout.
Paths are relative to the repository root.

## Files

- Plain text in UTF-8 (a byte-order mark is fine), any line endings. Files in another encoding, such as UTF-16, are
  not read: save them as UTF-8 first.
- Any number of hands per file. A hand starts at its header line; hands are separated by a blank line.
- Put the `.txt` files in one folder and pass the folder to step 01 with `--input`: every `.txt` file under it is read,
  sub-folders included (`.zip` archives in the folder are read as well, their `.txt` members: keep one copy of each
  file, or its hands are dropped as duplicates).
- No-limit hold'em only, written `Hold'em No Limit` (or `Hold'em NL`) in the header. Other games, and other spellings
  of this one such as `No Limit Hold'em`, are counted and dropped (`unsupported_game`).

## Two views of the same hands

| view | hole cards in the text | load with | what works |
|---|---|---|---|
| **full information**: the operator's export | a `Dealt to <player> [Xx Yy]` line for every dealt seat, folded hands included | `--source operator` | every test of the pipeline |
| **mined**: hands recorded by an observer at the tables | no `Dealt to` lines; cards appear only where a player shows them at the showdown | `--source mined` | money and showdowns, the pairs; everything else is distorted (next section) |

Step 01 stores the view of every hand in `hands.cards_visibility`: `full` when the text has a `Dealt to` line for
every dealt player, `showdown` otherwise. Cards shown at the showdown do not make a hand full, and neither does the
single `Dealt to` line of a player's own hand history (his own cards only).

## Without full information the results are distorted

When hole cards show only at showdowns, every test that needs the cards behind a fold goes blind or biased: the
range-adjusted test, the preflop and flop tests, the hand flags, the real-card share, the hidden-information signals,
the escalation rules built on them. The pipeline still runs every step, and the per-suspect report still prints every
section (the verdict, the hand flags, the scorecard, the signals, the metrics by street), computed on the hands that
reached a showdown. On such data those sections are distorted:

- hands are selected by their outcome: cards are known only where a player showed them, a player who loses at the
  showdown often mucks, and a fold never shows cards, so the hands with known cards lean toward the ones their owner
  won: a won river call reads as a "perfect call", and the folds say nothing;
- the field's predicted ranges are fitted on showdown hands only, so they come out too strong;
- a player can be flagged, escalated or ranked first by these sections for nothing.

The parts that stay meaningful are the money and showdown metrics per player (`obs_*`), the pairs' win-rate and
showdown parts and their pot-size columns (`docs/THIRD_PARTY_MINING.md`; the pairs' decision parts read showdown
hands only), and even they need many more hands than full-information data (the simulator results in
`docs/THIRD_PARTY_MINING.md`). For a verdict use full-information hands, or adapt the method to hands without
showdowns; on mined data treat every rank as a lead for a human reviewer, never as a finding.

## One hand, line by line

A heads-up hand from `data/sample_hands.txt` (the bundled sample, full information):

```text
Hand #h0069: Hold'em No Limit (25.00/50.00) - 2001/01/06 00:50:30 UTC
Table '3' 2-max Seat #1 is the button
Seat 1: 1002 (5000.00 in chips)
Seat 2: 1001 (9272.00 in chips)
1002: posts the ante 5.00
1001: posts the ante 5.00
1002: posts small blind 25.00
1001: posts big blind 50.00
*** HOLE CARDS ***
Dealt to 1002 [7h 6s]
Dealt to 1001 [Kh 2d]
1002: raises 75.00 to 125.00
1001: calls 75.00
*** FLOP *** [6d Qd Kd]
1001: checks
1002: checks
*** TURN *** [6d Qd Kd] [3d]
1001: checks
1002: checks
*** RIVER *** [6d Qd Kd 3d] [Tc]
1001: bets 195.00
1002: calls 195.00
*** SHOW DOWN ***
1001: shows [Kh 2d] (a flush, King high)
1002: shows [7h 6s] (a pair of Sixes)
1001 collected 650.00 from pot
*** SUMMARY ***
Total pot 650.00 | Rake 0.00
Board [6d Qd Kd 3d Tc]
Seat 1: 1002 (button) (small blind) showed [7h 6s] and lost with a pair of Sixes
Seat 2: 1001 (big blind) showed [Kh 2d] and won (650.00) with a flush, King high
```

A mined hand history of the same hand has no `Dealt to` lines: only the two `shows` lines at the showdown say which
cards the players held, and a hand that ends with a fold says nothing about them.

| part | form | notes |
|---|---|---|
| header | `[<site>]Hand #<id>: Hold'em No Limit (<sb>/<bb>[ - Ante <a>][ <CUR>]) - <yyyy/mm/dd hh:mm:ss>[ <TZ>]` | the hand id must be unique; the time is the hand's start |
| table | `Table '<name>' <N>-max Seat #<b> is the button` | `<N>` is the table size |
| seats | `Seat <s>: <name> (<stack> in chips)`, optionally followed by ` is sitting out` | a name may contain spaces |
| posts | `<name>: posts small blind <x>`, `posts big blind <x>`, `posts the ante <x>`, `posts straddle <x>`, dead blinds | any of them may end with `and is all-in` |
| hole cards | `*** HOLE CARDS ***`, then `Dealt to <name> [Xx Yy]` lines when the cards are known | cards are rank + suit: `2`-`9`, `T`, `J`, `Q`, `K`, `A` and `c`, `d`, `h`, `s` |
| actions | `<name>: folds`, `checks`, `calls <x>`, `bets <x>`, `raises <inc> to <total>` | a raise names the increment and the total |
| streets | `*** FLOP *** [a b c]`, `*** TURN *** [a b c] [d]`, `*** RIVER *** [a b c d] [e]` | |
| showdown | `*** SHOW DOWN ***`, `<name>: shows [..]`, `<name>: mucks hand` | the only place cards appear in a mined hand |
| money back | `Uncalled bet (<x>) returned to <name>`, `<name> collected <x> from pot` | also `from main pot`, `from side pot-1` |
| summary | `*** SUMMARY ***`, `Total pot <x> \| Rake <y>`, `Board [..]`, one `Seat <s>: ...` line per seat | collected + returned must equal what was put in minus the rake, to the cent |

The exact grammar, with every accepted wording, is the docstring at the top of `sudetect/hhparse.py`.

## What may vary

- **A site name before `Hand #`** (exports in the PokerStars layout start `PokerStars Hand #...`), or nothing. The
  hand id may hold letters, digits, `_` and `-`.
- **Currency and separators**: `($0.25/$0.50 USD)`, `$1,050.00`. Amounts are read with two decimals
  (`amount_decimals` in `config/defaults.toml`) and every check of the gate is exact in those units; results are in
  big blinds.
- **Antes** in the header (`- Ante 0.10`) or as posts.
- **Extra text in brackets after the table size** (`Table 'Big Table' 9-max (Real Money) Seat #2 is the button`) and
  **other drops after the rake** (`Total pot $607 | Rake $3 | Jackpot $1`): the drops count like the rake.
- **The time zone** after the time. No zone, `UTC` and `GMT` are read as UTC; any other zone is kept as written and
  the hand carries the remark `time_zone_not_utc` (in `hands.error_detail`; the hand is loaded). A second time in
  brackets after the first (`... 21:05:40 CET [2001/03/14 15:05:40 ET]`) is ignored. The time windows and the 15-day
  escalation read the times as written, so keep one zone per dataset.
- **A player's own hand history**, with one `Dealt to` line for himself: it loads, and the hand stays in the mined
  view (only his own cards are known), with the bias of a sample that one player chose to keep.
- **Straddles, dead blinds, a button blind, run-it-twice boards** (`*** FIRST FLOP ***` ...) and side pots.
- **Informational lines**: timeouts, time bank, disconnects, chat, players joining or leaving. They are ignored; a fold
  or check right after a timeout line is marked as automatic.

The same rules with a site name, a currency, an ante and a time zone:

```text
Site Hand #S-0001: Hold'em No Limit ($0.25/$0.50 - Ante $0.05 USD) - 2001/03/14 21:05:40 ET
Table 'Example II' 6-max Seat #4 is the button
Seat 1: alpha ($50.00 in chips)
Seat 2: bravo ($62.35 in chips)
Seat 4: charlie ($1,050.00 in chips)
alpha: posts the ante $0.05
bravo: posts the ante $0.05
charlie: posts the ante $0.05
alpha: posts small blind $0.25
bravo: posts big blind $0.50
*** HOLE CARDS ***
charlie: raises $1 to $1.50
alpha: folds
bravo: calls $1
*** FLOP *** [Ks 7d 2c]
bravo: checks
charlie: bets $2
bravo: calls $2
*** TURN *** [Ks 7d 2c] [Qh]
bravo: checks
charlie: bets $5.50
bravo: folds
Uncalled bet ($5.50) returned to charlie
charlie collected $7.03 from pot
*** SUMMARY ***
Total pot $7.40 | Rake $0.37
Board [Ks 7d 2c Qh]
Seat 1: alpha (small blind) folded before Flop
Seat 2: bravo (big blind) folded on the Turn
Seat 4: charlie (button) collected ($7.03)
```

`01_action_table/tests/test_step01_format_doc.py` parses every example hand on this page and checks the drop table
below against the parser, so the page and the parser cannot drift apart.

## What is dropped, and why

A hand that fails a check is not loaded. Every such hand is counted by reason, per file, in the table `parse_funnel`,
and step 01 prints the totals. The first failing check names the reason:

| reason | what it means |
|---|---|
| `cancelled` | the hand was cancelled (`*** HAND CANCELLED ***`) |
| `bad_header`, `bad_table_line` | the header or the table line does not follow the layout above |
| `unsupported_game` | not no-limit hold'em written `Hold'em No Limit` or `Hold'em NL` (`No Limit Hold'em` lands here too) |
| `unsupported_structure_no_sb`, `bomb_pot_no_blinds`, `no_big_blind_post` | ante-only tables, bomb pots, a hand without a big blind |
| `no_actions` | nobody acted although the betting was open |
| `unknown_grammar` | a line, post, verb or marker the grammar does not know |
| `bad_card` | an unreadable or repeated card |
| `payee_not_dealt` | a player who was not dealt in collects from the pot |
| `stack_inconsistent`, `amount_inconsistent`, `action_order_invalid` | the betting does not add up with the stacks or the amounts to call |
| `money_unreconciled` | collected + returned differs from what was put in minus the rake |
| `hidden_post_inferred` | the betting shows a blind that the text does not print |
| `duplicate`, `duplicate_diff` | the same hand id was already read (with the same or a different text) |
| `replay_error` | the hand passed the checks but could not be replayed; step 01 prints the first ten and exits with code 1 |
| `no_hand_header` | a whole file without a single hand header (another layout or encoding); counted with 0 hands |

A few drops need no action. Many drops of one reason mean the files differ from this layout in one
place, and the reason says where.

If no hand of the input passes these checks (no hand header at all, another layout or encoding, every hand dropped),
step 01 stops before it touches the warehouse, names the most frequent reasons and exits with code 2:
`error: no hand history found in ...` or `error: no hand in ... passes the checks of step 01 (dropped: ...)`. Files in
another layout have to be written in this one first.

## Check your files

```
# bash
export SU_DATA_DIR=/path/to/warehouses
python 01_action_table/run.py --dataset check --input /path/to/folder --source mined --limit 2000
```

```
# Windows PowerShell
$env:SU_DATA_DIR = "D:\warehouses"
python 01_action_table/run.py --dataset check --input D:\hands --source mined --limit 2000
```

`--limit 2000` stops after 2,000 loaded hands, enough for a check; the last file read may be cut short. `check` is a
dataset of its own, rebuilt by every such run. The output names the hands read and loaded, every drop by reason, the
files without a hand header, and the invariant checks of the loaded tables (`invariants: all 20 checks pass`). The
exit code is 0 when at least one hand loaded and every check passed (drops do not change it: read their counts), 1
when no hand was loaded, a check failed or a hand could not be replayed, and 2 when an input does not exist or no hand
of it passes the checks (the warehouse is not touched then). Per file:

```python
from sudetect import warehouse
con = warehouse.connect("check", read_only=True)
print(warehouse.query_df(con, "SELECT file, reason, n_hands FROM parse_funnel ORDER BY n_hands DESC"))
```

## Try it

- **The layout.** `data/sample_hands.txt`, the bundled sample (718 heads-up hands, full information), loads with the
  check command above:

  ```
  python 01_action_table/run.py --dataset check --input data/sample_hands.txt --source operator
  ```

  It prints `hands read: 718; loaded: 718` and `invariants: all 20 checks pass`.
- **The mined route without hand histories of your own.** The simulator writes the mined view of any of its datasets
  (`--mined`: the same hands without the `Dealt to` lines, and the same answer key in `ground_truth.json` next to the
  hands folder):

  ```
  python 00_simulator/simulate.py --dataset sim_small --mined --out "$SU_DATA_DIR/sim_mined/hands"
  ```

  Then run the steps of `docs/THIRD_PARTY_MINING.md` ("Step by step") on `--dataset sim_mined` with
  `--input "$SU_DATA_DIR/sim_mined/hands"`, and compare the ranks with the answer key.
