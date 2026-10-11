# From zero to the reports

This page goes from an empty data directory to three results on hand histories of your own: a report for each
suspect, every pair of the field scored and ranked in the warehouse, and the suspicious hands. Every command runs
from the repository root. Where bash and Windows PowerShell differ (setting and reading a variable, paths), both are
shown. `mine` is the name of the dataset: any name of letters, digits, `_`, `.` and `-`.

The times below were measured on the simulator and are quoted from the step READMEs; a step without a measured time
has none here.

## The path at a glance

| step | command | writes | measured on the simulator |
|---|---|---|---|
| 01 | `01_action_table/run.py` | `hands`, `hand_players`, `actions`, `parse_funnel`; views `decisions`, `hu_decisions` | |
| 02 | `02_equity/run.py` | `action_equity`, `hand_equity` | `sim_demo`: 1,006,782 decision rows in 131 s and 280,507 heads-up hands in 95 s (16 processes, an 8-core laptop); 200,000 hands at heads-up and six-max tables: about 2 minutes on 12 processes |
| 03, optional | `03_player_stats/run.py` | `player_stats` | |
| 04 | `04_predicted_range/run.py --steps fit,enrich,leakage` | the field tables `rng_*`, `action_range_equity`, `rng_leak_matrix`; the folder `<dataset>_step04/` | `sim_demo`: 750,635 decisions scored in 511 s (12 worker processes, an 8-core laptop) |
| 04 | `04_predicted_range/preflop.py` | `action_range_equity_pre` | about 40 seconds for 330,000 decisions |
| 05 | `05_detection/screen.py` | `player_scores`, `pair_scores`, `pair_outliers`, `window_scores`, `flagged_hands`, `escalations`, `street_flags`, `report_meta`, `scored_decisions` | |
| 05 | `05_detection/run.py` | one report folder per suspect | |

## 1. Install

Python 3.11 or newer. From the repository root:

```
pip install -e .
```

It installs the code (`sudetect`, `superuser`) with numpy, pandas, pyarrow, duckdb and tabulate. On Windows, when
`python` opens the app store instead of an interpreter, call the interpreter by its full path or through the `py`
launcher.

## 2. A data directory

```
# bash
export SU_DATA_DIR=/path/to/warehouses
```

```
# Windows PowerShell
$env:SU_DATA_DIR = "D:\warehouses"
```

Each dataset becomes one DuckDB file, `$SU_DATA_DIR/<dataset>.duckdb`, with two folders next to it: `<dataset>/` (the
converted input, step 02's staged parts, the default report folder) and `<dataset>_step04/` (step 04's scored chunks
and its `summary.json`). Without `SU_DATA_DIR` everything goes to `work/` in the repository. The variable lives in
the terminal: set it again in a new one.

## 3. The hands

Put them in one folder; its sub-folders are read too.

- **Text**: `.txt` files, or `.zip` archives of them (their `.txt` members are read), in the PokerStars-style layout
  of `docs/HAND_HISTORY_FORMAT.md` (line by line, what may vary, what is dropped and why; `data/sample_hands.txt` is a
  file in this layout). A hand is full-information (`hands.cards_visibility = 'full'`) when it has a `Dealt to <player> [Xx Yy]`
  line for every seat. Step 04 fits its tables only on full-information hands, and every test that needs the cards
  behind a fold needs them too: without them the results are distorted (the same page). Hand histories in another
  format need a converter first (`01_action_table/README.md`, "How an operator builds the tables from its own
  format").
- **JSON hand records** (one row per hand, in CSV, parquet or JSON lines; the fields are in
  `01_action_table/README.md`, input variant 2): convert them to text first. The converter also writes the decision
  times. Write both into `$SU_DATA_DIR/<dataset>/input`, where step 03 and the per-suspect report look for
  `decision_times.csv`:

  ```
  # bash
  python 01_action_table/from_replay_json.py --input /path/to/records.csv --out "$SU_DATA_DIR/mine/input"
  ```

  ```
  # Windows PowerShell
  python 01_action_table/from_replay_json.py --input D:\records.csv --out "$env:SU_DATA_DIR\mine\input"
  ```

  Step 01 then reads `$SU_DATA_DIR/mine/input/hands.txt`.

Text hand histories carry no decision times: without them no decision is a snap decision, and the reports show no
decision times.

## 4. Load: step 01

```
# bash
python 01_action_table/run.py --dataset mine --input /path/to/hands --source operator
```

```
# Windows PowerShell
python 01_action_table/run.py --dataset mine --input D:\hands --source operator
```

It writes `hands`, `hand_players`, `actions` and `parse_funnel`, and creates the views `decisions` and
`hu_decisions`. It prints the hands read and loaded, every dropped hand by reason (`parse_funnel` keeps the counts
by file) and the invariant checks, and exits with code 1 when no hand was loaded, a check fails or a hand could not
be replayed, and with code 2, before it touches the warehouse, when no input file holds a hand header (the layout is
in `docs/HAND_HISTORY_FORMAT.md`).
`--source operator` is only a label (`hands.source`). `--incremental` replaces only the days present in the input,
for adding new days later (load every file of a day in the same run); without it the tables are rebuilt.
`--hash-ids` replaces the ids by hashes (section 12).

## 5. Equity: step 02

```
python 02_equity/run.py --dataset mine
```

It writes `action_equity`, for every postflop decision and every preflop decision with one live opponent: the
equity against one random hand (`eq_vs_random`) and against the real cards of every live opponent (`eq_vs_real`);
and `hand_equity`, each player's preflop equity against the other's hand in the heads-up hands. On `sim_demo` it
scored 1,006,782 decision rows in 131 s and 280,507 heads-up hands in 95 s with 16 processes on an 8-core laptop; on
200,000 simulated hands at heads-up and six-max tables the whole step took about 2 minutes on 12 processes
(`02_equity/README.md`). A rerun reuses the staged parts whose inputs and settings did not change.

## 6. Player statistics: step 03, optional

```
python 03_player_stats/run.py --dataset mine
```

It writes `player_stats`: the 16 classic statistics and the 9 hidden-information signals per player. Neither the
screen nor the report reads it.

## 7. The predicted range: step 04

```
# bash
python 04_predicted_range/run.py --dataset mine --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset mine
```

```
# Windows PowerShell
python 04_predicted_range/run.py --dataset mine --steps "fit,enrich,leakage"
python 04_predicted_range/preflop.py --dataset mine
```

`fit` builds the field tables (`rng_*`) on the dataset itself, and stops with a message when not every hand is
full-information. `enrich` writes `action_range_equity`, the equity against the opponent's predicted range in
heads-up pots after the flop. `leakage` writes `rng_leak_matrix` and each player's real-card share into
`$SU_DATA_DIR/mine_step04/summary.json`. `preflop.py` writes `action_range_equity_pre`, the equity against the field's
range for the opponent's preflop line. `enrich` scored the 750,635 decisions of `sim_demo` in 511 s with 12 worker
processes on an 8-core laptop, and `preflop.py` takes about 40 seconds for 330,000 decisions
(`04_predicted_range/README.md`). To take the tables from another dataset, see section 13.

## 8. The screen: step 05

```
python 05_detection/screen.py --dataset mine
```

It scores every player in the windows `all`, `last30` and `last15` and writes `player_scores`, `pair_scores`,
`pair_outliers`, `window_scores`, `flagged_hands`, `escalations`, `street_flags`, `report_meta` and
`scored_decisions`, then prints `report_meta`: the population sizes, how many players each rule flags and how many
were tested. `--config my.toml` overrides any key of `config/step05_screening.toml` that the file repeats under
`[step05]`; `--dry-run` computes and prints without writing. Every table and column is described in
`05_detection/README.md`.

## 9. Pick the suspects with SQL

Open the warehouse read-only, with the DuckDB command-line client or from Python (root README, "Your local
database"):

```
# bash
duckdb -readonly "$SU_DATA_DIR/mine.duckdb"
```

```
# Windows PowerShell
duckdb -readonly "$env:SU_DATA_DIR\mine.duckdb"
```

Start from the escalations and the pairs:

```sql
-- players the street-tails rule escalates or puts up for review, with their most suspicious pair
SELECT player_id, hands, street_hits, pf_hit, fl_hit, tr_hit, esc_street_tails,
       victims_n, best_pair_opponent, best_pair_score, best_pair_rank
FROM player_scores
WHERE time_window = 'all' AND review_street_tails
ORDER BY street_hits DESC, z_b DESC;

-- players with at least one victim
SELECT player_id, hands, victims_n, best_pair_opponent, best_pair_score, best_pair_rank, street_hits
FROM player_scores
WHERE time_window = 'all' AND victims_n > 0
ORDER BY best_pair_score DESC;

-- every escalation verdict of one player
SELECT rule_set, rule, window_start, window_end, hands, flagged, ratio, threshold_ratio, threshold_count, escalate
FROM escalations
WHERE player_id = '<player id>'
ORDER BY rule_set, rule;
```

The other verdicts of the screen are columns of `player_scores` too: `rb_screen` (`z_b` at least 4), `pf_screen` and
`fl_screen` (the standardised preflop and flop tests at least 4), `s1_flag` and `s2_flag` (Stages 1 and 2), the
`esc_*` columns (Stage 5) and `cp_z` (the onset changepoint). A flag is a reason to look, not a verdict. On hands
with cards only at the showdowns most of them stay empty:
[When nothing passes the bars](THIRD_PARTY_MINING.md#when-nothing-passes-the-bars) in `docs/THIRD_PARTY_MINING.md`
says how to read the top pairs, Stage 1 and the observer ranks then.

## 10. One report per suspect

```
# bash: the suspects' player ids
for P in PLAYER_1 PLAYER_2; do
  python 05_detection/run.py --dataset mine --player "$P" --out "$SU_DATA_DIR/mine/reports/$P"
done
```

```
# Windows PowerShell
foreach ($p in "PLAYER_1", "PLAYER_2") {
  python 05_detection/run.py --dataset mine --player $p --out "$env:SU_DATA_DIR\mine\reports\$p"
}
```

The report empties its output folder before it writes, sub-folders included: give every suspect a folder of his own
and never point `--out` at a folder that holds anything else. Without `--out` it writes to `$SU_DATA_DIR/mine/report/`,
which the next report empties. Hashed ids (section 12) make safe folder names; a screen name with characters a folder
name cannot hold needs a folder name of your own.

Each folder holds:

- `report.html` and `report.md`: the verdict (the article's 15-day rule, the street-tails rule, the pairs), metrics
  by street against the field, the opponents he beat, the article's scorecard, the observer metrics, then his
  statistics, decision times, signals, escalation and top flagged hands;
- `suspicious_hands.html`: his victims at the top, then the tabs Postflop (the default: every hand flagged by the
  rules A-D and the range rule R in the order of the Equity review, then the rare flop decisions), Preflop (the rare
  preflop decisions with the opponent's preflop range) and All; a victim's id carries "pair #k of N (top x%)" in
  every hand;
- `suspicious_hands_all.html`, the CSVs (`suspicious_hands.csv`, `suspicious_hands_ranked.csv`, `rare_decisions.csv`,
  `pairs_victims.csv`, `metrics_summary.csv`, `decisions.csv`, `decision_times.csv`, `stats.csv`, `signals.csv`) and
  `summary.json` (`05_detection/README.md`, "Outputs").

Options: `--opponent <victim>` keeps the hands where the victim also sat; `--scope hu_tables` keeps two-player tables
and `--scope all` adds multiway pots; `--top-hands N` sets how many rare decisions each tab lists; `--field-dataset`
compares him with another field (section 13). To publish a report made against a private field, set
`show_field_size = false` in `config/step05_report.toml` (`05_detection/README.md`, "The report against a field").

## 11. Every pair of the field, and the suspicious hands of every player

`pair_outliers` holds every ordered pair "A against B" of the dataset that played at least one pair hand: the parts of
its score, its rank among the ranked pairs and the pot-size view (columns in `05_detection/README.md`):

```sql
-- the field's most suspicious pairs
SELECT A, B, hands, bb100, sd_won, k, score, field_rank, n_ranked, wr_z, sd_z, call_z, bet_z, fold_z
FROM pair_outliers
WHERE field_rank IS NOT NULL
ORDER BY field_rank
LIMIT 25;

-- one player's pairs as A, the most suspicious first
SELECT B AS opponent, hands, bb100, k, score, field_rank, n_ranked, wsd_big, wsd_big_n, stackoff, stackoff_n
FROM pair_outliers
WHERE A = '<player id>' AND field_rank IS NOT NULL
ORDER BY field_rank;
```

Without step 02 or step 04 the decision parts stay empty and only the win rate and the showdowns score, so a pair
needs 30 hands and 7 showdowns to be ranked.

The report shows the suspicious hands of one player; the screen keeps them for every player: `flagged_hands`
(Stages 3 and 4: the hands of rules A-D under both rule sets, with labels and risk scores) and `street_flags` (the
preflop and flop hand rules PF1-PF4 and FL1-FL2):

```sql
-- one player's flagged hands in review order, range-conditioned rule set
SELECT hand_id, street, action, category, label, q, r, req, pot_bb, risk, review_rank
FROM flagged_hands
WHERE player_id = '<player id>' AND rule_set = 'range_conditioned'
ORDER BY review_rank;

-- the preflop and flop hand rules, the largest gap between the real cards and the range first
SELECT player_id, hand_id, rule, rule_name, q, r, req, gap, tier
FROM street_flags
ORDER BY gap DESC
LIMIT 50;
```

## 12. Hashed ids

`--hash-ids` makes step 01 replace player, table and hand ids by salted HMAC-SHA256 hashes (`P_`, `T_` and `H_`
followed by hex digits). The salt is a secret of your own, read from the environment variable `SU_ID_SALT`. Keep it:
the same salt gives the same ids, so a later load or another dataset loaded with it matches.

```
# bash
export SU_ID_SALT='<a secret of your own>'
python 01_action_table/run.py --dataset mine --input /path/to/hands --source operator --hash-ids
```

```
# Windows PowerShell
$env:SU_ID_SALT = "<a secret of your own>"
python 01_action_table/run.py --dataset mine --input D:\hands --source operator --hash-ids
```

A player's hashed id from his screen name, with the same salt in the environment:

```
# bash
python -c 'import os; from sudetect.replay import IdMapper; print(IdMapper(os.environ["SU_ID_SALT"].encode("utf-8")).player("<screen name>"))'
```

```
# Windows PowerShell
python -c "import os; from sudetect.replay import IdMapper; print(IdMapper(os.environ['SU_ID_SALT'].encode('utf-8')).player('<screen name>'))"
```

It prints the id step 01 stores with the default widths of `[step01.hash_ids]` in `config/defaults.toml`. Without
`--hash-ids` the ids are the names as written in the hand histories, and
`SELECT player_id, count(*) AS hands FROM hand_players GROUP BY 1 ORDER BY 2 DESC` lists them by volume.

## 13. A reference field: `--field-dataset` and `--id-map`

When your dataset is an extract of a larger field, for example the hands of one case cut out of a population, the
report can rank the suspect against that population. Build the field once, full-information, and screen it:

```
# bash
python 01_action_table/run.py --dataset field --input /path/to/all_hands --source operator
python 02_equity/run.py --dataset field
python 04_predicted_range/run.py --dataset field --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset field
python 05_detection/screen.py --dataset field
```

Score the case with the field's range tables, then run the report against the field:

```
# bash
python 01_action_table/run.py --dataset case --input /path/to/case_hands --source operator
python 02_equity/run.py --dataset case
python 04_predicted_range/run.py --dataset case --fit-from field --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset case --fit-from field
python 05_detection/run.py --dataset case --player <id> --field-dataset field --id-map ids.csv --out "$SU_DATA_DIR/case/reports/<id>"
```

In PowerShell the commands are the same, with `"fit,enrich,leakage"` in quotes and `$env:SU_DATA_DIR` in the paths.

With `--field-dataset` the hands on the page stay those of the case, and the report reads from the field: the pool of
the rare decisions (every other player of the field), the street-tails verdict, the metrics by street, the article's
scorecard and the observer metrics from its `player_scores`, his pairs from its `pair_outliers` (computed by the
report itself when the field has no such table and at most `pairs_max_hands` hands), and, because the field is named
explicitly, a comparison of his real-card share with field players measured on as many responses (it reads the
field's `field_step04/` chunks). If the case's hands are part of the field already, the report can also run on the
field itself: `--dataset field --player <id>`.

`--id-map` is needed when the two datasets name the same player differently, for example a case whose ids were
replaced. It is a CSV with the columns `case_id` (the id in the case) and `ref_id` (the same player in the field):

```
case_id,ref_id
<suspect's id in the case>,<his id in the field>
<victim's id in the case>,<the victim's id in the field>
```

Every lookup in the field goes through it, and the victims table shows a player's case id when the map knows him, the
field's id otherwise. Datasets loaded with `--hash-ids` and the same salt share their ids and need no map.

## 14. Practice on the simulator

The simulator plays a population with planted superusers and writes the answer key next to the hands; only the
evaluation scripts `05_detection/eval_planted.py` and `05_detection/eval_quick.py` read it. `sim_small` has 60 players
and about 20,000 hands over 20 days.

```
# bash
python 00_simulator/simulate.py --dataset sim_small
python 01_action_table/run.py --dataset sim_small --input "$SU_DATA_DIR/sim_small/hands" --source sim
python 02_equity/run.py --dataset sim_small
python 04_predicted_range/run.py --dataset sim_small --steps fit,enrich,leakage
python 04_predicted_range/preflop.py --dataset sim_small
python 05_detection/screen.py --dataset sim_small
python 05_detection/eval_planted.py --dataset sim_small --out "$SU_DATA_DIR/eval"
```

```
# Windows PowerShell
python 00_simulator/simulate.py --dataset sim_small
python 01_action_table/run.py --dataset sim_small --input "$env:SU_DATA_DIR\sim_small\hands" --source sim
python 02_equity/run.py --dataset sim_small
python 04_predicted_range/run.py --dataset sim_small --steps "fit,enrich,leakage"
python 04_predicted_range/preflop.py --dataset sim_small
python 05_detection/screen.py --dataset sim_small
python 05_detection/eval_planted.py --dataset sim_small --out "$env:SU_DATA_DIR\eval"
```

`--out` keeps the evaluation out of the repository's `05_detection/results/`.

## 15. Troubleshooting

A step left out usually leaves empty columns and a note, not an error. The exception is the report without step 02.

| what you see | why | what to do |
|---|---|---|
| `missing_inputs` in the screen's printout, empty test columns (`z_b`, `pf_z`, `z_b_flop`) in `player_scores` | the screen ran without step 02 or step 04 | run them, then the screen again |
| a report section shows a note instead of a table, such as "mine has no player_scores table: run 05_detection/screen.py --dataset mine" | the field was not screened | run `screen.py` on the field, then the report |
| "mine has no pair_outliers table: the pairs were computed here" | the same, on a field of at most 500,000 hands (`pairs_max_hands`); above that the section asks for the screen | nothing, or run the screen |
| an empty Preflop tab that says "has no action_range_equity_pre: run 04_predicted_range/preflop.py" | `preflop.py` was not run | run it, then the report |
| "Street-tails rule: not in the field's player_scores" | he has no row in the field's `player_scores` (next row), or an older `screen.py` wrote it | the next row, or run the screen again |
| "Player ... has no row in the field's player_scores" | his id in this dataset is not his id in the field | `--id-map` (section 13) |
| `Table with name action_equity does not exist` from `run.py` | the report needs step 02 | run step 02 |
| step 04: "only x% of the hands show every player's cards" | the dataset is not full-information | fit on a full-information field (section 13), or see `docs/THIRD_PARTY_MINING.md` |
| step 01: "error: no hand history found in ..." (exit code 2) | no input file is in the PokerStars-style layout, or there is no `.txt` file | `docs/HAND_HISTORY_FORMAT.md`; the warehouse was not changed |
| step 01 exits with code 1 | no hand was loaded (the reasons are printed), an invariant check failed or a hand could not be replayed | the drop reasons: `docs/HAND_HISTORY_FORMAT.md`; otherwise fix the input; `--allow-violations` reports the last two and exits with 0 |
| step 01: "--hash-ids needs the secret salt in the environment variable SU_ID_SALT" | `SU_ID_SALT` is not set in this terminal | set it (section 12) |
| `run.py`: "player ... has no hands in scope" | the id is not his id in this dataset (hashed?), or he has no hands in the scope | look the id up (section 12); `--scope all` |
| `run.py`: "--field-dataset ...: no warehouse at ..." | `SU_DATA_DIR` points elsewhere, or the name is misspelt | check both |
| a step stops on a lock, or waits | another process has the file open: a DuckDB session, a notebook, another step | close it; step 02 waits up to 300 s and the screen up to 600 s for the lock |
| Windows: an encoding error when a non-Latin player name is printed to a redirected output | the console's code page | `$env:PYTHONIOENCODING = "utf-8"` before the run |

None of the settings is calibrated on your data. Test the steps on hands you have already reviewed, and keep a human
reviewer between a flag and any decision about a player.
