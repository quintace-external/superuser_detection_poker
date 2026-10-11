# Step 00: simulator

`simulate.py` plays a population of no-limit hold'em players with the rules engine and the hand-history writer of
`sudetect/sim/` and writes full-information hand histories (every hole card) that step 01 loads like any other
text. A few planted superusers play with card information: full table (they see every opponent), single victim
(they see one opponent) and onset (they start seeing cards partway through). `ground_truth.json` and
`population.json` next to the hand files are the answer key; only `05_detection/eval_planted.py` and the tests read
them, the pipeline never does.

```
python 00_simulator/simulate.py --dataset sim_small                 # 60 players, 20 days, about 20,000 hands
python 00_simulator/simulate.py --dataset sim_demo --procs 16       # 300 players, 60 days, about 400,000 hands
python 00_simulator/simulate.py --dataset sim_demo --calibrate --hands 50000 --no-write
python 00_simulator/simulate.py --dataset sim_small --mined        # the same hands as an observer records them
```

`--mined` writes the same hands as an observer at the tables records them: no `Dealt to` lines, hole cards only where
a player shows them at the showdown (`docs/HAND_HISTORY_FORMAT.md`). The answer key is the same as without it.

Settings are in `config/sim.toml` (`[sim]`): table sizes and stakes, player archetypes and their jitter, session
structure, the planted players (`kind`, the share `u` of decisions played with information, the victim's share of
hands, the onset day) and the calibration targets. `--calibrate` prints the honest field's HUD statistics (VPIP,
PFR, 3-bet, c-bet, WTSD, W$SD, AF) next to typical ranges for the format, so the field looks like a real one before
any detection runs on it. The hands are synthetic; nothing in them comes from a real room.
