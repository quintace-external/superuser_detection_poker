-- Step 02 tables. Both are rebuilt in full by every run (run.py fills them from the staged parquet parts).
-- Equities are the actor's share of the pot if the board were dealt to the end with no more betting: a won pot
-- counts 1, a pot split k ways 1/k. Keys (checked by checks.sql): action_equity(hand_id, action_index),
-- hand_equity(hand_id, player_id).
-- Postgres has no CREATE OR REPLACE TABLE: use DROP TABLE IF EXISTS + CREATE TABLE. Spark SQL (Delta) has it.

CREATE OR REPLACE TABLE action_equity (
    hand_id         VARCHAR,
    action_index    SMALLINT,
    street          VARCHAR,   -- flop | turn | river (preflop too with step02.include_preflop)
    n_live_opp      SMALLINT,  -- live opponents at the row: not folded, all-in players included
    eq_vs_random    DOUBLE,    -- equity vs ONE random hand ("equity vs all hands"); a heads-up yardstick
    eq_method       VARCHAR,   -- exact | mc: how eq_vs_random was computed
    eq_se           DOUBLE,    -- standard error of eq_vs_random: 0 when exact, sqrt(p(1-p)/n) when mc
    eq_vs_real      DOUBLE,    -- equity vs the real cards of every live opponent; NULL unless all are known
    eq_real_known   BOOLEAN,   -- every live opponent's hole cards are known
    eq_real_method  VARCHAR,   -- exact | mc: how eq_vs_real was computed (NULL when eq_vs_real is NULL)
    eq_real_se      DOUBLE,    -- standard error of eq_vs_real
    ahead_now       BOOLEAN    -- the actor's hand beats every live opponent's hand on the current board (a tie is
                               -- not ahead); NULL unless all are known, and preflop
);

CREATE OR REPLACE TABLE hand_equity (
    hand_id            VARCHAR,
    player_id          VARCHAR,
    opp_player_id      VARCHAR,
    preflop_eq_vs_opp  DOUBLE,   -- preflop equity of player_id vs opp_player_id's real hand; the two rows sum to 1
    eq_method          VARCHAR,  -- mc (numpy backend) | exact (pyrust backend)
    eq_se              DOUBLE    -- standard error, the same on both rows of a hand
);
