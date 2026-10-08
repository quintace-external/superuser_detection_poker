-- Output tables of step 04 that run.py and examples.py fill from Python (the fit tables are created by their own
-- SQL files). Each run replaces them.

-- One row per scored decision: the hero's equity against the predicted range(s) of his live opponent(s).
CREATE OR REPLACE TABLE action_range_equity (
    hand_id             VARCHAR,
    action_index        SMALLINT,
    n_live_opp          SMALLINT,  -- live opponents at the row, each with his own range
    eq_vs_range         DOUBLE,    -- share of the pot the hero wins on average against the range(s)
    eq_range_method     VARCHAR,   -- exact (enumeration) | mc (Monte Carlo)
    eq_range_se         DOUBLE,    -- standard error of eq_vs_range (0 when exact)
    n_combos_opp        INTEGER,   -- combos in the opponent's range (multiway: the smallest of the ranges)
    range_top_class     VARCHAR,   -- heaviest of the 169 classes in that range
    range_strong_share  DOUBLE,    -- share of that range holding two pair or better (flush, straight, set, ...)
    range_air_share     DOUBLE     -- share of that range holding no pair
);

-- Fold rates of the heads-up responses to a bet by street, range bin and real class (input of the leakage split).
CREATE OR REPLACE TABLE rng_leak_matrix (
    street      VARCHAR,
    range_bin   SMALLINT,          -- least(bins - 1, floor(bins * eq_vs_range))
    real_class  VARCHAR,           -- dead (eq_vs_real < dead_below) | mid | win (eq_vs_real >= win_from)
    n           BIGINT,            -- responses (fold, call or raise)
    folds       BIGINT
);

-- Ranges of the narrowing example (examples.py): 169 class weights per step for one opponent of one hand.
CREATE OR REPLACE TABLE action_range_opp (
    hand_id        VARCHAR,
    step           SMALLINT,       -- 0 = after preflop, then one step per postflop action of the opponent
    action_index   SMALLINT,       -- the opponent's row after which the range is shown (NULL for step 0)
    step_label     VARCHAR,        -- e.g. 'flop: bet 0.75 pot'
    opp_player_id  VARCHAR,
    hero_player_id VARCHAR,
    n_combos       INTEGER,
    hand_class     VARCHAR,        -- one of the 169 classes
    weight         DOUBLE          -- share of the range in that class (sums to 1 per step)
);
