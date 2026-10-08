-- Field likelihood of postflop actions with their size: the counts of rng_lik_post with every bet and raise split by
-- the tier of its effective size, size_pot = (amount_eff - to_call) / (pot_before + to_call), an all-in by its
-- effective size:
--   t1 < :e1 <= t2 < :e2 <= t3 < :e3 <= t4 < :e4 <= t5      (0.40, 0.60, 0.90 and 1.10 of the pot)
-- and the tier of the bet or raise the actor faces (faced_size_pot; 'none' when he faces no bet) added to every key.
-- The edges are passed in from sudetect.rangemodel.SIZE_EDGES. Levels:
--   S6 ... S1  the keys of K6 ... K1 in rng_lik_post, each with faced_tier added
--   F1         street, faced_tier      the field's size mix, the last back-off of the tier of a bet or a raise
--   F0         street
-- A column that a level does not use is NULL on that level's rows. A bet or raise of unknown size is left out.
-- Rows: as in fit_lik_post.sql, postflop decisions whose actor's cards are known.

CREATE OR REPLACE TABLE rng_lik_size AS
WITH a AS (
    SELECT street,
           situation,
           CASE WHEN is_ip THEN 'ip' ELSE 'oop' END AS pose,
           pot_type_now AS pot_type,
           made_hand,
           coalesce(draw, 'none') AS draw,
           CASE WHEN faced_size_pot IS NULL OR NOT isfinite(faced_size_pot) OR faced_size_pot <= 0 THEN 'none'
                WHEN faced_size_pot < :e1 THEN 't1'
                WHEN faced_size_pot < :e2 THEN 't2'
                WHEN faced_size_pot < :e3 THEN 't3'
                WHEN faced_size_pot < :e4 THEN 't4'
                ELSE 't5' END AS faced_tier,
           CASE WHEN action NOT IN ('bet', 'raise') THEN action
                WHEN size_pot IS NULL OR NOT isfinite(size_pot) OR size_pot <= 0 THEN NULL
                WHEN size_pot < :e1 THEN action || '_t1'
                WHEN size_pot < :e2 THEN action || '_t2'
                WHEN size_pot < :e3 THEN action || '_t3'
                WHEN size_pot < :e4 THEN action || '_t4'
                ELSE action || '_t5' END AS outcome
    FROM decisions
    WHERE street IN ('flop', 'turn', 'river')
      AND made_hand IS NOT NULL
      AND situation IS NOT NULL
      AND action IN ('fold', 'check', 'call', 'bet', 'raise')
),
g AS (
    SELECT grouping(situation) AS g_sit,
           grouping(pose) AS g_pose,
           grouping(pot_type) AS g_pot,
           grouping(made_hand) AS g_made,
           grouping(draw) AS g_draw,
           grouping(faced_tier) AS g_ft,
           street, situation, pose, pot_type, made_hand, draw, faced_tier,
           sum(CASE WHEN outcome = 'fold' THEN 1 ELSE 0 END) AS c_fold,
           sum(CASE WHEN outcome = 'check' THEN 1 ELSE 0 END) AS c_check,
           sum(CASE WHEN outcome = 'call' THEN 1 ELSE 0 END) AS c_call,
           sum(CASE WHEN outcome = 'bet_t1' THEN 1 ELSE 0 END) AS c_bet_t1,
           sum(CASE WHEN outcome = 'bet_t2' THEN 1 ELSE 0 END) AS c_bet_t2,
           sum(CASE WHEN outcome = 'bet_t3' THEN 1 ELSE 0 END) AS c_bet_t3,
           sum(CASE WHEN outcome = 'bet_t4' THEN 1 ELSE 0 END) AS c_bet_t4,
           sum(CASE WHEN outcome = 'bet_t5' THEN 1 ELSE 0 END) AS c_bet_t5,
           sum(CASE WHEN outcome = 'raise_t1' THEN 1 ELSE 0 END) AS c_raise_t1,
           sum(CASE WHEN outcome = 'raise_t2' THEN 1 ELSE 0 END) AS c_raise_t2,
           sum(CASE WHEN outcome = 'raise_t3' THEN 1 ELSE 0 END) AS c_raise_t3,
           sum(CASE WHEN outcome = 'raise_t4' THEN 1 ELSE 0 END) AS c_raise_t4,
           sum(CASE WHEN outcome = 'raise_t5' THEN 1 ELSE 0 END) AS c_raise_t5,
           count(*) AS n
    FROM a
    WHERE outcome IS NOT NULL
    GROUP BY GROUPING SETS (
        (street, situation, pose, pot_type, made_hand, draw, faced_tier),
        (street, situation, pose, made_hand, draw, faced_tier),
        (street, situation, made_hand, draw, faced_tier),
        (street, situation, made_hand, faced_tier),
        (street, situation, pose, pot_type, faced_tier),
        (street, situation, faced_tier),
        (street, faced_tier),
        (street)
    )
)
SELECT CASE
           WHEN g_sit = 1 AND g_ft = 1 THEN 'F0'
           WHEN g_sit = 1 THEN 'F1'
           WHEN g_pose = 0 AND g_pot = 0 AND g_made = 0 AND g_draw = 0 THEN 'S6'
           WHEN g_pose = 0 AND g_pot = 1 AND g_made = 0 AND g_draw = 0 THEN 'S5'
           WHEN g_pose = 1 AND g_pot = 1 AND g_made = 0 AND g_draw = 0 THEN 'S4'
           WHEN g_pose = 1 AND g_pot = 1 AND g_made = 0 AND g_draw = 1 THEN 'S3'
           WHEN g_pose = 0 AND g_pot = 0 AND g_made = 1 AND g_draw = 1 THEN 'S2'
           ELSE 'S1'
       END AS level,
       street, situation, pose, pot_type, made_hand, draw, faced_tier,
       c_fold, c_check, c_call, c_bet_t1, c_bet_t2, c_bet_t3, c_bet_t4, c_bet_t5,
       c_raise_t1, c_raise_t2, c_raise_t3, c_raise_t4, c_raise_t5, n
FROM g
ORDER BY level, street, situation, pose, pot_type, made_hand, draw, faced_tier;
