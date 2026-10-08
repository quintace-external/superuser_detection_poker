-- Field likelihood of postflop actions: how often the field folds, checks, calls, bets or raises with each class of
-- hand in each spot. Six levels, from the finest key to the coarsest:
--   K6  street, situation, pose, pot_type, made_hand, draw
--   K5  street, situation, pose,           made_hand, draw
--   K4  street, situation,                 made_hand, draw
--   K3  street, situation,                 made_hand
--   K2  street, situation, pose, pot_type
--   K1  street, situation
-- A column that a level does not use is NULL on that level's rows. K2 and K1 carry no hand class, so the range model
-- learns nothing about the hand from an action that has to back off that far.
-- Rows: postflop decisions whose actor's cards are known. Only full-information data qualifies: on showdown-only
-- data the hidden hands are the ones that folded, and counting the shown ones alone overstates strong hands.

CREATE OR REPLACE TABLE rng_lik_post AS
WITH a AS (
    SELECT street,
           situation,
           CASE WHEN is_ip THEN 'ip' ELSE 'oop' END AS pose,
           pot_type_now AS pot_type,
           made_hand,
           coalesce(draw, 'none') AS draw,
           action
    FROM decisions
    WHERE street IN ('flop', 'turn', 'river')
      AND made_hand IS NOT NULL
      AND situation IS NOT NULL
      AND action IN ('fold', 'check', 'call', 'bet', 'raise')
),
g AS (
    SELECT grouping(pose) AS g_pose,
           grouping(pot_type) AS g_pot,
           grouping(made_hand) AS g_made,
           grouping(draw) AS g_draw,
           street, situation, pose, pot_type, made_hand, draw,
           sum(CASE WHEN action = 'fold' THEN 1 ELSE 0 END) AS c_fold,
           sum(CASE WHEN action = 'check' THEN 1 ELSE 0 END) AS c_check,
           sum(CASE WHEN action = 'call' THEN 1 ELSE 0 END) AS c_call,
           sum(CASE WHEN action = 'bet' THEN 1 ELSE 0 END) AS c_bet,
           sum(CASE WHEN action = 'raise' THEN 1 ELSE 0 END) AS c_raise,
           count(*) AS n
    FROM a
    GROUP BY GROUPING SETS (
        (street, situation, pose, pot_type, made_hand, draw),
        (street, situation, pose, made_hand, draw),
        (street, situation, made_hand, draw),
        (street, situation, made_hand),
        (street, situation, pose, pot_type),
        (street, situation)
    )
)
SELECT CASE
           WHEN g_pose = 0 AND g_pot = 0 AND g_made = 0 AND g_draw = 0 THEN 'K6'
           WHEN g_pose = 0 AND g_pot = 1 AND g_made = 0 AND g_draw = 0 THEN 'K5'
           WHEN g_pose = 1 AND g_pot = 1 AND g_made = 0 AND g_draw = 0 THEN 'K4'
           WHEN g_pose = 1 AND g_pot = 1 AND g_made = 0 AND g_draw = 1 THEN 'K3'
           WHEN g_pose = 0 AND g_pot = 0 AND g_made = 1 AND g_draw = 1 THEN 'K2'
           ELSE 'K1'
       END AS level,
       street, situation, pose, pot_type, made_hand, draw,
       c_fold, c_check, c_call, c_bet, c_raise, n
FROM g
ORDER BY level, street, situation, pose, pot_type, made_hand, draw;
