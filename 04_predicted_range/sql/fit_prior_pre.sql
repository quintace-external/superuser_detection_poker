-- Preflop prior: which of the 169 starting-hand classes the players who saw a flop held, per preflop key.
-- The key is the one the range model looks up when it scores a decision, so the table and the lookup agree:
--   size_class          players dealt in: '2', '3-6' or '7+'
--   preflop_situation   situation of the player's last preflop betting line (unopened, vs_limp, vs_open,
--                       vs_open_call, vs_3bet, vs_4bet_plus; 'none' when he never acted, e.g. an all-in blind)
--   position_class      EP, MP, CO, BTN, SB, BB (heads-up: BTN and BB)
--   action              the action of that last line: check, call or raise
-- Levels, from the finest key to the coarsest:
--   P4  size_class, preflop_situation, position_class, action
--   P3              preflop_situation, position_class, action
--   P2              preflop_situation, position_class
--   P1                                 position_class, action
--   P0  every flop
-- An opener who called a 3-bet is keyed (vs_3bet, call) and a 4-bettor whose raise was called (vs_3bet, raise), so the
-- two never share a cell. Folded hands never enter: only players still in the hand when the flop was dealt count.

CREATE OR REPLACE TABLE rng_prior_pre AS
WITH last_line AS (
    SELECT hand_id,
           player_id,
           situation AS preflop_situation,
           action,
           row_number() OVER (PARTITION BY hand_id, player_id ORDER BY action_index DESC) AS rn
    FROM actions
    WHERE street = 'preflop'
      AND NOT is_post
),
seen AS (
    SELECT CASE WHEN h.n_dealt = 2 THEN '2' WHEN h.n_dealt <= 6 THEN '3-6' ELSE '7+' END AS size_class,
           coalesce(l.preflop_situation, 'none') AS preflop_situation,
           hp.position_class,
           coalesce(l.action, 'check') AS action,
           hp.hand_class
    FROM hand_players AS hp
    JOIN hands AS h ON h.hand_id = hp.hand_id
    LEFT JOIN last_line AS l ON l.hand_id = hp.hand_id AND l.player_id = hp.player_id AND l.rn = 1
    WHERE hp.saw_flop
      AND hp.hand_class IS NOT NULL
      AND NOT h.error_flag
      AND h.variant = 'standard'
),
g AS (
    SELECT grouping(size_class) AS g_size,
           grouping(preflop_situation) AS g_sit,
           grouping(position_class) AS g_pos,
           grouping(action) AS g_act,
           size_class, preflop_situation, position_class, action, hand_class,
           count(*) AS n
    FROM seen
    GROUP BY GROUPING SETS (
        (size_class, preflop_situation, position_class, action, hand_class),
        (preflop_situation, position_class, action, hand_class),
        (preflop_situation, position_class, hand_class),
        (position_class, action, hand_class),
        (hand_class)
    )
)
SELECT CASE
           WHEN g_size = 0 THEN 'P4'
           WHEN g_sit = 0 AND g_act = 0 THEN 'P3'
           WHEN g_sit = 0 THEN 'P2'
           WHEN g_pos = 0 THEN 'P1'
           ELSE 'P0'
       END AS level,
       size_class, preflop_situation, position_class, action, hand_class, n
FROM g
ORDER BY level, size_class, preflop_situation, position_class, action, hand_class;
