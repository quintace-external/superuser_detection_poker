-- Step 05, input 2: one row per player per hand (standard variant, no error), at tables of any size (parameter
-- :hu_tables_only true: hands dealt to exactly two players only, the article's scope).
-- Reads: hands, hand_players. A query; 05_detection/screen.py aggregates it (win rate, showdowns, VPIP / PFR, runs).
--   opponent_id  the other player of a heads-up confrontation: at a two-player table the other player, at a larger
--                table the other of exactly two players who saw the flop; NULL otherwise. Pairs are counted on it.
--   table_class  'hu' (two players dealt in), '3-6' or '7+'
--   walk         the big blind took no decision (everyone folded to him); left out of VPIP / PFR when configured
WITH flop_count AS (
    SELECT hand_id, count(*) FILTER (WHERE saw_flop) AS n_flop
    FROM hand_players
    GROUP BY hand_id
),
-- pairable: the player can have a heads-up opponent in this hand. The self-join below keys on an expression of it
-- (an equality DuckDB can hash): a condition on another table, or on one side only, inside the join's ON made DuckDB
-- compare every row with every row, hours on a large dataset
base AS (
    SELECT p.hand_id, p.player_id, p.net_bb, p.net_pre_rake_bb, p.vpip, p.pfr, p.is_bb, p.n_decisions,
           p.saw_flop, p.went_to_showdown, p.won_at_showdown, h.hand_date, h.started_at, h.n_dealt,
           (h.n_dealt = 2 OR (fc.n_flop = 2 AND p.saw_flop)) AS pairable
    FROM hand_players p
    JOIN hands h ON h.hand_id = p.hand_id
    JOIN flop_count fc ON fc.hand_id = p.hand_id
    WHERE (h.n_dealt = 2 OR NOT :hu_tables_only) AND h.variant = 'standard' AND NOT h.error_flag
)
SELECT p.hand_id, p.player_id, o.player_id AS opponent_id, p.hand_date, p.started_at,
       CASE WHEN p.n_dealt = 2 THEN 'hu' WHEN p.n_dealt <= 6 THEN '3-6' ELSE '7+' END AS table_class,
       p.net_bb, p.net_pre_rake_bb, p.vpip, p.pfr,
       (p.is_bb AND p.n_decisions = 0) AS walk,
       p.saw_flop, p.went_to_showdown, p.won_at_showdown
FROM base p
LEFT JOIN (SELECT hand_id, player_id FROM base WHERE pairable) o
       ON o.hand_id = CASE WHEN p.pairable THEN p.hand_id END
      AND o.player_id <> p.player_id
ORDER BY p.player_id, p.started_at, p.hand_id;
