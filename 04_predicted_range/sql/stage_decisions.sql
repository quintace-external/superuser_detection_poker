-- Decisions that get a predicted range: postflop rows of the decisions view whose actor's cards are known, with
-- 1 to :max_live_opp opponents still in the hand, at two-player tables only when :hu_only is true.
-- A temporary table for the two staging queries that follow (Postgres: CREATE TEMP TABLE; Spark: a temp view).

CREATE OR REPLACE TEMP TABLE rng_stage_dec AS
SELECT d.hand_id,
       d.action_index,
       d.player_id,
       d.street,
       d.hole_cards,
       d.board_at_action AS board,
       d.n_live - 1 AS n_live_opp,
       d.action,
       d.pot_odds,
       CASE WHEN d.table_players = 2 THEN '2' WHEN d.table_players <= 6 THEN '3-6' ELSE '7+' END AS size_class
FROM decisions AS d
WHERE d.street IN ('flop', 'turn', 'river')
  AND d.hole_cards IS NOT NULL
  AND d.n_live - 1 BETWEEN 1 AND :max_live_opp
  AND (d.is_hu_table OR NOT :hu_only);
