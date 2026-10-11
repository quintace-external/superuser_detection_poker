-- Preflop decisions with exactly one live opponent (everyone else has folded), at tables of any size, and the key
-- of the opponent's preflop line so far: what his range is read from in rng_prior_pre.
--   opp_situation, opp_action   situation and action of the opponent's last preflop action before this decision
--                               (NULL when he has not acted yet: his range is every hand)
--   size_class                  players at the table: '2', '3-6' or '7+' (the key step 04's enrich uses)
-- Spark SQL: d.live_opponents[1] becomes d.live_opponents[0] (arrays are 0-based there).
WITH dec AS (
    SELECT d.hand_id,
           d.action_index,
           d.player_id,
           d.hole_cards,
           d.hand_class,
           d.live_opponents[1].player_id AS opp_id,
           CASE WHEN d.table_players = 2 THEN '2' WHEN d.table_players <= 6 THEN '3-6' ELSE '7+' END AS size_class
    FROM decisions AS d
    WHERE d.street = 'preflop'
      AND d.n_live = 2
      AND d.hole_cards IS NOT NULL
),
opp_last AS (
    SELECT dec.hand_id,
           dec.action_index,
           a.situation,
           a.action,
           row_number() OVER (PARTITION BY dec.hand_id, dec.action_index ORDER BY a.action_index DESC) AS rn
    FROM dec
    JOIN actions AS a
      ON a.hand_id = dec.hand_id
     AND a.player_id = dec.opp_id
     AND a.street = 'preflop'
     AND NOT a.is_post
     AND a.action_index < dec.action_index
)
SELECT dec.hand_id,
       dec.action_index,
       dec.player_id,
       dec.opp_id,
       dec.hole_cards,
       dec.hand_class,
       dec.size_class,
       hp.position_class AS opp_position_class,
       ol.situation AS opp_situation,
       CASE WHEN ol.action = 'bet' THEN 'raise' ELSE ol.action END AS opp_action
FROM dec
JOIN hand_players AS hp
  ON hp.hand_id = dec.hand_id
 AND hp.player_id = dec.opp_id
LEFT JOIN opp_last AS ol
  ON ol.hand_id = dec.hand_id
 AND ol.action_index = dec.action_index
 AND ol.rn = 1
ORDER BY dec.hand_id, dec.action_index;
