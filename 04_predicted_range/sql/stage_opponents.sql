-- The live opponents of every staged decision, each with the key of his preflop prior and his tilt inputs.
-- An opponent is live at a row when he has not folded at an earlier row. His preflop key is the situation and
-- action of his last preflop betting line (as in fit_prior_pre.sql). :split_sizes picks the tilt row of the
-- decision's table-size class (true) or the pooled 'all' row (false).

WITH fold_at AS (
    SELECT hand_id, player_id, min(action_index) AS fold_index
    FROM actions
    WHERE action = 'fold'
      AND hand_id IN (SELECT hand_id FROM rng_stage_dec)
    GROUP BY hand_id, player_id
),
last_line AS (
    SELECT hand_id,
           player_id,
           situation AS preflop_situation,
           action,
           row_number() OVER (PARTITION BY hand_id, player_id ORDER BY action_index DESC) AS rn
    FROM actions
    WHERE street = 'preflop'
      AND NOT is_post
      AND hand_id IN (SELECT hand_id FROM rng_stage_dec)
)
SELECT d.hand_id,
       d.action_index,
       hp.player_id AS opp_id,
       hp.hole_cards AS opp_cards,
       coalesce(l.preflop_situation, 'none') AS pf_sit,
       coalesce(l.action, 'check') AS pf_act,
       hp.position_class,
       hp.pfr AS raised_pf,
       t.n AS t_n,
       t.vpip AS t_vpip,
       t.pfr AS t_pfr,
       t.agg AS t_agg
FROM rng_stage_dec AS d
JOIN hand_players AS hp ON hp.hand_id = d.hand_id AND hp.player_id <> d.player_id
LEFT JOIN fold_at AS f ON f.hand_id = hp.hand_id AND f.player_id = hp.player_id
LEFT JOIN last_line AS l ON l.hand_id = hp.hand_id AND l.player_id = hp.player_id AND l.rn = 1
LEFT JOIN rng_player_tilt AS t
       ON t.player_id = hp.player_id
      AND t.size_class = CASE WHEN :split_sizes THEN d.size_class ELSE 'all' END
WHERE f.fold_index IS NULL
   OR f.fold_index > d.action_index
ORDER BY d.hand_id, d.action_index, hp.seat;
