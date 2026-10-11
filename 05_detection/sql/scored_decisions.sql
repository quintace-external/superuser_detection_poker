-- Step 05, input 1: one row per decision of a heads-up confrontation, with everything the scorecard, the
-- range-adjusted test, the hand tests and the preflop tests need. The scope (parameter :hu_tables_only):
--   false (scope "flop_hu", the default)  tables of any size. Postflop: every decision of a hand that exactly two
--                                         players saw the flop of. Preflop: every decision with exactly one live
--                                         opponent (everyone else has folded), whether or not a flop follows.
--   true  (scope "hu_tables")             the same, on hands dealt to exactly two players only (the article).
-- Either way n_live = 2 at every row, so the opponent is the one live opponent of the row.
-- Reads: actions, hands, hand_players, action_equity (step 02), action_range_equity and action_range_equity_pre
-- (step 04). A query: 05_detection/screen.py computes the metrics on its rows; nothing is written here.
--
--   q              equity against the opponent's real cards (NULL unless they are known)
--   r              equity against the opponent's predicted range (NULL without step 04; preflop rows: the field's
--                  range for his preflop line so far, 04_predicted_range/preflop.py)
--   req            pot odds B / (P + 2B) of the bet faced (NULL when not facing a bet)
--   facing         the last bet or raise of the street before this action came from the opponent
--   bettor_q       the bettor's q at that bet or raise
--   bluff_success  bets and raises only: the opponent acted again on the street and every one of those actions
--                  was a fold (NULL for other actions)
--   pot_bb         the hand's contested pot: everything put in, the uncalled part of the last bet returned
--   table_class    'hu' (two players dealt in), '3-6' or '7+'
-- Spark SQL: live_opponents[1] becomes live_opponents[0] (arrays are 0-based there).
WITH flop_two AS (
    SELECT hand_id
    FROM hand_players
    GROUP BY hand_id
    HAVING count(*) FILTER (WHERE saw_flop) = 2
),
street_rows AS (
    SELECT a.hand_id, a.action_index, a.player_id, a.street,
           CASE WHEN a.action IN ('bet', 'raise') THEN 1 ELSE 0 END AS aggr,
           MAX(CASE WHEN a.action IN ('bet', 'raise') THEN a.action_index END) OVER (
               PARTITION BY a.hand_id, a.street ORDER BY a.action_index
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS last_aggr_index,
           COUNT(*) OVER (
               PARTITION BY a.hand_id, a.street ORDER BY a.action_index
               ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS later_all,
           COUNT(*) OVER (
               PARTITION BY a.hand_id, a.street, a.player_id ORDER BY a.action_index
               ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS later_own,
           SUM(CASE WHEN a.action = 'fold' THEN 1 ELSE 0 END) OVER (
               PARTITION BY a.hand_id, a.street ORDER BY a.action_index
               ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS later_fold_all,
           SUM(CASE WHEN a.action = 'fold' THEN 1 ELSE 0 END) OVER (
               PARTITION BY a.hand_id, a.street, a.player_id ORDER BY a.action_index
               ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS later_fold_own
    FROM actions a
    WHERE NOT a.is_post
),
eq AS (
    SELECT hand_id, action_index,
           CASE WHEN eq_real_known THEN eq_vs_real END AS q,
           eq_vs_random AS q_random
    FROM action_equity
),
dec AS (
    SELECT d.*, d.live_opponents[1].player_id AS opp_id
    FROM decisions d
    LEFT JOIN flop_two f ON f.hand_id = d.hand_id
    WHERE d.n_live = 2
      AND (d.is_hu_table OR NOT :hu_tables_only)
      AND (d.street = 'preflop' OR (d.street IN ('flop', 'turn', 'river') AND f.hand_id IS NOT NULL))
)
SELECT d.hand_id, d.action_index, d.player_id, o.player_id AS opponent_id,
       d.hand_date, h.started_at, d.street, d.action, d.is_allin, d.decision_s,
       d.hole_cards, o.hole_cards AS opp_hole_cards, d.board_at_action, d.made_hand,
       d.pot_odds AS req, d.pot_before_bb, d.to_call_bb, d.faced_size_pot,
       e.q, e.q_random, COALESCE(re.eq_vs_range, rp.eq_vs_range) AS r,
       (s.last_aggr_index IS NOT NULL AND b.player_id <> d.player_id) AS facing,
       CASE WHEN b.player_id <> d.player_id THEN b.player_id END AS bettor_id,
       CASE WHEN b.player_id <> d.player_id THEN be.q END AS bettor_q,
       CASE WHEN s.aggr = 1
            THEN (s.later_all - s.later_own) > 0
                 AND COALESCE(s.later_fold_all, 0) - COALESCE(s.later_fold_own, 0) = s.later_all - s.later_own
       END AS bluff_success,
       h.final_pot_bb AS pot_bb,
       p.went_to_showdown, p.won_at_showdown,
       d.situation, d.position, d.hand_class, rp.opp_situation, rp.opp_action, rp.prior_level AS pre_prior_level,
       CASE WHEN h.n_dealt = 2 THEN 'hu' WHEN h.n_dealt <= 6 THEN '3-6' ELSE '7+' END AS table_class
FROM dec d
JOIN street_rows s ON s.hand_id = d.hand_id AND s.action_index = d.action_index
JOIN hands h ON h.hand_id = d.hand_id
JOIN hand_players p ON p.hand_id = d.hand_id AND p.player_id = d.player_id
JOIN hand_players o ON o.hand_id = d.hand_id AND o.player_id = d.opp_id
LEFT JOIN street_rows b ON b.hand_id = s.hand_id AND b.action_index = s.last_aggr_index
LEFT JOIN eq e ON e.hand_id = d.hand_id AND e.action_index = d.action_index
LEFT JOIN eq be ON be.hand_id = b.hand_id AND be.action_index = b.action_index
LEFT JOIN action_range_equity re ON re.hand_id = d.hand_id AND re.action_index = d.action_index
LEFT JOIN action_range_equity_pre rp ON rp.hand_id = d.hand_id AND rp.action_index = d.action_index
ORDER BY d.hand_id, d.action_index;
