-- Invariants of the step-01 tables. One row per check: (check_name, violations). Every count must be 0.
-- Parameter :tol is the tolerance for amounts in big blinds (config: step01.invariant_tol).

WITH
dup_hands AS (
    SELECT hand_id FROM hands GROUP BY hand_id HAVING count(*) > 1
),
dup_players AS (
    SELECT hand_id, player_id FROM hand_players GROUP BY hand_id, player_id HAVING count(*) > 1
),
dup_actions AS (
    SELECT hand_id, action_index FROM actions GROUP BY hand_id, action_index HAVING count(*) > 1
),
seq AS (
    SELECT hand_id, player_id, action, action_index, is_post, amount_bb, amount_eff_bb, stack_before_bb,
           pot_before_bb,
           lag(pot_before_bb + amount_eff_bb) OVER (PARTITION BY hand_id ORDER BY action_index) AS pot_expect,
           lag(stack_before_bb - amount_bb) OVER (PARTITION BY hand_id, player_id ORDER BY action_index)
               AS stack_expect,
           max(CASE WHEN action = 'fold' THEN 1 ELSE 0 END) OVER (
               PARTITION BY hand_id, player_id ORDER BY action_index
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS folded_before
    FROM actions
),
per_hand_actions AS (
    SELECT hand_id, min(action_index) AS first_index, max(action_index) AS last_index, count(*) AS n_rows
    FROM actions GROUP BY hand_id
),
per_player_actions AS (
    SELECT hand_id, player_id,
           sum(amount_bb) AS put_in,
           max(CASE WHEN street = 'preflop' AND action IN ('call', 'bet', 'raise') THEN 1 ELSE 0 END) AS vpip_rows
    FROM actions GROUP BY hand_id, player_id
),
per_hand_players AS (
    SELECT hand_id, count(*) AS n_rows, sum(net_bb) AS net_sum, sum(committed_bb) AS committed_sum
    FROM hand_players GROUP BY hand_id
)
SELECT 'hands_key_unique' AS check_name, (SELECT count(*) FROM dup_hands) AS violations
UNION ALL
SELECT 'hand_players_key_unique', (SELECT count(*) FROM dup_players)
UNION ALL
SELECT 'actions_key_unique', (SELECT count(*) FROM dup_actions)
UNION ALL
SELECT 'action_index_contiguous',
       (SELECT count(*) FROM per_hand_actions WHERE first_index <> 1 OR last_index <> n_rows)
UNION ALL
SELECT 'pot_before_accumulates',
       (SELECT count(*) FROM seq WHERE pot_expect IS NOT NULL AND abs(pot_before_bb - pot_expect) > :tol)
UNION ALL
SELECT 'stack_before_accumulates',
       (SELECT count(*) FROM seq WHERE stack_expect IS NOT NULL AND abs(stack_before_bb - stack_expect) > :tol)
UNION ALL
SELECT 'no_action_after_fold', (SELECT count(*) FROM seq WHERE folded_before = 1)
UNION ALL
SELECT 'amount_within_stack',
       (SELECT count(*) FROM actions WHERE stack_before_bb < -:tol OR amount_bb > stack_before_bb + :tol)
UNION ALL
SELECT 'effective_not_above_actual',
       (SELECT count(*) FROM actions WHERE amount_eff_bb > amount_bb + :tol OR amount_eff_bb < -:tol)
UNION ALL
SELECT 'call_moves_amount_to_call',
       (SELECT count(*) FROM actions
        WHERE action = 'call' AND NOT is_post AND abs(amount_eff_bb - to_call_eff_bb) > :tol)
UNION ALL
SELECT 'check_faces_nothing',
       (SELECT count(*) FROM actions WHERE action = 'check' AND to_call_bb > :tol)
UNION ALL
SELECT 'pot_odds_in_range',
       (SELECT count(*) FROM actions
        WHERE NOT is_post AND to_call_eff_bb > :tol AND (pot_odds IS NULL OR pot_odds <= 0 OR pot_odds >= 1))
UNION ALL
SELECT 'money_identity',
       (SELECT count(*) FROM hands h JOIN per_hand_players p USING (hand_id)
        WHERE abs(p.net_sum + h.rake_bb + coalesce(h.fees_bb, 0)) > :tol)
UNION ALL
SELECT 'final_pot_from_contributions',
       (SELECT count(*) FROM hands h JOIN per_hand_players p USING (hand_id)
        WHERE abs(h.final_pot_bb - (p.committed_sum - h.uncalled_bb)) > :tol)
UNION ALL
SELECT 'committed_equals_action_amounts',
       (SELECT count(*) FROM hand_players hp LEFT JOIN per_player_actions a USING (hand_id, player_id)
        WHERE abs(hp.committed_bb - coalesce(a.put_in, 0)) > :tol)
UNION ALL
SELECT 'vpip_matches_actions',
       (SELECT count(*) FROM hand_players hp LEFT JOIN per_player_actions a USING (hand_id, player_id)
        WHERE hp.vpip <> (coalesce(a.vpip_rows, 0) = 1))
UNION ALL
SELECT 'every_dealt_player_has_a_row',
       (SELECT count(*) FROM hands h LEFT JOIN per_hand_players p USING (hand_id)
        WHERE coalesce(p.n_rows, 0) <> h.n_dealt)
UNION ALL
SELECT 'street_counts_monotone',
       (SELECT count(*) FROM hands WHERE players_flop < players_turn OR players_turn < players_river)
UNION ALL
SELECT 'showdown_has_two_players',
       (SELECT count(*) FROM hands WHERE went_to_showdown AND n_showdown < 2)
UNION ALL
SELECT 'hands_without_hard_errors',
       (SELECT count(*) FROM hands WHERE error_flag)
ORDER BY 1;
