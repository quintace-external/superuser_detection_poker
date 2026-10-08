-- Checks of the step-02 tables. One row per check: (check_name, violations). Every count must be 0.
-- Parameters: :include_preflop (preflop rows were scored), :hand_equity (hand_equity was built).

WITH
scored AS (
    SELECT d.hand_id, d.action_index, d.n_live
    FROM decisions AS d
    WHERE d.hole_cards IS NOT NULL
      AND (d.street <> 'preflop' OR :include_preflop)
),
hu_hands AS (
    SELECT hp.hand_id
    FROM hand_players AS hp
    JOIN hands AS h ON h.hand_id = hp.hand_id
    WHERE h.n_dealt = 2 AND h.variant = 'standard' AND NOT h.error_flag AND hp.hole_cards IS NOT NULL
    GROUP BY hp.hand_id
    HAVING count(*) = 2
),
hand_sums AS (
    SELECT hand_id, count(*) AS n_rows, sum(preflop_eq_vs_opp) AS total
    FROM hand_equity
    GROUP BY hand_id
)
SELECT 'one_row_per_decision' AS check_name, count(*) AS violations
FROM (SELECT hand_id, action_index FROM action_equity GROUP BY hand_id, action_index HAVING count(*) > 1) AS x
UNION ALL
SELECT 'every_decision_scored', count(*)
FROM scored AS s
LEFT JOIN action_equity AS e ON e.hand_id = s.hand_id AND e.action_index = s.action_index
WHERE e.hand_id IS NULL
UNION ALL
SELECT 'only_decisions_scored', count(*)
FROM action_equity AS e
LEFT JOIN scored AS s ON s.hand_id = e.hand_id AND s.action_index = e.action_index
WHERE s.hand_id IS NULL
UNION ALL
SELECT 'n_live_opp_is_n_live_minus_1', count(*)
FROM action_equity AS e
JOIN scored AS s ON s.hand_id = e.hand_id AND s.action_index = e.action_index
WHERE e.n_live_opp <> s.n_live - 1
UNION ALL
SELECT 'eq_vs_random_present', count(*) FROM action_equity WHERE eq_vs_random IS NULL OR eq_method IS NULL
UNION ALL
SELECT 'equities_in_0_1', count(*)
FROM action_equity
WHERE eq_vs_random < 0 OR eq_vs_random > 1 OR eq_vs_real < 0 OR eq_vs_real > 1
UNION ALL
SELECT 'eq_vs_real_null_iff_unknown', count(*) FROM action_equity WHERE (eq_vs_real IS NULL) = eq_real_known
UNION ALL
SELECT 'eq_real_method_null_iff_unknown', count(*)
FROM action_equity WHERE (eq_real_method IS NULL) = eq_real_known
UNION ALL
SELECT 'ahead_now_null_iff_unknown', count(*)
FROM action_equity WHERE street <> 'preflop' AND (ahead_now IS NULL) = eq_real_known
UNION ALL
SELECT 'ahead_now_null_preflop', count(*) FROM action_equity WHERE street = 'preflop' AND ahead_now IS NOT NULL
UNION ALL
-- A Monte Carlo value of exactly 0 or 1 has a standard error of 0 too, so only the exact side is checked.
SELECT 'se_zero_when_exact', count(*)
FROM action_equity
WHERE (eq_method = 'exact' AND eq_se <> 0) OR (eq_real_method = 'exact' AND eq_real_se <> 0)
   OR eq_se < 0 OR eq_real_se < 0
UNION ALL
SELECT 'river_ahead_iff_equity_1', count(*)
FROM action_equity
WHERE street = 'river' AND eq_real_known AND ahead_now <> (eq_vs_real > 1 - 1e-6)
UNION ALL
SELECT 'hand_equity_two_rows_per_hand', count(*) FROM hand_sums WHERE n_rows <> 2
UNION ALL
SELECT 'hand_equity_rows_sum_to_1', count(*) FROM hand_sums WHERE abs(total - 1) > 1e-9
UNION ALL
SELECT 'hand_equity_covers_hu_hands', count(*)
FROM hu_hands AS u
LEFT JOIN hand_sums AS s ON s.hand_id = u.hand_id
WHERE :hand_equity AND s.hand_id IS NULL;
