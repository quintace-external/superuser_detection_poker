-- Summary of a step-02 run. run.py prints every query below under the title on its first comment line.

-- Size: hands, postflop decision rows, those with the actor's cards known, rows scored, heads-up hands scored
SELECT (SELECT count(*) FROM hands) AS hands,
       (SELECT count(*) FROM decisions WHERE street <> 'preflop') AS postflop_decisions,
       (SELECT count(*) FROM decisions WHERE street <> 'preflop' AND hole_cards IS NOT NULL) AS actor_cards_known,
       (SELECT count(*) FROM action_equity) AS action_equity_rows,
       (SELECT count(DISTINCT hand_id) FROM hand_equity) AS hand_equity_hands;

-- Rows by number of live opponents
SELECT n_live_opp,
       count(*) AS n_rows,
       avg(eq_vs_random) AS mean_eq_random,
       avg(eq_vs_real) AS mean_eq_real,
       avg(CASE WHEN ahead_now THEN 1.0 WHEN NOT ahead_now THEN 0.0 END) AS share_ahead
FROM action_equity
GROUP BY n_live_opp
ORDER BY n_live_opp;

-- Rows, unknown real cards and the two equities by street
SELECT street,
       count(*) AS n_rows,
       avg(CASE WHEN eq_vs_real IS NULL THEN 1.0 ELSE 0.0 END) AS share_real_null,
       avg(CASE WHEN n_live_opp = 1 THEN 1.0 ELSE 0.0 END) AS share_hu,
       avg(eq_vs_random) AS mean_eq_random,
       avg(eq_vs_real) AS mean_eq_real,
       corr(eq_vs_random, eq_vs_real) AS corr_all,
       corr(CASE WHEN n_live_opp = 1 THEN eq_vs_random END,
            CASE WHEN n_live_opp = 1 THEN eq_vs_real END) AS corr_hu,
       avg(CASE WHEN ahead_now THEN 1.0 WHEN NOT ahead_now THEN 0.0 END) AS share_ahead
FROM action_equity
GROUP BY street
ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 ELSE 3 END;

-- Heads-up pots facing a bet: mean equities and pot odds by response
SELECT street,
       action,
       count(*) AS n_rows,
       avg(eq_vs_random) AS mean_eq_random,
       avg(eq_vs_real) AS mean_eq_real,
       avg(pot_odds) AS mean_pot_odds,
       avg(CASE WHEN eq_vs_real >= pot_odds THEN 1.0 ELSE 0.0 END) AS share_real_ge_pot_odds
FROM decisions_eq
WHERE n_live_opp = 1
  AND pot_odds IS NOT NULL
  AND eq_vs_real IS NOT NULL
GROUP BY street, action
ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 ELSE 3 END, action;

-- Heads-up identity in action_equity: first decisions of both players on one street (exact, so the sum is 1)
SELECT a.street,
       count(*) AS pairs,
       max(abs(a.eq_vs_real + b.eq_vs_real - 1)) AS max_abs_dev
FROM decisions_eq AS a
JOIN decisions_eq AS b
  ON b.hand_id = a.hand_id
 AND b.street = a.street
 AND b.player_id > a.player_id
WHERE a.n_live_opp = 1 AND b.n_live_opp = 1
  AND a.player_street_round = 1 AND b.player_street_round = 1
  AND a.eq_real_known AND b.eq_real_known
GROUP BY a.street
ORDER BY CASE a.street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 ELSE 3 END;

-- Heads-up hands: preflop equity vs the opponent's hand
SELECT count(*) AS hands,
       avg(eq_se) AS mean_se,
       max(eq_se) AS max_se,
       max(abs(total - 1)) AS max_abs_dev_of_sum,
       avg(eq_high) AS mean_eq_of_better_hand
FROM (
    SELECT hand_id, sum(preflop_eq_vs_opp) AS total, max(eq_se) AS eq_se, max(preflop_eq_vs_opp) AS eq_high
    FROM hand_equity
    GROUP BY hand_id
) AS x;

-- Methods and standard errors
SELECT street,
       eq_method,
       eq_real_method,
       count(*) AS n_rows,
       max(eq_se) AS max_eq_se,
       avg(eq_se) AS mean_eq_se,
       max(eq_real_se) AS max_eq_real_se
FROM action_equity
GROUP BY street, eq_method, eq_real_method
ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 ELSE 3 END, eq_method, eq_real_method;
