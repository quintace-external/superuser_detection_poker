-- Heads-up hands (two players dealt in) whose two hands are both known, one row per hand in hand order.
-- run.py cuts them into batches of step02.batch_hands hands.
SELECT hp.hand_id,
       1 AS n_rows
FROM hand_players AS hp
JOIN hands AS h
  ON h.hand_id = hp.hand_id
WHERE h.n_dealt = 2
  AND h.variant = 'standard'
  AND NOT h.error_flag
  AND hp.hole_cards IS NOT NULL
GROUP BY hp.hand_id
HAVING count(*) = 2
ORDER BY hp.hand_id;
