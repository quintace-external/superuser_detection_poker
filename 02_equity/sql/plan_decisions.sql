-- Decision rows per hand that step 02 scores: every decision whose actor's cards are known, on the flop, turn
-- and river (and preflop when :include_preflop is true: the preflop rows with at most :preflop_max_opp live
-- opponents, every preflop row when it is 0). run.py cuts the hands, in this order, into batches of
-- about step02.batch_rows rows without splitting a hand.
SELECT d.hand_id,
       count(*) AS n_rows
FROM decisions AS d
WHERE d.hole_cards IS NOT NULL
  AND (d.street <> 'preflop'
       OR (:include_preflop AND (:preflop_max_opp <= 0 OR d.n_live - 1 <= :preflop_max_opp)))
GROUP BY d.hand_id
ORDER BY d.hand_id;
