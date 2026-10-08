-- Decision rows per hand that step 02 scores: every decision whose actor's cards are known, on the flop, turn
-- and river (and preflop when :include_preflop is true). run.py cuts the hands, in this order, into batches of
-- about step02.batch_rows rows without splitting a hand.
SELECT d.hand_id,
       count(*) AS n_rows
FROM decisions AS d
WHERE d.hole_cards IS NOT NULL
  AND (d.street <> 'preflop' OR :include_preflop)
GROUP BY d.hand_id
ORDER BY d.hand_id;
