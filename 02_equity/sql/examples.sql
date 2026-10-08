-- Example rows for 02_equity/examples.py. Two statements: the rows, then the live opponents of one row.

-- Five decision rows of different kinds with the actor's cards known: the :pick-th of each kind in hand order.
-- Kind 6 stands in for kind 2 when a dataset has no turn fold with the actor's cards known: observer data shows
-- cards only at a showdown, which a player who folds never reaches.
WITH tagged AS (
    SELECT d.hand_id, d.action_index, d.player_id, d.position, d.street, d.action, d.hole_cards,
           d.board_at_action, d.n_live, d.pot_before_bb, d.to_call_eff_bb, d.amount_eff_bb, d.pot_odds,
           CASE
               WHEN d.street = 'flop' AND d.action = 'call' AND d.n_live = 2 THEN 1
               WHEN d.street = 'turn' AND d.action = 'fold' AND d.n_live = 2 THEN 2
               WHEN d.street = 'river' AND d.action = 'call' AND d.n_live = 2 THEN 3
               WHEN d.street = 'river' AND d.action = 'bet' AND d.n_live = 2 THEN 4
               WHEN d.street <> 'preflop' AND d.pot_odds IS NOT NULL AND d.n_live >= 3 THEN 5
               WHEN d.street = 'turn' AND d.n_live = 2 THEN 6
           END AS example
    FROM decisions AS d
    WHERE d.hole_cards IS NOT NULL
),
ranked AS (
    SELECT t.*, row_number() OVER (PARTITION BY t.example ORDER BY t.hand_id, t.action_index) AS rn
    FROM tagged AS t
    WHERE t.example IS NOT NULL
)
SELECT example, hand_id, action_index, player_id, position, street, action, hole_cards, board_at_action,
       n_live, pot_before_bb, to_call_eff_bb, amount_eff_bb, pot_odds
FROM ranked
WHERE rn = :pick
  AND (example <> 6 OR NOT EXISTS (SELECT 1 FROM ranked AS k WHERE k.example = 2 AND k.rn = :pick))
ORDER BY example;

-- The live opponents of one decision row, with their position and hole cards (NULL when never shown).
SELECT o.opp.player_id AS player_id,
       hp.position,
       hp.hole_cards
FROM decisions AS d
CROSS JOIN UNNEST(d.live_opponents) AS o(opp)
LEFT JOIN hand_players AS hp
       ON hp.hand_id = d.hand_id
      AND hp.player_id = o.opp.player_id
WHERE d.hand_id = :hand_id
  AND d.action_index = :action_index
ORDER BY o.opp.seat;
