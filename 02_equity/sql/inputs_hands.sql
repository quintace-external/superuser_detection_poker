-- The heads-up hands of one batch (hands :lo to :hi): player A (the lower seat) and player B with their cards.
SELECT h.hand_id,
       a.player_id AS player_a,
       a.hole_cards AS cards_a,
       b.player_id AS player_b,
       b.hole_cards AS cards_b
FROM hands AS h
JOIN hand_players AS a
  ON a.hand_id = h.hand_id
JOIN hand_players AS b
  ON b.hand_id = h.hand_id
 AND b.seat > a.seat
WHERE h.n_dealt = 2
  AND h.variant = 'standard'
  AND NOT h.error_flag
  AND a.hole_cards IS NOT NULL
  AND b.hole_cards IS NOT NULL
  AND h.hand_id BETWEEN :lo AND :hi
  AND a.hand_id BETWEEN :lo AND :hi   -- redundant, lets the engine skip the other hands
  AND b.hand_id BETWEEN :lo AND :hi
ORDER BY h.hand_id;
