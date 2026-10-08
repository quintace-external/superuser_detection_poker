-- Every postflop check, call, bet and raise in the hands of the staged decisions: the nodes at which the range
-- model updates an opponent's range. Each row carries the spot as the field tables key it (street, situation, pose,
-- pot type), the board seen at that row and the sizes the sizing-aware model reads (his own bet or raise as a fraction
-- of the pot, and the bet or raise he faced). Timeouts (is_decision = false) are left out: the system acted.

SELECT a.hand_id,
       a.action_index,
       a.player_id,
       a.street,
       a.action,
       a.situation,
       CASE WHEN a.is_ip THEN 'ip' ELSE 'oop' END AS pose,
       a.pot_type_now AS pot_type,
       a.board_at_action AS board,
       a.size_pot,
       a.faced_size_pot
FROM actions AS a
WHERE a.street IN ('flop', 'turn', 'river')
  AND NOT a.is_post
  AND a.is_decision
  AND a.action IN ('check', 'call', 'bet', 'raise')
  AND a.hand_id IN (SELECT hand_id FROM rng_stage_dec)
ORDER BY a.hand_id, a.action_index;
