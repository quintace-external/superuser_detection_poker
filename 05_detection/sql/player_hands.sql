-- Step 05, input 2: one row per player per hand dealt to exactly two players (standard variant, no error).
-- Reads: hands, hand_players. A query; 05_detection/screen.py aggregates it (win rate, showdowns, VPIP / PFR, runs).
--   walk  the big blind took no decision (everyone folded to him); left out of VPIP / PFR when configured
SELECT p.hand_id, p.player_id, o.player_id AS opponent_id, h.hand_date, h.started_at,
       p.net_bb, p.net_pre_rake_bb, p.vpip, p.pfr,
       (p.is_bb AND p.n_decisions = 0) AS walk,
       p.saw_flop, p.went_to_showdown, p.won_at_showdown
FROM hand_players p
JOIN hands h ON h.hand_id = p.hand_id
JOIN hand_players o ON o.hand_id = p.hand_id AND o.player_id <> p.player_id
WHERE h.n_dealt = 2 AND h.variant = 'standard' AND NOT h.error_flag
ORDER BY p.player_id, h.started_at, p.hand_id;
