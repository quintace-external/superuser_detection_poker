-- Fill hand_equity from the staged parquet parts (:parts is the list of part files); see load_action_equity.sql.
INSERT INTO hand_equity
SELECT hand_id, player_id, opp_player_id, preflop_eq_vs_opp, eq_method, eq_se
FROM read_parquet(:parts)
ORDER BY hand_id, player_id;
