-- Views that put the step-02 columns next to the step-01 rows. Rows step 02 did not score (posts, preflop by
-- default, decisions whose actor's cards are unknown) get NULL equity columns.

CREATE OR REPLACE VIEW actions_eq AS
SELECT a.*,
       e.n_live_opp, e.eq_vs_random, e.eq_method, e.eq_se,
       e.eq_vs_real, e.eq_real_known, e.eq_real_method, e.eq_real_se, e.ahead_now
FROM actions AS a
LEFT JOIN action_equity AS e
       ON e.hand_id = a.hand_id
      AND e.action_index = a.action_index;

CREATE OR REPLACE VIEW decisions_eq AS
SELECT d.*,
       e.n_live_opp, e.eq_vs_random, e.eq_method, e.eq_se,
       e.eq_vs_real, e.eq_real_known, e.eq_real_method, e.eq_real_se, e.ahead_now
FROM decisions AS d
LEFT JOIN action_equity AS e
       ON e.hand_id = d.hand_id
      AND e.action_index = d.action_index;

-- The heads-up decisions of step 01 (two players dealt in, both still in the hand) with the same columns.
CREATE OR REPLACE VIEW hu_decisions_eq AS
SELECT h.*,
       e.n_live_opp, e.eq_vs_random, e.eq_method, e.eq_se,
       e.eq_vs_real, e.eq_real_known, e.eq_real_method, e.eq_real_se, e.ahead_now
FROM hu_decisions AS h
LEFT JOIN action_equity AS e
       ON e.hand_id = h.hand_id
      AND e.action_index = h.action_index;

-- hand_players with each player's preflop equity vs the opponent's hand (heads-up hands only).
CREATE OR REPLACE VIEW hand_players_eq AS
SELECT hp.*,
       he.preflop_eq_vs_opp,
       he.eq_method AS preflop_eq_method,
       he.eq_se AS preflop_eq_se
FROM hand_players AS hp
LEFT JOIN hand_equity AS he
       ON he.hand_id = hp.hand_id
      AND he.player_id = hp.player_id;
