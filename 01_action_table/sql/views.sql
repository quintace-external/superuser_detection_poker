-- Views on the step-01 tables.

-- Decision rows that feed frequencies and outlier tests: no posts, no timeouts, no fold with nothing to call,
-- no hand with a hard error, standard hands only (bomb pots are kept only on request and never analysed).
CREATE OR REPLACE VIEW decisions AS
SELECT a.*
FROM actions AS a
WHERE a.is_decision
  AND NOT a.free_fold
  AND NOT a.error_flag
  AND a.hand_id NOT IN (SELECT hand_id FROM hands WHERE variant <> 'standard');

-- Decisions of hands dealt to exactly two players (heads-up tables), both still in the hand.
CREATE OR REPLACE VIEW hu_decisions AS
SELECT *
FROM decisions
WHERE is_hu_table
  AND n_live = 2;
