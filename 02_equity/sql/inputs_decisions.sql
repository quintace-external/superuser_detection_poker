-- The decision rows of one batch (hands :lo to :hi) with the real cards of every live opponent.
--
--   board        cards dealt before the decision, '' preflop ('AsKd7c')
--   n_live_opp   opponents not folded at the row, all-in players included
--   n_opp_known  how many of them have known hole cards in hand_players
--   opp_cards    their cards in seat order, 4 characters each ('AhKdQsQc'); read only when all are known
--
-- Folded players are not live, so their cards are never used, even when they are known.
-- Spark SQL: CROSS JOIN UNNEST(...) AS o(opp) becomes LATERAL VIEW explode(dec.live_opponents) o AS opp, and
-- string_agg(x, '' ORDER BY y) becomes array_join(transform(array_sort(collect_list(struct(y, x))), s -> s.x), '').
-- Postgres: CROSS JOIN LATERAL unnest(dec.live_opponents) AS o(opp).
WITH dec AS (
    SELECT d.hand_id, d.action_index, d.street, d.hole_cards, d.board_at_action, d.live_opponents
    FROM decisions AS d
    WHERE d.hole_cards IS NOT NULL
      AND (d.street <> 'preflop'
           OR (:include_preflop AND (:preflop_max_opp <= 0 OR d.n_live - 1 <= :preflop_max_opp)))
      AND d.hand_id BETWEEN :lo AND :hi
),
opp AS (
    SELECT dec.hand_id, dec.action_index, o.opp.seat AS opp_seat, hp.hole_cards AS opp_cards
    FROM dec
    CROSS JOIN UNNEST(dec.live_opponents) AS o(opp)
    LEFT JOIN hand_players AS hp
           ON hp.hand_id = dec.hand_id
          AND hp.player_id = o.opp.player_id
          AND hp.hand_id BETWEEN :lo AND :hi   -- redundant, lets the engine skip the other hands
)
SELECT dec.hand_id,
       dec.action_index,
       dec.street,
       dec.hole_cards,
       dec.board_at_action AS board,
       CAST(count(opp.opp_seat) AS SMALLINT) AS n_live_opp,
       CAST(count(opp.opp_cards) AS SMALLINT) AS n_opp_known,
       string_agg(opp.opp_cards, '' ORDER BY opp.opp_seat) AS opp_cards
FROM dec
LEFT JOIN opp
       ON opp.hand_id = dec.hand_id
      AND opp.action_index = dec.action_index
GROUP BY dec.hand_id, dec.action_index, dec.street, dec.hole_cards, dec.board_at_action
ORDER BY dec.hand_id, dec.action_index;
