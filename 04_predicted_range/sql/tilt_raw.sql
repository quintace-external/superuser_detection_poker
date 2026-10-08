-- Per player and table-size class: hands, VPIP, PFR and postflop aggression, the inputs of the two tilts.
--   vpip     share of hands with a voluntary preflop call or raise (hand_players.vpip)
--   pfr      share of hands with a preflop raise (hand_players.pfr)
--   n        postflop checks, calls, bets and raises (folds excluded): the sample size of both tilts
--   agg_raw  (bets + raises) / n
-- size_class 'all' pools every table size. None of these columns needs hole cards, so the view works on any dataset.

CREATE OR REPLACE VIEW rng_tilt_raw AS
WITH hp AS (
    SELECT hp.player_id,
           CASE WHEN h.n_dealt = 2 THEN '2' WHEN h.n_dealt <= 6 THEN '3-6' ELSE '7+' END AS size_class,
           hp.vpip,
           hp.pfr
    FROM hand_players AS hp
    JOIN hands AS h ON h.hand_id = hp.hand_id
    WHERE NOT h.error_flag
      AND h.variant = 'standard'
),
pre AS (
    SELECT player_id,
           coalesce(size_class, 'all') AS size_class,
           count(*) AS hands,
           avg(CASE WHEN vpip THEN 1.0 ELSE 0.0 END) AS vpip,
           avg(CASE WHEN pfr THEN 1.0 ELSE 0.0 END) AS pfr
    FROM hp
    GROUP BY GROUPING SETS ((player_id, size_class), (player_id))
),
post_rows AS (
    SELECT player_id,
           CASE WHEN table_players = 2 THEN '2' WHEN table_players <= 6 THEN '3-6' ELSE '7+' END AS size_class,
           action
    FROM decisions
    WHERE street IN ('flop', 'turn', 'river')
      AND action IN ('check', 'call', 'bet', 'raise')
),
post AS (
    SELECT player_id,
           coalesce(size_class, 'all') AS size_class,
           count(*) AS n,
           sum(CASE WHEN action IN ('bet', 'raise') THEN 1 ELSE 0 END) AS n_agg
    FROM post_rows
    GROUP BY GROUPING SETS ((player_id, size_class), (player_id))
)
SELECT pre.player_id,
       pre.size_class,
       pre.hands,
       coalesce(post.n, 0) AS n,
       pre.vpip,
       pre.pfr,
       CASE WHEN post.n > 0 THEN post.n_agg * 1.0 / post.n END AS agg_raw
FROM pre
LEFT JOIN post ON post.player_id = pre.player_id AND post.size_class = pre.size_class;
