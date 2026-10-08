-- Field baselines of the tilts, per table-size class: mean VPIP, PFR and postflop aggression of the "clean" players,
-- those with at least :min_hands hands whose VPIP and aggression lie within :sigma standard deviations of the mean
-- of the players with that many hands. Fitted with the likelihood and prior tables (a dataset scored with
-- --fit-from takes the baselines of the dataset the tables come from, so the tilts compare a player with the field
-- that produced the prior).

CREATE OR REPLACE TABLE rng_field AS
WITH eligible AS (
    SELECT *
    FROM rng_tilt_raw
    WHERE hands >= :min_hands
      AND agg_raw IS NOT NULL
),
moments AS (
    SELECT size_class,
           count(*) AS n_eligible,
           avg(vpip) AS mean_vpip,
           coalesce(stddev_samp(vpip), 0) AS sd_vpip,
           avg(agg_raw) AS mean_agg,
           coalesce(stddev_samp(agg_raw), 0) AS sd_agg
    FROM eligible
    GROUP BY size_class
),
clean AS (
    SELECT e.size_class, e.player_id, e.vpip, e.pfr, e.agg_raw
    FROM eligible AS e
    JOIN moments AS m ON m.size_class = e.size_class
    WHERE abs(e.vpip - m.mean_vpip) <= :sigma * m.sd_vpip
      AND abs(e.agg_raw - m.mean_agg) <= :sigma * m.sd_agg
)
SELECT m.size_class,
       m.n_eligible,
       count(c.player_id) AS n_clean,
       avg(c.vpip) AS field_vpip,
       avg(c.pfr) AS field_pfr,
       avg(c.agg_raw) AS field_agg
FROM moments AS m
LEFT JOIN clean AS c ON c.size_class = m.size_class
GROUP BY m.size_class, m.n_eligible
ORDER BY m.size_class;
