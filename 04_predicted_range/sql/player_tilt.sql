-- Tilt inputs per player and table-size class, with the aggression smoothed toward the field:
--   w   = clip((n - :agg_min_n) / (:agg_full_n - :agg_min_n), 0, 1)
--   agg = w * agg_raw + (1 - w) * field_agg      (field_agg alone below :agg_min_n postflop actions)
-- Needs rng_tilt_raw (tilt_raw.sql) and rng_field (fit_field.sql, or copied from the dataset the tables come from).

CREATE OR REPLACE TABLE rng_player_tilt AS
WITH r AS (
    SELECT *,
           least(1.0, greatest(0.0, (n - :agg_min_n) * 1.0 / (:agg_full_n - :agg_min_n))) AS w
    FROM rng_tilt_raw
)
SELECT r.player_id,
       r.size_class,
       r.hands,
       r.n,
       r.vpip,
       r.pfr,
       r.agg_raw,
       CASE
           WHEN f.field_agg IS NULL THEN r.agg_raw
           WHEN r.agg_raw IS NULL THEN f.field_agg
           ELSE r.w * r.agg_raw + (1 - r.w) * f.field_agg
       END AS agg
FROM r
LEFT JOIN rng_field AS f ON f.size_class = r.size_class
ORDER BY r.player_id, r.size_class;
