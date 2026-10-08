-- Fill action_equity from the staged parquet parts (:parts is the list of part files).
-- DuckDB reads parquet directly. Spark SQL: INSERT INTO action_equity SELECT ... FROM parquet.`<stage folder>`;
-- Postgres: load the parts with COPY (after converting them) or an extension such as pg_parquet.
INSERT INTO action_equity
SELECT hand_id, action_index, street, n_live_opp,
       eq_vs_random, eq_method, eq_se,
       eq_vs_real, eq_real_known, eq_real_method, eq_real_se,
       ahead_now
FROM read_parquet(:parts)
ORDER BY hand_id, action_index;
