-- Runs once after all batches are loaded.

-- Order of every hand at its table.
UPDATE hands
SET table_seq = s.seq
FROM (
    SELECT hand_id,
           row_number() OVER (PARTITION BY table_id ORDER BY started_at, hand_id) AS seq
    FROM hands
) AS s
WHERE hands.hand_id = s.hand_id;
