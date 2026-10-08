-- Step 01 tables: hands, hand_players, actions, parse_funnel.
--
-- Three grains of the same hands. Count hands and money from `hands` / `hand_players`, never from `actions`.
-- Amounts are DOUBLE in big blinds, computed in integer chip units first and rounded to 1e-6.
-- Columns describing the end of the hand (showdown, net result) are repeated on action rows for filtering
-- only: they were not known when the player acted.
-- Keys (checked by invariants.sql, not declared, so bulk loads stay fast):
--   hands(hand_id), hand_players(hand_id, player_id), actions(hand_id, action_index).
-- The loader creates these tables if they do not exist; a full rebuild drops them first.

CREATE TABLE IF NOT EXISTS hands (
    hand_id            VARCHAR,   -- hand number as written (salted hash with --hash-ids)
    source             VARCHAR,   -- fixture | sim | mined
    table_id           VARCHAR,   -- table name as written (salted hash with --hash-ids)
    table_seq          INTEGER,   -- order of the hand at its table (by started_at, hand_id)
    game               VARCHAR,   -- NLHE
    variant            VARCHAR,   -- standard | bomb_pot (analysis uses standard)
    table_size         SMALLINT,  -- seats at the table (the N of "N-max")
    n_dealt            SMALLINT,  -- players dealt in
    started_at         TIMESTAMP, -- hand start, UTC
    hand_date          DATE,      -- UTC date of started_at (partition key)
    sb_bb              DOUBLE,    -- small blind
    ante_bb            DOUBLE,    -- per-player ante (0 when none)
    straddle_bb        DOUBLE,    -- largest straddle, NULL when none
    n_straddles        SMALLINT,
    button_blind_bb    DOUBLE,    -- forced button post, NULL when none
    button_seat        SMALLINT,  -- may be an empty seat (dead button)
    sb_seat            SMALLINT,  -- NULL when there is no small blind (dead small blind)
    bb_seat            SMALLINT,
    board              VARCHAR,   -- board of run 1 as dealt, '' without a flop, e.g. 'AsKd7c2h9s'
    board_run2         VARCHAR,   -- full board of run 2 (run it twice), NULL otherwise
    n_runs             SMALLINT,  -- boards dealt (1, or 2-3 with run it twice)
    runout_street      VARCHAR,   -- street where the runs differ, NULL with one run
    pot_flop_bb        DOUBLE,    -- pot when the flop is dealt, uncalled excess removed; NULL if not dealt
    pot_turn_bb        DOUBLE,
    pot_river_bb       DOUBLE,
    final_pot_bb       DOUBLE,    -- all chips committed minus the uncalled excess, before rake
    uncalled_bb        DOUBLE,    -- part of the last bet or raise nobody matched (dead posts never count)
    uncalled_seat      SMALLINT,  -- who got it back, NULL when 0
    rake_bb            DOUBLE,    -- printed rake (negative = a bonus booked as rake)
    fees_bb            DOUBLE,    -- other drops printed next to the rake
    n_side_pots        SMALLINT,  -- pots beyond the main pot
    preflop_raises     SMALLINT,  -- effective raises before the flop (straddles are posts)
    pot_type           VARCHAR,   -- walk | limped | single_raised | 3bet | 4bet_plus (whole hand: hindsight)
    last_street        VARCHAR,   -- last street dealt: preflop | flop | turn | river
    players_flop       SMALLINT,  -- players in the hand when the flop was dealt (all-in included); 0 if not dealt
    players_turn       SMALLINT,
    players_river      SMALLINT,
    went_to_showdown   BOOLEAN,   -- two or more players left when the betting ended
    n_showdown         SMALLINT,  -- players at the showdown (0 without one)
    n_shown            SMALLINT,  -- players whose cards were revealed at the table
    cards_visibility   VARCHAR,   -- full (every dealt player's cards in the text) | showdown
    error_flag         BOOLEAN,   -- a hard error code is present; feature builders skip these hands
    error_detail       VARCHAR,   -- comma-separated codes (hard and informational)
    parser_version     VARCHAR
);

CREATE TABLE IF NOT EXISTS hand_players (
    hand_id             VARCHAR,
    player_id           VARCHAR,
    seat                SMALLINT,
    hand_date           DATE,
    position            VARCHAR,  -- BTN, SB, BB, CO, HJ, LJ, UTG+2, UTG+1, UTG (heads-up: BTN and BB)
    position_class      VARCHAR,  -- EP | MP | CO | BTN | SB | BB
    is_sb               BOOLEAN,  -- posted the small blind (heads-up: the button)
    is_bb               BOOLEAN,  -- posted the big blind
    is_straddle         BOOLEAN,
    preflop_order       SMALLINT, -- 1 = first to act before the flop
    postflop_order      SMALLINT, -- 1 = first to act after the flop
    stack_start_bb      DOUBLE,   -- stack before antes and blinds
    eff_stack_start_bb  DOUBLE,   -- min(own stack, largest other stack)
    ante_bb             DOUBLE,   -- antes posted
    blind_bb            DOUBLE,   -- live blinds and straddles posted
    dead_bb             DOUBLE,   -- dead posts (missed blinds, button blind)
    post_kind           VARCHAR,  -- main post of the player: sb | bb | straddle | post | dead | ante | NULL
    hole_cards          VARCHAR,  -- 'AhKd', higher card first; NULL when unknown
    hand_class          VARCHAR,  -- one of the 169 preflop classes: 'AKo', '77', 'T9s'
    hand_bucket10       VARCHAR,  -- one of 10 preflop buckets (sudetect.handclass)
    cards_known         BOOLEAN,
    cards_source        VARCHAR,  -- dealt (full-information text) | showdown | shown
    vpip                BOOLEAN,  -- put chips in voluntarily before the flop
    pfr                 BOOLEAN,  -- raised before the flop
    preflop_raises_made SMALLINT,
    saw_flop            BOOLEAN,  -- in the hand when the flop was dealt (all-in counts)
    saw_turn            BOOLEAN,
    saw_river           BOOLEAN,
    folded_street       VARCHAR,  -- NULL if he never folded
    allin_street        VARCHAR,  -- NULL if he never went all-in
    n_decisions         SMALLINT, -- his decision rows (posts and timeouts excluded)
    committed_bb        DOUBLE,   -- chips put in: antes, posts and betting lines
    returned_bb         DOUBLE,   -- uncalled chips returned (as printed)
    won_bb              DOUBLE,   -- chips collected from all pots and runs
    net_bb              DOUBLE,   -- won + returned - committed (after rake)
    rake_share_bb       DOUBLE,   -- rake attributed to him in proportion to what he collected
    net_pre_rake_bb     DOUBLE,   -- net_bb + rake_share_bb
    went_to_showdown    BOOLEAN,  -- had not folded when the betting ended with 2+ players left
    cards_shown         BOOLEAN,  -- his cards were revealed at the table
    won_at_showdown     BOOLEAN,  -- went to showdown and collected chips
    error_flag          BOOLEAN,
    error_detail        VARCHAR
);

-- live_opponents is a list of structs (Spark SQL: ARRAY<STRUCT<...>>; Postgres: an array of a composite type).
CREATE TABLE IF NOT EXISTS actions (
    hand_id                  VARCHAR,
    action_index             SMALLINT, -- 1.. over posts and betting lines, in hand-history order
    player_id                VARCHAR,
    seat                     SMALLINT,
    hand_date                DATE,
    acted_at                 TIMESTAMP, -- time of the action when the source has it (NULL in text)
    decision_s               DOUBLE,    -- think time in seconds when the source has it
    auto_kind                VARCHAR,   -- timeout (the system acted) | unknown (no timing in the source)
    street                   VARCHAR,   -- preflop | flop | turn | river
    street_action_index      SMALLINT,  -- 1.. over betting lines of the street (NULL on posts)
    player_street_round      SMALLINT,  -- Nth betting line of this player on this street
    is_post                  BOOLEAN,
    post_kind                VARCHAR,   -- ante | sb | bb | straddle | post | dead | button_blind | hidden
    action                   VARCHAR,   -- post | fold | check | call | bet | raise (effective)
    action_raw               VARCHAR,   -- verb as written (an all-in "raise" nobody can call more is a call)
    is_allin                 BOOLEAN,   -- the row put the player's last chip in
    is_capped                BOOLEAN,   -- the effective amount is below the chips moved
    free_fold                BOOLEAN,   -- a fold with nothing to call
    is_decision              BOOLEAN,   -- not a post and not a timeout
    position                 VARCHAR,
    position_class           VARCHAR,
    n_dealt                  SMALLINT,
    table_players            SMALLINT,  -- = hands.n_dealt
    is_hu_table              BOOLEAN,   -- two players dealt in
    n_live                   SMALLINT,  -- players not folded before the row, actor included
    n_live_start_street      SMALLINT,  -- players not folded when the street began (all-in included)
    live_opponents           STRUCT(player_id VARCHAR, seat SMALLINT, behind_bb DOUBLE, street_put_bb DOUBLE,
                                    is_allin BOOLEAN)[], -- opponents not folded (betting lines only)
    n_behind                 SMALLINT,  -- live opponents with chips who act after him on this street
    is_ip                    BOOLEAN,   -- after the flop: last to act among players able to act at the street start
    last_aggressor_id        VARCHAR,   -- last bettor or raiser on this street before the row
    prev_street_aggressor_id VARCHAR,   -- last bettor or raiser of the previous street
    preflop_aggressor_id     VARCHAR,   -- last preflop raiser as of the row
    faced_bettor_id          VARCHAR,   -- who made the bet or raise the actor faces
    bets_before              SMALLINT,  -- bets and raises on this street before the row
    calls_before             SMALLINT,
    checks_before            SMALLINT,
    n_limpers                SMALLINT,  -- calls before the first raise (preflop)
    pot_type_now             VARCHAR,   -- unopened (preflop rows) | limped | single_raised | 3bet | 4bet_plus
    situation                VARCHAR,   -- see sudetect/situations.py
    line                     VARCHAR,   -- turn and river: x/b per earlier postflop street, e.g. 'bx'
    hole_cards               VARCHAR,   -- actor's cards when known
    hand_class               VARCHAR,
    hand_bucket10            VARCHAR,
    made_hand                VARCHAR,   -- actor's made-hand class on board_at_action
    draw                     VARCHAR,   -- actor's draw class on board_at_action
    board_at_action          VARCHAR,   -- board seen at the row ('' before the flop)
    board_texture            VARCHAR,   -- texture of the flop (postflop rows)
    texture_class            VARCHAR,   -- coarse flop texture class
    board_change             VARCHAR,   -- what the turn or river card changed
    stack_before_bb          DOUBLE,    -- actual chips behind
    amount_bb                DOUBLE,    -- actual chips moved by the row
    amount_eff_bb            DOUBLE,    -- effective chips (capped)
    street_put_before_bb     DOUBLE,    -- his street contribution before the row (effective)
    level_before_bb          DOUBLE,    -- highest street contribution before the row (effective)
    to_call_bb               DOUBLE,    -- level - own street put, not capped
    to_call_eff_bb           DOUBLE,    -- the most he can be asked to put in now
    raise_to_eff_bb          DOUBLE,    -- his street total after a bet or raise
    pot_before_bb            DOUBLE,    -- every chip committed before the row (effective)
    eff_stack_bb             DOUBLE,    -- the most he can still lose to anyone
    spr                      DOUBLE,    -- eff_stack / pot_before
    size_pot                 DOUBLE,    -- (amount_eff - to_call) / (pot_before + to_call), bets and raises
    size_x                   DOUBLE,    -- raise_to_eff / level_before, raises
    faced_size_pot           DOUBLE,    -- size_pot of the bet or raise faced; NULL when the level is only posts
    pot_odds                 DOUBLE,    -- equity a call needs (side-pot aware); NULL with nothing to call
    went_to_showdown         BOOLEAN,   -- hindsight
    cards_shown              BOOLEAN,   -- hindsight
    net_bb                   DOUBLE,    -- hindsight
    error_flag               BOOLEAN,
    error_detail             VARCHAR
);

CREATE TABLE IF NOT EXISTS parse_funnel (
    dataset  VARCHAR,
    file     VARCHAR,  -- input file (archive member as 'archive.zip:member')
    reason   VARCHAR,  -- loaded | duplicate | duplicate_diff | replay_error | a gate reason
    n_hands  BIGINT
);
