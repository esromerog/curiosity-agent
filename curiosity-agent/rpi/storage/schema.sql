-- Curiosity Agent database schema

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- -----------------------------------------------------------------------
-- curiosities  – one row per "train of thought" session
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS curiosities (
    id              TEXT    PRIMARY KEY,          -- UUID
    started_at      REAL    NOT NULL,             -- Unix timestamp
    ended_at        REAL,
    status          TEXT    NOT NULL DEFAULT 'active',
        -- 'active' | 'completed' | 'saved' | 'ignored' | 'rejected'
    scene_context   TEXT,                         -- JSON: objects detected in trigger image
    trigger_question TEXT   NOT NULL,             -- the opening question Claude asked
    turn_count      INTEGER NOT NULL DEFAULT 0,
    summary         TEXT                          -- Claude-generated summary (on save/end)
);

-- -----------------------------------------------------------------------
-- turns  – individual exchanges within a curiosity
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    curiosity_id    TEXT    NOT NULL REFERENCES curiosities(id) ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    role            TEXT    NOT NULL,   -- 'assistant' | 'user'
    content         TEXT    NOT NULL,
    timestamp       REAL    NOT NULL
);

-- -----------------------------------------------------------------------
-- analytics_events  – fine-grained behavioural log
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analytics_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
        -- 'curiosity_started' | 'curiosity_ended' | 'curiosity_ignored'
        -- | 'curiosity_rejected' | 'curiosity_saved' | 'turn_answered'
        -- | 'turn_deferred' | 'cooldown_started' | 'cooldown_ended'
    curiosity_id    TEXT    REFERENCES curiosities(id) ON DELETE SET NULL,
    timestamp       REAL    NOT NULL,
    hour_of_day     INTEGER NOT NULL,   -- 0-23
    day_of_week     INTEGER NOT NULL,   -- 0=Mon … 6=Sun
    metadata        TEXT                -- JSON blob
);

-- -----------------------------------------------------------------------
-- interest_scores  – running tally per category (upserted after each turn)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interest_scores (
    category        TEXT    PRIMARY KEY,
    score           REAL    NOT NULL DEFAULT 0.0,
    mention_count   INTEGER NOT NULL DEFAULT 0,
    last_seen_at    REAL
);

-- -----------------------------------------------------------------------
-- user_profile  – key/value store for the hidden profile
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_profile (
    key             TEXT    PRIMARY KEY,
    value           TEXT    NOT NULL,   -- JSON value
    updated_at      REAL    NOT NULL
);

-- -----------------------------------------------------------------------
-- cooldown_log  – tracks when cooldowns were imposed
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cooldown_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      REAL    NOT NULL,
    ends_at         REAL    NOT NULL,
    reason          TEXT    NOT NULL    -- 'ignored' | 'rejected'
);

-- indices
CREATE INDEX IF NOT EXISTS idx_turns_curiosity    ON turns(curiosity_id);
CREATE INDEX IF NOT EXISTS idx_events_type        ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_hour        ON analytics_events(hour_of_day);
CREATE INDEX IF NOT EXISTS idx_events_curiosity   ON analytics_events(curiosity_id);
