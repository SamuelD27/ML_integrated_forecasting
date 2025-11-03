-- Drop tables if they exist (for clean setup)
DROP MATERIALIZED VIEW IF EXISTS ohlcv_1h CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ohlcv_5m CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ohlcv_1m CASCADE;
DROP TABLE IF EXISTS ticks CASCADE;

-- Tick data table
CREATE TABLE ticks (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL CHECK (price > 0),
    volume DOUBLE PRECISION CHECK (volume >= 0),
    bid DOUBLE PRECISION CHECK (bid >= 0),
    ask DOUBLE PRECISION CHECK (ask >= 0),
    CONSTRAINT valid_spread CHECK (ask >= bid)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

-- Create index for efficient symbol queries
CREATE INDEX idx_ticks_symbol_time ON ticks (symbol, time DESC);

-- 1-minute OHLCV continuous aggregate
CREATE MATERIALIZED VIEW ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume
FROM ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- 5-minute OHLCV
CREATE MATERIALIZED VIEW ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    symbol,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume
FROM ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- 1-hour OHLCV
CREATE MATERIALIZED VIEW ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume
FROM ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- Refresh policies (update aggregates every minute)
SELECT add_continuous_aggregate_policy('ohlcv_1m',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('ohlcv_5m',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Retention policy (keep raw ticks for 30 days)
SELECT add_retention_policy('ticks', INTERVAL '30 days', if_not_exists => TRUE);
