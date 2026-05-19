-- FinSight AI — PostgreSQL Schema
-- Run: psql -U postgres -d finsight -f backend/database/schema.sql

CREATE DATABASE IF NOT EXISTS finsight;
\c finsight;

-- ── Users ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(20) PRIMARY KEY,
    name            VARCHAR(100),
    email           VARCHAR(150) UNIQUE NOT NULL,
    annual_income   NUMERIC(12, 2),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Transactions ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id          VARCHAR(50) PRIMARY KEY,
    user_id                 VARCHAR(20) REFERENCES users(user_id),
    amount                  NUMERIC(12, 2) NOT NULL,
    merchant_category       VARCHAR(50),
    geo_distance_km         NUMERIC(8, 2),
    device_fingerprint_match SMALLINT DEFAULT 1,
    is_international        SMALLINT DEFAULT 0,
    is_night_txn            SMALLINT DEFAULT 0,
    fraud_score             NUMERIC(5, 4),      -- 0.0 to 1.0
    fraud_flag              BOOLEAN DEFAULT FALSE,
    risk_flag               VARCHAR(10),        -- LOW / MEDIUM / HIGH
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_txn_user ON transactions(user_id);
CREATE INDEX idx_txn_fraud ON transactions(fraud_flag);
CREATE INDEX idx_txn_created ON transactions(created_at DESC);

-- ── Fraud Alerts ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id        SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(50) REFERENCES transactions(transaction_id),
    user_id         VARCHAR(20),
    fraud_score     NUMERIC(5, 4),
    alert_reason    TEXT,
    status          VARCHAR(20) DEFAULT 'OPEN', -- OPEN / REVIEWED / CLOSED
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Stock Predictions ──────────────────────────────────
CREATE TABLE IF NOT EXISTS stock_predictions (
    prediction_id   SERIAL PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    target_date     DATE NOT NULL,
    predicted_close NUMERIC(10, 2),
    actual_close    NUMERIC(10, 2),
    confidence_low  NUMERIC(10, 2),
    confidence_high NUMERIC(10, 2),
    model_version   VARCHAR(20) DEFAULT 'lstm_v1',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, prediction_date, target_date)
);
CREATE INDEX idx_stock_symbol ON stock_predictions(symbol, prediction_date);

-- ── Credit Scores ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS credit_scores (
    score_id                SERIAL PRIMARY KEY,
    user_id                 VARCHAR(20) REFERENCES users(user_id),
    credit_score            INTEGER NOT NULL,           -- 300-850
    risk_category           VARCHAR(20) NOT NULL,       -- Low / Medium / High
    approval_status         VARCHAR(20),
    debt_to_income_ratio    NUMERIC(5, 3),
    credit_utilization      NUMERIC(5, 3),
    missed_payments_2y      INTEGER,
    top_negative_factor     VARCHAR(100),
    top_positive_factor     VARCHAR(100),
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_credit_user ON credit_scores(user_id);

-- ── Budget Reports ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS budget_reports (
    report_id       SERIAL PRIMARY KEY,
    user_id         VARCHAR(20) REFERENCES users(user_id),
    report_month    VARCHAR(7),                 -- e.g. '2025-06'
    total_spent     NUMERIC(12, 2),
    food_dining     NUMERIC(10, 2) DEFAULT 0,
    shopping        NUMERIC(10, 2) DEFAULT 0,
    transport       NUMERIC(10, 2) DEFAULT 0,
    entertainment   NUMERIC(10, 2) DEFAULT 0,
    healthcare      NUMERIC(10, 2) DEFAULT 0,
    utilities       NUMERIC(10, 2) DEFAULT 0,
    education       NUMERIC(10, 2) DEFAULT 0,
    groceries       NUMERIC(10, 2) DEFAULT 0,
    savings_tips    TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, report_month)
);

-- ── Stock OHLCV Cache (written by Spark) ──────────────
CREATE TABLE IF NOT EXISTS stock_ohlcv_1m (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    open        NUMERIC(10, 2),
    high        NUMERIC(10, 2),
    low         NUMERIC(10, 2),
    close       NUMERIC(10, 2),
    volume      BIGINT,
    UNIQUE (symbol, window_start)
);
CREATE INDEX idx_ohlcv_symbol_time ON stock_ohlcv_1m(symbol, window_start DESC);
