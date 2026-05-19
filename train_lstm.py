"""
FinSight AI — LSTM Stock Price Prediction Model
Trains a stacked LSTM to predict next 7-day closing prices.
Run: python backend/ml/stock/train_lstm.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/stock_prices.csv")

# Try importing TensorFlow, fallback gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Run: pip install tensorflow")


SEQUENCE_LEN = 60     # Look-back window: 60 trading days
PREDICT_DAYS = 7      # Forecast horizon: 7 days
SYMBOL = "AAPL"       # Train on this symbol (change as needed)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 20+ technical indicators from OHLCV data."""
    # Exponential Moving Averages
    for span in [9, 21, 50]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    # RSI (14-period)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-period)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # VWAP (daily approximation)
    df["vwap"] = (df["close"] * df["volume"]).rolling(14).sum() / df["volume"].rolling(14).sum()

    # Average True Range (ATR)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["atr_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Price change features
    df["returns_1d"] = df["close"].pct_change()
    df["returns_5d"] = df["close"].pct_change(5)
    df["high_low_ratio"] = df["high"] / (df["low"] + 1e-9)

    df.dropna(inplace=True)
    return df


def make_sequences(data: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon, 0])  # close price index=0
    return np.array(X), np.array(y)


def build_lstm(input_shape, output_steps):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(output_steps)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="huber",
                  metrics=["mae"])
    model.summary()
    return model


def main():
    if not TF_AVAILABLE:
        print("Install TensorFlow to train LSTM: pip install tensorflow")
        return

    print(f"Training LSTM for {SYMBOL}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["symbol"] == SYMBOL].sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} trading days loaded")

    df = add_technical_indicators(df)

    feature_cols = ["close", "open", "high", "low", "volume",
                    "ema_9", "ema_21", "ema_50", "rsi_14",
                    "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
                    "vwap", "atr_14", "obv", "volume_ratio",
                    "returns_1d", "returns_5d", "high_low_ratio"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    # Train/val/test split (70/15/15)
    n = len(scaled)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = make_sequences(scaled[:train_end], SEQUENCE_LEN, PREDICT_DAYS)
    X_val,   y_val   = make_sequences(scaled[train_end:val_end], SEQUENCE_LEN, PREDICT_DAYS)
    X_test,  y_test  = make_sequences(scaled[val_end:], SEQUENCE_LEN, PREDICT_DAYS)
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = build_lstm((SEQUENCE_LEN, len(feature_cols)), PREDICT_DAYS)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, monitor="val_loss")
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=100,
              batch_size=32,
              callbacks=callbacks,
              verbose=1)

    # Evaluate
    preds = model.predict(X_test)
    # De-scale close price column only
    close_idx = feature_cols.index("close")
    dummy = np.zeros((preds.shape[0] * PREDICT_DAYS, len(feature_cols)))
    dummy[:, close_idx] = preds.flatten()
    preds_descaled = scaler.inverse_transform(dummy)[:, close_idx].reshape(-1, PREDICT_DAYS)

    dummy2 = np.zeros((y_test.shape[0] * PREDICT_DAYS, len(feature_cols)))
    dummy2[:, close_idx] = y_test.flatten()
    y_descaled = scaler.inverse_transform(dummy2)[:, close_idx].reshape(-1, PREDICT_DAYS)

    mae = mean_absolute_error(y_descaled.flatten(), preds_descaled.flatten())
    rmse = np.sqrt(mean_squared_error(y_descaled.flatten(), preds_descaled.flatten()))
    print(f"\nTest MAE:  ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")

    # Save
    model.save(os.path.join(BASE_DIR, "lstm_stock_model.h5"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "stock_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "stock_features.pkl"))
    print("\nModel saved to backend/ml/stock/")


if __name__ == "__main__":
    main()
