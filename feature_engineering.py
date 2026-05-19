"""
FinSight AI — Fraud Feature Engineering
Shared utilities for feature extraction from raw transactions.
"""
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "merchant_category_enc",
    "geo_distance_km", "transaction_velocity_7d", "device_fingerprint_match",
    "time_since_last_txn_mins", "amount_vs_avg_ratio", "is_international"
]


def extract_features(txn: dict, user_history: list = None) -> dict:
    """
    Converts a raw transaction dict into ML-ready features.
    txn keys: amount, timestamp, merchant_category, geo_lat, geo_lon,
               device_id, user_id, is_international
    user_history: list of past transactions for velocity/avg computation
    """
    import datetime
    ts = pd.Timestamp(txn.get("timestamp", pd.Timestamp.now()))

    # Time features
    hour = ts.hour
    dow = ts.dayofweek

    # Velocity: count txns in last 7 days
    velocity = 1
    last_txn_mins = 60.0
    avg_amount = txn["amount"]

    if user_history:
        seven_days_ago = ts - pd.Timedelta(days=7)
        recent = [h for h in user_history
                  if pd.Timestamp(h["timestamp"]) > seven_days_ago]
        velocity = len(recent) + 1

        if len(user_history) > 0:
            amounts = [h["amount"] for h in user_history]
            avg_amount = np.mean(amounts)
            last_ts = pd.Timestamp(user_history[-1]["timestamp"])
            last_txn_mins = (ts - last_ts).total_seconds() / 60

    # Merchant category encoding (simple hash-based, consistent)
    cat_map = {
        "grocery": 0, "electronics": 1, "restaurant": 2,
        "travel": 3, "atm": 4, "online_retail": 5,
        "pharmacy": 6, "entertainment": 7
    }
    cat_enc = cat_map.get(txn.get("merchant_category", "online_retail"), 5)

    features = {
        "amount": float(txn["amount"]),
        "hour_of_day": hour,
        "day_of_week": dow,
        "merchant_category_enc": cat_enc,
        "geo_distance_km": float(txn.get("geo_distance_km", 0)),
        "transaction_velocity_7d": velocity,
        "device_fingerprint_match": int(txn.get("device_fingerprint_match", 1)),
        "time_since_last_txn_mins": round(last_txn_mins, 1),
        "amount_vs_avg_ratio": round(txn["amount"] / max(avg_amount, 1), 3),
        "is_international": int(txn.get("is_international", 0))
    }
    return features


def features_to_df(features: dict) -> pd.DataFrame:
    return pd.DataFrame([features])[FEATURE_COLS]
