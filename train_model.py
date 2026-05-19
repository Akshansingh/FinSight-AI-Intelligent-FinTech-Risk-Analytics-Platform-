"""
FinSight AI — Fraud Detection Model
Trains XGBoost + Isolation Forest ensemble.
Run: python backend/ml/fraud/train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, accuracy_score)
import xgboost as xgb
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/fraud_transactions.csv")
MODEL_DIR = BASE_DIR


def load_and_preprocess(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} transactions | Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

    le = LabelEncoder()
    df["merchant_category_enc"] = le.fit_transform(df["merchant_category"])

    features = [
        "amount", "hour_of_day", "day_of_week", "merchant_category_enc",
        "geo_distance_km", "transaction_velocity_7d", "device_fingerprint_match",
        "time_since_last_txn_mins", "amount_vs_avg_ratio", "is_international"
    ]
    X = df[features]
    y = df["is_fraud"]
    return X, y, le


def train_isolation_forest(X_train):
    print("\nTraining Isolation Forest (anomaly detection)...")
    iso = IsolationForest(n_estimators=200, contamination=0.08,
                          max_samples="auto", random_state=42, n_jobs=-1)
    iso.fit(X_train)
    return iso


def train_xgboost(X_train, y_train, X_val, y_val):
    print("Training XGBoost classifier...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=50)
    return model


def evaluate(xgb_model, iso_model, X_test, y_test):
    print("\n── Model Evaluation ──")
    # XGBoost predictions
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba > 0.5).astype(int)

    # Isolation Forest scores (normalized to 0-1)
    iso_scores = iso_model.decision_function(X_test)
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    iso_scores_inverted = 1 - iso_scores_norm  # higher = more anomalous

    # Ensemble: 70% XGBoost + 30% Isolation Forest
    ensemble_proba = 0.7 * xgb_proba + 0.3 * iso_scores_inverted
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    print(f"\nXGBoost ROC-AUC:  {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"Ensemble ROC-AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")
    print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, ensemble_pred,
                                target_names=["Legitimate", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))


def main():
    X, y, le = load_and_preprocess(DATA_PATH)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    iso_model = train_isolation_forest(X_train[y_train == 0])
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    evaluate(xgb_model, iso_model, X_test, y_test)

    # Save models
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_fraud_model.pkl"))
    joblib.dump(iso_model, os.path.join(MODEL_DIR, "iso_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "fraud_scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "merchant_encoder.pkl"))
    print("\nModels saved to backend/ml/fraud/")


if __name__ == "__main__":
    main()
