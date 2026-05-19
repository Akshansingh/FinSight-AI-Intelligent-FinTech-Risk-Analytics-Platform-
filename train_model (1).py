"""
FinSight AI — Credit Risk Scoring Model
Trains Random Forest + Logistic Regression ensemble with SHAP explainability.
Run: python backend/ml/credit/train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                              roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/credit_risk.csv")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")


FEATURE_COLS = [
    "age", "annual_income", "employment_years", "existing_loans",
    "credit_utilization", "missed_payments_2y",
    "loan_amount_requested", "debt_to_income_ratio"
]
TARGET = "risk_category"


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")
    print(df[TARGET].value_counts())
    le = LabelEncoder()
    df["risk_encoded"] = le.fit_transform(df[TARGET])
    return df, le


def train_models(X_train, y_train):
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)

    print("Training Logistic Regression...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   multi_class="ovr", random_state=42))
    ])
    lr.fit(X_train, y_train)

    return rf, gb, lr


def ensemble_predict(models, X, weights=(0.5, 0.3, 0.2)):
    rf, gb, lr = models
    p_rf = rf.predict_proba(X)
    p_gb = gb.predict_proba(X)
    p_lr = lr.predict_proba(X)
    ensemble_prob = weights[0]*p_rf + weights[1]*p_gb + weights[2]*p_lr
    return ensemble_prob, np.argmax(ensemble_prob, axis=1)


def evaluate(models, X_test, y_test, le):
    print("\n── Evaluation ──")
    proba, preds = ensemble_predict(models, X_test)
    acc = accuracy_score(y_test, preds)
    # ROC-AUC for multiclass
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
    print(f"Ensemble Accuracy:  {acc:.4f}")
    print(f"Ensemble ROC-AUC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


def generate_shap_explainer(rf_model, X_train, feature_names):
    if not SHAP_AVAILABLE:
        print("SHAP not available — skipping explainer generation")
        return None
    print("\nGenerating SHAP explainer...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train[:500])  # sample for speed
    print("  SHAP explainer ready")
    return explainer


def credit_score_from_features(features: dict, scaler, rf_model, gb_model, le) -> dict:
    """
    Given raw user financial features, returns a credit score and risk tier.
    """
    X = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLS}])
    proba_rf = rf_model.predict_proba(X)[0]
    risk_idx = np.argmax(proba_rf)
    risk_label = le.inverse_transform([risk_idx])[0]

    # Map risk → score band
    score_bands = {"Low": (720, 850), "Medium": (580, 719), "High": (300, 579)}
    low, high = score_bands.get(risk_label, (300, 850))
    confidence = proba_rf[risk_idx]
    score = int(low + (high - low) * confidence)

    return {
        "credit_score": score,
        "risk_category": risk_label,
        "probability": {le.classes_[i]: round(float(p), 3)
                        for i, p in enumerate(proba_rf)},
        "approval_status": "Approved" if risk_label in ["Low", "Medium"] else "Manual Review"
    }


def main():
    df, le = load_data(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["risk_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = train_models(X_train, y_train)
    evaluate(models, X_test, y_test, le)

    rf, gb, lr = models

    # Feature importance
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop feature importances (Random Forest):")
    print(fi.to_string(index=False))

    explainer = generate_shap_explainer(rf, X_train, FEATURE_COLS)

    # Save
    joblib.dump(rf, os.path.join(BASE_DIR, "credit_rf_model.pkl"))
    joblib.dump(gb, os.path.join(BASE_DIR, "credit_gb_model.pkl"))
    joblib.dump(lr, os.path.join(BASE_DIR, "credit_lr_model.pkl"))
    joblib.dump(le, os.path.join(BASE_DIR, "credit_label_encoder.pkl"))
    if explainer:
        joblib.dump(explainer, os.path.join(BASE_DIR, "shap_explainer.pkl"))
    print("\nModels saved to backend/ml/credit/")


if __name__ == "__main__":
    main()
