"""
FinSight AI — Budget Categorizer & Spending Cluster Model
TF-IDF + Naive Bayes for merchant classification.
K-Means for spending pattern clustering.
Run: python backend/ml/budget/categorizer.py
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../data/budget_transactions.csv")

CATEGORIES = [
    "Food & Dining", "Shopping", "Transport", "Entertainment",
    "Healthcare", "Utilities", "Education", "Groceries"
]


def train_categorizer(df: pd.DataFrame):
    print("Training TF-IDF + Naive Bayes categorizer...")
    X = df["description"].str.lower().str.strip()
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            strip_accents="unicode",
            analyzer="word"
        )),
        ("clf", MultinomialNB(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return pipeline


def train_spending_clusters(df: pd.DataFrame, n_clusters: int = 5):
    """
    Clusters users by their spending patterns across categories.
    """
    print("\nTraining K-Means spending cluster model...")
    user_spending = df.pivot_table(
        index="user_id", columns="category", values="amount",
        aggfunc="sum", fill_value=0
    ).reset_index()

    feature_cols = [c for c in user_spending.columns if c != "user_id"]
    X = user_spending[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method to confirm k
    inertias = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans.fit(X_scaled)

    user_spending["cluster"] = kmeans.labels_

    # Describe clusters
    cluster_profiles = user_spending.groupby("cluster")[feature_cols].mean().round(1)
    print("Cluster Spending Profiles (avg monthly ₹):")
    print(cluster_profiles.T.to_string())

    return kmeans, scaler, feature_cols


def categorize_transaction(description: str, pipeline) -> dict:
    """Predict category for a single transaction description."""
    desc_clean = description.lower().strip()
    category = pipeline.predict([desc_clean])[0]
    proba = pipeline.predict_proba([desc_clean])[0]
    classes = pipeline.classes_
    confidence = float(proba.max())
    return {
        "category": category,
        "confidence": round(confidence, 3),
        "top_3": [
            {"category": classes[i], "probability": round(float(proba[i]), 3)}
            for i in np.argsort(proba)[::-1][:3]
        ]
    }


def generate_budget_report(user_txns: pd.DataFrame) -> dict:
    """
    Generates a monthly budget summary for a user.
    user_txns: DataFrame with columns: date, description, amount, category
    """
    user_txns["date"] = pd.to_datetime(user_txns["date"])
    user_txns["month"] = user_txns["date"].dt.to_period("M")

    monthly = user_txns.groupby(["month", "category"])["amount"].sum().reset_index()
    total_spent = user_txns["amount"].sum()
    by_category = user_txns.groupby("category")["amount"].sum().sort_values(ascending=False)

    report = {
        "total_spent": round(total_spent, 2),
        "by_category": by_category.round(2).to_dict(),
        "top_category": by_category.index[0],
        "savings_opportunities": []
    }

    # Simple rules-based savings tips
    if by_category.get("Entertainment", 0) > total_spent * 0.15:
        report["savings_opportunities"].append(
            "Entertainment spending is above 15% of total — consider reviewing subscriptions.")
    if by_category.get("Food & Dining", 0) > total_spent * 0.30:
        report["savings_opportunities"].append(
            "Food & Dining is your largest expense category — home cooking could save 20-30%.")
    if by_category.get("Shopping", 0) > 5000:
        report["savings_opportunities"].append(
            "Shopping spend is high — try a 48-hour rule before non-essential purchases.")

    return report


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions | {df['category'].nunique()} categories")

    pipeline = train_categorizer(df)
    kmeans, scaler, feature_cols = train_spending_clusters(df)

    # Quick test
    test_descs = ["McDonald's outlet", "Amazon purchase", "Uber ride", "Netflix subscription"]
    print("\nSample predictions:")
    for desc in test_descs:
        result = categorize_transaction(desc, pipeline)
        print(f"  '{desc}' → {result['category']} ({result['confidence']:.0%} confidence)")

    # Save
    joblib.dump(pipeline, os.path.join(BASE_DIR, "budget_categorizer.pkl"))
    joblib.dump(kmeans, os.path.join(BASE_DIR, "spending_kmeans.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "spending_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "spending_features.pkl"))
    print("\nModels saved to backend/ml/budget/")


if __name__ == "__main__":
    main()
