"""
FinSight AI — Dataset Generator
Generates all training datasets needed for every ML model.
Run: python data/generate_datasets.py
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────
# 1. FRAUD DETECTION DATASET
# ─────────────────────────────────────────
def generate_fraud_dataset(n=20000):
    print("Generating fraud detection dataset...")
    merchant_categories = ["grocery", "electronics", "restaurant", "travel",
                            "atm", "online_retail", "pharmacy", "entertainment"]
    records = []
    for i in range(n):
        is_fraud = 1 if random.random() < 0.08 else 0
        if is_fraud:
            amount = round(random.uniform(500, 5000), 2)
            hour = random.choice([0, 1, 2, 3, 22, 23])
            geo_distance = random.uniform(300, 5000)
            velocity = random.randint(8, 30)
            device_match = 0
        else:
            amount = round(random.uniform(5, 800), 2)
            hour = random.randint(7, 21)
            geo_distance = random.uniform(0, 50)
            velocity = random.randint(1, 6)
            device_match = 1

        records.append({
            "transaction_id": f"TXN{i:07d}",
            "user_id": f"USR{random.randint(1000, 9999)}",
            "amount": amount,
            "hour_of_day": hour,
            "day_of_week": random.randint(0, 6),
            "merchant_category": random.choice(merchant_categories),
            "geo_distance_km": round(geo_distance, 2),
            "transaction_velocity_7d": velocity,
            "device_fingerprint_match": device_match,
            "time_since_last_txn_mins": round(random.uniform(0.5, 2880), 1),
            "amount_vs_avg_ratio": round(amount / random.uniform(50, 300), 3),
            "is_international": random.choice([0, 0, 0, 1]),
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "fraud_transactions.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ─────────────────────────────────────────
# 2. STOCK PRICE DATASET (OHLCV)
# ─────────────────────────────────────────
def generate_stock_dataset(n_days=1000):
    print("Generating stock price dataset...")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    all_records = []
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    for symbol in symbols:
        price = random.uniform(100, 400)
        for date in dates:
            change = np.random.normal(0.0005, 0.018)
            price = max(10, price * (1 + change))
            daily_range = price * random.uniform(0.01, 0.04)
            open_p = round(price * random.uniform(0.99, 1.01), 2)
            high_p = round(price + daily_range, 2)
            low_p = round(price - daily_range, 2)
            close_p = round(price, 2)
            volume = random.randint(5_000_000, 80_000_000)

            all_records.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume
            })

    df = pd.DataFrame(all_records)
    path = os.path.join(DATA_DIR, "stock_prices.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ─────────────────────────────────────────
# 3. CREDIT RISK DATASET
# ─────────────────────────────────────────
def generate_credit_dataset(n=15000):
    print("Generating credit risk dataset...")
    records = []
    for i in range(n):
        income = random.randint(15000, 250000)
        age = random.randint(21, 70)
        existing_loans = random.randint(0, 6)
        credit_util = round(random.uniform(0, 1), 3)
        payment_history = random.randint(0, 10)   # missed payments in 2 years
        employment_years = random.randint(0, 30)
        loan_amount = random.randint(5000, 200000)
        debt_to_income = round(loan_amount / income, 3)

        # Score logic (approximate)
        base = 750
        base -= payment_history * 25
        base -= int(credit_util * 150)
        base -= existing_loans * 15
        base += min(employment_years * 3, 60)
        base += min(income // 5000, 50)
        base += random.randint(-30, 30)
        credit_score = max(300, min(850, base))

        risk = "Low" if credit_score >= 720 else ("Medium" if credit_score >= 580 else "High")

        records.append({
            "customer_id": f"CUST{i:06d}",
            "age": age,
            "annual_income": income,
            "employment_years": employment_years,
            "existing_loans": existing_loans,
            "credit_utilization": credit_util,
            "missed_payments_2y": payment_history,
            "loan_amount_requested": loan_amount,
            "debt_to_income_ratio": debt_to_income,
            "credit_score": credit_score,
            "risk_category": risk
        })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "credit_risk.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


# ─────────────────────────────────────────
# 4. BUDGET / TRANSACTIONS DATASET
# ─────────────────────────────────────────
def generate_budget_dataset(n=10000):
    print("Generating budget transactions dataset...")
    categories = {
        "Food & Dining": ["McDonald's", "Swiggy", "Zomato", "restaurant", "cafe", "pizza hut", "dominos", "biryani"],
        "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "clothing store", "shoes"],
        "Transport": ["Uber", "Ola", "Metro", "petrol pump", "fuel", "parking", "rapido"],
        "Entertainment": ["Netflix", "Amazon Prime", "BookMyShow", "Spotify", "gaming", "PVR cinema"],
        "Healthcare": ["pharmacy", "Apollo", "Medplus", "doctor", "hospital", "lab test"],
        "Utilities": ["electricity bill", "water bill", "internet", "Airtel", "Jio", "gas"],
        "Education": ["Udemy", "Coursera", "books", "stationery", "tuition"],
        "Groceries": ["DMart", "BigBasket", "grocery", "vegetables", "fruits", "supermarket"]
    }

    records = []
    for i in range(n):
        category = random.choice(list(categories.keys()))
        merchant_raw = random.choice(categories[category])
        noise = random.choice(["", " store", " outlet", " #" + str(random.randint(1, 99)), ""])
        description = merchant_raw + noise

        if category in ["Shopping", "Healthcare"]:
            amount = round(random.uniform(200, 5000), 2)
        elif category == "Entertainment":
            amount = round(random.uniform(100, 1500), 2)
        elif category == "Utilities":
            amount = round(random.uniform(300, 3000), 2)
        else:
            amount = round(random.uniform(30, 800), 2)

        records.append({
            "txn_id": f"B{i:06d}",
            "user_id": f"USR{random.randint(100, 500)}",
            "date": fake.date_between(start_date="-12m", end_date="today").strftime("%Y-%m-%d"),
            "description": description,
            "amount": amount,
            "category": category
        })

    df = pd.DataFrame(records)
    path = os.path.join(DATA_DIR, "budget_transactions.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return df


if __name__ == "__main__":
    generate_fraud_dataset()
    generate_stock_dataset()
    generate_credit_dataset()
    generate_budget_dataset()
    print("\nAll datasets generated successfully!")
