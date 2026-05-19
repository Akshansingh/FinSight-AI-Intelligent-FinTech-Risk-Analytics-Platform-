"""
FinSight AI — Kafka Producer
Simulates a live stream of financial transactions into Kafka topics.
Run: python backend/streaming/kafka_producer.py
"""
import json
import time
import random
import uuid
from datetime import datetime

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed. Run: pip install kafka-python")
    print("Running in SIMULATION MODE (printing events to console)\n")

KAFKA_BROKER = "localhost:9092"
TOPICS = {
    "transactions": "finsight-transactions",
    "stock_tick":   "finsight-stock-ticks",
    "user_events":  "finsight-user-events",
}

MERCHANT_CATEGORIES = ["grocery", "electronics", "restaurant", "travel",
                        "atm", "online_retail", "pharmacy", "entertainment"]

STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]

# Seed prices for simulation
_prices = {s: round(random.uniform(100, 400), 2) for s in STOCK_SYMBOLS}


def make_transaction():
    is_fraud_sim = random.random() < 0.05  # 5% fraud rate
    return {
        "event_type": "transaction",
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"USR{random.randint(1000, 9999)}",
        "amount": round(random.uniform(500, 4000) if is_fraud_sim else random.uniform(10, 800), 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "geo_distance_km": round(random.uniform(500, 3000) if is_fraud_sim else random.uniform(0, 30), 1),
        "device_fingerprint_match": 0 if is_fraud_sim else 1,
        "is_international": 1 if is_fraud_sim and random.random() > 0.5 else 0,
        "transaction_velocity_7d": random.randint(10, 25) if is_fraud_sim else random.randint(1, 5),
        "timestamp": datetime.utcnow().isoformat(),
        "is_fraud_label": int(is_fraud_sim)   # for evaluation only
    }


def make_stock_tick(symbol: str):
    global _prices
    change_pct = random.gauss(0, 0.003)
    _prices[symbol] = round(max(10, _prices[symbol] * (1 + change_pct)), 2)
    return {
        "event_type": "stock_tick",
        "symbol": symbol,
        "price": _prices[symbol],
        "volume": random.randint(10000, 500000),
        "bid": round(_prices[symbol] - random.uniform(0.01, 0.10), 2),
        "ask": round(_prices[symbol] + random.uniform(0.01, 0.10), 2),
        "timestamp": datetime.utcnow().isoformat()
    }


def make_user_event():
    actions = ["login", "view_portfolio", "view_credit_score",
               "run_budget_report", "request_loan", "logout"]
    return {
        "event_type": "user_event",
        "user_id": f"USR{random.randint(1000, 9999)}",
        "action": random.choice(actions),
        "session_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat()
    }


def send_event(producer, topic: str, event: dict):
    key = event.get("user_id", event.get("symbol", "global")).encode()
    value = json.dumps(event).encode()
    if KAFKA_AVAILABLE and producer:
        producer.send(topic, key=key, value=value)
    else:
        print(f"[{topic}] {json.dumps(event, indent=None)}")


def main():
    producer = None
    if KAFKA_AVAILABLE:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                acks="all",
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            print(f"Connected to Kafka broker at {KAFKA_BROKER}")
        except Exception as e:
            print(f"Kafka connection failed: {e}\nRunning in simulation mode.")

    print("Starting event stream... (Ctrl+C to stop)\n")
    tick = 0
    try:
        while True:
            # Send a transaction every ~1.5s
            txn = make_transaction()
            send_event(producer, TOPICS["transactions"], txn)

            # Stock tick for a random symbol every 2 events
            if tick % 2 == 0:
                symbol = random.choice(STOCK_SYMBOLS)
                tick_event = make_stock_tick(symbol)
                send_event(producer, TOPICS["stock_tick"], tick_event)

            # User event occasionally
            if tick % 7 == 0:
                user_event = make_user_event()
                send_event(producer, TOPICS["user_events"], user_event)

            if producer:
                producer.flush()

            tick += 1
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\nProducer stopped.")
    finally:
        if producer:
            producer.close()


if __name__ == "__main__":
    main()
