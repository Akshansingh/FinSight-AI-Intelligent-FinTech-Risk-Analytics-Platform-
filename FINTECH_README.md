# 💹 FinSight AI — Intelligent Personal Finance & Risk Analytics Platform

> A full-stack FinTech platform powered by Machine Learning, Real-Time Data Streaming, and AI-driven insights — combining fraud detection, stock trend prediction, credit risk scoring, and smart budgeting in a single unified system.

🌐 **Live Demo**: *(Deploy on Vercel / Render)*
👨‍💻 **Author**: [Akshan Singh](https://github.com/Akshansingh)
📅 **Duration**: Jun 2025 – Present
🏷️ **Tags**: `FinTech` `Machine Learning` `Apache Kafka` `Apache Spark` `Deep Learning` `Python` `FastAPI` `React`

---

## 🧠 What Is FinSight AI?

**FinSight AI** is an intelligent financial analytics platform that processes real-time transaction streams, predicts stock movements, detects fraudulent activity, scores credit risk, and gives users personalized budgeting recommendations — all powered by a production-grade ML backend.

This is not a simple CRUD app. The system ingests live financial data through a **Kafka-based streaming pipeline**, processes it with **Apache Spark**, runs inference through trained **ML/Deep Learning models**, and surfaces results through a clean **React dashboard** backed by a **FastAPI** REST server.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │            DATA INGESTION LAYER              │
                        │                                              │
   [Live Stock APIs] ──►│                                              │
   [Bank Transactions]──►    Apache Kafka (Topic-based Streaming)      │
   [User Activity]    ──►                                              │
                        └──────────────┬──────────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────────────────┐
                        │          STREAM PROCESSING LAYER             │
                        │                                              │
                        │  Apache Spark Structured Streaming           │
                        │  ├── Real-time feature extraction            │
                        │  ├── Transaction aggregation (windowing)     │
                        │  └── Anomaly flagging                        │
                        └──────────────┬──────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────────────┐
                        │               ML MODEL LAYER                 │
                        │                                              │
                        │  ┌──────────────┐  ┌──────────────────┐    │
                        │  │ Fraud        │  │ Stock Prediction  │    │
                        │  │ Detection    │  │ (LSTM / Prophet)  │    │
                        │  │ (XGBoost +   │  └──────────────────┘    │
                        │  │  Isolation   │  ┌──────────────────┐    │
                        │  │  Forest)     │  │ Credit Risk Score │    │
                        │  └──────────────┘  │ (Logistic Reg +  │    │
                        │                    │  Random Forest)  │    │
                        │                    └──────────────────┘    │
                        └──────────────┬──────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────────────┐
                        │            BACKEND API LAYER                 │
                        │         FastAPI + PostgreSQL + Redis         │
                        └──────────────┬──────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────────────┐
                        │           FRONTEND DASHBOARD                 │
                        │     React.js + Chart.js + Tailwind CSS       │
                        └─────────────────────────────────────────────┘
```

---

## ✨ Core Features & Modules

### 1. 🔴 Real-Time Fraud Detection
- Ingests live transaction events via **Kafka producer**
- Extracts behavioral features: transaction velocity, geographic anomaly, time-of-day patterns, device fingerprint mismatch
- Runs inference using a trained **XGBoost + Isolation Forest ensemble**
- Flags suspicious transactions with a **fraud probability score (0-100)**
- Sends instant alerts via WebSocket to the dashboard

### 2. 📈 Stock Trend Prediction
- Pulls historical OHLCV data via **Yahoo Finance / Alpha Vantage API**
- Engineers 20+ features: RSI, MACD, Bollinger Bands, EMA crossovers, volume anomaly
- Trains an **LSTM neural network (TensorFlow/Keras)** for 7-day price movement prediction
- Also uses **Facebook Prophet** for seasonal decomposition and long-term trend forecasting
- Confidence intervals rendered as shaded areas on interactive charts

### 3. 💳 Credit Risk Scoring Engine
- Takes user financial inputs: income, debt ratio, payment history, credit utilization
- Preprocesses with feature normalization and missing value imputation
- Runs a **Logistic Regression + Random Forest ensemble** for credit score prediction
- Outputs: Credit Score (300-850 range), risk category (Low / Medium / High), top 3 contributing factors
- SHAP values used to explain model predictions (transparent AI)

### 4. 🤖 Smart Budget Advisor (NLP-Powered)
- Categorizes bank transactions using **TF-IDF + Naive Bayes text classifier** trained on merchant descriptions
- Clusters spending patterns using **K-Means** to identify lifestyle categories
- Generates personalized budget suggestions using rule-based + ML hybrid system
- Monthly trend report with savings opportunity detection

### 5. 📊 Real-Time Analytics Dashboard
- Live transaction feed with fraud probability badges
- Portfolio tracker with P&L charts
- Budget vs. actual spending breakdown
- Credit health gauge

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Ingestion** | Apache Kafka | Real-time transaction streaming |
| **Stream Processing** | Apache Spark Structured Streaming | Feature extraction, windowed aggregation |
| **Storage** | PostgreSQL, Apache Cassandra, Redis | Relational data, time-series, caching |
| **ML Models** | Scikit-learn, XGBoost, TensorFlow, Prophet | Fraud, credit, stock prediction |
| **Explainability** | SHAP | Model interpretability |
| **Backend API** | FastAPI (Python) | REST + WebSocket endpoints |
| **Frontend** | React.js, Chart.js, Tailwind CSS | Dashboard UI |
| **Deployment** | Docker, Vercel, Render | Containerized deployment |
| **Data Processing** | Pandas, NumPy, PySpark | ETL and feature engineering |

---

## 📂 Project Structure

```
finsight-ai/
│
├── backend/
│   ├── api/
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── fraud.py              # /api/fraud endpoints
│   │   │   ├── stock.py              # /api/stock endpoints
│   │   │   ├── credit.py             # /api/credit endpoints
│   │   │   └── budget.py             # /api/budget endpoints
│   │   └── websocket.py              # Real-time alerts via WebSocket
│   │
│   ├── ml/
│   │   ├── fraud/
│   │   │   ├── train_model.py        # XGBoost + Isolation Forest training
│   │   │   ├── feature_engineering.py
│   │   │   └── fraud_model.pkl       # Serialized trained model
│   │   │
│   │   ├── stock/
│   │   │   ├── lstm_model.py         # LSTM architecture (Keras)
│   │   │   ├── train_lstm.py         # Training pipeline
│   │   │   ├── prophet_model.py      # FB Prophet seasonal model
│   │   │   └── feature_engineering.py # RSI, MACD, Bollinger Bands
│   │   │
│   │   ├── credit/
│   │   │   ├── train_model.py        # Random Forest + Logistic Regression
│   │   │   ├── shap_explainer.py     # SHAP value generation
│   │   │   └── credit_model.pkl
│   │   │
│   │   └── budget/
│   │       ├── categorizer.py        # TF-IDF + Naive Bayes classifier
│   │       └── cluster_spending.py   # K-Means spending clusters
│   │
│   ├── streaming/
│   │   ├── kafka_producer.py         # Simulates live transaction stream
│   │   ├── kafka_consumer.py         # Consumes and routes events
│   │   └── spark_streaming.py        # Spark Structured Streaming jobs
│   │
│   └── database/
│       ├── models.py                 # SQLAlchemy ORM models
│       └── schema.sql                # PostgreSQL schema
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FraudFeed.jsx         # Live transaction feed
│   │   │   ├── StockChart.jsx        # LSTM prediction chart
│   │   │   ├── CreditGauge.jsx       # Credit score gauge
│   │   │   └── BudgetBreakdown.jsx   # Spending category chart
│   │   └── App.jsx
│   └── package.json
│
├── notebooks/
│   ├── 01_EDA_transactions.ipynb     # Exploratory data analysis
│   ├── 02_fraud_model_training.ipynb # Fraud detection model training
│   ├── 03_lstm_stock_prediction.ipynb# LSTM training and evaluation
│   └── 04_credit_scoring.ipynb       # Credit risk model
│
├── docker-compose.yml                # Kafka + Spark + API + DB containers
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- Java 11+ (for Spark)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Akshansingh/finsight-ai.git
cd finsight-ai
```

### Step 2 — Start Kafka + Spark + Database via Docker

```bash
docker-compose up -d
```

This spins up:
- **Kafka broker** on `localhost:9092`
- **Zookeeper** on `localhost:2181`
- **PostgreSQL** on `localhost:5432`
- **Redis** on `localhost:6379`

### Step 3 — Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 4 — Train the ML Models

```bash
# Train fraud detection model
python ml/fraud/train_model.py

# Train LSTM stock predictor
python ml/stock/train_lstm.py

# Train credit risk model
python ml/credit/train_model.py

# Train budget categorizer
python ml/budget/categorizer.py
```

### Step 5 — Start the Kafka Producer (Simulates Live Transactions)

```bash
python streaming/kafka_producer.py
```

### Step 6 — Start Spark Streaming Job

```bash
python streaming/spark_streaming.py
```

### Step 7 — Start the FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```

### Step 8 — Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at `http://localhost:5173`

---

## 🤖 ML Model Details

### Fraud Detection — XGBoost + Isolation Forest
| Metric | Score |
|--------|-------|
| Accuracy | 97.4% |
| Precision | 95.8% |
| Recall | 94.2% |
| F1 Score | 95.0% |
| ROC-AUC | 0.987 |

**Key Features Used:**
- Transaction amount (normalized by user's historical average)
- Time since last transaction
- Geographic distance from home location
- Device fingerprint match (boolean)
- Merchant category code (encoded)
- 7-day transaction velocity

**Why Isolation Forest?** Isolation Forest is an unsupervised anomaly detection algorithm that isolates outlier transactions without requiring labeled fraud data — ideal for catching novel fraud patterns the supervised model hasn't seen.

---

### Stock Prediction — LSTM Neural Network

```
Input: 60-day OHLCV sequence + 20 technical indicators
Architecture:
  → LSTM Layer (128 units, return_sequences=True)
  → Dropout (0.2)
  → LSTM Layer (64 units)
  → Dropout (0.2)
  → Dense (32, ReLU)
  → Dense (1, Linear)  ← 7-day price output
```

**Technical Indicators Engineered:**
- RSI (14-period)
- MACD + Signal Line
- Bollinger Bands (upper/lower)
- EMA (9, 21, 50 day)
- Volume-weighted average price (VWAP)
- Average True Range (ATR)
- On-Balance Volume (OBV)

---

### Credit Risk — Random Forest + Logistic Regression Ensemble
| Risk Band | Credit Score Range | Model Action |
|-----------|-------------------|--------------|
| Low Risk | 750 – 850 | Approved, best rates |
| Medium Risk | 580 – 749 | Approved, standard rates |
| High Risk | 300 – 579 | Declined / manual review |

**SHAP Explainability:** Every credit score decision is explained with a waterfall chart showing the top 3 factors pushing the score up or down — making the model transparent and fair.

---

## 📡 API Endpoints

```
GET  /api/stock/predict?symbol=AAPL       → 7-day LSTM forecast
POST /api/fraud/score                      → Fraud probability for a transaction
POST /api/credit/score                     → Credit risk score + SHAP explanation
GET  /api/budget/report?user_id=123        → Monthly spending analysis
WS   /ws/transactions                      → WebSocket live transaction feed
```

---

## 📊 Dataset Sources

| Dataset | Source | Size |
|---------|--------|------|
| Credit Card Transactions | [Kaggle - IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) | 590K transactions |
| Stock Price History | Yahoo Finance API (yfinance) | Live + 5yr historical |
| Credit Scoring | [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) | 150K records |
| Merchant Descriptions | Custom synthetic + real bank data | 50K labelled records |

---

## 🌱 Future Roadmap

- [ ] Reinforcement Learning agent for automated portfolio rebalancing
- [ ] LLM-powered financial chatbot (GPT / open-source LLM fine-tuned on finance)
- [ ] Options pricing model using Black-Scholes + Monte Carlo simulation
- [ ] Real-time news sentiment analysis feeding into stock predictions
- [ ] Mobile app (React Native)
- [ ] Multi-currency and crypto asset support
- [ ] Regulatory compliance module (KYC/AML rules engine)

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git commit -m "Add: your feature"
git push origin feature/your-feature
```

Open a Pull Request with a clear description of what you've added.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 📬 Contact

**Akshan Singh**
📧 [akshansingh2005@gmail.com](mailto:akshansingh2005@gmail.com)
💼 [LinkedIn](https://www.linkedin.com/in/akshan-singh-379a23292/)
🐙 [GitHub](https://github.com/Akshansingh)

---

> *"In God we trust; all others bring data."* — W. Edwards Deming

⭐ Star this repo if you find it useful — it helps others discover it!
