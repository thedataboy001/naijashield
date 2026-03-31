# 🇳🇬 NaijaShield: Nigerian Financial Fraud Detection Platform

> A production-grade, end-to-end ML pipeline for real-time fraud detection
> in the Nigerian fintech ecosystem — from raw transaction streams to a
> deployed, monitored inference API.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Monitoring & Observability](#monitoring--observability)
- [Nigerian Fintech Context](#nigerian-fintech-context)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Project Overview  

[Table of Contents](#table-of-contents)

NaijaShield is a full end-to-end machine learning platform designed to detect
financial fraud in Nigerian payment systems in real time (<100ms latency).

The platform addresses three core challenges unique to the Nigerian fintech
landscape:

- **Payment channel diversity**: USSD, Mobile Apps, Card, and Bank Transfer
  each carry distinct risk profiles
- **Behavioral heterogeneity**: Salary earners, students, and traders exhibit
  fundamentally different transaction patterns
- **Emerging fraud vectors**: SIM swap fraud, BVN identity fraud, and
  geospatial velocity attacks are increasingly common in Nigerian markets

This is not a notebook project. It is a system, with streaming ingestion,
a feature store, automated retraining, drift monitoring, and a low-latency
serving API.

---

## Dataset

[Table of Contents](#table-of-contents)

**Source**: [Nigerian Financial Transactions and Fraud Detection Dataset](https://huggingface.co/datasets/kidnextdoor57/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset)  
**Host**: Hugging Face Datasets

### Statistics

| Property | Value |
|----------|-------|
| Total Transactions | 5,000,000 |
| Total Features | 45 |
| Fraud Rate | ~15% |
| Unique Senders | ~500,000 |
| Time Span | 12-month simulation |
| Currency | Nigerian Naira (NGN) |
| Nigerian Cities | 20 across 6 geo-regions |

### Feature Categories

| Category | Count | Examples |
|----------|-------|---------|
| Core Transaction | 15 | amount_ngn, payment_channel, merchant_category |
| User Behavior | 5 | user_avg_txn_amt, user_txn_frequency_24h |
| Device & IP Intelligence | 5 | is_device_shared, ip_seen_count |
| Transaction Window | 4 | txn_count_last_1h, total_amount_last_1h |
| Risk Scoring | 4 | merchant_fraud_rate, channel_risk_score |
| Temporal | 4 | txn_hour, is_night_txn, is_salary_week |
| Technical/Derived | 8 | new_device_for_sender, time_since_last |

### Fraud Types Covered

- Account Takeover
- Identity Fraud
- Impossible Travel / Geospatial Velocity
- SIM Swap Fraud
- Card-Not-Present Fraud
- Deposit Fraud
- Money Laundering

### ⚠️ Important Notes on This Dataset

1. **Synthetic data**: Patterns are engineered. Real-world generalization
   should be validated before production deployment on live Nigerian data.

2. **15% fraud rate**: Significantly higher than real-world rates (0.1–2%).
   Models will need threshold calibration before live deployment.

3. **Temporal integrity**: Train/test splits MUST respect chronological order.
   Random splits will cause data leakage via window features.

---

## System Architecture

[Table of Contents](#table-of-contents)

    DATA LAYER
    Transactions → Kafka → Stream Processor → Feature Store
                                      ├── Redis (Online)
                                      └── S3 (Offline)      
              ↓
        TRAINING LAYER
        - XGBoost / LightGBM
        - Autoencoder
        - SMOTE / Focal Loss
        - MLflow Registry
        - Airflow DAGs

                ↓
        SERVING LAYER
        - FastAPI (<100ms SLA)
        - Feature Fetcher
        - Decision Engine

                ↓
        MONITORING LAYER
        - Evidently AI (Drift Detection)
        - Grafana Dashboards
        - A/B Testing Framework

## Project Structure

[Table of Contents](#table-of-contents)


    naijashield/
    │
    ├── data/
    │   ├── raw/                  # Raw datasets
    │   ├── processed/            # Cleaned and split datasets
    │   └── features/             # Offline feature store exports
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │   └── 04_model_evaluation.ipynb
    │
    ├── src/
    │   ├── ingestion/
    │   │   ├── kafka_producer.py
    │   │   └── kafka_consumer.py
    │   │
    │   ├── features/
    │   │   ├── feature_definitions.py
    │   │   ├── feature_engineering.py
    │   │   └── feature_store.py
    │   │
    │   ├── training/
    │   │   ├── train_xgboost.py
    │   │   ├── train_autoencoder.py
    │   │   ├── imbalance.py
    │   │   └── evaluate.py
    │   │
    │   ├── serving/
    │   │   ├── main.py
    │   │   ├── model_loader.py
    │   │   ├── predictor.py
    │   │   └── schemas.py
    │   │
    │   └── monitoring/
    │       ├── drift_detector.py
    │       ├── performance_tracker.py
    │       └── ab_testing.py
    │
    ├── dags/
    │   ├── retraining_dag.py
    │   └── data_validation_dag.py
    │
    ├── infrastructure/
    │   ├── docker-compose.yml
    │   ├── Dockerfile.api
    │   ├── Dockerfile.training
    │   └── prometheus/
    │       └── prometheus.yml
    │
    ├── monitoring/
    │   └── grafana/
    │       └── dashboards/
    │           ├── fraud_overview.json
    │           └── model_performance.json
    │
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │   └── performance/
    │
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       └── cd.yml
    │
    ├── feature_store/
    │   └── feature_repo/
    │
    ├── mlflow/
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    ├── Makefile
    └── README.md


## Key Features

[Table of Contents](#table-of-contents)

### 🇳🇬 Nigerian-Specific Fraud Intelligence
- BVN (Bank Verification Number) linkage as a fraud signal
- USSD channel risk profiling (highest risk: 0.8)
- Geospatial velocity detection across 20 Nigerian cities
- SIM swap detection via device change patterns
- Salary week transaction pattern analysis (last 5 days of month)

### ⚡ Real-Time Performance
- Sub-100ms end-to-end inference latency
- Redis-backed online feature store for <10ms feature lookups
- Asynchronous prediction logging (non-blocking)

### 🔄 Automated Retraining
- Airflow DAGs trigger retraining on drift detection or schedule
- MLflow tracks all experiments, parameters, and model versions
- Staged promotion: Staging → A/B Test → Production

### 📊 Full Observability
- Grafana dashboards for real-time fraud KPIs
- Evidently AI for data and concept drift monitoring
- A/B testing framework for safe model rollouts

---

## Tech Stack

[Table of Contents](#table-of-contents)

| Layer | Technology |
|-------|-----------|
| Stream Ingestion | Apache Kafka |
| Feature Store | Feast + Redis (online) + S3 (offline) |
| Model Training | XGBoost, LightGBM, PyTorch (Autoencoder) |
| Imbalance Handling | imbalanced-learn (SMOTE), Focal Loss |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow |
| Serving API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Drift Monitoring | Evidently AI |
| Dashboards | Grafana + Prometheus |
| CI/CD | GitHub Actions |
| Language | Python 3.10+ |

---

## Getting Started

[Table of Contents](#table-of-contents)

### Prerequisites

```bash
# Required
Docker >= 24.0
Docker Compose >= 2.0
Python >= 3.10
Git 
```

### Clone the repository
```bash
git clone https://github.com/thedataboy001/naijashield
cd naijashield
```

### Start the full stack
```bash
docker-compose up -d
```

### Verify services

| Service | URL |
|---------|-----------|
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI    | http://localhost:5000      |
| Airflow UI   | http://localhost:8080      |
| Grafana      | http://localhost:3000      |
| Kafka UI     | http://localhost:8090      |

	
### Run the Training Pipeline
	
```bash	
make train
```

### Test the Inference API

```bash	

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender_account": "0123456789",
    "amount_ngn": 250000,
    "payment_channel": "USSD",
    "merchant_category": "Bet9ja Stake",
    "location": "Lagos",
    "timestamp": "2024-01-15T23:45:00"
  }'
  
  ```

## Pipeline Components

[Table of Contents](#table-of-contents)

### Data Ingestion

Transactions are streamed through Apache Kafka, simulating real-time payment events from Nigerian payment channels (USSD, Mobile, Card, Bank Transfer).

### Feature Store

Feast manages feature definitions with strict separation between online
(Redis) and offline (S3/Parquet) storage. All features use expanding windows with ```.shift(1)``` to prevent data leakage.

### Model Training

Two complementary models are trained:

- **XGBoost**: Supervised classifier on labeled fraud transactions
- **Autoencoder**: Unsupervised anomaly detector trained on legitimate transactions only — catches novel fraud patterns unseen during training

### Inference API

FastAPI serves predictions with a strict <100ms SLA:

- Parse Request → Fetch Features (Redis) → Inference → Log → Respond

### Automated Retraining

Airflow DAGs monitor model performance and trigger retraining when:

- F1 score drops below defined threshold
- Data drift score (PSI) exceeds 0.2
- Scheduled weekly retraining window


## Model Performance

[Table of Contents](#table-of-contents)

Metrics will be populated after initial training run.

| Metric                       | XGBoost     | Autoencoder |  Ensemble   |
|------------------------------|-------------|-------------|-------------|
| AUC-PR                       | TBD         | TBD         | TBD         |
| F1 Score                     | TBD         | TBD         | TBD         |
| Precision@Recall90           | TBD         | TBD         | TBD         |
| Inference Latency (p99)      | TBD         | TBD         | TBD         |

Primary metric: AUC-PR (preferred over AUC-ROC for imbalanced fraud data)


## Monitoring & Observability

[Table of Contents](#table-of-contents)

### Grafana Dashboards

- **Fraud Overview:** Real-time fraud rate, volume by channel, geographic heatmap
- **Model Performance:** Precision, recall, F1 over time, score distributions
- **System Health:** API latency percentiles, throughput, error rates

### Drift Detection (Evidently AI)

- **Data Drift:** PSI on all 45 input features, weekly reports
- **Concept Drift:** Performance metric degradation tracking
- **Alerts:** Slack/email notifications on threshold breach

### A/B Testing
Champion vs Challenger routing with configurable traffic split.
Automatic promotion based on statistical significance testing.

## Nigerian Fintech Context

[Table of Contents](#table-of-contents)

### Why This Problem Matters

Nigeria is Africa's largest economy with one of the fastest-growing fintech
ecosystems. The CBN (Central Bank of Nigeria) reported over ₦9.5 billion lost
to fraud in 2023. Key fraud vectors in the Nigerian context include:

- **SIM Swap Fraud:** Fraudsters port victim's number to gain OTP access
- **BVN Exploitation:** Stolen BVN used to open fake accounts
- **USSD Interception:** Highest-risk channel due to unencrypted sessions
- **Salary Period Attacks:** Fraud spikes in the last week of the month
- **Betting Platform Abuse:** Bet9ja, NairaBet used for money laundering

### Regulatory Considerations

- CBN fraud reporting requirements
- NDPR (Nigeria Data Protection Regulation) compliance for user data
- BVN verification as mandated KYC signal

## Roadmap

[Table of Contents](#table-of-contents)

 - Dataset acquisition and documentation 
 - Exploratory data analysis (EDA)
 - Feature engineering pipeline
 - XGBoost baseline model
 - Autoencoder anomaly detector
 - MLflow experiment tracking
 - FastAPI inference server
 - Docker Compose full stack
 - Feast feature store integration
 - Kafka streaming pipeline
 - Airflow retraining DAGs
 - Evidently drift monitoring
 - Grafana dashboards
 - GitHub Actions CI/CD
 - A/B testing framework
 - Performance testing suite
 - Full system integration test


## Contributing

[Table of Contents](#table-of-contents)

This project is currently in active development.
Contribution guidelines will be added as the project matures.

## License

[Table of Contents](#table-of-contents)

MIT License — see LICENSE for details.
