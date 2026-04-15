# NaijaShield — Full Stack Fraud Detection Platform

A comprehensive machine learning platform for detecting fraudulent transactions in real-time, built with FastAPI, MLflow, Prometheus, and Grafana.

## 📋 Project Overview

NaijaShield is an end-to-end fraud detection system designed to:
- **Detect fraudulent transactions** using trained ML models
- **Track experiments** with MLflow for reproducibility
- **Monitor performance** with Prometheus and Grafana
- **Serve predictions** via a production-ready FastAPI API
- **Visualize insights** through an interactive web dashboard

## 🏗️ Architecture

### Services

| Service | Port | Purpose |
|---------|------|---------|
| **naijashield-api** | 8000 | FastAPI inference server |
| **mlflow-server** | 5001 | Experiment tracking & model registry |
| **prometheus** | 9090 | Metrics collection & storage |
| **grafana** | 3000 | Dashboards & visualizations |
| **frontend** | 8080 | Web interface for testing predictions |

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/thedataboy001/naijashield
   cd naijashield
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your settings (Grafana credentials, etc.)

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**
   ```bash
   docker-compose ps
   ```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5001
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Frontend**: http://localhost:8080

## 📁 Project Structure

```
fraud-detection-project/
├── src/
│   ├── models/          # Model training & evaluation
│   ├── features/        # Feature engineering
│   ├── serving/         # FastAPI application
│   └── utils/           # Helper functions
├── notebooks/           # Jupyter notebooks for EDA & experiments
├── models/              # Trained model files (.joblib, .pkl)
├── data/
│   ├── raw/             # Original datasets
│   ├── processed/       # Cleaned & processed data
│   └── datasets/        # Additional datasets
├── infrastructure/
│   ├── prometheus/      # Prometheus configuration
│   ├── grafana/         # Grafana dashboards
│   └── nginx/           # Nginx reverse proxy config
├── monitoring/          # Monitoring & drift detection
├── frontend/            # HTML/CSS/JS web interface
├── docker-compose.yml   # Container orchestration
├── Dockerfile.api       # API service image
├── .gitignore           # Git ignore rules
├── .env                 # Environment variables
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Grafana Admin Credentials
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=your_secure_password

# API Configuration
ENV=development
LOG_LEVEL=info

# Model Configuration
MODEL_PATH=/app/models/best_model.joblib
CONFIG_PATH=/app/models/model_config.json
```

## 🤖 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.50,
    "merchant_id": "MRC123",
    "customer_id": "CUST456",
    "transaction_type": "online"
  }'
```

## 📊 Monitoring & Dashboards

### Grafana

1. Login at http://localhost:3000 (default: admin/admin)
2. Prometheus data source is pre-configured
3. Import fraud detection dashboards from `monitoring/grafana/dashboards/`

### Prometheus

- Metrics endpoint: http://localhost:9090
- Query language: PromQL
- Retention: 15 days

## 🧪 Development

### Local Setup (Without Docker)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API locally
python -m uvicorn src.serving.main:app --reload --port 8000
```

### Training a Model

```bash
python -m src.models.train \
  --data-path data/processed/train.csv \
  --output-path models/new_model.joblib
```

## 📝 Logging

- **API Logs**: Container logs via `docker-compose logs naijashield-api`
- **MLflow Logs**: http://localhost:5001
- **Prometheus**: http://localhost:9090

## 🔒 Security Best Practices

- [ ] Never commit `.env` files
- [ ] Use strong Grafana credentials
- [ ] Enable HTTPS in production
- [ ] Implement API authentication/authorization
- [ ] Rotate secrets regularly
- [ ] Use read-only volumes for models

## 📦 Dependencies

Key packages:
- **FastAPI**: Web framework for API
- **MLflow**: Experiment tracking & model registry
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **Prometheus**: Metrics collection
- **Grafana**: Visualization

See `requirements.txt` for full list.

## 🐛 Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs -f service_name

# Restart all services
docker-compose restart
```

### Port conflicts
```bash
# Find process using port
lsof -i :8000
# Kill process
kill -9 <PID>
```

### Models not loading
- Verify model path in `.env`
- Check file permissions
- Ensure model format matches (joblib/pkl)

## 📈 Performance Metrics

Track these metrics in Grafana:
- Prediction latency (p50, p95, p99)
- API throughput (requests/sec)
- Model accuracy scores
- False positive/negative rates
- Transaction volume by type

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit a pull request

## 📄 License

MIT License — see LICENSE for details.

## 👤 Author

**The Data Boy** - Fraud Detection Project

## 📧 Support

For issues, questions, or suggestions, please open an GitHub issue or contact the maintainers.

---
