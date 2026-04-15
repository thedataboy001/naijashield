# ============================================================
# main.py
# ============================================================
# FastAPI application.
# Defines all endpoints and wires everything together.
# ============================================================

import time
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)
from fastapi.responses import Response
import time

from .schemas import (
    TransactionRequest,
    TransactionResponse,
    BatchTransactionRequest,
    BatchTransactionResponse,
    HealthResponse,
)
from .model_loader import model_loader
from .predictor import predict

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App startup / shutdown ────────────────────────────────────
APP_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup, cleanup on shutdown.
    Lifespan replaces deprecated @app.on_event("startup")
    """
    # ── STARTUP ──────────────────────────────────────────────
    logger.info("NaijaShield API starting up...")
    model_loader.load(
        model_path  = "models/best_model.joblib",
        config_path = "models/model_config.json",
    )
    logger.info("✅ Model loaded — API ready")
    yield

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("NaijaShield API shutting down...")


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "NaijaShield Fraud Detection API",
    description = """
## Nigerian Financial Fraud Detection

Real-time fraud scoring API for Nigerian payment transactions.

### Features
- **Sub-100ms latency** — designed for payment authorization windows
- **Three-tier decisions** — APPROVE / REVIEW / DECLINE
- **Explainability** — top features driving each decision
- **Nigerian context** — USSD, Mobile App, Card, Bank Transfer channels

### Fraud Score
- `0.0` = very likely legitimate
- `1.0` = very likely fraud
- Threshold: configured per model version
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  # Restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Define metrics ────────────────────────────────────────────
# These are the metrics Prometheus will scrape every 15s

# Total predictions counter
PREDICTIONS_TOTAL = Counter(
    "naijashield_predictions_total",
    "Total number of fraud predictions made",
    ["decision", "model_version"]
)

# Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    "naijashield_prediction_latency_ms",
    "End-to-end prediction latency in milliseconds",
    buckets=[5, 10, 20, 30, 50, 75, 100, 150, 200, 500]
)

# Fraud score distribution
FRAUD_SCORE = Histogram(
    "naijashield_fraud_score",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Current fraud rate gauge (rolling)
FRAUD_RATE_GAUGE = Gauge(
    "naijashield_fraud_rate_current",
    "Current fraud rate (last 1000 predictions)"
)

# API errors counter
API_ERRORS = Counter(
    "naijashield_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"]
)

# Model info gauge
MODEL_INFO = Gauge(
    "naijashield_model_info",
    "Model metadata",
    ["model_version", "model_type"]
)

# Rolling window for fraud rate calculation
from collections import deque
RECENT_DECISIONS = deque(maxlen=1000)


# ── Add metrics endpoint ──────────────────────────────────────
@app.get("/metrics", tags=["Operations"])
async def metrics():
    """
    Prometheus metrics endpoint.
    Scraped every 15 seconds by Prometheus.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )



# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "NaijaShield Fraud Detection API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"]
)
async def health_check():
    """
    Health check endpoint.
    Used by Docker, load balancers, and monitoring systems.
    Returns 200 if model is loaded and ready.
    Returns 503 if model failed to load.
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable"
        )

    return HealthResponse(
        status         = "healthy",
        model_loaded   = True,
        model_version  = model_loader.version,
        uptime_seconds = round(time.time() - APP_START_TIME, 1),
    )


@app.post(
    "/predict",
    response_model=TransactionResponse,
    tags=["Inference"],
)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
):
    if not model_loader.is_loaded:
        API_ERRORS.labels(
            endpoint="predict",
            error_type="model_not_loaded"
        ).inc()
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        response = predict(transaction)

        # ── Record Prometheus metrics ─────────────────────
        PREDICTIONS_TOTAL.labels(
            decision      = response.decision,
            model_version = response.model_version
        ).inc()

        PREDICTION_LATENCY.observe(response.latency_ms)
        FRAUD_SCORE.observe(response.fraud_score)

        # Update rolling fraud rate
        RECENT_DECISIONS.append(
            1 if response.decision == "DECLINE" else 0
        )
        if len(RECENT_DECISIONS) > 0:
            FRAUD_RATE_GAUGE.set(
                sum(RECENT_DECISIONS) / len(RECENT_DECISIONS)
            )

        # Record model info (once)
        MODEL_INFO.labels(
            model_version = response.model_version,
            model_type    = "XGBoost"
        ).set(1)

        background_tasks.add_task(
            log_prediction, transaction, response
        )

        return response

    except Exception as e:
        API_ERRORS.labels(
            endpoint="predict",
            error_type=type(e).__name__
        ).inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchTransactionResponse,
    tags=["Inference"],
    summary="Score multiple transactions (max 100)",
)
async def predict_batch(batch: BatchTransactionRequest):
    """
    Score up to 100 transactions in a single request.
    More efficient than individual calls for bulk processing.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    results    = []
    total_ms   = 0.0

    for txn in batch.transactions:
        try:
            result = predict(txn)
            results.append(result)
            total_ms += result.latency_ms
        except Exception as e:
            logger.error(f"Batch prediction failed for {txn.transaction_id}: {e}")

    decisions = [r.decision for r in results]

    return BatchTransactionResponse(
        results       = results,
        total         = len(results),
        approved      = decisions.count("APPROVE"),
        declined      = decisions.count("DECLINE"),
        review        = decisions.count("REVIEW"),
        avg_latency_ms= round(total_ms / len(results), 2) if results else 0,
    )


@app.post(
    "/model/reload",
    tags=["Operations"],
    summary="Hot-reload model without restarting server",
)
async def reload_model():
    """
    Reload the model from disk without restarting the API.
    Use this after promoting a new model version.
    """
    try:
        model_loader.reload()
        return {
            "status":  "success",
            "message": "Model reloaded successfully",
            "version": model_loader.version,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed: {str(e)}"
        )


@app.get(
    "/model/info",
    tags=["Operations"],
    summary="Current model metadata",
)
async def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "version":   model_loader.version,
        "threshold": model_loader.threshold,
        "n_features":len(model_loader.features),
        "features":  model_loader.features,
        "is_loaded": model_loader.is_loaded,
        "uptime_s":  round(model_loader.uptime, 1),
    }


# ── Background task: log predictions ─────────────────────────
async def log_prediction(
    txn: TransactionRequest,
    response: TransactionResponse,
) -> None:
    """
    Async logging of predictions.
    Non-blocking — does not affect response latency.
    In production: writes to Kafka topic for monitoring.
    """
    logger.info(
        f"PREDICTION | "
        f"id={response.transaction_id} | "
        f"score={response.fraud_score:.4f} | "
        f"decision={response.decision} | "
        f"latency={response.latency_ms:.1f}ms | "
        f"model={response.model_version}"
    )


