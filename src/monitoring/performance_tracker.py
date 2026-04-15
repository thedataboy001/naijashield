# ============================================================
# performance_tracker.py
# ============================================================
# Tracks prediction logs and computes rolling metrics.
# In production this reads from Kafka.
# In development it reads from a local log file.
# ============================================================

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs predictions to a local JSONL file.
    In production: writes to Kafka topic.
    """

    def __init__(
        self,
        log_path: str = "../monitoring/predictions.jsonl"
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        transaction_id : str,
        fraud_score    : float,
        decision       : str,
        latency_ms     : float,
        model_version  : str,
        features       : dict = None,
    ) -> None:
        """Append prediction to log file."""
        record = {
            "timestamp":      datetime.now().isoformat(),
            "transaction_id": transaction_id,
            "fraud_score":    fraud_score,
            "decision":       decision,
            "latency_ms":     latency_ms,
            "model_version":  model_version,
            "features":       features or {},
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_recent(
        self,
        hours: int = 24
    ) -> pd.DataFrame:
        """Load predictions from last N hours."""
        if not self.log_path.exists():
            return pd.DataFrame()

        records = []
        cutoff  = datetime.now() - timedelta(hours=hours)

        with open(self.log_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    ts = datetime.fromisoformat(record["timestamp"])
                    if ts >= cutoff:
                        records.append(record)
                except Exception:
                    continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def compute_rolling_metrics(
        self,
        window_hours: int = 1,
    ) -> dict:
        """Compute metrics over a rolling time window."""
        df = self.load_recent(hours=window_hours)

        if df.empty:
            return {
                "window_hours":   window_hours,
                "n_predictions":  0,
                "approve_rate":   None,
                "decline_rate":   None,
                "review_rate":    None,
                "avg_score":      None,
                "avg_latency_ms": None,
                "p99_latency_ms": None,
                "sla_breach_rate":None,
            }

        n = len(df)
        return {
            "window_hours":    window_hours,
            "n_predictions":   n,
            "approve_rate":    float((df["decision"]=="APPROVE").mean()),
            "decline_rate":    float((df["decision"]=="DECLINE").mean()),
            "review_rate":     float((df["decision"]=="REVIEW").mean()),
            "avg_score":       float(df["fraud_score"].mean()),
            "p50_score":       float(df["fraud_score"].median()),
            "p95_score":       float(df["fraud_score"].quantile(0.95)),
            "avg_latency_ms":  float(df["latency_ms"].mean()),
            "p99_latency_ms":  float(df["latency_ms"].quantile(0.99)),
            "sla_breach_rate": float((df["latency_ms"] > 100).mean()),
            "timestamp":       datetime.now().isoformat(),
        }