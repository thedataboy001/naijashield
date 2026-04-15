# ============================================================
# model_loader.py
# ============================================================
# Handles loading the model from MLflow Model Registry
# and keeping it in memory for fast inference.
#
# WHY SINGLETON PATTERN?
#   Loading a model takes ~500ms
#   We load ONCE at startup, reuse for every request
#   This is how we stay under 100ms per request
#
# WHY MLFLOW REGISTRY?
#   Single source of truth for model versions
#   Stage-based promotion (Staging → Production)
#   Audit trail of which model served which predictions
# ============================================================

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader.
    Loads model once at startup, serves from memory.
    """

    _instance  : Optional["ModelLoader"] = None
    _model     = None
    _config    : dict = {}
    _loaded_at : float = 0.0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        model_path  : str = "../models/best_model.joblib",
        config_path : str = "../models/model_config.json",
        mlflow_uri  : Optional[str] = None,
        use_registry: bool = False,
    ) -> None:
        """
        Load model either from local file or MLflow registry.

        Args:
            model_path  : Path to joblib model file
            config_path : Path to model config JSON
            mlflow_uri  : MLflow tracking URI (if use_registry=True)
            use_registry: Load from MLflow registry instead of file
        """
        start = time.time()

        # ── Load config ───────────────────────────────────────
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}\n"
                f"Run 03_model_training.ipynb first."
            )

        with open(config_path) as f:
            self._config = json.load(f)

        logger.info(f"Config loaded from {config_path}")

        # ── Load model ────────────────────────────────────────
        if use_registry and mlflow_uri:
            self._load_from_registry(mlflow_uri)
        else:
            self._load_from_file(model_path)

        self._loaded_at = time.time()
        load_time = (self._loaded_at - start) * 1000

        logger.info(
            f"Model loaded in {load_time:.0f}ms | "
            f"Version: {self.version} | "
            f"Features: {len(self.features)}"
        )
        print(f"✅ Model loaded in {load_time:.0f}ms")
        print(f"   Version   : {self.version}")
        print(f"   Type      : {self._config['model_type']}")
        print(f"   Threshold : {self.threshold}")
        print(f"   Features  : {len(self.features)}")

    def _load_from_file(self, model_path: str) -> None:
        """Load model from local joblib file."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Run 03_model_training.ipynb first."
            )
        self._model = joblib.load(model_path)
        logger.info(f"Model loaded from file: {model_path}")

    def _load_from_registry(self, mlflow_uri: str) -> None:
        """Load model from MLflow Model Registry (Staging stage)."""
        mlflow.set_tracking_uri(mlflow_uri)
        model_name = "naijashield-xgboost"
        model_uri  = f"models:/{model_name}/Staging"

        try:
            self._model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Model loaded from registry: {model_uri}")
        except Exception as e:
            logger.warning(
                f"Registry load failed ({e}), "
                f"falling back to local file"
            )
            self._load_from_file("../models/best_model.joblib")

    def reload(self) -> None:
        """Hot-reload the model without restarting the server."""
        logger.info("Hot-reloading model...")
        self.load()

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call ModelLoader().load() first."
            )
        return self._model

    @property
    def features(self) -> list:
        return self._config.get("features", [])

    @property
    def threshold(self) -> float:
        return self._config.get("threshold", 0.5)

    @property
    def version(self) -> str:
        return self._config.get(
            "mlflow_run_id", "unknown"
        )[:8]  # First 8 chars of run ID

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def uptime(self) -> float:
        if self._loaded_at == 0:
            return 0.0
        return time.time() - self._loaded_at


# Global singleton instance
model_loader = ModelLoader()