# ============================================================
# drift_detector.py
# ============================================================
# Uses Evidently AI to detect:
#   1. DATA DRIFT    → input feature distributions shifted
#   2. TARGET DRIFT  → fraud rate changed significantly
#   3. MODEL QUALITY → performance degraded over time
#
# HOW IT WORKS:
#   Reference dataset = training data distribution
#   Current dataset   = recent production predictions
#   Evidently compares them and raises alerts
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    ClassificationPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
    ColumnDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
)

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors data drift and model performance degradation.
    Generates Evidently HTML reports and JSON summaries.
    """

    def __init__(
        self,
        reference_data_path : str = "../data/processed/X_train.parquet",
        reports_dir         : str = "../monitoring/reports",
        key_features        : list = None,
    ):
        self.reference_data_path = reference_data_path
        self.reports_dir         = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Key features to monitor individually
        # These are the top features from feature importance
        self.key_features = key_features or [
            "new_device_transaction",
            "has_prior_transaction",
            "time_since_last_transaction",
            "amount_ngn",
            "txn_count_last_1h",
            "device_seen_count",
            "spending_deviation_score",
            "velocity_score",
            "amount_vs_user_avg",
            "user_txn_count_total",
        ]

        self._reference_data = None
        logger.info("DriftDetector initialised")

    @property
    def reference_data(self) -> pd.DataFrame:
        """Load reference data lazily."""
        if self._reference_data is None:
            self._reference_data = pd.read_parquet(
                self.reference_data_path
            )
            # Sample for speed (10k rows is sufficient for drift detection)
            if len(self._reference_data) > 10_000:
                self._reference_data = self._reference_data.sample(
                    n=10_000, random_state=42
                )
            logger.info(
                f"Reference data loaded: {self._reference_data.shape}"
            )
        return self._reference_data

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        report_name : str = None,
    ) -> dict:
        """
        Generate full data drift report comparing
        reference (training) vs current (production) data.

        Returns drift summary dict for alerting.
        """
        if report_name is None:
            report_name = datetime.now().strftime(
                "drift_report_%Y%m%d_%H%M%S"
            )

        # Align columns
        common_cols = [
            c for c in self.key_features
            if c in self.reference_data.columns
            and c in current_data.columns
        ]

        ref = self.reference_data[common_cols].copy()
        cur = current_data[common_cols].copy()

        logger.info(
            f"Generating drift report: "
            f"ref={len(ref)} rows, cur={len(cur)} rows"
        )

        # ── Full drift report ─────────────────────────────────
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesSummaryMetric(),
        ])

        report.run(
            reference_data = ref,
            current_data   = cur,
        )

        # Save HTML report
        html_path = self.reports_dir / f"{report_name}.html"
        report.save_html(str(html_path))
        logger.info(f"HTML report saved: {html_path}")

        # Extract summary as dict
        report_dict = report.as_dict()

        # ── Parse drift summary ───────────────────────────────
        drift_summary = self._parse_drift_summary(report_dict)
        drift_summary["report_name"] = report_name
        drift_summary["report_path"] = str(html_path)
        drift_summary["timestamp"]   = datetime.now().isoformat()
        drift_summary["n_reference"] = len(ref)
        drift_summary["n_current"]   = len(cur)

        # Save JSON summary
        json_path = self.reports_dir / f"{report_name}.json"
        with open(json_path, "w") as f:
            json.dump(drift_summary, f, indent=2, default=str)

        logger.info(f"Drift summary: {drift_summary}")

        return drift_summary

    def _parse_drift_summary(self, report_dict: dict) -> dict:
        """Extract key drift metrics from Evidently report dict."""
        summary = {
            "dataset_drift_detected": False,
            "n_drifted_features":     0,
            "share_drifted_features": 0.0,
            "drifted_features":       [],
        }

        try:
            for metric in report_dict.get("metrics", []):
                result = metric.get("result", {})

                if "dataset_drift" in result:
                    summary["dataset_drift_detected"] = (
                        result["dataset_drift"]
                    )

                if "number_of_drifted_columns" in result:
                    summary["n_drifted_features"] = (
                        result["number_of_drifted_columns"]
                    )

                if "share_of_drifted_columns" in result:
                    summary["share_drifted_features"] = (
                        result["share_of_drifted_columns"]
                    )

                if "drift_by_columns" in result:
                    for col, col_result in (
                        result["drift_by_columns"].items()
                    ):
                        if col_result.get("drift_detected", False):
                            summary["drifted_features"].append({
                                "feature":  col,
                                "p_value":  col_result.get("p_value"),
                                "stattest": col_result.get(
                                    "stattest_name"
                                ),
                            })
        except Exception as e:
            logger.error(f"Failed to parse drift summary: {e}")

        return summary

    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
    ) -> dict:
        """
        Run automated drift tests with pass/fail outcomes.
        Used for automated alerts and retraining triggers.

        Tests:
          - Less than 30% of features drifted
          - Key features (new_device_transaction etc.) not drifted
        """
        common_cols = [
            c for c in self.key_features
            if c in self.reference_data.columns
            and c in current_data.columns
        ]

        ref = self.reference_data[common_cols].copy()
        cur = current_data[common_cols].copy()

        # Build tests
        tests = [
            TestShareOfDriftedColumns(lt=0.3),     # < 30% features drift
            TestNumberOfDriftedColumns(lt=5),       # < 5 features drift
        ]

        # Add individual tests for top features
        for feat in self.key_features[:3]:
            if feat in common_cols:
                tests.append(TestColumnDrift(column_name=feat))

        test_suite = TestSuite(tests=tests)
        test_suite.run(reference_data=ref, current_data=cur)

        # Parse results
        results = test_suite.as_dict()
        passed  = all(
            t.get("status") == "SUCCESS"
            for t in results.get("tests", [])
        )

        test_summary = {
            "all_tests_passed":  passed,
            "timestamp":         datetime.now().isoformat(),
            "n_tests":           len(results.get("tests", [])),
            "test_results":      [
                {
                    "name":   t.get("name"),
                    "status": t.get("status"),
                    "detail": t.get("description", ""),
                }
                for t in results.get("tests", [])
            ],
        }

        if not passed:
            logger.warning(
                f"DRIFT ALERT: {sum(1 for t in results.get('tests',[]) if t.get('status') != 'SUCCESS')} "
                f"tests failed → consider retraining"
            )

        return test_summary

    def check_performance_degradation(
        self,
        recent_predictions : pd.DataFrame,
        threshold_auc_pr   : float = 0.035,
        threshold_fraud_rate: float = 0.10,
    ) -> dict:
        """
        Check if model performance has degraded.

        recent_predictions DataFrame must have:
          - fraud_score   : model output probability
          - is_fraud      : ground truth label (if available)
          - decision      : APPROVE/REVIEW/DECLINE
          - timestamp     : prediction timestamp
        """
        summary = {
            "timestamp":             datetime.now().isoformat(),
            "n_predictions":         len(recent_predictions),
            "performance_ok":        True,
            "alerts":                [],
        }

        # ── Fraud rate check ──────────────────────────────────
        if "decision" in recent_predictions.columns:
            decline_rate = (
                recent_predictions["decision"] == "DECLINE"
            ).mean()

            summary["current_decline_rate"] = float(decline_rate)

            if decline_rate > threshold_fraud_rate:
                alert = (
                    f"HIGH DECLINE RATE: {decline_rate:.1%} "
                    f"(threshold: {threshold_fraud_rate:.1%})"
                )
                summary["alerts"].append(alert)
                summary["performance_ok"] = False
                logger.warning(f"PERFORMANCE ALERT: {alert}")

        # ── Score distribution check ──────────────────────────
        if "fraud_score" in recent_predictions.columns:
            avg_score = recent_predictions["fraud_score"].mean()
            summary["avg_fraud_score"] = float(avg_score)

            # If average score suddenly changes → distribution shift
            EXPECTED_AVG_SCORE = 0.45  # From training distribution
            if abs(avg_score - EXPECTED_AVG_SCORE) > 0.15:
                alert = (
                    f"SCORE DISTRIBUTION SHIFT: "
                    f"avg={avg_score:.3f} "
                    f"(expected≈{EXPECTED_AVG_SCORE:.3f})"
                )
                summary["alerts"].append(alert)
                summary["performance_ok"] = False

        # ── Latency check ─────────────────────────────────────
        if "latency_ms" in recent_predictions.columns:
            p99_latency = recent_predictions["latency_ms"].quantile(0.99)
            summary["p99_latency_ms"] = float(p99_latency)

            if p99_latency > 100:
                alert = (
                    f"SLA BREACH: p99 latency = {p99_latency:.1f}ms "
                    f"(SLA: 100ms)"
                )
                summary["alerts"].append(alert)
                summary["performance_ok"] = False

        return summary