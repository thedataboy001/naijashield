# ============================================================
# run_drift_check.py
# ============================================================
# Standalone script to run drift detection.
# Called by Airflow DAG weekly or on-demand.
#
# Usage:
#   python src/monitoring/run_drift_check.py
#   python src/monitoring/run_drift_check.py --hours 48
# ============================================================

import argparse
import json
import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

from drift_detector import DriftDetector
from performance_tracker import PredictionLogger

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main(hours: int = 24):
    print("=" * 65)
    print("NAIJASHIELD DRIFT CHECK")
    print(f"Checking last {hours} hours of predictions")
    print("=" * 65)

    # ── Load recent predictions ───────────────────────────────
    pred_logger = PredictionLogger()
    recent_preds = pred_logger.load_recent(hours=hours)

    if recent_preds.empty:
        print(f"\n⚠️  No predictions found in last {hours} hours")
        print("   Skipping drift check — no current data")
        return {"status": "skipped", "reason": "no_recent_predictions"}

    print(f"\n✅ Loaded {len(recent_preds):,} recent predictions")

    # ── Extract feature data from logs ───────────────────────
    if "features" in recent_preds.columns:
        feature_data = pd.json_normalize(recent_preds["features"])
    else:
        print("⚠️  No feature data in logs — skipping feature drift")
        feature_data = pd.DataFrame()

    # ── Run drift detection ───────────────────────────────────
    detector = DriftDetector()

    results = {
        "timestamp": datetime.now().isoformat(),
        "hours_checked": hours,
        "n_predictions": len(recent_preds),
    }

    # Feature drift
    if not feature_data.empty:
        print("\n⏳ Running feature drift check...")
        drift_summary = detector.generate_data_drift_report(
            current_data=feature_data,
            report_name=f"drift_{datetime.now().strftime('%Y%m%d_%H')}"
        )
        results["drift"] = drift_summary

        print(f"\n  Dataset drift detected : {drift_summary['dataset_drift_detected']}")
        print(f"  Drifted features       : {drift_summary['n_drifted_features']}")
        if drift_summary["drifted_features"]:
            print(f"  Drifted columns:")
            for f in drift_summary["drifted_features"]:
                print(f"    - {f['feature']} (p={f['p_value']:.4f})")

        # Run automated tests
        print("\n⏳ Running drift tests...")
        test_results = detector.run_drift_tests(feature_data)
        results["tests"] = test_results

        status = "✅ PASSED" if test_results["all_tests_passed"] else "❌ FAILED"
        print(f"\n  Drift tests: {status}")
        for t in test_results["test_results"]:
            icon = "✅" if t["status"] == "SUCCESS" else "❌"
            print(f"    {icon} {t['name']}: {t['status']}")

    # Performance check
    print("\n⏳ Running performance check...")
    perf_results = detector.check_performance_degradation(recent_preds)
    results["performance"] = perf_results

    perf_status = "✅ OK" if perf_results["performance_ok"] else "❌ DEGRADED"
    print(f"\n  Performance: {perf_status}")

    if perf_results.get("alerts"):
        print("  ALERTS:")
        for alert in perf_results["alerts"]:
            print(f"    🚨 {alert}")

    # Rolling metrics
    metrics = pred_logger.compute_rolling_metrics(window_hours=hours)
    results["rolling_metrics"] = metrics

    print(f"\n{'='*65}")
    print("ROLLING METRICS SUMMARY")
    print(f"{'='*65}")
    print(f"  Predictions   : {metrics['n_predictions']:,}")
    print(f"  Approve rate  : {metrics.get('approve_rate', 0)*100:.1f}%")
    print(f"  Decline rate  : {metrics.get('decline_rate', 0)*100:.1f}%")
    print(f"  Avg score     : {metrics.get('avg_score', 0):.4f}")
    print(f"  Avg latency   : {metrics.get('avg_latency_ms', 0):.1f}ms")
    print(f"  p99 latency   : {metrics.get('p99_latency_ms', 0):.1f}ms")
    print(f"  SLA breach    : {metrics.get('sla_breach_rate', 0)*100:.2f}%")

    # ── Save results ──────────────────────────────────────────
    output_path = Path("../monitoring/reports/latest_check.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved: {output_path}")

    # ── Trigger alert if needed ───────────────────────────────
    should_retrain = (
        results.get("drift", {}).get("dataset_drift_detected", False) or
        not results.get("tests", {}).get("all_tests_passed", True) or
        not results.get("performance", {}).get("performance_ok", True)
    )

    if should_retrain:
        print(f"\n🚨 RETRAINING RECOMMENDED")
        print(f"   Drift or performance degradation detected")
        print(f"   Trigger Airflow retraining DAG")
        results["retraining_recommended"] = True
    else:
        print(f"\n✅ No retraining needed")
        results["retraining_recommended"] = False

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Hours of recent predictions to analyse"
    )
    args = parser.parse_args()
    main(hours=args.hours)