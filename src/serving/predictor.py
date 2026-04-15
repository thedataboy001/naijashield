# ============================================================
# predictor.py
# ============================================================
# Core inference logic.
# Takes raw transaction data → assembles features →
# runs model → returns decision with explanation.
#
# THIS IS THE CRITICAL PATH FOR LATENCY.
# Every operation here directly affects the 100ms SLA.
# ============================================================

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Tuple

from .model_loader import model_loader
from .schemas import (
    TransactionRequest,
    TransactionResponse,
    FraudDecision,
)

logger = logging.getLogger(__name__)

# ── Nigerian city → target encoded fraud rate ─────────────────
# Pre-computed from training data (Cell 6, feature engineering)
# In production this comes from the feature store
LOCATION_FRAUD_RATES = {
    "Lagos":         0.0359,
    "Abuja":         0.0363,
    "Kano":          0.0360,
    "Kaduna":        0.0364,
    "Port Harcourt": 0.0354,
    "Ibadan":        0.0359,
    "Enugu":         0.0356,
    "Benin City":    0.0359,
    "Onitsha":       0.0361,
    "Aba":           0.0356,
}
GLOBAL_FRAUD_RATE = 0.0359  # Fallback for unknown cities

MERCHANT_FRAUD_RATES = {
    # Pre-computed from training data
    # In production: fetched from feature store (Redis)
    "Bet9ja Stake":        0.0370,
    "NairaBet Gaming":     0.0368,
    "MTN Airtime Top-up":  0.0355,
    "Jumia Purchase":      0.0358,
    "Konga Shopping":      0.0357,
    "Uber Ride":           0.0360,
    "Bolt Transport":      0.0359,
    "Opay Transfer":       0.0361,
    "PalmPay Service":     0.0358,
    "Flutterwave Payment": 0.0362,
}


def assemble_features(
    txn: TransactionRequest,
) -> pd.DataFrame:
    """
    Transform raw transaction request into the feature vector
    expected by the model.

    This MUST produce the exact same features in the exact
    same order as the training feature matrix.
    Feature mismatch = silent wrong predictions.
    """

    # ── Extract temporal features ─────────────────────────────
    ts = txn.timestamp
    txn_hour        = ts.hour
    txn_day_of_week = ts.weekday()
    txn_day_of_month= ts.day
    txn_month       = ts.month
    txn_is_weekend  = int(txn_day_of_week >= 5)
    txn_is_salary_week = int(txn_day_of_month >= 26)
    is_night_hour   = int(txn_hour >= 23 or txn_hour <= 5)

    # ── Amount features ───────────────────────────────────────
    EPSILON = 1e-6
    log_amount = np.log1p(txn.amount_ngn)

    amount_vs_user_avg = (
        txn.amount_ngn / (txn.user_avg_txn_amt + EPSILON)
    )
    amount_zscore_user = (
        (txn.amount_ngn - txn.user_avg_txn_amt) /
        (txn.user_std_txn_amt + EPSILON)
    )

    # ── Device features ───────────────────────────────────────
    new_device = int(txn.new_device_transaction)
    device_novelty_score = 1 / (txn.device_seen_count + 1)

    # ── Missingness features ──────────────────────────────────
    has_prior = int(txn.time_since_last_transaction != -1)
    tsl_value = txn.time_since_last_transaction

    # ── User history features ─────────────────────────────────
    is_new_user          = int(txn.user_txn_count_total <= 3)
    user_maturity_score  = np.log1p(txn.user_txn_count_total)

    velocity_vs_typical = (
        txn.txn_count_last_1h /
        (max(txn.avg_gap_between_txns, 1) + EPSILON)
    )
    hourly_amount_vs_avg = (
        (txn.amount_ngn * txn.txn_count_last_1h) /
        (txn.user_avg_txn_amt * txn.txn_count_last_1h + EPSILON)
    )

    # ── Interaction features ──────────────────────────────────
    high_velocity = int(txn.txn_count_last_1h > 3)
    new_device_high_velocity = new_device * high_velocity
    large_amount  = int(amount_vs_user_avg > 2)
    new_device_large_amount  = new_device * large_amount
    night_new_device = is_night_hour * new_device

    # ── Target encoded categoricals ───────────────────────────
    location_enc = LOCATION_FRAUD_RATES.get(
        txn.location, GLOBAL_FRAUD_RATE
    )
    merchant_enc = MERCHANT_FRAUD_RATES.get(
        txn.merchant_category, GLOBAL_FRAUD_RATE
    )
    # user_top_category — use global rate as default
    user_top_cat_enc = GLOBAL_FRAUD_RATE

    # ── One-hot encoded categoricals ─────────────────────────
    # Payment channel
    pc_ussd    = int(txn.payment_channel == "USSD")
    pc_card    = int(txn.payment_channel == "Card")
    pc_mobile  = int(txn.payment_channel == "Mobile App")
    pc_bank    = int(txn.payment_channel == "Bank Transfer")

    # Transaction type
    tt_deposit    = int(txn.transaction_type == "deposit")
    tt_withdrawal = int(txn.transaction_type == "withdrawal")
    tt_payment    = int(txn.transaction_type == "payment")
    tt_transfer   = int(txn.transaction_type == "transfer")

    # Device used
    du_mobile = int(txn.device_used == "mobile")
    du_web    = int(txn.device_used == "web")
    du_atm    = int(txn.device_used == "atm")
    du_pos    = int(txn.device_used == "pos")

    # Sender persona
    sp_salary  = int(txn.sender_persona == "Salary Earner")
    sp_student = int(txn.sender_persona == "Student")
    sp_trader  = int(txn.sender_persona == "Trader")

    # Geo region
    gr_sw  = int(txn.ip_geo_region == "South West")
    gr_se  = int(txn.ip_geo_region == "South East")
    gr_ss  = int(txn.ip_geo_region == "South South")
    gr_nc  = int(txn.ip_geo_region == "North Central")
    gr_nw  = int(txn.ip_geo_region == "North West")

    # ── Assemble feature dict ─────────────────────────────────
    # ORDER MUST MATCH TRAINING FEATURE LIST EXACTLY
    feature_dict = {
        "new_device_transaction":     new_device,
        "bvn_linked":                 int(txn.bvn_linked),
        "spending_deviation_score":   txn.spending_deviation_score,
        "velocity_score":             txn.velocity_score,
        "geo_anomaly_score":          txn.geo_anomaly_score,
        "amount_ngn":                 txn.amount_ngn,
        "device_seen_count":          txn.device_seen_count,
        "ip_seen_count":              txn.ip_seen_count,
        "user_txn_count_total":       txn.user_txn_count_total,
        "user_avg_txn_amt":           txn.user_avg_txn_amt,
        "user_std_txn_amt":           txn.user_std_txn_amt,
        "user_txn_frequency_24h":     txn.txn_count_last_1h,
        "txn_count_last_1h":          txn.txn_count_last_1h,
        "time_since_last":            txn.time_since_last,
        "avg_gap_between_txns":       txn.avg_gap_between_txns,
        "merchant_fraud_rate":        merchant_enc,
        "txn_hour":                   txn_hour,
        "time_since_last_transaction":tsl_value,
        "has_prior_transaction":      has_prior,
        "log_amount_ngn":             log_amount,
        "amount_vs_user_avg":         amount_vs_user_avg,
        "amount_zscore_user":         amount_zscore_user,
        "velocity_vs_typical":        velocity_vs_typical,
        "hourly_amount_vs_avg":       hourly_amount_vs_avg,
        "new_device_high_velocity":   new_device_high_velocity,
        "new_device_large_amount":    new_device_large_amount,
        "device_novelty_score":       device_novelty_score,
        "is_new_user":                is_new_user,
        "user_maturity_score":        user_maturity_score,
        "is_night_hour":              is_night_hour,
        "night_new_device":           night_new_device,
        "txn_is_weekend":             txn_is_weekend,
        "txn_is_salary_week":         txn_is_salary_week,
        "txn_day_of_week":            txn_day_of_week,
        "txn_day_of_month":           txn_day_of_month,
        "txn_month":                  txn_month,
        "merchant_category_target_enc": merchant_enc,
        "location_target_enc":        location_enc,
        "user_top_category_target_enc": user_top_cat_enc,
        "payment_channel_USSD":       pc_ussd,
        "payment_channel_Card":       pc_card,
        "payment_channel_Mobile App": pc_mobile,
        "payment_channel_Bank Transfer": pc_bank,
        "transaction_type_deposit":   tt_deposit,
        "transaction_type_withdrawal":tt_withdrawal,
        "transaction_type_payment":   tt_payment,
        "transaction_type_transfer":  tt_transfer,
        "device_used_mobile":         du_mobile,
        "device_used_web":            du_web,
        "device_used_atm":            du_atm,
        "device_used_pos":            du_pos,
        "sender_persona_Salary Earner": sp_salary,
        "sender_persona_Student":     sp_student,
        "sender_persona_Trader":      sp_trader,
        "ip_geo_region_South West":   gr_sw,
        "ip_geo_region_South East":   gr_se,
        "ip_geo_region_South South":  gr_ss,
        "ip_geo_region_North Central":gr_nc,
        "ip_geo_region_North West":   gr_nw,
    }

    # Build DataFrame in correct feature order
    ordered_features = model_loader.features
    feature_df = pd.DataFrame(
        [{k: feature_dict.get(k, 0) for k in ordered_features}]
    )

    return feature_df


def get_top_features(
    feature_df: pd.DataFrame,
    n: int = 5
) -> dict:
    """
    Return the top N features driving the fraud decision.
    Uses feature importances from the model (fast, no SHAP needed).
    """
    importances = model_loader.model.feature_importances_
    feature_names = model_loader.features

    top_idx = np.argsort(importances)[::-1][:n]

    return {
        feature_names[i]: {
            "value":      float(feature_df.iloc[0, i]),
            "importance": float(importances[i])
        }
        for i in top_idx
    }


def make_decision(fraud_score: float, threshold: float) -> FraudDecision:
    """
    Convert fraud probability to a business decision.

    Three-tier decision system:
      APPROVE  : score < threshold × 0.8  (clearly legitimate)
      REVIEW   : threshold × 0.8 ≤ score < threshold (borderline)
      DECLINE  : score ≥ threshold (likely fraud)
    """
    review_threshold = threshold * 0.8

    if fraud_score >= threshold:
        return FraudDecision.DECLINE
    elif fraud_score >= review_threshold:
        return FraudDecision.REVIEW
    else:
        return FraudDecision.APPROVE


def predict(txn: TransactionRequest) -> TransactionResponse:
    """
    Main prediction function.
    Raw transaction → fraud decision in <100ms.
    """
    start_time = time.time()

    # ── 1. Assemble feature vector ────────────────────────────
    feature_df = assemble_features(txn)

    # ── 2. Model inference ────────────────────────────────────
    fraud_score = float(
        model_loader.model.predict_proba(feature_df)[0, 1]
    )

    # ── 3. Decision ───────────────────────────────────────────
    threshold = model_loader.threshold
    decision  = make_decision(fraud_score, threshold)

    # ── 4. Top feature explanation (lightweight) ──────────────
    explanation = get_top_features(feature_df, n=5)

    # ── 5. Compute latency ────────────────────────────────────
    latency_ms = (time.time() - start_time) * 1000

    if latency_ms > 100:
        logger.warning(
            f"SLA BREACH: {latency_ms:.1f}ms > 100ms "
            f"for transaction {txn.transaction_id}"
        )

    return TransactionResponse(
        transaction_id = txn.transaction_id,
        decision       = decision,
        fraud_score    = round(fraud_score, 6),
        threshold_used = threshold,
        model_version  = model_loader.version,
        latency_ms     = round(latency_ms, 2),
        explanation    = explanation,
    )