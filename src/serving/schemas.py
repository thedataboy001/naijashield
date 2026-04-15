# ============================================================
# schemas.py
# ============================================================
# Pydantic models for request and response validation.
#
# WHY PYDANTIC?
#   Automatic type validation — wrong input type = clear error
#   Auto-generated API documentation (FastAPI uses this)
#   Serialization/deserialization handled automatically
#   Catches bad requests BEFORE they reach the model
# ============================================================

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum


# ── Enums for categorical fields ─────────────────────────────
class PaymentChannel(str, Enum):
    USSD          = "USSD"
    MOBILE_APP    = "Mobile App"
    CARD          = "Card"
    BANK_TRANSFER = "Bank Transfer"


class DeviceUsed(str, Enum):
    MOBILE = "mobile"
    WEB    = "web"
    ATM    = "atm"
    POS    = "pos"


class TransactionType(str, Enum):
    DEPOSIT    = "deposit"
    WITHDRAWAL = "withdrawal"
    PAYMENT    = "payment"
    TRANSFER   = "transfer"


class SenderPersona(str, Enum):
    SALARY_EARNER = "Salary Earner"
    STUDENT       = "Student"
    TRADER        = "Trader"


class GeoRegion(str, Enum):
    SOUTH_WEST    = "South West"
    SOUTH_EAST    = "South East"
    SOUTH_SOUTH   = "South South"
    NORTH_CENTRAL = "North Central"
    NORTH_WEST    = "North West"
    NORTH_EAST    = "North East"


# ── Transaction Request Schema ────────────────────────────────
class TransactionRequest(BaseModel):
    """
    Incoming transaction payload for fraud scoring.
    All fields mirror the dataset features before
    feature engineering is applied.
    """

    # Core transaction fields
    transaction_id   : str = Field(..., example="T1234567")
    sender_account   : str = Field(..., example="1000018177")
    amount_ngn       : float = Field(..., gt=0, example=250000.0)
    timestamp        : datetime = Field(..., example="2024-01-15T23:45:00")
    payment_channel  : PaymentChannel
    transaction_type : TransactionType
    device_used      : DeviceUsed
    location         : str = Field(..., example="Lagos")
    ip_geo_region    : GeoRegion
    merchant_category: str = Field(..., example="Bet9ja Stake")

    # Account signals
    bvn_linked            : bool = Field(..., example=True)
    new_device_transaction: bool = Field(..., example=True)
    sender_persona        : SenderPersona

    # User history features (from feature store in production)
    user_avg_txn_amt     : float = Field(..., ge=0, example=45000.0)
    user_std_txn_amt     : float = Field(..., ge=0, example=12000.0)
    user_txn_count_total : int   = Field(..., ge=0, example=5)
    txn_count_last_1h    : int   = Field(..., ge=0, example=3)
    avg_gap_between_txns : float = Field(..., ge=-1, example=120.5)

    # Device intelligence
    device_seen_count : int   = Field(..., ge=1, example=1)
    ip_seen_count     : int   = Field(..., ge=1, example=5)

    # Risk scores
    spending_deviation_score: float = Field(..., example=1.2)
    velocity_score          : int   = Field(..., ge=0, example=3)
    geo_anomaly_score       : float = Field(..., ge=0, example=0.0)

    # Time since last transaction (-1 if first transaction)
    time_since_last_transaction: float = Field(..., example=-1.0)
    time_since_last            : float = Field(..., example=-1.0)

    @validator("amount_ngn")
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("amount_ngn must be positive")
        return v

    @validator("timestamp")
    def timestamp_not_future(cls, v):
        if v > datetime.now(timezone.utc):
            raise ValueError("timestamp cannot be in the future")
        return v

    class Config:
        use_enum_values = True


# ── Fraud Decision Response Schema ────────────────────────────
class FraudDecision(str, Enum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    REVIEW  = "REVIEW"   # Human review queue


class TransactionResponse(BaseModel):
    """
    Fraud scoring response returned to the caller.
    """
    transaction_id : str
    decision       : FraudDecision
    fraud_score    : float = Field(
        ..., ge=0, le=1,
        description="Probability of fraud (0=legit, 1=fraud)"
    )
    threshold_used : float
    model_version  : str
    latency_ms     : float
    explanation    : dict   # Top features driving the decision

    class Config:
        use_enum_values = True


# ── Health Check Response ─────────────────────────────────────
class HealthResponse(BaseModel):
    status        : str
    model_loaded  : bool
    model_version : str
    uptime_seconds: float


# ── Batch Request Schema ──────────────────────────────────────
class BatchTransactionRequest(BaseModel):
    """
    Batch scoring for up to 100 transactions at once.
    """
    transactions: list[TransactionRequest] = Field(
        ..., min_items=1, max_items=100
    )

    class Config:
        use_enum_values = True


class BatchTransactionResponse(BaseModel):
    results      : list[TransactionResponse]
    total        : int
    approved     : int
    declined     : int
    review       : int
    avg_latency_ms: float