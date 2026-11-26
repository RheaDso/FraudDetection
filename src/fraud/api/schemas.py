"""API request and response schemas (M10)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """One raw transaction. Any other raw column (C*, D*, V*, id_*...) may be added."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    TransactionID: int
    TransactionDT: int = Field(ge=0, description="Seconds from the dataset's reference time")
    TransactionAmt: float = Field(gt=0)
    ProductCD: str
    card1: float
    card2: float | None = None
    card5: float | None = None
    addr1: float | None = None
    D1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceInfo: str | None = None


class Reason(BaseModel):
    feature: str
    message: str


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    expected_loss: float
    alert: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: list[Reason]
    model_version: str


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    store_rows: int
