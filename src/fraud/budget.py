"""Alert selection and business metrics (M6).

Input frames have one row per transaction with columns TransactionID, day, TransactionAmt,
isFraud, probability, rule_score and optionally raw_score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud.config import Costs


def select_alerts(
    df: pd.DataFrame, policy: str, k: int, *, threshold: float | None = None
) -> pd.DataFrame:
    """Return the reviewed alerts, per day, with a 1-based ``rank`` (FR-6.1).

    Ties are broken by raw score, then TransactionID. The fixed-threshold policy works in
    arrival order, and anything past the K-th alert of a day is over capacity and not reviewed.
    """
    raw = df["raw_score"] if "raw_score" in df else df["probability"]
    keys = df.assign(_raw=raw)
    if policy == "probability":
        keys["_key"] = df["probability"]
        order, ascending = ["day", "_key", "_raw", "TransactionID"], [True, False, False, True]
    elif policy == "expected_loss":
        keys["_key"] = df["probability"] * df["TransactionAmt"]
        order, ascending = ["day", "_key", "_raw", "TransactionID"], [True, False, False, True]
    elif policy == "rules":
        order = ["day", "rule_score", "TransactionAmt", "TransactionID"]
        ascending = [True, False, False, True]
    elif policy == "fixed_threshold":
        if threshold is None:
            raise ValueError("fixed_threshold needs a threshold")
        keys = keys[keys["probability"] >= threshold]
        arrival = ["day", "TransactionDT"] if "TransactionDT" in keys else ["day"]
        order = [*arrival, "TransactionID"]
        ascending = [True] * len(order)
    else:
        raise ValueError(f"unknown policy: {policy}")
    chosen = keys.sort_values(order, ascending=ascending, kind="stable")
    chosen = chosen.groupby("day", sort=False).head(k)
    chosen = chosen.drop(columns=[c for c in ("_raw", "_key") if c in chosen])
    return chosen.assign(rank=chosen.groupby("day").cumcount() + 1)


def net_benefit(value_alerted: float, alerts: float, false_alerts: float, costs: Costs) -> float:
    """Section 10.4, in rupees."""
    return (
        costs.recovery_rate * value_alerted * costs.inr_per_unit
        - costs.review_cost_per_alert_inr * alerts
        - costs.friction_cost_per_fp_inr * false_alerts
    )


def daily_metrics(df: pd.DataFrame, alerts: pd.DataFrame, costs: Costs) -> pd.DataFrame:
    """Per-day alert metrics (FR-6.2). Every day in ``df`` gets a row."""
    days = pd.Index(np.sort(df["day"].unique()), name="day")

    def per_day(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
        return values.groupby(frame["day"]).sum().reindex(days, fill_value=0)

    fraud = df["isFraud"] == 1
    hit = alerts["isFraud"] == 1
    table = pd.DataFrame(index=days)
    table["alerts"] = per_day(alerts, pd.Series(1, index=alerts.index)).astype(int)
    table["true_frauds"] = per_day(alerts, hit.astype(int)).astype(int)
    table["frauds"] = per_day(df, fraud.astype(int)).astype(int)
    table["value_alerted"] = per_day(alerts, alerts["TransactionAmt"].where(hit, 0.0))
    table["fraud_value"] = per_day(df, df["TransactionAmt"].where(fraud, 0.0))
    table["precision"] = table["true_frauds"] / table["alerts"].replace(0, np.nan)
    table["recall"] = table["true_frauds"] / table["frauds"].replace(0, np.nan)
    table["vdr"] = table["value_alerted"] / table["fraud_value"].replace(0, np.nan)
    table["net_benefit"] = net_benefit(
        table["value_alerted"], table["alerts"], table["alerts"] - table["true_frauds"], costs
    )
    return table.reset_index()


def period_metrics(daily: pd.DataFrame) -> dict[str, float]:
    """Totals over days, never averages of daily value ratios (FR-6.3)."""
    alerts = int(daily["alerts"].sum())
    caught = int(daily["true_frauds"].sum())
    frauds = int(daily["frauds"].sum())
    value = float(daily["value_alerted"].sum())
    fraud_value = float(daily["fraud_value"].sum())
    return {
        "days": len(daily),
        "alerts": alerts,
        "true_frauds": caught,
        "frauds": frauds,
        "value_alerted": value,
        "fraud_value": fraud_value,
        "precision": caught / alerts if alerts else np.nan,
        "recall": caught / frauds if frauds else np.nan,
        "vdr": value / fraud_value if fraud_value else np.nan,
        "net_benefit": float(daily["net_benefit"].sum()),
        "mean_daily_precision": float(daily["precision"].mean()),
        "p10_daily_precision": float(daily["precision"].quantile(0.10)),
    }


def choose_threshold(validation: pd.DataFrame, k: int) -> float:
    """Threshold whose average daily alert count on VALIDATION is closest to K (policy P3)."""
    scores = np.sort(validation["probability"].to_numpy())
    candidates = np.unique(scores)
    counts = len(scores) - np.searchsorted(scores, candidates, side="left")
    average = counts / validation["day"].nunique()
    return float(candidates[np.argmin(np.abs(average - k))])


@dataclass(frozen=True)
class Breaches:
    breach_days: int
    max_overflow: int
    underused_days: int


def capacity_breaches(df: pd.DataFrame, threshold: float, k: int) -> Breaches:
    """Days where a static threshold exceeds capacity or wastes more than half (FR-6.5)."""
    counts = (df["probability"] >= threshold).groupby(df["day"]).sum()
    over = counts - k
    return Breaches(int((over > 0).sum()), int(max(over.max(), 0)), int((counts < 0.5 * k).sum()))


def evaluate_policy(
    df: pd.DataFrame, policy: str, k: int, costs: Costs, *, threshold: float | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select alerts and return (daily table, period totals)."""
    daily = daily_metrics(df, select_alerts(df, policy, k, threshold=threshold), costs)
    return daily, period_metrics(daily)
