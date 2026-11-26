"""Day-by-day replay of TEST (M9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.budget import daily_metrics, select_alerts
from fraud.config import Config
from fraud.explain import Explainer


def format_reasons(reasons: list[dict[str, str]]) -> dict[str, str]:
    return {f"reason_{i}": r["message"] for i, r in enumerate(reasons, start=1)}


def replay(
    test: pd.DataFrame,
    features: pd.DataFrame,
    policy: str,
    threshold: float | None,
    explainer: Explainer,
    config: Config,
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write one queue per TEST day and a summary (FR-9.1, FR-9.2).

    Returns (all alerts with reasons, summary). ``features`` is indexed by TransactionID.
    """
    k = config.budget.default_k
    alerts = select_alerts(test, policy, k, threshold=threshold).reset_index(drop=True)
    reasons = explainer.explain(features.loc[alerts["TransactionID"]])
    reason_columns = pd.DataFrame([format_reasons(r) for r in reasons], index=alerts.index)
    alerts = pd.concat([alerts, reason_columns], axis=1)

    queue_dir = directory / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    for old in queue_dir.glob("day_*.csv"):
        old.unlink()
    columns = [
        "rank",
        "TransactionID",
        "amount",
        "score",
        "expected_loss",
        *reason_columns.columns,
        "true_label",
    ]
    table = alerts.assign(
        amount=alerts["TransactionAmt"],
        score=alerts["probability"],
        expected_loss=alerts["probability"] * alerts["TransactionAmt"],
        true_label=alerts["isFraud"],
    )
    for day in sorted(test["day"].unique()):
        table.loc[table["day"] == day, columns].to_csv(
            queue_dir / f"day_{int(day):03d}.csv", index=False
        )

    daily = daily_metrics(test, alerts, config.costs)
    summary = pd.DataFrame(
        {
            "day": daily["day"],
            "alerts": daily["alerts"],
            "frauds_caught": daily["true_frauds"],
            "value_caught": daily["value_alerted"],
            "value_missed": daily["fraud_value"] - daily["value_alerted"],
            "net_benefit": daily["net_benefit"],
            "cumulative_net_benefit": daily["net_benefit"].cumsum(),
        }
    )
    summary.to_csv(directory / "queue_summary.csv", index=False)
    return alerts, summary
