"""Train-serve parity (FR-10.4) and latency (FR-10.6), measured through the real API."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud.api.main import create_app
from fraud.api.scorer import Scorer
from fraud.config import Config
from fraud.feature_store import FeatureStore
from fraud.features.history import HistoryStates, compute_history
from fraud.features.uid import build_uid
from fraud.split import TEST


def payloads(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """JSON-ready raw transactions, with missing values left out."""
    records = frame[columns].astype(object).to_dict("records")
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items() if pd.notna(v)}
        for record in records
    ]


class _Recording(Scorer):
    """Keeps the feature rows the API computed."""

    seen: list[pd.DataFrame]

    def features(self, raw: dict[str, Any]) -> pd.DataFrame:
        matrix = super().features(raw)
        self.seen.append(matrix)
        return matrix


def check(
    interim: pd.DataFrame,
    split: pd.Series,
    batch_features: pd.DataFrame,
    batch_probability: pd.Series,
    config: Config,
) -> dict[str, float]:
    """Warm a store with everything before TEST, replay through POST /score and compare.

    ``interim`` must be sorted by time; ``batch_probability`` is indexed by TransactionID.
    """
    start = int(np.flatnonzero((split == TEST).to_numpy())[0])
    count = max(config.api.parity_rows, config.api.latency_requests)
    prior, replayed = interim.iloc[:start], interim.iloc[start : start + count]
    states = HistoryStates()
    compute_history(prior, build_uid(prior), config, states)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        store = FeatureStore(Path(directory) / "parity.sqlite")
        store.load(states)
        scorer = _Recording(config, store)
        scorer.seen = []
        probabilities, latencies = [], []
        with TestClient(create_app(config, scorer)) as client:
            for body in payloads(replayed, scorer.encoders.raw_columns):
                began = time.perf_counter()
                response = client.post("/score", json=body)
                latencies.append((time.perf_counter() - began) * 1_000)
                response.raise_for_status()
                probabilities.append(response.json()["fraud_probability"])
        online = pd.concat(scorer.seen)
        store.close()

    rows = min(config.api.parity_rows, len(replayed))
    online = online.iloc[:rows]
    batch = batch_features.loc[online.index, online.columns]
    mismatches, worst = 0, 0.0
    for column in online.columns:
        left, right = online[column], batch[column]
        if isinstance(right.dtype, pd.CategoricalDtype):
            a, b = left.astype(object), right.astype(object)
            mismatches += int((~((a == b) | (a.isna() & b.isna()))).sum())
            continue
        a_num, b_num = left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64")
        both_missing = np.isnan(a_num) & np.isnan(b_num)
        difference = np.where(both_missing, 0.0, np.abs(a_num - b_num))
        mismatches += int((~(difference <= config.api.parity_tolerance)).sum())
        worst = max(worst, float(np.nanmax(difference, initial=0.0)))
    expected = batch_probability.loc[online.index].to_numpy()
    probability_gap = float(np.max(np.abs(np.array(probabilities[:rows]) - expected)))
    return {
        "parity_rows": rows,
        "feature_mismatches": mismatches,
        "max_feature_difference": worst,
        "max_probability_difference": probability_gap,
        "latency_requests": len(latencies[: config.api.latency_requests]),
        "latency_p95_ms": float(np.percentile(latencies[: config.api.latency_requests], 95)),
    }
