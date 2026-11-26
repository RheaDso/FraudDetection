"""Ingestion and validation (M1)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from fraud.config import Config

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
INTEGER_COLUMNS = {"TransactionID": "int64", "TransactionDT": "int64", "isFraud": "int8"}
DERIVED_COLUMNS = ["day", "hour", "dow", "HAS_IDENTITY"]


class DataValidationError(Exception):
    """A required file or column is missing, or the data breaks the schema."""


def dtype_for(column: str, config: Config) -> str:
    """Return the configured dtype of a raw column."""
    if column in INTEGER_COLUMNS:
        return INTEGER_COLUMNS[column]
    if column in config.data.dtypes.category:
        return "category"
    return "float64" if column in config.data.dtypes.float64 else "float32"


def _read_csv(path: Path, required: Iterable[str], config: Config) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"required file is missing: {path}")
    header = pd.read_csv(path, nrows=0).columns
    # The Kaggle test identity file writes id-01 where the training file writes id_01.
    names = {column: column.replace("id-", "id_") for column in header}
    missing = sorted(set(required) - set(names.values()))
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {', '.join(missing)}")
    dtypes = {original: dtype_for(name, config) for original, name in names.items()}
    try:
        frame = pd.read_csv(path, dtype=dtypes)
    except ValueError as error:
        raise DataValidationError(f"{path.name}: {error}") from error
    return frame.rename(columns=names)


def validate(frame: pd.DataFrame) -> None:
    """Check the key columns (FR-1.2). ``frame`` must already be sorted by time."""
    columns = {
        "TransactionID": pa.Column("int64", unique=True),
        "TransactionDT": pa.Column(
            "int64", pa.Check(lambda s: s.is_monotonic_increasing, error="not sorted by time")
        ),
        "TransactionAmt": pa.Column("float64", pa.Check.gt(0), nullable=False),
    }
    if "isFraud" in frame:
        columns["isFraud"] = pa.Column("int8", pa.Check.isin([0, 1]))
    try:
        pa.DataFrameSchema(columns).validate(frame[list(columns)], lazy=True)
    except pa.errors.SchemaErrors as error:
        summary = error.failure_cases[["column", "check"]].drop_duplicates()
        raise DataValidationError(
            f"schema check failed:\n{summary.to_string(index=False)}"
        ) from None


def add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add day, hour and dow derived from TransactionDT (FR-1.3)."""
    seconds = frame["TransactionDT"]
    day = seconds // SECONDS_PER_DAY
    derived = pd.DataFrame(
        {
            "day": day.astype("int32"),
            "hour": (seconds % SECONDS_PER_DAY // 3_600).astype("int8"),
            "dow": (day % 7).astype("int8"),
        },
        index=frame.index,
    )
    return pd.concat([frame, derived], axis=1)


def load_raw(
    config: Config, transaction_file: str, identity_file: str, *, labelled: bool = True
) -> pd.DataFrame:
    """Read one transaction/identity pair, left-join, validate and derive columns."""
    required = [c for c in config.data.required_columns.transaction if labelled or c != "isFraud"]
    transactions = _read_csv(config.paths.raw / transaction_file, required, config)
    identity = _read_csv(
        config.paths.raw / identity_file, config.data.required_columns.identity, config
    )
    if identity["TransactionID"].duplicated().any():
        raise DataValidationError(f"{identity_file} has duplicate TransactionID values")
    has_identity = transactions["TransactionID"].isin(identity["TransactionID"])
    frame = transactions.merge(identity, on="TransactionID", how="left")
    frame["HAS_IDENTITY"] = has_identity.to_numpy().astype(np.int8)
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable", ignore_index=True)
    validate(frame)
    return add_time_columns(frame)


def data_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing %, unique count, min and max."""
    rows = []
    for name in frame.columns:
        column = frame[name]
        numeric = pd.api.types.is_numeric_dtype(column)
        rows.append(
            {
                "column": name,
                "dtype": str(column.dtype),
                "missing_pct": round(float(column.isna().mean()) * 100, 4),
                "unique": int(column.nunique()),
                "min": column.min() if numeric else None,
                "max": column.max() if numeric else None,
            }
        )
    return pd.DataFrame(rows)


def daily_volume(frame: pd.DataFrame) -> pd.DataFrame:
    """Transactions, frauds, fraud rate and fraud amount per day."""
    fraud_amount = frame["TransactionAmt"].where(frame["isFraud"] == 1, 0.0)
    table = (
        frame.assign(fraud_amount=fraud_amount)
        .groupby("day")
        .agg(
            transactions=("TransactionID", "size"),
            frauds=("isFraud", "sum"),
            fraud_amount=("fraud_amount", "sum"),
        )
        .reset_index()
    )
    table["fraud_rate"] = table["frauds"] / table["transactions"]
    return table[["day", "transactions", "frauds", "fraud_rate", "fraud_amount"]]


def ingest(config: Config) -> pd.DataFrame:
    """Run M1 and write the interim table and data reports."""
    transaction_file, identity_file = config.data.required_files
    frame = load_raw(config, transaction_file, identity_file)
    log.info(
        "ingested %d rows, %.0f MB in memory",
        len(frame),
        frame.memory_usage(deep=True).sum() / 1e6,
    )
    config.paths.interim.mkdir(parents=True, exist_ok=True)
    config.paths.reports.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.paths.interim / "train.parquet", index=False)
    data_quality(frame).to_csv(config.paths.reports / "data_quality.csv", index=False)
    daily_volume(frame).to_csv(config.paths.reports / "daily_volume.csv", index=False)
    return frame
