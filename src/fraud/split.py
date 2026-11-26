"""Time-based split (M2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.config import Config

TRAIN, VALIDATION, TEST, GAP = "train", "validation", "test", "gap"
RANDOM_SPLIT_FILE = "split_random_LEAKAGE_DEMO_ONLY.parquet"


def time_split(day: pd.Series, config: Config) -> pd.Series:
    """Label each row train, validation, test or gap by its day (FR-2.1).

    The gap days are taken from the start of the validation and test windows.
    """
    first, last = int(day.min()), int(day.max())
    n_days = last - first + 1
    validation_start = first + round(n_days * config.split.train)
    test_start = first + round(n_days * (config.split.train + config.split.validation))
    gap = config.split.gap_days
    if validation_start + gap >= test_start or test_start + gap > last:
        raise ValueError(f"gap_days={gap} leaves no data in validation or test ({n_days} days)")
    labels = np.select(
        [
            day < validation_start,
            day < validation_start + gap,
            day < test_start,
            day < test_start + gap,
        ],
        [TRAIN, GAP, VALIDATION, GAP],
        default=TEST,
    )
    return pd.Series(labels, index=day.index, name="split")


def random_split(n_rows: int, config: Config) -> np.ndarray:
    """LEAKAGE DEMONSTRATION ONLY (FR-2.4). Never use this for model selection."""
    rng = np.random.default_rng(config.seed)
    shares = [config.split.train, config.split.validation, config.split.test]
    return rng.choice(np.array([TRAIN, VALIDATION, TEST]), size=n_rows, p=shares)


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per split: rows, days, fraud rate, fraud amount and UID overlap with TRAIN (FR-2.3).

    ``frame`` needs columns split, day, isFraud, TransactionAmt and UID.
    """
    train_uids = set(frame.loc[frame["split"] == TRAIN, "UID"])
    rows = []
    for name in (TRAIN, VALIDATION, TEST, GAP):
        part = frame[frame["split"] == name]
        if part.empty:
            continue
        uids = set(part["UID"])
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "days": part["day"].nunique(),
                "fraud_rate": part["isFraud"].mean(),
                "fraud_amount": part.loc[part["isFraud"] == 1, "TransactionAmt"].sum(),
                "uid_share_seen_in_train": len(uids & train_uids) / len(uids)
                if name in (VALIDATION, TEST)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
