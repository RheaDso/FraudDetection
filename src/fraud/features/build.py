"""Assemble the model matrix. The batch pipeline and the API both go through ``assemble``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.config import Config
from fraud.features.encoders import Encoders
from fraud.features.history import HistoryStates, compute_history
from fraud.features.static import STRING_FEATURES, row_features
from fraud.features.uid import build_uid
from fraud.split import TRAIN


def assemble(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    uid: pd.Series,
    encoders: Encoders,
    static: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine raw, row-level, history and encoded columns, indexed by TransactionID."""
    static = row_features(frame) if static is None else static
    matrix = pd.concat(
        [
            frame[encoders.numeric_columns],
            static.drop(columns=STRING_FEATURES),
            history,
            encoders.transform(frame, static, uid),
        ],
        axis=1,
    )
    numbers = matrix.select_dtypes("number")
    # Checked in blocks so the full data set is never copied into one large array.
    # A single row is checked in one go; the slicing itself is slow per request.
    width = 25 if len(numbers) > 1_000 else max(numbers.shape[1], 1)
    for first in range(0, numbers.shape[1], width):
        block = numbers.iloc[:, first : first + width]
        infinite = np.isinf(block.to_numpy(dtype="float64")).any(axis=0)
        for column in block.columns[infinite]:
            matrix[column] = matrix[column].replace([np.inf, -np.inf], np.nan)
    matrix.index = pd.Index(frame["TransactionID"].to_numpy(), name="TransactionID")
    return matrix


def build_features(
    frame: pd.DataFrame,
    split: pd.Series,
    config: Config,
    states: HistoryStates | None = None,
) -> tuple[pd.DataFrame, Encoders]:
    """Fit encoders on TRAIN and build features for every row (M3)."""
    uid = build_uid(frame)
    static = row_features(frame)
    history = compute_history(frame, uid, config, states)
    train = split == TRAIN
    encoders = Encoders().fit(frame[train], static[train], uid[train], config)
    return assemble(frame, history, uid, encoders, static), encoders


_FORMULAS = {
    "AMT_LOG": ("log1p(TransactionAmt)", "Size of the purchase on a log scale"),
    "AMT_CENTS": ("fractional part of TransactionAmt", "Odd cents can mean currency conversion"),
    "AMT_ROUND_FLAG": ("1 if amount is whole", "Round amounts"),
    "HOUR": ("hour from TransactionDT", "Time of day"),
    "DOW": ("day mod 7", "Day of week"),
    "IS_NIGHT": ("hour in 0-5", "Night-time purchase"),
    "EMAIL_MATCH": ("P_emaildomain == R_emaildomain", "Buyer and recipient email match"),
    "P_EMAIL_PROVIDER": ("domain before the first dot", "Email provider"),
    "P_EMAIL_TLD": ("domain after the first dot", "Email suffix"),
    "DEVICE_BRAND": ("first token of DeviceInfo", "Device family"),
    "M_TRUE_COUNT": ("count of T in M1-M9", "Matching identity checks"),
    "M_FALSE_COUNT": ("count of F in M1-M9", "Failed identity checks"),
    "M_NAN_COUNT": ("count of missing M1-M9", "Checks not available"),
    "UID_AMT_SUM_24H": ("sum of prior amounts, 24h", "Recent spend"),
    "UID_AMT_MEAN_PRIOR": ("expanding mean of prior amounts", "Usual spend"),
    "UID_AMT_STD_PRIOR": ("expanding std of prior amounts", "Spend variability"),
    "AMT_ZSCORE_UID": ("(amount - prior mean) / prior std", "Unusual amount for this card"),
    "AMT_TO_MAX_PRIOR": ("amount / max prior amount", "Compared with largest past purchase"),
    "SECS_SINCE_LAST_TXN": ("seconds since previous UID transaction", "Burst detection"),
    "UID_NEW_DEVICE": ("DeviceInfo not seen before for UID", "New device"),
    "UID_NEW_EMAIL": ("P_emaildomain not seen before for UID", "New email domain"),
    "UID_DISTINCT_ADDR_7D": ("distinct addr1 in prior 7 days", "Address changes"),
    "UID_IS_FIRST_TXN": ("no prior UID transaction", "No history"),
    "CARD1_TXN_COUNT_24H": ("prior card1 transactions, 24h", "Card velocity"),
}


def feature_dictionary(columns: list[str]) -> pd.DataFrame:
    """One row per model feature: name, formula, business meaning and safety flag."""
    rows = []
    for name in columns:
        if name in _FORMULAS:
            formula, meaning = _FORMULAS[name]
        elif name.startswith("UID_TXN_COUNT_"):
            formula, meaning = "prior UID transactions in window", "Velocity"
        elif name.startswith("D_NORM_"):
            formula, meaning = f"day - D{name.rsplit('_', 1)[1]}", "Stable anchor date"
        elif name.startswith("FREQ_"):
            formula, meaning = "TRAIN count of value / TRAIN rows", "How common the value is"
        else:
            formula, meaning = "raw column", "Supplied by the data provider (anonymised)"
        rows.append(
            {
                "name": name,
                "formula": formula,
                "business_meaning": meaning,
                "point_in_time_safe": True,
            }
        )
    return pd.DataFrame(rows)
