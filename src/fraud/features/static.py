"""Row-level features (M3). Each uses only the row itself."""

from __future__ import annotations

import numpy as np
import pandas as pd

M_COLUMNS = [f"M{i}" for i in range(1, 10)]
D_NORM_KEYS = [1, 4, 10, 15]
STRING_FEATURES = ["P_EMAIL_PROVIDER", "P_EMAIL_TLD", "DEVICE_BRAND"]


def _split_domain(domain: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = domain.astype(object).str.split(".", n=1, expand=False)
    return parts.str[0], parts.str[1]


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the section 7.1 features for each row of ``frame``."""
    amount = frame["TransactionAmt"].astype("float64")
    cents = (amount - np.floor(amount)).round(3)
    purchaser = frame["P_emaildomain"].astype(object)
    recipient = frame["R_emaildomain"].astype(object)
    both = purchaser.notna() & recipient.notna()
    provider, suffix = _split_domain(purchaser)
    device = frame["DeviceInfo"].astype(object)
    brand = device.str.lower().str.split(r"[\s\-_/;]+", n=1, regex=True).str[0]
    flags = frame.reindex(columns=M_COLUMNS).astype(object)
    out = {
        "AMT_LOG": np.log1p(amount),
        "AMT_CENTS": cents,
        "AMT_ROUND_FLAG": (cents == 0).astype("float32"),
        "HOUR": frame["hour"].astype("float32"),
        "DOW": frame["dow"].astype("float32"),
        "IS_NIGHT": (frame["hour"] <= 5).astype("float32"),
        "EMAIL_MATCH": (purchaser == recipient).astype("float32").where(both),
        "P_EMAIL_PROVIDER": provider,
        "P_EMAIL_TLD": suffix,
        "DEVICE_BRAND": brand,
        "M_TRUE_COUNT": (flags == "T").sum(axis=1).astype("float32"),
        "M_FALSE_COUNT": (flags == "F").sum(axis=1).astype("float32"),
        "M_NAN_COUNT": flags.isna().sum(axis=1).astype("float32"),
    }
    day = frame["day"].astype("float64")
    for k in D_NORM_KEYS:
        column = f"D{k}"
        delta = frame[column].astype("float64") if column in frame else np.nan
        out[f"D_NORM_{k}"] = day - delta
    return pd.DataFrame(out, index=frame.index)
