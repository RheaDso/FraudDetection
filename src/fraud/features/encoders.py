"""Train-fitted encoders (M3): frequencies, category levels and V-column reduction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud.config import Config
from fraud.features.static import STRING_FEATURES

log = logging.getLogger(__name__)

OTHER = "OTHER"
NOT_FEATURES = {"TransactionID", "isFraud", "TransactionDT", "day", "hour", "dow"}


def is_v_column(name: str) -> bool:
    return name.startswith("V") and name[1:].isdigit()


def reduce_v_columns(train: pd.DataFrame, threshold: float) -> tuple[list[str], pd.DataFrame]:
    """Keep one representative per correlated cluster within each missing pattern (FR-3.6).

    Returns the kept names and a log of every V column with the reason.
    """
    names = [c for c in train.columns if is_v_column(c)]
    groups: dict[bytes, list[str]] = {}
    for name in names:
        groups.setdefault(np.packbits(train[name].isna().to_numpy()).tobytes(), []).append(name)
    kept: list[str] = []
    records = []
    for number, columns in enumerate(groups.values()):
        correlation = train[columns].corr().abs()
        representatives: list[str] = []
        for name in columns:
            if train[name].nunique() < 2:
                records.append((name, number, "dropped", "constant on train"))
                continue
            match = next((r for r in representatives if correlation.at[name, r] > threshold), None)
            if match is None:
                representatives.append(name)
                records.append((name, number, "kept", ""))
            else:
                records.append((name, number, "dropped", f"correlated with {match}"))
        kept.extend(representatives)
    table = pd.DataFrame(records, columns=["column", "missing_group", "action", "reason"])
    return sorted(kept, key=lambda c: int(c[1:])), table


@dataclass
class Encoders:
    """Everything learned from TRAIN that is needed to build model features."""

    raw_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    frequencies: dict[str, dict[object, float]] = field(default_factory=dict)
    levels: dict[str, list[str]] = field(default_factory=dict)
    v_kept: list[str] = field(default_factory=list)
    v_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(
        self, frame: pd.DataFrame, static: pd.DataFrame, uid: pd.Series, config: Config
    ) -> Encoders:
        """Fit on TRAIN rows only. All three inputs must already be restricted to TRAIN."""
        features = config.features
        derived = {"day", "hour", "dow", "HAS_IDENTITY"}
        self.raw_columns = [c for c in frame.columns if c not in derived and c != "isFraud"]
        self.v_kept, self.v_log = reduce_v_columns(frame, features.v_corr_threshold)
        log.info("V reduction kept %d of %d columns", len(self.v_kept), len(self.v_log))
        categories = set(config.data.dtypes.category)
        kept = set(self.v_kept)
        usable = [
            c for c in frame.columns if c not in NOT_FEATURES and (not is_v_column(c) or c in kept)
        ]
        self.categorical_columns = [c for c in usable if c in categories] + STRING_FEATURES
        self.numeric_columns = [c for c in usable if c not in categories]
        for column in features.frequency_encode:
            values = uid if column == "UID" else frame[column]
            share = values.value_counts() / len(values)
            self.frequencies[column] = share[share > 0].to_dict()
        for column in self.categorical_columns:
            source = static if column in STRING_FEATURES else frame
            share = source[column].astype(object).value_counts() / len(source)
            if len(share) > features.max_levels_without_grouping:
                share = share[share >= features.rare_level_share]
            self.levels[column] = sorted(str(v) for v in share.index if v != OTHER)
        return self

    def transform(self, frame: pd.DataFrame, static: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
        """Frequency-encoded and level-grouped columns for any rows."""
        out: dict[str, object] = {}
        # Plain dict lookups: Series.map would rebuild the (large) UID table on every request.
        for column, table in self.frequencies.items():
            values = uid if column == "UID" else frame[column]
            out[f"FREQ_{column}"] = np.array([table.get(v, 0.0) for v in values.tolist()])
        for column in self.categorical_columns:
            source = static if column in STRING_FEATURES else frame
            values = source[column].to_numpy(dtype=object)
            categories = pd.Index([*self.levels[column], OTHER])
            codes = categories.get_indexer(values)
            codes[codes < 0] = len(categories) - 1  # unseen or rare
            codes[pd.isna(values)] = -1
            out[column] = pd.Categorical.from_codes(codes, categories=categories)
        return pd.DataFrame(out, index=frame.index)
