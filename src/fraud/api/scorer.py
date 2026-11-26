"""Score one transaction with the same feature code as the batch pipeline."""

from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.config import Config
from fraud.explain import Explainer, load_reason_codes
from fraud.feature_store import FeatureStore
from fraud.features.build import assemble
from fraud.features.uid import build_uid, card1_key
from fraud.ingest import SECONDS_PER_DAY, dtype_for
from fraud.models import calibrate, load_pickle, model_input, predict_lgbm

STORE_FILE = "feature_store.sqlite"


class InvalidField(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


class Scorer:
    def __init__(self, config: Config, store: FeatureStore | None = None) -> None:
        artifacts = config.paths.artifacts
        self.config = config
        self.booster = lgb.Booster(model_file=artifacts / "lgbm.txt")
        self.calibrator = load_pickle(artifacts / "calibrator.pkl")
        self.encoders = load_pickle(artifacts / "encoders.pkl")
        decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
        self.threshold = float(decision["fixed_threshold"])
        self.explainer = Explainer(self.booster, load_reason_codes(), config.explain.top_reasons)
        self.store = store or FeatureStore(artifacts / STORE_FILE)
        self._dtypes = {c: dtype_for(c, config) for c in self.encoders.raw_columns}

    def to_frame(self, raw: dict[str, Any]) -> pd.DataFrame:
        """A one-row frame with every training column in its training dtype."""
        columns: dict[str, Any] = {}
        for name, dtype in self._dtypes.items():
            value = raw.get(name)
            if dtype == "category":
                if value is not None and not isinstance(value, str):
                    raise InvalidField(name, "expected a string")
                columns[name] = np.array([value], dtype=object)
                continue
            if isinstance(value, str | bool):
                raise InvalidField(name, "expected a number")
            if value is not None and not np.isfinite(value):
                raise InvalidField(name, "expected a finite number")
            columns[name] = np.array([np.nan if value is None else value], dtype=dtype)
        seconds = int(raw["TransactionDT"])
        day = seconds // SECONDS_PER_DAY
        identity = any(
            raw.get(c) is not None
            for c in self.encoders.raw_columns
            if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")
        )
        columns["day"] = np.array([day], dtype="int32")
        columns["hour"] = np.array([seconds % SECONDS_PER_DAY // 3_600], dtype="int8")
        columns["dow"] = np.array([day % 7], dtype="int8")
        columns["HAS_IDENTITY"] = np.array([identity], dtype="int8")
        return pd.DataFrame(columns)

    def features(self, raw: dict[str, Any]) -> pd.DataFrame:
        """Build features from the store, then add this transaction to it."""
        frame = self.to_frame(raw)
        uid = build_uid(frame)
        windows = self.config.features.history_windows_hours
        ts = int(frame["TransactionDT"].iat[0])
        amount = float(frame["TransactionAmt"].iat[0])
        address, device, email = (
            None if pd.isna(v) else v
            for v in (frame[c].tolist()[0] for c in ("addr1", "DeviceInfo", "P_emaildomain"))
        )
        with self.store.transaction(uid.iat[0], card1_key(frame).iat[0]) as (state, card):
            values = state.features(
                ts, amount, device, email, windows, self.config.features.min_prior_for_zscore
            )
            values["CARD1_TXN_COUNT_24H"] = card.count_24h(ts)
            matrix = assemble(frame, pd.DataFrame([values]), uid, self.encoders)
            state.add(ts, amount, address, device, email)
            card.add(ts)
        return matrix

    def score(self, raw: dict[str, Any]) -> dict[str, Any]:
        matrix = self.features(raw)
        values = model_input(matrix)
        probability = float(calibrate(self.calibrator, predict_lgbm(self.booster, values))[0])
        levels = self.config.api.risk_levels
        risk = (
            "HIGH"
            if probability >= levels.high
            else ("MEDIUM" if probability >= levels.medium else "LOW")
        )
        return {
            "fraud_probability": probability,
            "expected_loss": probability * float(raw["TransactionAmt"]),
            "alert": probability >= self.threshold,
            "risk_level": risk,
            "reasons": self.explainer.explain(matrix, values)[0],
            "model_version": self.config.api.model_version,
        }
