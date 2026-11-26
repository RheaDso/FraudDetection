"""Rules baseline (M4)."""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    op: Literal[">", ">=", "<", "<=", "==", "in"]
    value: float | list[str] | None = None
    quantile: float | None = None


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    when: list[Condition]


def load_rules(path: Path) -> list[Rule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Rule.model_validate(item) for item in raw["rules"]]


def triggers(features: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """Boolean column per rule. Quantile thresholds must already be resolved."""
    out = {}
    for rule in rules:
        fired = np.ones(len(features), dtype=bool)
        for condition in rule.when:
            column = features[condition.feature]
            if condition.op == "in":
                assert isinstance(condition.value, list)
                met = column.isin(condition.value)
            else:
                if condition.value is None:
                    raise ValueError(f"rule {rule.name}: unfitted threshold")
                met = _OPERATORS[condition.op](column.astype("float64"), condition.value)
            fired &= met.to_numpy(dtype=bool)
        out[rule.name] = fired
    return pd.DataFrame(out, index=features.index)


class RuleSet:
    """Rules with thresholds and lift weights fitted on TRAIN."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules
        self.weights: dict[str, float] = {}

    def fit(self, features: pd.DataFrame, is_fraud: pd.Series) -> RuleSet:
        """Resolve quantiles, then weight each rule by its lift (FR-4.2)."""
        for rule in self.rules:
            for condition in rule.when:
                if condition.quantile is not None:
                    values = features[condition.feature].astype("float64")
                    condition.value = float(values.quantile(condition.quantile))
        fired = triggers(features, self.rules)
        base = float(is_fraud.mean())
        labels = is_fraud.to_numpy()
        for name in fired.columns:
            mask = fired[name].to_numpy()
            self.weights[name] = float(labels[mask].mean() / base) if mask.any() and base else 0.0
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Sum of the weights of the triggered rules."""
        fired = triggers(features, self.rules)
        weights = np.array([self.weights[name] for name in fired.columns])
        return fired.to_numpy(dtype="float64") @ weights

    def performance(
        self, features: pd.DataFrame, is_fraud: pd.Series, amount: pd.Series
    ) -> pd.DataFrame:
        """Trigger rate, precision and fraud value caught per rule (FR-4.4)."""
        fired = triggers(features, self.rules)
        labels = is_fraud.to_numpy() == 1
        amounts = amount.to_numpy()
        rows = []
        for name in fired.columns:
            mask = fired[name].to_numpy()
            rows.append(
                {
                    "rule": name,
                    "weight_lift": self.weights[name],
                    "trigger_rate": mask.mean(),
                    "precision": labels[mask].mean() if mask.any() else np.nan,
                    "fraud_value_caught": amounts[mask & labels].sum(),
                }
            )
        return pd.DataFrame(rows)
