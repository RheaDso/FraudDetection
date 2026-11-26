"""Typed loader for config/config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
Policy = Literal["probability", "expected_loss", "fixed_threshold", "rules"]


class _Section(BaseModel):
    # Unknown keys are errors, so a typo cannot be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Section):
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path


class RequiredColumns(_Section):
    transaction: list[str]
    identity: list[str]


class Dtypes(_Section):
    float64: list[str]
    category: list[str]


class Data(_Section):
    required_files: list[str]
    optional_files: list[str]
    required_columns: RequiredColumns
    dtypes: Dtypes


class Split(_Section):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    gap_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> Split:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split shares must sum to 1, got {total}")
        return self


class LabelFeatures(_Section):
    enabled: bool
    label_delay_days: int = Field(ge=0)


class Features(_Section):
    rare_level_share: float = Field(ge=0, lt=1)
    max_levels_without_grouping: int = Field(gt=0)
    frequency_encode: list[str]
    v_corr_threshold: float = Field(gt=0, le=1)
    history_windows_hours: list[int]
    min_prior_for_zscore: int = Field(ge=2)
    label_features: LabelFeatures


class Rules(_Section):
    file: Path


class Models(_Section):
    imbalance_strategies: list[Literal["none", "scale_pos_weight", "undersample_1_5"]]
    undersample_ratio: float = Field(gt=0)
    selection_policy: Literal["probability", "expected_loss"]
    optuna_trials: int = Field(ge=0, le=40)
    calibration: Literal["isotonic"]
    num_boost_round: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    logreg_max_iter: int = Field(gt=0)
    lgbm: dict[str, float | int]


class Budget(_Section):
    default_k: int = Field(gt=0)
    sensitivity_k: list[int]
    policies: list[Policy]


class Costs(_Section):
    review_cost_per_alert_inr: float = Field(ge=0)
    friction_cost_per_fp_inr: float = Field(ge=0)
    recovery_rate: float = Field(ge=0, le=1)
    inr_per_unit: float = Field(gt=0)


class Explain(_Section):
    shap_background: int = Field(gt=0)
    top_reasons: int = Field(gt=0)
    top_global: int = Field(gt=0)


class Evaluation(_Section):
    psi_warn: float
    psi_alert: float
    amount_bands: int = Field(gt=1)
    segment_flag_ratio: float
    missed_top_n: int = Field(gt=0)
    review_cost_multipliers: list[float]


class Targets(_Section):
    pr_auc: float
    roc_auc: float
    vdr_uplift_vs_rules_pp: float


class RiskLevels(_Section):
    medium: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> RiskLevels:
        if self.medium >= self.high:
            raise ValueError("risk_levels.medium must be below risk_levels.high")
        return self


class Api(_Section):
    model_version: str
    risk_levels: RiskLevels
    allow_reset: bool
    p95_latency_ms: float = Field(gt=0)
    parity_rows: int = Field(gt=0)
    parity_tolerance: float = Field(gt=0)
    latency_requests: int = Field(gt=0)


class Smoke(_Section):
    transactions: int
    uids: int
    days: int
    optuna_trials: int
    num_boost_round: int
    default_k: int
    sensitivity_k: list[int]
    shap_background: int
    parity_rows: int
    latency_requests: int


class Config(_Section):
    seed: int
    paths: Paths
    data: Data
    split: Split
    features: Features
    rules: Rules
    models: Models
    budget: Budget
    costs: Costs
    explain: Explain
    evaluation: Evaluation
    targets: Targets
    api: Api
    smoke: Smoke


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    """Load and validate the config; relative paths are resolved against ``root``.

    ``FRAUD_CONFIG`` and ``FRAUD_ROOT`` override the defaults, so the API and the app can be
    pointed at another run.
    """
    path = path or os.environ.get("FRAUD_CONFIG") or DEFAULT_CONFIG_PATH
    root = root or Path(os.environ.get("FRAUD_ROOT") or PROJECT_ROOT)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = Config.model_validate(raw)
    paths = Paths(**{name: root / value for name, value in config.paths})
    # Rules are code-like and stay with the project even when data lives elsewhere.
    rules = Rules(file=PROJECT_ROOT / config.rules.file)
    return config.model_copy(update={"paths": paths, "rules": rules})


def smoke_config(config: Config) -> Config:
    """Apply the ``smoke`` overrides to a loaded config."""
    s = config.smoke
    return config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"optuna_trials": s.optuna_trials, "num_boost_round": s.num_boost_round}
            ),
            "budget": config.budget.model_copy(
                update={"default_k": s.default_k, "sensitivity_k": s.sensitivity_k}
            ),
            "explain": config.explain.model_copy(update={"shap_background": s.shap_background}),
            "api": config.api.model_copy(
                update={"parity_rows": s.parity_rows, "latency_requests": s.latency_requests}
            ),
        }
    )
