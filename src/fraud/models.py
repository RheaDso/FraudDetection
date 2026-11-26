"""Models, imbalance handling and calibration (M5)."""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud.budget import evaluate_policy
from fraud.config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)


def _compact(features: pd.DataFrame) -> pd.DataFrame:
    """float32 numbers halve the memory the logistic regression needs."""
    return features.astype({c: "float32" for c in features.select_dtypes("float64").columns})


def fit_logreg(features: pd.DataFrame, is_fraud: pd.Series, config: Config) -> Pipeline:
    """Imputed, scaled and one-hot logistic regression, no class weights (FR-5.1)."""
    numeric = features.select_dtypes("number").columns.tolist()
    low_cardinality = [
        c
        for c in features.select_dtypes("category").columns
        if len(features[c].cat.categories) <= config.features.max_levels_without_grouping + 1
    ]
    prepare = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                low_cardinality,
            ),
        ]
    )
    model = Pipeline(
        [
            ("prepare", prepare),
            (
                "model",
                LogisticRegression(
                    max_iter=config.models.logreg_max_iter, random_state=config.seed
                ),
            ),
        ]
    )
    return model.fit(_compact(features), is_fraud)


def predict_logreg(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(_compact(features))[:, 1])


def _params(config: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "seed": config.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        **config.models.lgbm,
        **(extra or {}),
    }


def _training_rows(
    is_fraud: pd.Series, strategy: str, config: Config
) -> tuple[np.ndarray, dict[str, Any]]:
    """Row positions and extra parameters for one imbalance strategy (FR-5.3)."""
    labels = is_fraud.to_numpy()
    rows = np.arange(len(labels))
    positives, negatives = rows[labels == 1], rows[labels == 0]
    if strategy == "scale_pos_weight":
        return rows, {"scale_pos_weight": len(negatives) / len(positives)}
    if strategy == "undersample_1_5":
        rng = np.random.default_rng(config.seed)
        keep = min(len(negatives), int(len(positives) * config.models.undersample_ratio))
        chosen = rng.choice(negatives, size=keep, replace=False)
        return np.sort(np.concatenate([positives, chosen])), {}
    return rows, {}


def train_lgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    config: Config,
    strategy: str = "none",
    overrides: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Fit with early stopping on VALIDATION average precision (FR-5.2)."""
    rows, extra = _training_rows(train_y, strategy, config)
    params = _params(config, {**extra, **(overrides or {})})
    full = len(rows) == len(train_x)
    train_set = lgb.Dataset(train_x if full else train_x.iloc[rows], label=train_y.to_numpy()[rows])
    valid_set = lgb.Dataset(valid_x, label=valid_y.to_numpy(), reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.models.num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(config.models.early_stopping_rounds, verbose=False)],
    )
    return booster, params


def model_input(features: pd.DataFrame) -> np.ndarray:
    """The matrix LightGBM builds from a DataFrame, with categories as their codes.

    Valid because the encoders give every categorical column the same category list for
    training and serving. It avoids LightGBM's slow per-request DataFrame conversion.
    """
    categorical = np.array([isinstance(d, pd.CategoricalDtype) for d in features.dtypes])
    values = np.empty(features.shape)
    # One block conversion: reading hundreds of columns one at a time is slow per request.
    values[:, ~categorical] = features.loc[:, ~categorical].to_numpy(dtype="float64")
    for position in np.flatnonzero(categorical):
        codes = features.iloc[:, position].cat.codes.to_numpy().astype("float64")
        codes[codes < 0] = np.nan
        values[:, position] = codes
    return values


def predict_lgbm(booster: lgb.Booster, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    iteration = booster.best_iteration if booster.best_iteration > 0 else None
    threads = 1 if len(features) == 1 else 0
    return np.asarray(booster.predict(features, num_iteration=iteration, num_threads=threads))


def fit_calibrator(scores: np.ndarray, is_fraud: pd.Series) -> IsotonicRegression:
    """Isotonic regression on VALIDATION predictions (FR-5.5)."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, is_fraud)


def calibrate(calibrator: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return np.asarray(calibrator.predict(scores))


def compare_imbalance(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    valid_meta: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, str]:
    """Same parameters and early stopping for each strategy; pick by validation VDR at K."""
    rows = []
    for strategy in config.models.imbalance_strategies:
        booster, _ = train_lgbm(train_x, train_y, valid_x, valid_y, config, strategy)
        raw = predict_lgbm(booster, valid_x)
        probability = calibrate(fit_calibrator(raw, valid_y), raw)
        scored = valid_meta.assign(probability=probability, raw_score=raw)
        row: dict[str, Any] = {
            "strategy": strategy,
            "trees": booster.best_iteration,
            "pr_auc": average_precision_score(valid_y, raw),
        }
        for policy in ("probability", "expected_loss"):
            _, totals = evaluate_policy(scored, policy, config.budget.default_k, config.costs)
            row[f"vdr_{policy}"] = totals["vdr"]
        rows.append(row)
        log.info("imbalance %s: %s", strategy, row)
    table = pd.DataFrame(rows)
    column = f"vdr_{config.models.selection_policy}"
    best = table.sort_values([column, "pr_auc"], ascending=False, kind="stable").iloc[0]
    table["chosen"] = table["strategy"] == best["strategy"]
    return table, str(best["strategy"])


def tune_lgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    config: Config,
    strategy: str,
) -> tuple[lgb.Booster, dict[str, Any], pd.DataFrame]:
    """Optuna search on validation average precision (FR-5.4).

    The configured defaults are the first trial, so tuning cannot end up worse than them.
    """
    best: dict[str, Any] = {}
    trials: list[dict[str, Any]] = []

    def record(booster: lgb.Booster, params: dict[str, Any], chosen: dict[str, Any]) -> float:
        score = float(average_precision_score(valid_y, predict_lgbm(booster, valid_x)))
        trials.append(
            {"trial": len(trials), "pr_auc": score, "trees": booster.best_iteration, **chosen}
        )
        if not best or score > best["score"]:
            best.update(score=score, booster=booster, params=params)
        return score

    booster, params = train_lgbm(train_x, train_y, valid_x, valid_y, config, strategy)
    record(booster, params, {})

    def objective(trial: optuna.Trial) -> float:
        chosen = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 10.0, log=True),
        }
        model, used = train_lgbm(train_x, train_y, valid_x, valid_y, config, strategy, chosen)
        return record(model, used, chosen)

    if config.models.optuna_trials > 1:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
        )
        # The default run above counts towards the trial limit.
        study.optimize(objective, n_trials=config.models.optuna_trials - 1)
    return best["booster"], best["params"], pd.DataFrame(trials)


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"
    return result.stdout.strip()


@dataclass
class ModelBundle:
    """Fitted objects needed to score a transaction."""

    booster: lgb.Booster
    calibrator: IsotonicRegression
    logreg: Pipeline
    logreg_calibrator: IsotonicRegression

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "lgbm.txt")
        for name in ("calibrator", "logreg", "logreg_calibrator"):
            with (directory / f"{name}.pkl").open("wb") as handle:
                pickle.dump(getattr(self, name), handle)


def load_pickle(path: Path) -> Any:
    """Load a pickle written by this pipeline. Never point this at a file from elsewhere."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_metadata(directory: Path, metadata: dict[str, Any]) -> None:
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
