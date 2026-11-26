"""SHAP reason codes (M7).

LightGBM's ``pred_contrib`` returns exact TreeSHAP values, so per-alert explanations use it
directly; the ``shap`` library is used for the global plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

from fraud.config import PROJECT_ROOT

log = logging.getLogger(__name__)

REASON_CODES_PATH = PROJECT_ROOT / "config" / "reason_codes.yaml"


def load_reason_codes(path: Path = REASON_CODES_PATH) -> dict[str, str]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def contributions(booster: lgb.Booster, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """SHAP values in log-odds, shape (rows, features)."""
    iteration = booster.best_iteration if booster.best_iteration > 0 else None
    values = np.asarray(booster.predict(features, pred_contrib=True, num_iteration=iteration))
    return values[:, :-1]  # the last column is the base value


class Explainer:
    """Turns SHAP values into the top reasons pushing a score up."""

    def __init__(self, booster: lgb.Booster, codes: dict[str, str], top: int) -> None:
        self.booster = booster
        self.codes = codes
        self.top = top
        self.unmapped: set[str] = set()

    def message(self, feature: str) -> str:
        text = self.codes.get(feature)
        if text:
            return text
        if feature not in self.unmapped:
            self.unmapped.add(feature)
            log.info("no reason code for feature %s", feature)
        return feature

    def explain(
        self, features: pd.DataFrame, values_for_model: np.ndarray | None = None
    ) -> list[list[dict[str, str]]]:
        """One list of ``top`` reasons per row, strongest upward push first (FR-7.2).

        ``values_for_model`` is an optional pre-converted copy of ``features``.
        """
        if features.empty:
            return []
        values = contributions(
            self.booster, features if values_for_model is None else values_for_model
        )
        names = features.columns
        order = np.argsort(-values, axis=1, kind="stable")[:, : self.top]
        return [
            [{"feature": names[j], "message": self.message(names[j])} for j in row] for row in order
        ]


def global_plots(
    booster: lgb.Booster, sample: pd.DataFrame, directory: Path, top: int, seed: int
) -> list[str]:
    """Importance bar chart and beeswarm for a TRAIN sample (FR-7.1)."""
    directory.mkdir(parents=True, exist_ok=True)
    values = contributions(booster, sample)
    # The beeswarm colours by value, so categories become their codes.
    display = sample.copy()
    for column in display.select_dtypes("category").columns:
        display[column] = display[column].cat.codes
    paths = []
    for kind, name in (("bar", "shap_importance.png"), ("dot", "shap_beeswarm.png")):
        shap.summary_plot(
            values,
            display,
            plot_type=kind,
            max_display=top,
            show=False,
            rng=np.random.default_rng(seed),
        )
        plt.gcf().savefig(directory / name, dpi=110, bbox_inches="tight")
        plt.close("all")
        paths.append(name)
    return paths
