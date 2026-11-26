"""Evaluation and error analysis (M6 reporting, M8).

``scored`` frames hold one row per VALIDATION or TEST transaction with the columns that
``budget`` needs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from fraud.budget import (
    capacity_breaches,
    choose_threshold,
    daily_metrics,
    evaluate_policy,
    select_alerts,
)
from fraud.config import Config, Costs

MODELS = ("lightgbm", "logreg")
RULES = "rules"


def model_metrics(
    labels: dict[str, np.ndarray], scores: dict[str, dict[str, tuple[np.ndarray, bool]]]
) -> pd.DataFrame:
    """PR-AUC, ROC-AUC and Brier per split and model (FR-8.1).

    ``scores[split][model]`` is (values, is_probability). Rules have no Brier score.
    """
    rows = []
    for split, by_model in scores.items():
        for model, (values, is_probability) in by_model.items():
            y = labels[split]
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "pr_auc": average_precision_score(y, values),
                    "roc_auc": roc_auc_score(y, values),
                    "brier": brier_score_loss(y, values) if is_probability else np.nan,
                }
            )
    return pd.DataFrame(rows)


def combinations(config: Config) -> list[tuple[str, str]]:
    """Every (model, policy) pair that is evaluated."""
    pairs: list[tuple[str, str]] = [
        (m, p) for m in MODELS for p in config.budget.policies if p != RULES
    ]
    if RULES in config.budget.policies:
        pairs.append((RULES, RULES))
    return pairs


def _frame_for(scored: dict[str, pd.DataFrame], model: str) -> pd.DataFrame:
    return scored["lightgbm" if model == RULES else model]


def budget_table(
    validation: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    ks: list[int],
    config: Config,
    costs: Costs | None = None,
) -> pd.DataFrame:
    """TEST business metrics for every model, policy and K (FR-6.4, FR-8.2).

    Thresholds for the fixed-threshold policy are chosen on VALIDATION for each K.
    """
    costs = costs or config.costs
    rows = []
    for k in ks:
        for model, policy in combinations(config):
            threshold = None
            if policy == "fixed_threshold":
                threshold = choose_threshold(validation[model], k)
            _, totals = evaluate_policy(
                _frame_for(test, model), policy, k, costs, threshold=threshold
            )
            rows.append({"model": model, "policy": policy, "k": k, **totals})
    return pd.DataFrame(rows)


def best_row(table: pd.DataFrame, k: int) -> pd.Series:
    """Highest net benefit at ``k``, excluding the rules baseline (FR-6.6)."""
    options = table[(table["k"] == k) & (table["model"] != RULES)]
    return options.sort_values("net_benefit", ascending=False, kind="stable").iloc[0]


def recommendation(sensitivity: pd.DataFrame, config: Config) -> str:
    """Paragraph generated from the numbers (FR-6.6)."""
    k = config.budget.default_k
    best = best_row(sensitivity, k)
    rules = sensitivity[(sensitivity["k"] == k) & (sensitivity["model"] == RULES)]
    text = (
        f"At K = {k} alerts per day, the highest TEST net benefit comes from {best['model']} "
        f"with the {best['policy']} policy: ₹{best['net_benefit']:,.0f}, value detection rate "
        f"{best['vdr']:.1%}, precision {best['precision']:.1%}."
    )
    if not rules.empty:
        gap = best["net_benefit"] - rules.iloc[0]["net_benefit"]
        direction = "more" if gap >= 0 else "less"
        text += (
            f" That is ₹{abs(gap):,.0f} {direction} than the rules baseline "
            f"(value detection rate {rules.iloc[0]['vdr']:.1%})."
        )
    changes = []
    for other in sorted(sensitivity["k"].unique()):
        winner = best_row(sensitivity, other)
        if (winner["model"], winner["policy"]) != (best["model"], best["policy"]):
            changes.append(f"K = {other}: {winner['model']} / {winner['policy']}")
    text += (
        f" The winner changes at {'; '.join(changes)}."
        if changes
        else " The same combination wins at every other K tested."
    )
    return text + (
        " Costs are assumptions. The winner is picked using TEST results, as the specification "
        "requires, so its figures are likely to be somewhat optimistic."
    )


def cost_sensitivity(
    validation: dict[str, pd.DataFrame], test: dict[str, pd.DataFrame], config: Config
) -> pd.DataFrame:
    """Winner when the review cost is halved or doubled (section 10.4)."""
    rows = []
    k = config.budget.default_k
    for multiplier in config.evaluation.review_cost_multipliers:
        costs = config.costs.model_copy(
            update={
                "review_cost_per_alert_inr": config.costs.review_cost_per_alert_inr * multiplier
            }
        )
        table = budget_table(validation, test, [k], config, costs)
        best = best_row(table, k)
        rules = table[table["model"] == RULES]
        rows.append(
            {
                "review_cost_inr": costs.review_cost_per_alert_inr,
                "winner_model": best["model"],
                "winner_policy": best["policy"],
                "net_benefit": best["net_benefit"],
                "rules_net_benefit": rules.iloc[0]["net_benefit"] if not rules.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def segments(
    test: pd.DataFrame, alerts: pd.DataFrame, train_amounts: pd.Series, config: Config
) -> pd.DataFrame:
    """Fraud rate, precision and VDR by segment (FR-8.4).

    ``test`` needs ProductCD, HAS_IDENTITY, TransactionAmt, isFraud and seen_uid.
    """
    quantiles = np.linspace(0, 1, config.evaluation.amount_bands + 1)
    edges = np.unique(train_amounts.quantile(quantiles).to_numpy())
    edges[0], edges[-1] = -np.inf, np.inf
    labelled = test.assign(
        amount_band=pd.cut(test["TransactionAmt"], edges).astype(str),
        uid_status=np.where(test["seen_uid"], "seen", "new"),
        alerted=test["TransactionID"].isin(alerts["TransactionID"]),
    )
    fraud_value = labelled["TransactionAmt"].where(labelled["isFraud"] == 1, 0.0)
    labelled["fraud_value"] = fraud_value
    labelled["value_caught"] = fraud_value.where(labelled["alerted"], 0.0)
    overall = labelled["value_caught"].sum() / fraud_value.sum() if fraud_value.sum() else np.nan
    rows = []
    for dimension in ("ProductCD", "amount_band", "HAS_IDENTITY", "uid_status"):
        for value, part in labelled.groupby(dimension, observed=True):
            alerted = part[part["alerted"]]
            total = part["fraud_value"].sum()
            vdr = part["value_caught"].sum() / total if total else np.nan
            rows.append(
                {
                    "dimension": dimension,
                    "segment": value,
                    "rows": len(part),
                    "fraud_rate": part["isFraud"].mean(),
                    "alerts": len(alerted),
                    "precision": alerted["isFraud"].mean() if len(alerted) else np.nan,
                    "vdr": vdr,
                    "flag_low_vdr": bool(vdr < config.evaluation.segment_flag_ratio * overall),
                }
            )
    return pd.DataFrame(rows)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index, with bins from the expected distribution's quantiles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]
    e = np.bincount(np.searchsorted(inner, expected, side="right"), minlength=len(inner) + 1)
    a = np.bincount(np.searchsorted(inner, actual, side="right"), minlength=len(inner) + 1)
    e_share = np.clip(e / len(expected), 1e-6, None)
    a_share = np.clip(a / len(actual), 1e-6, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_table(
    train_scores: np.ndarray, others: dict[str, np.ndarray], config: Config
) -> pd.DataFrame:
    """PSI of the score distribution against TRAIN (FR-8.5)."""
    rows = []
    for name, scores in others.items():
        value = psi(train_scores, scores)
        flag = "ok"
        if value >= config.evaluation.psi_alert:
            flag = "alert"
        elif value >= config.evaluation.psi_warn:
            flag = "warn"
        rows.append({"comparison": f"train vs {name}", "psi": value, "flag": flag})
    return pd.DataFrame(rows)


def weekly_pr_auc(test: pd.DataFrame) -> pd.DataFrame:
    week = (test["day"] - test["day"].min()) // 7 + 1
    rows = []
    for number, part in test.groupby(week):
        frauds = int(part["isFraud"].sum())
        rows.append(
            {
                "week": number,
                "rows": len(part),
                "frauds": frauds,
                "pr_auc": average_precision_score(part["isFraud"], part["raw_score"])
                if frauds
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missed_frauds(test: pd.DataFrame, alerts: pd.DataFrame, top: int) -> pd.DataFrame:
    """Highest-value frauds that were not alerted (FR-8.6)."""
    missed = test[(test["isFraud"] == 1) & ~test["TransactionID"].isin(alerts["TransactionID"])]
    return missed.sort_values(["TransactionAmt", "TransactionID"], ascending=[False, True]).head(
        top
    )


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close("all")


def plot_pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(y, values)
        plt.plot(recall, precision, label=f"{name} ({average_precision_score(y, values):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall on TEST")
    plt.legend()
    _save(path)


def plot_vs_k(sensitivity: pd.DataFrame, column: str, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for (model, policy), part in sensitivity.groupby(["model", "policy"]):
        label = policy if model == RULES else f"{model} / {policy}"
        plt.plot(part["k"], part[column], marker="o", label=label)
    plt.xlabel("Alert budget K per day")
    plt.ylabel(title)
    plt.legend(fontsize=8)
    _save(path)


def plot_calibration(y: np.ndarray, before: np.ndarray, after: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, values in (("before", before), ("after", after)):
        observed, predicted = calibration_curve(
            y, np.clip(values, 0, 1), n_bins=10, strategy="quantile"
        )
        plt.plot(predicted, observed, marker="o", label=f"{name} isotonic")
    limit = plt.gca().get_xlim()[1]
    plt.plot([0, limit], [0, limit], linestyle="--", color="grey", label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.legend()
    _save(path)


def plot_daily_alerts(test: pd.DataFrame, threshold: float, k: int, path: Path) -> None:
    counts = (test["probability"] >= threshold).groupby(test["day"]).sum()
    plt.figure(figsize=(9, 4))
    plt.bar(counts.index, counts.to_numpy())
    plt.axhline(k, color="red", linestyle="--", label=f"capacity K = {k}")
    plt.xlabel("Day")
    plt.ylabel("Alerts raised by the fixed threshold")
    plt.legend()
    _save(path)


def plot_daily_performance(
    test: pd.DataFrame,
    rules_test: pd.DataFrame,
    policy: str,
    threshold: float | None,
    config: Config,
    path: Path,
) -> None:
    """Daily precision@K and VDR for the recommended setup against rules (FR-8.3)."""
    k = config.budget.default_k
    chosen = daily_metrics(test, select_alerts(test, policy, k, threshold=threshold), config.costs)
    rules = daily_metrics(rules_test, select_alerts(rules_test, RULES, k), config.costs)
    _, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, column, title in zip(axes, ["precision", "vdr"], ["Precision@K", "VDR"], strict=True):
        axis.plot(chosen["day"], chosen[column], label="recommended")
        axis.plot(rules["day"], rules[column], label="rules")
        axis.set_ylabel(title)
        axis.legend()
    axes[-1].set_xlabel("Day")
    _save(path)


def breaches_table(
    validation: pd.DataFrame, test: pd.DataFrame, k: int
) -> tuple[float, pd.DataFrame]:
    """Fixed threshold from VALIDATION and its capacity breaches on TEST (FR-6.5)."""
    threshold = choose_threshold(validation, k)
    result = capacity_breaches(test, threshold, k)
    table = pd.DataFrame(
        [
            {
                "k": k,
                "threshold": threshold,
                "test_days": test["day"].nunique(),
                "breach_days": result.breach_days,
                "max_overflow": result.max_overflow,
                "underused_days": result.underused_days,
            }
        ]
    )
    return threshold, table
