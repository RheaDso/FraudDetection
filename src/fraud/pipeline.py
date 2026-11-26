"""Command-line entry point: check-data, run (M1-M11) and smoke (synthetic data)."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from fraud import evaluate, models, parity, report
from fraud.api.scorer import STORE_FILE
from fraud.budget import select_alerts
from fraud.config import PROJECT_ROOT, Config, load_config, smoke_config
from fraud.explain import Explainer, global_plots, load_reason_codes
from fraud.feature_store import FeatureStore
from fraud.features.build import assemble, build_features, feature_dictionary
from fraud.features.history import HistoryStates, compute_history
from fraud.features.uid import build_uid
from fraud.ingest import ingest, load_raw
from fraud.rules import RuleSet, load_rules
from fraud.simulate import format_reasons, replay
from fraud.split import (
    RANDOM_SPLIT_FILE,
    TEST,
    TRAIN,
    VALIDATION,
    random_split,
    split_summary,
    time_split,
)

log = logging.getLogger(__name__)

DOWNLOAD_STEPS = """Download steps:
  1. Sign in to Kaggle and open the competition "IEEE-CIS Fraud Detection".
  2. Accept the competition rules.
  3. Download the data and unzip the CSV files into: {raw}
The pipeline never downloads data or reads Kaggle credentials."""

CHECKS = {
    "fraud": "confirm with the cardholder, review recent activity on the card and block if "
    "unrecognised.",
    "false": "compare with the card's past purchases and devices; a genuine unusual purchase "
    "should be cleared quickly.",
}


def missing_data_files(config: Config) -> list[str]:
    """Return the required raw files that are not present."""
    return [name for name in config.data.required_files if not (config.paths.raw / name).is_file()]


def check_data(config: Config) -> int:
    """Report whether the required raw files exist; non-zero exit code if not."""
    missing = missing_data_files(config)
    if not missing:
        print(f"All required files found in {config.paths.raw}")
        return 0
    print(f"Missing required files: {', '.join(missing)}", file=sys.stderr)
    print(DOWNLOAD_STEPS.format(raw=config.paths.raw), file=sys.stderr)
    return 1


@contextmanager
def _stage(name: str, runtimes: list[dict[str, Any]]) -> Iterator[None]:
    log.info("stage: %s", name)
    began = time.perf_counter()
    yield
    seconds = round(time.perf_counter() - began, 1)
    runtimes.append({"stage": name, "seconds": seconds})
    log.info("stage %s took %.1f s", name, seconds)


def _case_studies(alerts: pd.DataFrame) -> list[dict[str, Any]]:
    """Three caught frauds and two false alerts, highest scores first (FR-7.4)."""
    cases = []
    reason_columns = [c for c in alerts.columns if c.startswith("reason_")]
    for label, key, title, count in (
        (1, "fraud", "Caught fraud", 3),
        (0, "false", "False alert", 2),
    ):
        chosen = alerts[alerts["isFraud"] == label].sort_values(
            ["probability", "TransactionID"], ascending=[False, True]
        )
        for number, row in enumerate(chosen.head(count).to_dict("records"), start=1):
            cases.append(
                {
                    "title": f"{title} {number}",
                    "id": row["TransactionID"],
                    "amount": row["TransactionAmt"],
                    "score": row["probability"],
                    "reasons": [row[c] for c in reason_columns],
                    "checks": CHECKS[key],
                }
            )
    return cases


def _score_unlabelled(
    config: Config, states: HistoryStates, encoders: Any, booster: Any
) -> np.ndarray | None:
    """Raw scores for the optional unlabelled Kaggle test file, used only for PSI."""
    transaction_file, identity_file = config.data.optional_files
    if not all((config.paths.raw / f).is_file() for f in config.data.optional_files):
        return None
    frame = load_raw(config, transaction_file, identity_file, labelled=False)
    for column in encoders.raw_columns:
        if column not in frame:
            frame[column] = np.nan
    uid = build_uid(frame)
    history = compute_history(frame, uid, config, states)
    return models.predict_lgbm(booster, assemble(frame, history, uid, encoders))


def run(config: Config, *, synthetic: bool = False) -> None:
    """Run the whole pipeline."""
    paths = config.paths
    for directory in (paths.processed, paths.artifacts, paths.reports / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    figures = paths.reports / "figures"
    runtimes: list[dict[str, Any]] = []

    def write(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(paths.reports / name, index=False)

    with _stage("ingest", runtimes):
        data = ingest(config)

    with _stage("split", runtimes):
        split = time_split(data["day"], config)
        pd.DataFrame({"TransactionID": data["TransactionID"], "split": split}).to_parquet(
            paths.processed / "split.parquet", index=False
        )
        random_labels = random_split(len(data), config)
        pd.DataFrame({"TransactionID": data["TransactionID"], "split": random_labels}).to_parquet(
            paths.processed / RANDOM_SPLIT_FILE, index=False
        )

    with _stage("features", runtimes):
        states = HistoryStates()
        features, encoders = build_features(data, split, config, states)
        features.to_parquet(paths.processed / "features.parquet")
        with (paths.artifacts / "encoders.pkl").open("wb") as handle:
            pickle.dump(encoders, handle)
        write(feature_dictionary(list(features.columns)), "feature_dictionary.csv")
        write(encoders.v_log, "v_reduction.csv")

    meta = data[
        [
            "TransactionID",
            "TransactionDT",
            "day",
            "TransactionAmt",
            "isFraud",
            "ProductCD",
            "HAS_IDENTITY",
        ]
    ].copy()
    # Same labels as the feature index, left unnamed so TransactionID stays unambiguous.
    meta.index = features.index.rename(None)
    meta["UID"] = build_uid(data).to_numpy()
    meta["split"] = split.to_numpy()
    summary = split_summary(meta)
    write(summary, "split_summary.csv")
    parts = {name: features[meta["split"] == name] for name in (TRAIN, VALIDATION, TEST)}
    info = {name: meta[meta["split"] == name] for name in (TRAIN, VALIDATION, TEST)}
    y = {name: info[name]["isFraud"] for name in info}

    # The raw table is large and not needed again until the end; reload it then.
    n_rows = len(data)
    del data

    with _stage("rules", runtimes):
        ruleset = RuleSet(load_rules(config.rules.file)).fit(parts[TRAIN], y[TRAIN])
        rule_columns = sorted({c.feature for r in ruleset.rules for c in r.when})
        rule_scores = {
            name: ruleset.score(parts[name][rule_columns]) for name in (VALIDATION, TEST)
        }
        rules_performance = ruleset.performance(
            parts[VALIDATION], y[VALIDATION], info[VALIDATION]["TransactionAmt"]
        )
        write(rules_performance, "rules_performance.csv")

    def scored(name: str, probability: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
        return info[name].assign(
            probability=probability, raw_score=raw, rule_score=rule_scores[name]
        )

    with _stage("logistic regression", runtimes):
        logreg = models.fit_logreg(parts[TRAIN], y[TRAIN], config)
        logreg_raw = {n: models.predict_logreg(logreg, parts[n]) for n in (VALIDATION, TEST)}
        logreg_calibrator = models.fit_calibrator(logreg_raw[VALIDATION], y[VALIDATION])

    with _stage("imbalance comparison", runtimes):
        imbalance, strategy = models.compare_imbalance(
            parts[TRAIN],
            y[TRAIN],
            parts[VALIDATION],
            y[VALIDATION],
            info[VALIDATION].assign(rule_score=rule_scores[VALIDATION]),
            config,
        )
        write(imbalance, "imbalance_comparison.csv")

    with _stage("tuning", runtimes):
        booster, params, trials = models.tune_lgbm(
            parts[TRAIN], y[TRAIN], parts[VALIDATION], y[VALIDATION], config, strategy
        )
        write(trials, "optuna_trials.csv")

    with _stage("calibration and saving", runtimes):
        raw = {n: models.predict_lgbm(booster, parts[n]) for n in (TRAIN, VALIDATION, TEST)}
        calibrator = models.fit_calibrator(raw[VALIDATION], y[VALIDATION])
        models.ModelBundle(booster, calibrator, logreg, logreg_calibrator).save(paths.artifacts)
        validation = {
            "lightgbm": scored(
                VALIDATION, models.calibrate(calibrator, raw[VALIDATION]), raw[VALIDATION]
            ),
            "logreg": scored(
                VALIDATION,
                models.calibrate(logreg_calibrator, logreg_raw[VALIDATION]),
                logreg_raw[VALIDATION],
            ),
        }
        test = {
            "lightgbm": scored(TEST, models.calibrate(calibrator, raw[TEST]), raw[TEST]),
            "logreg": scored(
                TEST,
                models.calibrate(logreg_calibrator, logreg_raw[TEST]),
                logreg_raw[TEST],
            ),
        }
        models.save_metadata(
            paths.artifacts,
            {
                "model_version": config.api.model_version,
                "features": list(features.columns),
                "params": params,
                "imbalance_strategy": strategy,
                "best_iteration": booster.best_iteration,
                "train_days": [int(info[TRAIN]["day"].min()), int(info[TRAIN]["day"].max())],
                "seed": config.seed,
                "git_hash": models.git_hash(),
                "rows": {name: len(part) for name, part in parts.items()},
                "synthetic_data": synthetic,
            },
        )

    with _stage("evaluation", runtimes):
        metrics = evaluate.model_metrics(
            {n: y[n].to_numpy() for n in (VALIDATION, TEST)},
            {
                name: {
                    "lightgbm": (frames["lightgbm"]["probability"].to_numpy(), True),
                    "logreg": (frames["logreg"]["probability"].to_numpy(), True),
                    "rules": (rule_scores[name], False),
                }
                for name, frames in ((VALIDATION, validation), (TEST, test))
            },
        )
        # Ranking metrics use raw scores, which have no isotonic ties.
        for split_name in (VALIDATION, TEST):
            for model_name, values in (("lightgbm", raw), ("logreg", logreg_raw)):
                row = (metrics["split"] == split_name) & (metrics["model"] == model_name)
                metrics.loc[row, "pr_auc"] = average_precision_score(
                    y[split_name], values[split_name]
                )
        write(metrics, "model_metrics.csv")

        k = config.budget.default_k
        ks = sorted(set(config.budget.sensitivity_k) | {k})
        sensitivity = evaluate.budget_table(validation, test, ks, config)
        write(sensitivity, "budget_sensitivity.csv")
        budget_results = sensitivity[sensitivity["k"] == k]
        write(budget_results, "budget_results.csv")
        costs_table = evaluate.cost_sensitivity(validation, test, config)
        write(costs_table, "cost_sensitivity.csv")
        best = evaluate.best_row(sensitivity, k)
        model_name, policy = str(best["model"]), str(best["policy"])
        # The API serves LightGBM, so its threshold is the one that is saved.
        api_threshold, breaches = evaluate.breaches_table(
            validation["lightgbm"], test["lightgbm"], k
        )
        write(breaches, "capacity_breaches.csv")
        threshold, _ = evaluate.breaches_table(validation[model_name], test[model_name], k)
        (paths.artifacts / "decision.json").write_text(
            json.dumps(
                {"model": model_name, "policy": policy, "k": k, "fixed_threshold": api_threshold},
                indent=2,
            ),
            encoding="utf-8",
        )

    with _stage("explanations and queues", runtimes):
        explainer = Explainer(booster, load_reason_codes(), config.explain.top_reasons)
        sample = parts[TRAIN].sample(
            min(config.explain.shap_background, len(parts[TRAIN])), random_state=config.seed
        )
        global_plots(booster, sample, figures, config.explain.top_global, config.seed)
        chosen = test[model_name]
        alerts, queue = replay(
            chosen,
            parts[TEST],
            policy,
            threshold if policy == "fixed_threshold" else None,
            explainer,
            config,
            paths.reports,
        )

    with _stage("error analysis", runtimes):
        train_uids = set(info[TRAIN]["UID"])
        segment_input = chosen.assign(seen_uid=chosen["UID"].isin(train_uids))
        segments = evaluate.segments(segment_input, alerts, info[TRAIN]["TransactionAmt"], config)
        write(segments, "segments.csv")
        others = {TEST: raw[TEST]}
        unlabelled = _score_unlabelled(config, states, encoders, booster)
        if unlabelled is not None:
            others["kaggle test file"] = unlabelled
        psi = evaluate.psi_table(raw[TRAIN], others, config)
        write(psi, "psi_table.csv")
        weekly = evaluate.weekly_pr_auc(test["lightgbm"])
        write(weekly, "weekly_pr_auc.csv")
        missed = evaluate.missed_frauds(chosen, alerts, config.evaluation.missed_top_n)
        missed_reasons = explainer.explain(parts[TEST].loc[missed["TransactionID"]])
        missed_table = pd.concat(
            [
                missed[["TransactionID", "day", "TransactionAmt", "probability"]].reset_index(
                    drop=True
                ),
                pd.DataFrame([format_reasons(r) for r in missed_reasons]),
            ],
            axis=1,
        )
        write(missed_table, "missed_frauds.csv")

    with _stage("leakage demonstration", runtimes):
        demo = {n: random_labels == n for n in (TRAIN, VALIDATION, TEST)}
        labels = meta["isFraud"]
        leaky, _ = models.train_lgbm(
            features[demo[TRAIN]],
            labels[demo[TRAIN]],
            features[demo[VALIDATION]],
            labels[demo[VALIDATION]],
            config,
            strategy,
            {k_: v for k_, v in params.items() if k_ in config.models.lgbm},
        )
        leakage = pd.DataFrame(
            [
                {
                    "split": "time (honest)",
                    "test_pr_auc": average_precision_score(y[TEST], raw[TEST]),
                },
                {
                    "split": "random (LEAKS, demo only)",
                    "test_pr_auc": average_precision_score(
                        labels[demo[TEST]], models.predict_lgbm(leaky, features[demo[TEST]])
                    ),
                },
            ]
        )
        write(leakage, "leakage_demo.csv")

    with _stage("charts", runtimes):
        test_y = y[TEST].to_numpy()
        evaluate.plot_pr_curves(
            test_y,
            {"lightgbm": raw[TEST], "logreg": logreg_raw[TEST], "rules": rule_scores[TEST]},
            figures / "pr_curves.png",
        )
        evaluate.plot_vs_k(sensitivity, "vdr", "Value detection rate", figures / "vdr_vs_k.png")
        evaluate.plot_vs_k(
            sensitivity, "net_benefit", "Net benefit (₹)", figures / "net_benefit_vs_k.png"
        )
        evaluate.plot_calibration(
            test_y,
            raw[TEST],
            test["lightgbm"]["probability"].to_numpy(),
            figures / "calibration.png",
        )
        evaluate.plot_daily_alerts(
            test["lightgbm"], api_threshold, k, figures / "daily_alerts_fixed_threshold.png"
        )
        evaluate.plot_daily_performance(
            chosen,
            test["lightgbm"],
            policy,
            threshold if policy == "fixed_threshold" else None,
            config,
            figures / "daily_performance.png",
        )

    with _stage("feature store, parity and latency", runtimes):
        data = pd.read_parquet(paths.interim / "train.parquet")
        store = FeatureStore(paths.artifacts / STORE_FILE)
        store.load(states if unlabelled is None else _train_states(data, config))
        store.close()
        batch_probability = pd.Series(
            test["lightgbm"]["probability"].to_numpy(), index=parts[TEST].index
        )
        api_result = parity.check(data, split, features, batch_probability, config)
        pd.DataFrame([api_result]).to_csv(paths.reports / "api_checks.csv", index=False)

    with _stage("report", runtimes):
        rules_row = budget_results[budget_results["model"] == "rules"].iloc[0]
        month = 30 / best["days"]
        final_selection = select_alerts(
            chosen, policy, k, threshold=threshold if policy == "fixed_threshold" else None
        )
        assert len(final_selection) == len(alerts)
        report.render(
            {
                "synthetic": synthetic,
                "recommendation": evaluate.recommendation(sensitivity, config),
                "monthly_benefit": best["net_benefit"] * month,
                "monthly_uplift": (best["net_benefit"] - rules_row["net_benefit"]) * month,
                "costs": config.costs,
                "k": k,
                "gap_days": config.split.gap_days,
                "split_summary": summary,
                "n_features": features.shape[1],
                "v_kept": len(encoders.v_kept),
                "v_total": len(encoders.v_log),
                "rules_performance": rules_performance,
                "strategy": strategy,
                "imbalance": imbalance,
                "model_metrics": metrics,
                "brier_before": brier_score_loss(y[TEST], np.clip(raw[TEST], 0, 1)),
                "brier_after": brier_score_loss(y[TEST], test["lightgbm"]["probability"]),
                "budget_results": budget_results[
                    [
                        "model",
                        "policy",
                        "alerts",
                        "precision",
                        "recall",
                        "vdr",
                        "net_benefit",
                        "p10_daily_precision",
                    ]
                ],
                "breaches": breaches,
                "cost_sensitivity": costs_table,
                "cases": _case_studies(alerts),
                "segments": segments,
                "weekly": weekly,
                "psi": psi,
                "missed": missed_table,
                "leakage": leakage,
                "git_hash": models.git_hash(),
                "n_rows": n_rows,
                "seed": config.seed,
                "targets": report.targets_table(metrics, sensitivity, best, psi, config),
                "parity": api_result,
                "p95_target": config.api.p95_latency_ms,
                "runtimes": pd.DataFrame(runtimes),
                "tests": report.test_status(PROJECT_ROOT / "build"),
            },
            paths.reports,
        )
    log.info("queue days: %d, reports in %s", len(queue), paths.reports)
    if api_result["feature_mismatches"] or (
        api_result["max_probability_difference"] > config.api.parity_tolerance
    ):
        raise RuntimeError(f"train-serve parity failed (release blocker): {api_result}")


def _train_states(data: pd.DataFrame, config: Config) -> HistoryStates:
    """State from the training file only, without the unlabelled rows."""
    states = HistoryStates()
    compute_history(data, build_uid(data), config, states)
    return states


def smoke(config: Config, workdir: Path) -> Config:
    """Run the pipeline on freshly generated synthetic data under ``workdir``."""
    # The generator lives with the test fixtures.
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.fixtures.synthetic_txns import write_csvs

    config = smoke_config(config)
    paths = config.paths.model_copy(
        update={name: workdir / value.relative_to(PROJECT_ROOT) for name, value in config.paths}
    )
    config = config.model_copy(update={"paths": paths})
    s = config.smoke
    write_csvs(
        config.paths.raw,
        config.seed,
        n_txns=s.transactions,
        n_uids=s.uids,
        n_days=s.days,
    )
    run(config, synthetic=True)
    return config


def main(argv: list[str] | None = None) -> int:
    """Run the requested command."""
    parser = argparse.ArgumentParser(prog="fraud.pipeline")
    parser.add_argument("command", choices=["check-data", "run", "smoke"])
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    # One line per replayed request would bury everything else.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config()
    if args.trials is not None:
        config = config.model_copy(
            update={"models": config.models.model_copy(update={"optuna_trials": args.trials})}
        )
    if args.command == "check-data":
        return check_data(config)
    if args.command == "smoke":
        smoke(config, PROJECT_ROOT / "build" / "smoke")
        return 0
    if check_data(config):
        return 1
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
