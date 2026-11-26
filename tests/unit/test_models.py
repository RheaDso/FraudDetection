import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from fraud import budget, explain, models, rules
from fraud.config import Config, load_config

FEATURES_FOR_RULES: dict[str, object] = {
    "TransactionAmt": 10.0,
    "UID_NEW_DEVICE": 0.0,
    "UID_IS_FIRST_TXN": 0.0,
    "UID_TXN_COUNT_1H": 0.0,
    "AMT_ZSCORE_UID": 0.0,
    "EMAIL_MATCH": 1.0,
    "P_EMAIL_PROVIDER": "gmail",
    "IS_NIGHT": 0.0,
    "UID_DISTINCT_ADDR_7D": 1.0,
}
TRIGGERS: dict[str, dict[str, object]] = {
    "high_amount": {"TransactionAmt": 5_000.0},
    "new_device": {"UID_NEW_DEVICE": 1.0},
    "velocity_1h": {"UID_TXN_COUNT_1H": 3.0},
    "amount_zscore": {"AMT_ZSCORE_UID": 4.0},
    "email_mismatch_risky_provider": {"EMAIL_MATCH": 0.0, "P_EMAIL_PROVIDER": "anonymous"},
    "night_high_amount": {"IS_NIGHT": 1.0, "TransactionAmt": 950.0},
    "first_txn_high_amount": {"UID_IS_FIRST_TXN": 1.0, "TransactionAmt": 980.0},
    "many_addresses_7d": {"UID_DISTINCT_ADDR_7D": 2.0},
}


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def _resolved_rules(config: Config) -> list[rules.Rule]:
    loaded = rules.load_rules(config.rules.file)
    train = pd.DataFrame([FEATURES_FOR_RULES] * 1_000)
    train["TransactionAmt"] = np.arange(1.0, 1_001.0)
    return rules.RuleSet(loaded).fit(train, pd.Series(np.zeros(1_000))).rules


def test_fr_4_1_rules_load_and_each_fires(config: Config) -> None:
    loaded = _resolved_rules(config)
    assert 6 <= len(loaded) <= 10
    assert {r.name for r in loaded} == set(TRIGGERS)
    rows = pd.DataFrame(
        [FEATURES_FOR_RULES] + [{**FEATURES_FOR_RULES, **TRIGGERS[r.name]} for r in loaded]
    )
    fired = rules.triggers(rows, loaded)
    assert not fired.iloc[0].any()
    for position, rule in enumerate(loaded, start=1):
        assert fired[rule.name].iloc[position], rule.name


def test_fr_4_2_weights_equal_lift() -> None:
    rule = rules.Rule(name="big", when=[rules.Condition(feature="x", op=">", value=5)])
    features = pd.DataFrame({"x": [10, 10, 1, 1, 1, 1, 1, 1]})
    labels = pd.Series([1, 0, 1, 0, 0, 0, 0, 0])
    fitted = rules.RuleSet([rule]).fit(features, labels)
    assert fitted.weights["big"] == pytest.approx(0.5 / 0.25)
    assert fitted.score(features).tolist() == [2.0, 2.0, 0, 0, 0, 0, 0, 0]


def test_fr_4_3_rules_share_the_budget_with_amount_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "day": 0,
            "TransactionAmt": [5.0, 50.0, 500.0],
            "isFraud": 0,
            "probability": 0.5,
            "rule_score": [1.0, 1.0, 0.0],
        }
    )
    assert budget.select_alerts(frame, "rules", 2)["TransactionID"].tolist() == [2, 1]


def test_fr_4_4_rules_performance_report(run_config: Config) -> None:
    table = pd.read_csv(run_config.paths.reports / "rules_performance.csv")
    assert {"rule", "trigger_rate", "precision", "fraud_value_caught"} <= set(table.columns)
    assert len(table) == 8


def _sample(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    group = pd.Categorical(rng.choice(["a", "b"], n), categories=["a", "b", "OTHER"])
    probability = 1 / (1 + np.exp(-(2 * x - 3 + (group == "b"))))
    return (
        pd.DataFrame({"x": x, "noise": rng.normal(size=n), "group": group}),
        pd.Series((rng.random(n) < probability).astype(int)),
    )


@pytest.fixture(scope="module")
def trained(config: Config) -> tuple[lgb.Booster, pd.DataFrame, pd.Series]:
    quick = config.model_copy(
        update={"models": config.models.model_copy(update={"num_boost_round": 60})}
    )
    train_x, train_y = _sample(4_000, 1)
    valid_x, valid_y = _sample(2_000, 2)
    booster, _ = models.train_lgbm(train_x, train_y, valid_x, valid_y, quick, "scale_pos_weight")
    return booster, valid_x, valid_y


def test_fr_5_1_logreg_saved(run_config: Config) -> None:
    model = models.load_pickle(run_config.paths.artifacts / "logreg.pkl")
    features = pd.read_parquet(run_config.paths.processed / "features.parquet").head(20)
    probability = models.predict_logreg(model, features)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert model.named_steps["model"].class_weight is None


def test_fr_5_2_saved_model_gives_identical_predictions(
    trained: tuple[lgb.Booster, pd.DataFrame, pd.Series], tmp_path: pytest.TempPathFactory
) -> None:
    booster, features, _ = trained
    booster.save_model(f"{tmp_path}/lgbm.txt")
    loaded = lgb.Booster(model_file=f"{tmp_path}/lgbm.txt")
    expected = models.predict_lgbm(booster, features)
    assert np.array_equal(loaded.predict(features), expected)
    # The fast array path used by the API must agree with the DataFrame path.
    assert np.array_equal(
        models.predict_lgbm(loaded, models.model_input(features)), loaded.predict(features)
    )


def test_fr_5_3_imbalance_comparison(run_config: Config) -> None:
    table = pd.read_csv(run_config.paths.reports / "imbalance_comparison.csv")
    assert set(table["strategy"]) == {"none", "scale_pos_weight", "undersample_1_5"}
    assert table["chosen"].sum() == 1


def test_fr_5_3_undersampling_ratio(config: Config) -> None:
    labels = pd.Series([1] * 10 + [0] * 990)
    rows, _ = models._training_rows(labels, "undersample_1_5", config)
    assert labels.iloc[rows].sum() == 10 and len(rows) == 60
    assert models._training_rows(labels, "scale_pos_weight", config)[1] == {"scale_pos_weight": 99}


def test_fr_5_4_trial_log(run_config: Config) -> None:
    trials = pd.read_csv(run_config.paths.reports / "optuna_trials.csv")
    assert 1 <= len(trials) <= 40
    assert len(trials) == run_config.models.optuna_trials


def test_fr_5_5_calibration(trained: tuple[lgb.Booster, pd.DataFrame, pd.Series]) -> None:
    booster, features, labels = trained
    raw = models.predict_lgbm(booster, features)
    calibrated = models.calibrate(models.fit_calibrator(raw, labels), raw)
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert brier_score_loss(labels, calibrated) < brier_score_loss(labels, raw)


def test_fr_5_6_metadata(run_config: Config) -> None:
    metadata = json.loads(
        (run_config.paths.artifacts / "metadata.json").read_text(encoding="utf-8")
    )
    assert {"features", "params", "train_days", "seed", "git_hash", "rows"} <= set(metadata)
    assert metadata["seed"] == 42


def test_fr_7_1_global_shap_plots(run_config: Config) -> None:
    for name in ("shap_importance.png", "shap_beeswarm.png"):
        assert (run_config.paths.reports / "figures" / name).stat().st_size > 1_000


def test_fr_7_2_three_reasons_with_fallback(
    trained: tuple[lgb.Booster, pd.DataFrame, pd.Series],
) -> None:
    booster, features, _ = trained
    explainer = explain.Explainer(booster, {"x": "Mapped message"}, 3)
    reasons = explainer.explain(features.head(5))
    assert all(len(r) == 3 and all(item["message"] for item in r) for r in reasons)
    messages = {item["feature"]: item["message"] for r in reasons for item in r}
    assert messages["x"] == "Mapped message"
    assert messages["noise"] == "noise"
    assert "noise" in explainer.unmapped
    assert explainer.explain(features.head(0)) == []


def test_fr_7_3_explanation_latency(run_config: Config) -> None:
    booster = lgb.Booster(model_file=run_config.paths.artifacts / "lgbm.txt")
    features = pd.read_parquet(run_config.paths.processed / "features.parquet").head(1)
    explainer = explain.Explainer(booster, {}, 3)
    values = models.model_input(features)
    explainer.explain(features, values)
    timings = []
    for _ in range(20):
        began = time.perf_counter()
        explainer.explain(features, values)
        timings.append(time.perf_counter() - began)
    assert np.median(timings) < 0.050
