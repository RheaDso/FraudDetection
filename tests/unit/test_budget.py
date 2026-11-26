import numpy as np
import pandas as pd
import pytest

from fraud import budget, evaluate
from fraud.config import Config, load_config
from tests.fixtures.hand_fixture import hand_fixture


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def _ids(frame: pd.DataFrame, day: int) -> list[int]:
    return frame.loc[frame["day"] == day, "TransactionID"].tolist()


def test_fr_6_1_probability_policy() -> None:
    alerts = budget.select_alerts(hand_fixture(), "probability", 2)
    assert [_ids(alerts, d) for d in (0, 1, 2)] == [[1, 2], [6, 7], [11, 12]]


def test_fr_6_1_expected_loss_prefers_high_amount_moderate_risk() -> None:
    alerts = budget.select_alerts(hand_fixture(), "expected_loss", 2)
    assert [_ids(alerts, d) for d in (0, 1)] == [[3, 4], [6, 8]]


def test_fr_6_1_rules_and_threshold_policies() -> None:
    assert _ids(budget.select_alerts(hand_fixture(), "rules", 2), 0) == [3, 1]
    alerts = budget.select_alerts(hand_fixture(), "fixed_threshold", 3, threshold=0.25)
    # Arrival order, capped at capacity.
    assert _ids(alerts, 0) == [1, 2, 3]
    with pytest.raises(ValueError):
        budget.select_alerts(hand_fixture(), "fixed_threshold", 2)
    with pytest.raises(ValueError):
        budget.select_alerts(hand_fixture(), "nonsense", 2)


def test_fr_6_1_day_with_fewer_than_k_rows() -> None:
    alerts = budget.select_alerts(hand_fixture(), "probability", 3)
    assert _ids(alerts, 2) == [11, 12]
    assert alerts.loc[alerts["day"] == 0, "rank"].tolist() == [1, 2, 3]


def test_fr_6_2_daily_metrics_match_hand_calculation(config: Config) -> None:
    daily, _ = budget.evaluate_policy(hand_fixture(), "probability", 2, config.costs)
    assert daily["alerts"].tolist() == [2, 2, 2]
    assert daily["true_frauds"].tolist() == [1, 1, 0]
    assert daily["value_alerted"].tolist() == [10, 200, 0]
    assert daily["precision"].tolist() == pytest.approx([0.5, 0.5, 0.0], abs=1e-9)
    assert daily["recall"].iloc[:2].tolist() == pytest.approx([0.5, 0.5], abs=1e-9)
    assert daily["vdr"].iloc[:2].tolist() == pytest.approx([10 / 1010, 0.4], abs=1e-9)
    # Zero-fraud day: recall and VDR are undefined, the cost is still paid.
    assert np.isnan(daily["recall"].iloc[2]) and np.isnan(daily["vdr"].iloc[2])
    # 0.9 * value * 85 - 150 per alert - 50 per false alert
    assert daily["net_benefit"].tolist() == pytest.approx([415, 14_950, -400], abs=1e-9)


def test_fr_6_3_period_metrics_sum_over_days(config: Config) -> None:
    _, totals = budget.evaluate_policy(hand_fixture(), "probability", 2, config.costs)
    assert totals["alerts"] == 6 and totals["true_frauds"] == 2 and totals["frauds"] == 4
    assert totals["vdr"] == pytest.approx(210 / 1510, abs=1e-9)
    assert totals["precision"] == pytest.approx(1 / 3, abs=1e-9)
    assert totals["recall"] == pytest.approx(0.5, abs=1e-9)
    assert totals["net_benefit"] == pytest.approx(14_965, abs=1e-9)
    assert totals["mean_daily_precision"] == pytest.approx(1 / 3, abs=1e-9)
    assert totals["p10_daily_precision"] == pytest.approx(0.1, abs=1e-9)


def test_fr_6_4_budget_sensitivity_outputs(run_config: Config) -> None:
    table = pd.read_csv(run_config.paths.reports / "budget_sensitivity.csv")
    assert set(run_config.budget.sensitivity_k) <= set(table["k"])
    assert len(table[table["k"] == run_config.budget.default_k]) == 7
    assert (run_config.paths.reports / "figures" / "vdr_vs_k.png").is_file()


def test_fr_6_5_capacity_breaches() -> None:
    result = budget.capacity_breaches(hand_fixture(), 0.25, 2)
    assert (result.breach_days, result.max_overflow, result.underused_days) == (1, 2, 0)
    assert budget.capacity_breaches(hand_fixture(), 0.25, 4).underused_days == 1


def test_fr_6_5_threshold_targets_average_k() -> None:
    frame = hand_fixture()
    threshold = budget.choose_threshold(frame, 2)
    assert (frame["probability"] >= threshold).sum() == 6


def _sensitivity() -> pd.DataFrame:
    rows = []
    for k, winner in ((100, "expected_loss"), (200, "probability")):
        for policy in ("expected_loss", "probability"):
            rows.append(("lightgbm", policy, k, 100.0 if policy == winner else 50.0, 0.5, 0.4))
        rows.append(("rules", "rules", k, 500.0, 0.2, 0.1))
    return pd.DataFrame(rows, columns=["model", "policy", "k", "net_benefit", "vdr", "precision"])


def test_fr_6_6_recommendation_from_numbers(config: Config) -> None:
    text = evaluate.recommendation(_sensitivity(), config)
    assert "expected_loss" in text
    assert "₹400 less than the rules baseline" in text
    assert "K = 200: lightgbm / probability" in text
