from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from fraud import evaluate
from fraud.config import PROJECT_ROOT, Config

# Spec section 15.2.
ARTIFACTS = ["logreg.pkl", "lgbm.txt", "calibrator.pkl", "encoders.pkl", "metadata.json"]
REPORTS = [
    "data_quality.csv",
    "daily_volume.csv",
    "feature_dictionary.csv",
    "rules_performance.csv",
    "model_metrics.csv",
    "imbalance_comparison.csv",
    "budget_results.csv",
    "budget_sensitivity.csv",
    "cost_sensitivity.csv",
    "segments.csv",
    "psi_table.csv",
    "missed_frauds.csv",
    "leakage_demo.csv",
    "queue_summary.csv",
    "model_report.md",
    "model_report.html",
    "RUN_SUMMARY.md",
]


def _report(config: Config, name: str) -> pd.DataFrame:
    return pd.read_csv(config.paths.reports / name)


def test_smoke_produces_every_output(run_config: Config) -> None:
    missing = [n for n in ARTIFACTS if not (run_config.paths.artifacts / n).is_file()]
    missing += [n for n in REPORTS if not (run_config.paths.reports / n).is_file()]
    assert not missing
    assert list((run_config.paths.reports / "queues").glob("day_*.csv"))
    assert list((run_config.paths.reports / "figures").glob("*.png"))


def test_fr_7_4_case_studies(run_config: Config) -> None:
    report = (run_config.paths.reports / "model_report.md").read_text(encoding="utf-8")
    assert "Case studies" in report and "Caught fraud 1" in report
    assert "SYNTHETIC DATA" in report


def test_fr_8_1_model_metrics(run_config: Config) -> None:
    table = _report(run_config, "model_metrics.csv")
    assert len(table) == 6
    assert table.loc[table["model"] == "rules", "brier"].isna().all()
    assert table[["pr_auc", "roc_auc"]].apply(lambda c: c.between(0, 1)).all().all()


def test_fr_8_2_budget_results(run_config: Config) -> None:
    table = _report(run_config, "budget_results.csv")
    assert set(zip(table["model"], table["policy"], strict=True)) == set(
        evaluate.combinations(run_config)
    )


def test_fr_8_3_daily_chart(run_config: Config) -> None:
    assert (run_config.paths.reports / "figures" / "daily_performance.png").is_file()


def test_fr_8_4_segments(run_config: Config) -> None:
    table = _report(run_config, "segments.csv")
    assert {"ProductCD", "amount_band", "HAS_IDENTITY", "uid_status"} == set(table["dimension"])
    assert "flag_low_vdr" in table


def test_fr_8_5_psi_and_weekly_stability(run_config: Config) -> None:
    assert "train vs test" in _report(run_config, "psi_table.csv")["comparison"].tolist()
    assert len(_report(run_config, "weekly_pr_auc.csv")) >= 1
    sample = np.random.default_rng(0).normal(size=5_000)
    assert evaluate.psi(sample, sample) < 1e-9
    assert evaluate.psi(sample, sample + 1) > 0.25
    flags = evaluate.psi_table(sample, {"same": sample, "shifted": sample + 1}, run_config)
    assert flags["flag"].tolist() == ["ok", "alert"]


def test_fr_8_6_missed_frauds(run_config: Config) -> None:
    table = _report(run_config, "missed_frauds.csv")
    assert len(table) <= 20
    assert table["TransactionAmt"].is_monotonic_decreasing


def test_fr_8_7_leakage_demo(run_config: Config) -> None:
    table = _report(run_config, "leakage_demo.csv")
    assert len(table) == 2
    assert table["test_pr_auc"].between(0, 1).all()


def test_fr_9_1_one_queue_per_test_day(run_config: Config) -> None:
    summary = _report(run_config, "queue_summary.csv")
    files = sorted((run_config.paths.reports / "queues").glob("day_*.csv"))
    assert len(files) == len(summary)
    for path in files:
        queue = pd.read_csv(path)
        assert len(queue) <= run_config.budget.default_k
        assert {
            "rank",
            "TransactionID",
            "amount",
            "score",
            "expected_loss",
            "reason_1",
            "reason_2",
            "reason_3",
            "true_label",
        } <= set(queue.columns)
        assert queue[["reason_1", "reason_2", "reason_3"]].notna().all().all()


def test_fr_9_2_queue_summary_is_cumulative_in_time_order(run_config: Config) -> None:
    summary = _report(run_config, "queue_summary.csv")
    assert summary["day"].is_monotonic_increasing
    assert np.allclose(summary["cumulative_net_benefit"], summary["net_benefit"].cumsum())


def test_fr_9_3_investigator_app_renders(run_config: Config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRAUD_ROOT", str(Path(run_config.paths.reports).parent))
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "investigator.py"), default_timeout=60)
    app.run()
    assert not app.exception and not app.error
    assert "true_label" not in app.dataframe[0].value.columns
    assert len(app.metric) == 1
    app.toggle[0].set_value(True).run()
    assert "true_label" in app.dataframe[0].value.columns
    assert len(app.metric) == 4


def test_investigator_app_without_outputs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRAUD_ROOT", str(tmp_path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "investigator.py"), default_timeout=60)
    app.run()
    assert "Run the pipeline" in app.error[0].value
