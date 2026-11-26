from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import ingest, split
from fraud.config import Config, load_config
from tests.fixtures import synthetic_txns as synthetic


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    config = load_config(root=tmp_path_factory.mktemp("data"))
    synthetic.write_csvs(config.paths.raw, config.seed, n_txns=2_000, n_uids=200, n_days=60)
    return config


@pytest.fixture(scope="module")
def frame(config: Config) -> pd.DataFrame:
    return ingest.ingest(config)


def test_fr_1_1_load_downcast_and_join(config: Config, frame: pd.DataFrame) -> None:
    assert len(frame) == 2_000
    assert frame["V1"].dtype == "float32"
    assert isinstance(frame["ProductCD"].dtype, pd.CategoricalDtype)
    assert frame["TransactionAmt"].dtype == "float64"
    assert len(pd.read_parquet(config.paths.interim / "train.parquet")) == 2_000


def test_fr_1_2_schema_passes_and_fails_on_duplicate_id(frame: pd.DataFrame) -> None:
    ingest.validate(frame)
    duplicated = frame.copy()
    duplicated.loc[1, "TransactionID"] = duplicated.loc[0, "TransactionID"]
    with pytest.raises(ingest.DataValidationError, match="TransactionID"):
        ingest.validate(duplicated)


@pytest.mark.parametrize(("column", "value"), [("TransactionAmt", 0.0), ("isFraud", 2)])
def test_fr_1_2_schema_rejects_bad_values(frame: pd.DataFrame, column: str, value: float) -> None:
    broken = frame.copy()
    broken.loc[0, column] = value
    with pytest.raises(ingest.DataValidationError, match=column):
        ingest.validate(broken)


def test_fr_1_3_derived_columns(frame: pd.DataFrame) -> None:
    assert {"day", "hour", "dow", "HAS_IDENTITY"} <= set(frame.columns)
    assert frame["hour"].between(0, 23).all()
    assert (frame["day"] == frame["TransactionDT"] // 86_400).all()
    assert 0 < frame["HAS_IDENTITY"].mean() < 1


def test_fr_1_4_reports_written(config: Config, frame: pd.DataFrame) -> None:
    quality = pd.read_csv(config.paths.reports / "data_quality.csv")
    assert len(quality) == frame.shape[1]
    volume = pd.read_csv(config.paths.reports / "daily_volume.csv")
    assert volume["transactions"].sum() == len(frame)
    assert volume["frauds"].sum() == frame["isFraud"].sum()


def test_fr_1_5_missing_file_or_column_fails_fast(tmp_path: Path) -> None:
    config = load_config(root=tmp_path)
    with pytest.raises(ingest.DataValidationError, match="missing"):
        ingest.ingest(config)
    synthetic.write_csvs(config.paths.raw, config.seed, n_txns=200, n_uids=20, n_days=5)
    path = config.paths.raw / "train_transaction.csv"
    pd.read_csv(path).drop(columns="card1").to_csv(path, index=False)
    with pytest.raises(ingest.DataValidationError, match="card1"):
        ingest.ingest(config)


def _days(config: Config) -> tuple[pd.Series, pd.Series]:
    day = pd.Series(np.repeat(np.arange(1, 101), 3))
    return day, split.time_split(day, config)


def test_fr_2_1_time_order_and_gap(config: Config) -> None:
    day, labels = _days(config)
    low = day.groupby(labels).min()
    high = day.groupby(labels).max()
    assert high["train"] < low["validation"] < high["validation"] < low["test"]
    assert low["validation"] - high["train"] - 1 == config.split.gap_days
    assert low["test"] - high["validation"] - 1 == config.split.gap_days
    assert set(labels) == {"train", "validation", "test", "gap"}


def test_fr_2_1_gap_too_large_is_rejected(config: Config) -> None:
    with pytest.raises(ValueError, match="gap_days"):
        split.time_split(pd.Series(np.arange(20)), config)


def test_fr_2_2_no_id_in_two_splits(run_config: Config) -> None:
    saved = pd.read_parquet(run_config.paths.processed / "split.parquet")
    assert saved["TransactionID"].is_unique


def test_fr_2_3_split_summary() -> None:
    meta = pd.DataFrame(
        {
            "split": ["train", "validation", "validation", "test"],
            "day": [1, 10, 10, 20],
            "isFraud": [1, 0, 1, 0],
            "TransactionAmt": [5.0, 6.0, 7.0, 8.0],
            "UID": ["a", "a", "b", "a"],
        }
    )
    table = split.split_summary(meta).set_index("split")
    assert table.loc["validation", "uid_share_seen_in_train"] == 0.5
    assert table.loc["test", "uid_share_seen_in_train"] == 1.0
    assert table.loc["validation", "fraud_amount"] == 7.0
    assert table.loc["validation", "rows"] == 2


def test_fr_2_4_random_split_is_flagged_and_separate(run_config: Config) -> None:
    assert "LEAKAGE" in split.RANDOM_SPLIT_FILE
    assert (run_config.paths.processed / split.RANDOM_SPLIT_FILE).is_file()
    report = (run_config.paths.reports / "model_report.md").read_text(encoding="utf-8")
    assert "Leakage demonstration" in report
    assert "LEAKS" in report
