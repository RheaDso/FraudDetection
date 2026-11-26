from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import synthetic_txns as synthetic
from tests.fixtures.edge_cases import SINGLE_TXN_CARD, UNSEEN_PRODUCT, edge_cases
from tests.fixtures.hand_fixture import hand_fixture

Frames = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def data() -> Frames:
    return synthetic.generate(seed=42)


def test_shape_and_columns(data: Frames) -> None:
    transactions, identity = data
    assert len(transactions) == 20_000
    assert list(transactions.columns) == synthetic.TRANSACTION_COLUMNS
    assert list(identity.columns) == synthetic.IDENTITY_COLUMNS
    assert len(synthetic.TRANSACTION_COLUMNS) == 394
    assert len(synthetic.IDENTITY_COLUMNS) == 41


def test_keys_time_and_amounts(data: Frames) -> None:
    transactions, identity = data
    assert transactions["TransactionID"].is_unique
    assert identity["TransactionID"].is_unique
    assert identity["TransactionID"].isin(transactions["TransactionID"]).all()
    assert transactions["TransactionDT"].is_monotonic_increasing
    assert (transactions["TransactionAmt"] > 0).all()
    assert (transactions["TransactionDT"] // 86_400).nunique() == 60


def test_fraud_rate_identity_share_and_customers(data: Frames) -> None:
    transactions, identity = data
    assert transactions["isFraud"].mean() == pytest.approx(0.035, abs=0.001)
    assert 0.2 < len(identity) / len(transactions) < 0.3
    assert transactions["card1"].nunique() == 1_500


def test_fraud_signal_is_present(data: Frames) -> None:
    transactions, identity = data
    frame = transactions.merge(identity[["TransactionID", "DeviceInfo"]], how="left")
    typical = frame.groupby("card1")["TransactionAmt"].transform("median")
    high_amount = frame["TransactionAmt"] > 3 * typical
    new_device = frame["DeviceInfo"].str.startswith("NEWDEV", na=False)
    hour = frame["TransactionDT"] // 3_600
    burst = frame.groupby([frame["card1"], hour])["TransactionID"].transform("size") >= 2
    base = frame["isFraud"].mean()
    for signal in (high_amount, new_device, burst):
        assert frame.loc[signal, "isFraud"].mean() > 4 * base


def test_same_seed_same_data_different_seed_differs() -> None:
    sizes = {"n_txns": 500, "n_uids": 50, "n_days": 5}
    first = synthetic.generate(seed=1, **sizes)
    again = synthetic.generate(seed=1, **sizes)
    other = synthetic.generate(seed=2, **sizes)
    pd.testing.assert_frame_equal(first[0], again[0])
    pd.testing.assert_frame_equal(first[1], again[1])
    assert not first[0]["TransactionAmt"].equals(other[0]["TransactionAmt"])


def test_write_csvs(tmp_path: Path) -> None:
    synthetic.write_csvs(tmp_path, seed=42, n_txns=500, n_uids=50, n_days=5)
    assert len(pd.read_csv(tmp_path / "train_transaction.csv")) == 500
    assert (tmp_path / "train_identity.csv").is_file()


def test_hand_fixture_shape() -> None:
    frame = hand_fixture()
    assert len(frame) == 12
    assert frame.groupby("day").size().tolist() == [5, 5, 2]
    assert frame.groupby("day")["isFraud"].sum().tolist() == [2, 2, 0]


def test_edge_cases_present() -> None:
    frame = edge_cases()
    assert frame.duplicated(["card1", "TransactionDT"]).any()
    assert (frame["card1"] == SINGLE_TXN_CARD).sum() == 1
    assert frame["DeviceInfo"].isna().any()
    assert (frame["ProductCD"] == UNSEEN_PRODUCT).sum() == 1
