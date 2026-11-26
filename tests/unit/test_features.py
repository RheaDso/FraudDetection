import numpy as np
import pandas as pd
import pytest

from fraud import ingest
from fraud.config import Config, load_config
from fraud.features import build, encoders, history, static, uid
from tests.fixtures import synthetic_txns as synthetic


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def frame(config: Config, tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    local = load_config(root=tmp_path_factory.mktemp("features"))
    synthetic.write_csvs(local.paths.raw, local.seed, n_txns=1_500, n_uids=100, n_days=40)
    return ingest.load_raw(local, "train_transaction.csv", "train_identity.csv")


def _split(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(frame["day"] <= 25, "train", "test"), index=frame.index)


def _history(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    return history.compute_history(frame, uid.build_uid(frame), config)


def _tiny(times: list[int], amounts: list[float], **columns: list) -> pd.DataFrame:
    n = len(times)
    data = {
        "TransactionID": range(1, n + 1),
        "TransactionDT": times,
        "TransactionAmt": amounts,
        "card1": [1000] * n,
        "addr1": [200.0] * n,
        "DeviceInfo": ["a"] * n,
        "P_emaildomain": ["x.com"] * n,
    }
    return pd.DataFrame({**data, **columns})


def test_fr_3_1_frequency_encoding_uses_train_only(config: Config, frame: pd.DataFrame) -> None:
    split = _split(frame)
    _, fitted = build.build_features(frame, split, config)
    changed = frame.copy()
    changed.loc[split == "test", "card1"] = 99_999.0
    _, refitted = build.build_features(changed, split, config)
    assert fitted.frequencies["card1"] == refitted.frequencies["card1"]
    train = frame[split == "train"]
    value = train["card1"].iloc[0]
    assert fitted.frequencies["card1"][value] == pytest.approx((train["card1"] == value).mean())
    matrix, _ = build.build_features(changed, split, config)
    unseen = changed.loc[split == "test", "TransactionID"]
    assert (matrix.loc[unseen, "FREQ_card1"] == 0).all()


def test_fr_3_2_rare_and_unseen_levels_become_other() -> None:
    fitted = encoders.Encoders(categorical_columns=["x"], levels={"x": ["a", "b"]})
    values = pd.DataFrame({"x": ["a", "zzz", None, "b"]})
    out = fitted.transform(values, values, pd.Series(["u"] * 4))
    assert out["x"].astype(object).tolist()[:2] == ["a", "OTHER"]
    assert pd.isna(out["x"].iloc[2])
    assert out["x"].cat.categories.tolist() == ["a", "b", "OTHER"]


def test_fr_3_2_grouping_applies_above_level_limit(config: Config, frame: pd.DataFrame) -> None:
    many = frame.copy()
    names = ["common"] * 1_400 + [f"rare{i}" for i in range(100)]
    many["DeviceInfo"] = pd.Series(names, dtype="category")
    strict = config.model_copy(
        update={"features": config.features.model_copy(update={"rare_level_share": 0.01})}
    )
    split = pd.Series("train", index=many.index)
    _, fitted = build.build_features(many, split, strict)
    assert fitted.levels["DeviceInfo"] == ["common"]


def test_fr_3_3_single_pass_matches_brute_force(config: Config, frame: pd.DataFrame) -> None:
    keys = uid.build_uid(frame)
    out = history.compute_history(frame, keys, config)
    for i in range(0, len(frame), 37):
        row = frame.iloc[i]
        prior = frame[(keys == keys.iloc[i]) & (frame["TransactionDT"] < row["TransactionDT"])]
        recent = prior[prior["TransactionDT"] >= row["TransactionDT"] - 86_400]
        assert out["UID_TXN_COUNT_24H"].iloc[i] == len(recent)
        assert out["UID_AMT_SUM_24H"].iloc[i] == pytest.approx(recent["TransactionAmt"].sum())
        assert out["UID_IS_FIRST_TXN"].iloc[i] == float(prior.empty)
        if len(prior) >= 2:
            assert out["UID_AMT_MEAN_PRIOR"].iloc[i] == pytest.approx(
                prior["TransactionAmt"].mean()
            )
            assert out["UID_AMT_STD_PRIOR"].iloc[i] == pytest.approx(prior["TransactionAmt"].std())
        card = frame[
            (frame["card1"] == row["card1"])
            & (frame["TransactionDT"] < row["TransactionDT"])
            & (frame["TransactionDT"] >= row["TransactionDT"] - 86_400)
        ]
        assert out["CARD1_TXN_COUNT_24H"].iloc[i] == len(card)


@pytest.mark.parametrize("cut", [100, 700, 1_300])
def test_fr_3_4_future_rows_never_change_the_past(
    config: Config, frame: pd.DataFrame, cut: int
) -> None:
    before = _history(frame, config)
    changed = frame.copy()
    changed.loc[changed.index[cut:], "TransactionAmt"] *= 100
    changed.loc[changed.index[cut:], "addr1"] = 1.0
    perturbed = _history(changed, config)
    deleted = _history(frame.iloc[:cut], config)
    # addr1 is part of the UID, so compare only the untouched past.
    pd.testing.assert_frame_equal(before.iloc[:cut], perturbed.iloc[:cut])
    pd.testing.assert_frame_equal(before.iloc[:cut], deleted)


def test_fr_3_5_tied_timestamps_do_not_see_each_other(config: Config) -> None:
    frame = _tiny(
        [100, 500, 500, 600],
        [10.0, 20.0, 30.0, 40.0],
        day=[0] * 4,
        D1=[0.0] * 4,
        card2=[1.0] * 4,
        card5=[1.0] * 4,
    )
    out = _history(frame, config)
    assert out["UID_TXN_COUNT_1H"].tolist() == [0, 1, 1, 3]
    assert out["UID_AMT_SUM_24H"].tolist() == [0, 10, 10, 60]
    assert out["CARD1_TXN_COUNT_24H"].tolist() == [0, 1, 1, 3]
    assert out["SECS_SINCE_LAST_TXN"].iloc[2] == 400


def test_fr_3_6_v_reduction_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    train = pd.DataFrame(
        {
            "V1": base,
            "V2": base + rng.normal(0, 0.01, 500),
            "V3": rng.normal(size=500),
            "V4": np.ones(500),
            "V5": np.where(np.arange(500) < 100, np.nan, base),
        }
    )
    kept, table = encoders.reduce_v_columns(train, 0.75)
    assert kept == ["V1", "V3", "V5"]  # V5 has its own missing pattern
    assert dict(zip(table["column"], table["action"], strict=True))["V2"] == "dropped"
    assert encoders.reduce_v_columns(train, 0.75)[0] == kept


def test_fr_3_7_feature_matrix_is_clean(config: Config, frame: pd.DataFrame) -> None:
    matrix, _ = build.build_features(frame, _split(frame), config)
    assert not {"isFraud", "TransactionID", "TransactionDT", "UID", "day"} & set(matrix.columns)
    assert matrix.index.name == "TransactionID"
    assert matrix.index.is_unique and len(matrix) == len(frame)
    assert not np.isinf(matrix.select_dtypes("number").to_numpy(dtype="float64")).any()
    assert build.feature_dictionary(list(matrix.columns))["point_in_time_safe"].all()


def test_uid_formula_and_fallback() -> None:
    frame = pd.DataFrame(
        {
            "card1": [10.0, 10.0],
            "card2": [2.0, np.nan],
            "card5": [5.0, 5.0],
            "addr1": [7.0, np.nan],
            "D1": [3.0, 1.0],
            "day": [100, 100],
        }
    )
    assert uid.build_uid(frame).tolist() == ["10_7_97", "fb_10_na_5"]


def test_row_features_values() -> None:
    frame = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 3.25],
            "P_emaildomain": ["yahoo.com.mx", None],
            "R_emaildomain": ["yahoo.com.mx", "a.com"],
            "DeviceInfo": ["SAMSUNG SM-G960F", None],
            "M1": ["T", None],
            "M2": ["F", "T"],
            "hour": [3, 12],
            "dow": [1, 2],
            "day": [50, 50],
            "D1": [10.0, np.nan],
        }
    )
    out = static.row_features(frame)
    assert out["AMT_ROUND_FLAG"].tolist() == [1, 0]
    assert out["AMT_CENTS"].tolist() == [0, 0.25]
    assert out["IS_NIGHT"].tolist() == [1, 0]
    assert out["P_EMAIL_PROVIDER"].iloc[0] == "yahoo"
    assert out["P_EMAIL_TLD"].iloc[0] == "com.mx"
    assert out["DEVICE_BRAND"].iloc[0] == "samsung"
    assert out["EMAIL_MATCH"].iloc[0] == 1 and pd.isna(out["EMAIL_MATCH"].iloc[1])
    assert out["M_TRUE_COUNT"].tolist() == [1, 1]
    assert out["M_NAN_COUNT"].tolist() == [7, 8]
    assert out["D_NORM_1"].iloc[0] == 40
