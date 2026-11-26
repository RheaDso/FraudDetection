"""Synthetic IEEE-CIS-shaped data (spec section 16.1).

The values are random. Only the column layout and a deliberate fraud signal matter: fraud is
more likely on a new device, with an unusually high amount, and in bursts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400
FIRST_TRANSACTION_ID = 2_987_000
FRAUD_RATE = 0.035
IDENTITY_SHARE_LEGIT = 0.23
IDENTITY_SHARE_FRAUD = 0.70

PRODUCTS = ["W", "C", "R", "H", "S"]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "yahoo.com.mx"]
DEVICES = ["Windows", "iOS Device", "MacOS", "SAMSUNG SM-G960F", "Moto G", "HUAWEI P20"]

C_COLUMNS = [f"C{i}" for i in range(1, 15)]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]
V_COLUMNS = [f"V{i}" for i in range(1, 340)]
ID_COLUMNS = [f"id_{i:02d}" for i in range(1, 39)]

TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    *[f"card{i}" for i in range(1, 7)],
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *C_COLUMNS,
    *D_COLUMNS,
    *M_COLUMNS,
    *V_COLUMNS,
]
IDENTITY_COLUMNS = ["TransactionID", *ID_COLUMNS, "DeviceType", "DeviceInfo"]

# id_12 onwards is text in the real data apart from these.
NUMERIC_ID_NUMBERS = {*range(1, 12), 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 32}


def _customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer": np.arange(n),
            "card1": rng.choice(np.arange(1_000, 19_000), size=n, replace=False),
            "card2": rng.integers(100, 600, n).astype(float),
            "card3": rng.choice([150.0, 185.0], n, p=[0.9, 0.1]),
            "card4": rng.choice(["visa", "mastercard", "discover"], n, p=[0.6, 0.35, 0.05]),
            "card5": rng.choice([102.0, 117.0, 166.0, 226.0], n),
            "card6": rng.choice(["debit", "credit"], n, p=[0.7, 0.3]),
            "addr1": rng.integers(100, 540, n).astype(float),
            "addr2": 87.0,
            "email": rng.choice(EMAILS, n),
            "device": rng.choice(DEVICES, n),
            "product": rng.choice(PRODUCTS, n, p=[0.74, 0.12, 0.06, 0.06, 0.02]),
            "open_day": rng.integers(-300, 1, n),
            "log_amount": rng.normal(4.0, 0.7, n),
        }
    )


def _legit_events(
    rng: np.random.Generator, customers: pd.DataFrame, n: int, n_days: int
) -> pd.DataFrame:
    # Every customer appears at least once; the rest follow a skewed activity level.
    activity = rng.gamma(1.0, 1.0, len(customers))
    extra = rng.choice(len(customers), n - len(customers), p=activity / activity.sum())
    customer = np.concatenate([np.arange(len(customers)), extra])
    amount = np.exp(customers["log_amount"].to_numpy()[customer] + rng.normal(0, 0.35, n))
    # Occasional honest splurges keep the amount signal from being perfect.
    amount = np.where(rng.random(n) < 0.02, amount * rng.uniform(3, 6, n), amount)
    return pd.DataFrame(
        {
            "customer": customer,
            "seconds": rng.integers(0, n_days * SECONDS_PER_DAY, n),
            "amount": amount,
            "new_device": rng.random(n) < 0.03,
            "isFraud": 0,
        }
    )


def _fraud_events(
    rng: np.random.Generator, customers: pd.DataFrame, n: int, n_days: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    while len(rows) < n:
        customer = int(rng.integers(len(customers)))
        start = int(rng.integers(0, n_days * SECONDS_PER_DAY - 3_600))
        size = min(int(rng.integers(1, 6)), n - len(rows))
        new_device = bool(rng.random() < 0.8)
        multiplier = rng.uniform(3, 8) if rng.random() < 0.75 else rng.uniform(0.5, 1.5)
        for offset in np.sort(rng.integers(0, 3_600, size)):
            amount = np.exp(customers["log_amount"].iat[customer] + rng.normal(0, 0.25))
            rows.append(
                {
                    "customer": customer,
                    "seconds": start + int(offset),
                    "amount": amount * multiplier,
                    "new_device": new_device,
                    "isFraud": 1,
                }
            )
    return pd.DataFrame(rows)


def _v_block(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """V columns in groups that share a missing pattern and contain correlated clusters."""
    columns: dict[str, np.ndarray] = {}
    for start in range(0, len(V_COLUMNS), 20):
        group = V_COLUMNS[start : start + 20]
        missing = rng.random(n) < rng.uniform(0.0, 0.6)
        factors = rng.normal(0, 1, (n, 4))
        for position, name in enumerate(group):
            values = factors[:, position % 4] + rng.normal(0, 0.2, n)
            values[missing] = np.nan
            columns[name] = values
    return pd.DataFrame(columns)


def _identity(rng: np.random.Generator, events: pd.DataFrame, devices: np.ndarray) -> pd.DataFrame:
    n = len(events)
    # A new device must be visible somewhere, so those rows always carry identity.
    share = np.where(events["isFraud"] == 1, IDENTITY_SHARE_FRAUD, IDENTITY_SHARE_LEGIT)
    has_identity = (rng.random(n) < share) | events["new_device"].to_numpy()
    ids = events.loc[has_identity, "TransactionID"].to_numpy()
    m = len(ids)
    columns: dict[str, object] = {"TransactionID": ids}
    for number, name in enumerate(ID_COLUMNS, start=1):
        if number in NUMERIC_ID_NUMBERS:
            columns[name] = np.round(rng.normal(0, 10, m), 1)
        else:
            columns[name] = rng.choice(["Found", "NotFound", "New"], m)
    columns["DeviceType"] = rng.choice(["desktop", "mobile"], m)
    columns["DeviceInfo"] = devices[has_identity]
    return pd.DataFrame(columns)[IDENTITY_COLUMNS]


def generate(
    seed: int, n_txns: int = 20_000, n_uids: int = 1_500, n_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, identity) with the raw Kaggle column names."""
    rng = np.random.default_rng(seed)
    customers = _customers(rng, n_uids)
    n_fraud = round(n_txns * FRAUD_RATE)
    events = pd.concat(
        [
            _legit_events(rng, customers, n_txns - n_fraud, n_days),
            _fraud_events(rng, customers, n_fraud, n_days),
        ],
        ignore_index=True,
    )
    events = events.sort_values("seconds", kind="stable").reset_index(drop=True)
    events["TransactionID"] = np.arange(FIRST_TRANSACTION_ID, FIRST_TRANSACTION_ID + n_txns)
    rows = events.merge(customers, on="customer", how="left", validate="many_to_one")

    day = events["seconds"] // SECONDS_PER_DAY
    devices = np.where(
        events["new_device"],
        [f"NEWDEV-{number}" for number in rng.integers(0, 500, n_txns)],
        rows["device"],
    )
    d1 = (day - rows["open_day"]).astype(float)

    transactions = pd.DataFrame(
        {
            "TransactionID": events["TransactionID"],
            "isFraud": events["isFraud"].astype(int),
            # The real data starts one day after its hidden reference time.
            "TransactionDT": events["seconds"] + SECONDS_PER_DAY,
            "TransactionAmt": np.round(events["amount"].clip(lower=1.0), 2),
            "ProductCD": rows["product"],
            **{f"card{i}": rows[f"card{i}"] for i in range(1, 7)},
            "addr1": rows["addr1"].where(rng.random(n_txns) >= 0.10),
            "addr2": rows["addr2"],
            "dist1": np.where(rng.random(n_txns) < 0.6, np.nan, rng.exponential(50, n_txns)),
            "dist2": np.where(rng.random(n_txns) < 0.9, np.nan, rng.exponential(200, n_txns)),
            "P_emaildomain": rows["email"].where(rng.random(n_txns) >= 0.15),
            "R_emaildomain": rows["email"].where(rng.random(n_txns) >= 0.75),
        }
    )
    counts = {name: rng.poisson(2, n_txns).astype(float) for name in C_COLUMNS}
    deltas = {
        name: np.where(rng.random(n_txns) < 0.4, np.nan, rng.integers(0, 400, n_txns))
        for name in D_COLUMNS
    }
    deltas["D1"] = np.where(rng.random(n_txns) < 0.02, np.nan, d1)
    flags = {
        name: rng.choice(np.array(["T", "F", None], dtype=object), n_txns, p=[0.4, 0.2, 0.4])
        for name in M_COLUMNS
    }
    transactions = pd.concat(
        [transactions, pd.DataFrame({**counts, **deltas, **flags}), _v_block(rng, n_txns)], axis=1
    )[TRANSACTION_COLUMNS]
    return transactions, _identity(rng, events, devices)


def write_csvs(directory: Path, seed: int, **sizes: int) -> None:
    """Write the two training files the pipeline requires."""
    transactions, identity = generate(seed, **sizes)
    directory.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(directory / "train_transaction.csv", index=False)
    identity.to_csv(directory / "train_identity.csv", index=False)
