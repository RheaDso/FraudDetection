"""Small raw-column edge cases (spec section 16.1).

Days with fewer than K transactions or no fraud are covered by ``hand_fixture``.
"""

from __future__ import annotations

import pandas as pd

SINGLE_TXN_CARD = 2_000
UNSEEN_PRODUCT = "Z"


def edge_cases() -> pd.DataFrame:
    """Tied timestamps, a one-transaction card, missing identity and a late unseen category."""
    columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "addr1",
        "D1",
        "DeviceInfo",
    ]
    rows = [
        (1, 0, 86_400, 10.0, "W", 1_000, 200.0, 0.0, "Windows"),
        # 2 and 3 share a card and a timestamp, so neither may see the other.
        (2, 0, 90_000, 20.0, "W", 1_000, 200.0, 0.0, "Windows"),
        (3, 1, 90_000, 30.0, "W", 1_000, 200.0, 0.0, None),
        (4, 0, 95_000, 40.0, "C", SINGLE_TXN_CARD, 300.0, 5.0, None),
        # Late in time with a category that earlier rows never have.
        (5, 0, 50 * 86_400, 50.0, UNSEEN_PRODUCT, 1_000, 200.0, 49.0, "Windows"),
    ]
    return pd.DataFrame(rows, columns=columns)
