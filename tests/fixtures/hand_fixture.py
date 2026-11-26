"""Hand-built 3-day, 12-transaction fixture for exact metric checks (spec section 16.1).

With K = 2:
  day 0: probability alerts 1, 2; expected loss alerts 3, 4 (a high-amount, moderate-risk
         transaction outranks low-amount, high-risk ones)
  day 1: five transactions, two frauds
  day 2: two transactions and no fraud
With K = 3, day 2 has fewer than K transactions.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["TransactionID", "day", "TransactionAmt", "isFraud", "probability", "rule_score"]

ROWS = [
    (1, 0, 10.0, 1, 0.90, 2.0),
    (2, 0, 20.0, 0, 0.80, 0.0),
    (3, 0, 1000.0, 1, 0.40, 5.0),
    (4, 0, 500.0, 0, 0.30, 1.0),
    (5, 0, 50.0, 0, 0.05, 0.0),
    (6, 1, 200.0, 1, 0.95, 4.0),
    (7, 1, 40.0, 0, 0.60, 3.0),
    (8, 1, 300.0, 1, 0.20, 0.0),
    (9, 1, 60.0, 0, 0.10, 0.0),
    (10, 1, 80.0, 0, 0.02, 1.0),
    (11, 2, 100.0, 0, 0.70, 2.0),
    (12, 2, 30.0, 0, 0.15, 0.0),
]


def hand_fixture() -> pd.DataFrame:
    """Return a fresh copy of the fixture."""
    return pd.DataFrame(ROWS, columns=COLUMNS)
