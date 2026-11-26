"""UID construction (M3). The UID is a proxy customer key and is never a model feature."""

from __future__ import annotations

import pandas as pd


def _part(values: pd.Series) -> list[str]:
    """Format a numeric key part as an integer string, 'na' when missing.

    Plain Python on purpose: pandas string operations cost milliseconds per API request.
    """
    return [
        "na" if v != v else str(int(v)) if v.is_integer() else str(v)
        for v in values.astype("float64").tolist()
    ]


def build_uid(frame: pd.DataFrame) -> pd.Series:
    """card1_addr1_(day - D1), or fb_card1_card2_card5 when addr1 or D1 is missing."""
    anchor = frame["day"].astype("float64") - frame["D1"].astype("float64")
    parts = zip(
        _part(frame["card1"]),
        _part(frame["card2"]),
        _part(frame["card5"]),
        _part(frame["addr1"]),
        _part(anchor),
        strict=True,
    )
    # The prefix stops a fallback key from colliding with a primary key.
    keys = [
        f"fb_{card1}_{card2}_{card5}" if "na" in (addr1, opened) else f"{card1}_{addr1}_{opened}"
        for card1, card2, card5, addr1, opened in parts
    ]
    return pd.Series(keys, index=frame.index, dtype=object, name="UID")


def card1_key(frame: pd.DataFrame) -> pd.Series:
    """Key used for card-level velocity."""
    return pd.Series(_part(frame["card1"]), index=frame.index, dtype=object, name="card1_key")
