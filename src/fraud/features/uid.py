"""UID construction (M3). The UID is a proxy customer key and is never a model feature."""

from __future__ import annotations

import pandas as pd


def _part(values: pd.Series) -> pd.Series:
    """Format a numeric key part as an integer string, 'na' when missing."""
    return values.astype("float64").astype("Int64").astype("string").fillna("na")


def build_uid(frame: pd.DataFrame) -> pd.Series:
    """card1_addr1_(day - D1), or fb_card1_card2_card5 when addr1 or D1 is missing."""
    card1 = _part(frame["card1"])
    anchor = frame["day"].astype("float64") - frame["D1"].astype("float64")
    primary = card1 + "_" + _part(frame["addr1"]) + "_" + _part(anchor)
    # The prefix stops a fallback key from colliding with a primary key.
    fallback = "fb_" + card1 + "_" + _part(frame["card2"]) + "_" + _part(frame["card5"])
    complete = frame["addr1"].notna() & frame["D1"].notna()
    return primary.where(complete, fallback).astype(object).rename("UID")


def card1_key(frame: pd.DataFrame) -> pd.Series:
    """Key used for card-level velocity."""
    return _part(frame["card1"]).astype(object).rename("card1_key")
