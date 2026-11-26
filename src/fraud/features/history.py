"""Point-in-time history features (M3).

The same state classes serve the batch pipeline and the online store, so the two cannot drift.
An event only becomes visible to queries with a strictly later timestamp, which covers ties
both in batch and online. Events must arrive in time order per key.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import Config
from fraud.features.uid import card1_key

HOUR = 3_600
NAN = float("nan")


@dataclass
class UidState:
    """Everything remembered about one UID."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    max_amount: float = 0.0
    last_ts: int | None = None
    devices: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    # Recent events as parallel lists, kept only as long as the longest window.
    times: list[int] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    addresses: list[float | None] = field(default_factory=list)
    # Events at the newest timestamp: [ts, amount, addr1, device, email].
    pending: list[list[Any]] = field(default_factory=list)

    def _absorb(self, before: int) -> None:
        while self.pending and self.pending[0][0] < before:
            ts, amount, address, device, email = self.pending.pop(0)
            self.count += 1
            delta = amount - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (amount - self.mean)
            self.max_amount = max(self.max_amount, amount)
            self.last_ts = ts
            if device is not None and device not in self.devices:
                self.devices.append(device)
            if email is not None and email not in self.emails:
                self.emails.append(email)
            self.times.append(ts)
            self.amounts.append(amount)
            self.addresses.append(address)

    def _evict(self, oldest: int) -> None:
        drop = bisect_left(self.times, oldest)
        if drop:
            del self.times[:drop], self.amounts[:drop], self.addresses[:drop]

    def features(
        self,
        ts: int,
        amount: float,
        device: str | None,
        email: str | None,
        windows: list[int],
        min_prior: int,
    ) -> dict[str, float]:
        """Features for a transaction at ``ts``, using only strictly earlier events."""
        self._absorb(ts)
        self._evict(ts - max(windows) * HOUR)
        out: dict[str, float] = {}
        for hours in windows:
            start = bisect_left(self.times, ts - hours * HOUR)
            out[f"UID_TXN_COUNT_{_label(hours)}"] = float(len(self.times) - start)
        day_start = bisect_left(self.times, ts - 24 * HOUR)
        out["UID_AMT_SUM_24H"] = float(sum(self.amounts[day_start:]))
        std = math.sqrt(max(self.m2, 0.0) / (self.count - 1)) if self.count > 1 else NAN
        out["UID_AMT_MEAN_PRIOR"] = self.mean if self.count else NAN
        out["UID_AMT_STD_PRIOR"] = std
        usable = self.count >= min_prior and std > 0
        out["AMT_ZSCORE_UID"] = (amount - self.mean) / std if usable else NAN
        out["AMT_TO_MAX_PRIOR"] = amount / self.max_amount if self.max_amount > 0 else NAN
        out["SECS_SINCE_LAST_TXN"] = NAN if self.last_ts is None else float(ts - self.last_ts)
        out["UID_NEW_DEVICE"] = NAN if device is None else float(device not in self.devices)
        out["UID_NEW_EMAIL"] = NAN if email is None else float(email not in self.emails)
        week_start = bisect_left(self.times, ts - 168 * HOUR)
        out["UID_DISTINCT_ADDR_7D"] = float(
            len({a for a in self.addresses[week_start:] if a is not None})
        )
        out["UID_IS_FIRST_TXN"] = float(self.count == 0)
        return out

    def add(
        self, ts: int, amount: float, address: float | None, device: str | None, email: str | None
    ) -> None:
        """Remember a transaction. It stays invisible to queries at the same timestamp."""
        self._absorb(ts)
        self.pending.append([ts, amount, address, device, email])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CardState:
    """Recent timestamps for one card1 value."""

    times: list[int] = field(default_factory=list)

    def count_24h(self, ts: int) -> float:
        del self.times[: bisect_left(self.times, ts - 24 * HOUR)]
        return float(bisect_left(self.times, ts))

    def add(self, ts: int) -> None:
        self.times.append(ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label(hours: int) -> str:
    return f"{hours // 24}D" if hours > 24 and hours % 24 == 0 else f"{hours}H"


def history_columns(config: Config) -> list[str]:
    """Names of the history features, in output order."""
    sample = UidState().features(0, 1.0, None, None, config.features.history_windows_hours, 3)
    return [*sample, "CARD1_TXN_COUNT_24H"]


def _optional(values: pd.Series) -> list[Any]:
    return [None if pd.isna(v) else v for v in values.tolist()]


@dataclass
class HistoryStates:
    """In-memory state for all keys."""

    uids: dict[str, UidState] = field(default_factory=dict)
    cards: dict[str, CardState] = field(default_factory=dict)


def compute_history(
    frame: pd.DataFrame, uid: pd.Series, config: Config, states: HistoryStates | None = None
) -> pd.DataFrame:
    """One sorted pass over ``frame`` (FR-3.3). ``states`` is updated in place if given."""
    if config.features.label_features.enabled:
        raise NotImplementedError("label-delay features are not implemented; keep them disabled")
    states = states if states is not None else HistoryStates()
    windows = config.features.history_windows_hours
    min_prior = config.features.min_prior_for_zscore
    order = np.lexsort((frame["TransactionID"].to_numpy(), frame["TransactionDT"].to_numpy()))
    times = frame["TransactionDT"].tolist()
    amounts = frame["TransactionAmt"].tolist()
    addresses = _optional(frame["addr1"])
    devices = _optional(frame["DeviceInfo"])
    emails = _optional(frame["P_emaildomain"])
    uids = uid.tolist()
    cards = card1_key(frame).tolist()
    names = history_columns(config)
    out = np.empty((len(frame), len(names)))
    for i in order:
        ts, amount = int(times[i]), float(amounts[i])
        state = states.uids.get(uids[i])
        if state is None:
            state = states.uids[uids[i]] = UidState()
        card = states.cards.get(cards[i])
        if card is None:
            card = states.cards[cards[i]] = CardState()
        values = state.features(ts, amount, devices[i], emails[i], windows, min_prior)
        out[i, :-1] = list(values.values())
        out[i, -1] = card.count_24h(ts)
        state.add(ts, amount, addresses[i], devices[i], emails[i])
        card.add(ts)
    return pd.DataFrame(out, columns=names, index=frame.index)
