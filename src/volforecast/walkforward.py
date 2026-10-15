"""Expanding-window walk-forward engine with purging.

At each refit date T (information set: closes up to and including T):
- training samples are dates t <= T - h (in trading days), so that every training
  target, which spans t+1 .. t+h, is fully observed at T (purging);
- the model then forecasts from every origin T, T+1, ..., T+refit_every-1, using
  the features available at each origin.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Split:
    refit_date: pd.Timestamp
    train_end: pd.Timestamp        # last training sample date (inclusive)
    test_dates: pd.DatetimeIndex   # forecast origins served by this fit


def make_splits(calendar: pd.DatetimeIndex, test_start: str, refit_every: int, horizon: int) -> list[Split]:
    first = calendar.searchsorted(pd.Timestamp(test_start))
    if first < horizon:
        raise ValueError("Not enough history before the first test date")
    splits = []
    for pos in range(first, len(calendar), refit_every):
        splits.append(Split(
            refit_date=calendar[pos],
            train_end=calendar[pos - horizon],
            test_dates=calendar[pos:pos + refit_every],
        ))
    return splits


def run_supervised(model_factory: Callable, X: pd.DataFrame, y: pd.Series, splits: list[Split]) -> pd.Series:
    """Walk-forward for a model with fit(X, y) / predict(X) -> forecast variance.

    X and y share the same DatetimeIndex (one ticker). Rows with missing features
    or target are dropped from training; rows with missing features get no forecast.
    """
    valid_X = X.notna().all(axis=1)
    trainable = valid_X & y.notna()
    preds = []
    for s in splits:
        train = trainable & (X.index <= s.train_end)
        model = model_factory()
        model.fit(X[train], y[train])
        test = X.loc[X.index.isin(s.test_dates) & valid_X]
        if len(test):
            preds.append(pd.Series(model.predict(test), index=test.index))
    return pd.concat(preds).rename("forecast_var") if preds else pd.Series(dtype=float, name="forecast_var")
