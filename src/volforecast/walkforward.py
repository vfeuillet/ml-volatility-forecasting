"""Expanding-window walk-forward engine with purging.

At each refit date T (information set: closes up to and including T):
- training samples are dates t <= T - h (in trading days), so that every training
  target, which spans t+1 .. t+h, is fully observed at T (purging);
- the model then forecasts from every origin T, T+1, ..., T+refit_every-1, using
  the features available at each origin.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
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


def yearly_splits(calendar: pd.DatetimeIndex, test_start: str, horizon: int) -> list[Split]:
    """One refit on the first trading day of each year from `test_start` on."""
    test = calendar[calendar >= pd.Timestamp(test_start)]
    firsts = pd.Series(test, index=test).groupby(test.year).first()
    splits = []
    for i, refit in enumerate(firsts):
        pos = calendar.get_loc(refit)
        end = calendar.get_loc(firsts.iloc[i + 1]) if i + 1 < len(firsts) else len(calendar)
        splits.append(Split(refit_date=refit, train_end=calendar[pos - horizon],
                            test_dates=calendar[pos:end]))
    return splits


def purged_holdout(dates: pd.DatetimeIndex, val_days: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (fit, validation) inside a training window.

    The last `val_days` dates validate; the fitting part stops `horizon` days
    earlier so that no fitting target (t+1..t+h) overlaps a validation target.
    """
    unique = dates.unique().sort_values()
    val_start = unique[-val_days]
    fit_end = unique[len(unique) - val_days - horizon]
    return np.asarray(dates <= fit_end), np.asarray(dates >= val_start)


def tuning_masks(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int,
                 train_end: str, val_end: str) -> tuple[np.ndarray, np.ndarray]:
    """Purged (train, validation) masks for hyperparameter selection before the test.

    Training targets end by the first validation origin; validation targets end by
    the last trading day of `val_end`, i.e. before the first test origin.
    """
    cal = calendar[calendar <= pd.Timestamp(val_end)]
    val_first = cal[cal.searchsorted(pd.Timestamp(train_end), side="right")]
    train = dates <= cal[cal.get_loc(val_first) - horizon]
    val = (dates >= val_first) & (dates <= cal[-1 - horizon])
    return np.asarray(train), np.asarray(val)


def dates_of(index: pd.Index) -> pd.DatetimeIndex:
    """Date component of a DatetimeIndex or of a (date, ticker) MultiIndex."""
    return index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index


def run_supervised(model_factory: Callable, X: pd.DataFrame, y: pd.Series, splits: list[Split]) -> pd.Series:
    """Walk-forward for a model with fit(X, y) / predict(X) -> forecast variance.

    X and y share the same index: a DatetimeIndex (one ticker) or a (date, ticker)
    MultiIndex (pooled model). Rows with missing features or target are dropped from
    training; rows with missing features get no forecast.
    """
    dates = dates_of(X.index)
    valid_X = X.notna().all(axis=1).to_numpy()
    trainable = valid_X & y.notna().to_numpy()
    preds = []
    for s in splits:
        train = trainable & (dates <= s.train_end)
        model = model_factory()
        model.fit(X[train], y[train])
        test = X[dates.isin(s.test_dates) & valid_X]
        if len(test):
            preds.append(pd.Series(model.predict(test), index=test.index))
    return pd.concat(preds).rename("forecast_var") if preds else pd.Series(dtype=float, name="forecast_var")
