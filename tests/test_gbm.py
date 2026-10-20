"""Pooled GBM: purged time-ordered validation, pooled walk-forward, reproducibility."""

import numpy as np
import pandas as pd

from volforecast.models.gbm import GBM
from volforecast.walkforward import make_splits, run_supervised

H = 5
PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0}


def pooled(n_days=900, tickers=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, list(tickers)], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    y = pd.Series(-1.6 + 0.3 * X["a"] - 0.2 * X["b"] ** 2 + rng.normal(0, 0.2, len(idx)), index=idx)
    return X, y


def model(seed=0, val_days=100):
    return GBM(PARAMS, ["a", "b", "c"], val_days, H, 300, 10, seed, importance=True)


def test_internal_validation_is_last_dates_and_purged():
    X, _ = pooled()
    dates = X.index.get_level_values("date")
    is_fit, is_val = model().split(dates)
    unique = dates.unique()
    assert set(dates[is_val]) == set(unique[-100:])
    last_fit = unique.get_loc(dates[is_fit].max())
    first_val = unique.get_loc(dates[is_val].min())
    assert last_fit + H < first_val + 1   # fitting targets end before validation targets start
    assert first_val - last_fit == H
    assert not (is_fit & is_val).any()


def test_gbm_is_reproducible_and_learns():
    X, y = pooled()
    p1 = model().fit(X, y).predict(X)
    p2 = model().fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)
    m = model().fit(X, y)
    assert m.importances_["a"] > m.importances_["c"] and m.importances_["b"] > m.importances_["c"]


def test_pooled_walkforward_purges_by_date():
    X, y = pooled()
    cal = X.index.get_level_values("date").unique()
    splits = make_splits(cal, str(cal[600].date()), 100, H)
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(X.index.get_level_values("date").max())
            return self

        def predict(self, X):
            return np.ones(len(X))

    preds = run_supervised(Recorder, X, y, splits)
    assert seen == [s.train_end for s in splits]
    assert preds.index.get_level_values("date").min() == splits[0].refit_date
    assert len(preds) == 3 * (len(cal) - 600)  # every ticker forecast at every test date
