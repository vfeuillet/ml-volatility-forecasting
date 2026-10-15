"""Features at date t must use data up to t only."""

import numpy as np
import pandas as pd
import pytest

from volforecast.features import backward_realized_vol, har_features

ANN = 252


def daily_var(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(1e-5, 4e-4, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_backward_vol_formula():
    v = daily_var()
    t = 50
    expected = np.sqrt(ANN / 22 * v.iloc[t - 21:t + 1].sum())
    assert backward_realized_vol(v, 22, ANN).iloc[t] == pytest.approx(expected)


def test_warmup_is_nan():
    X = har_features(daily_var(), ANN)
    assert X["rv_m"].iloc[:21].isna().all() and X["rv_m"].iloc[21:].notna().all()


@pytest.mark.parametrize("offset", [1, 2, 5, 30])
def test_features_ignore_the_future(offset):
    v = daily_var()
    t = 60
    base = har_features(v, ANN).iloc[t]
    shocked = v.copy()
    shocked.iloc[t + offset] *= 50
    pd.testing.assert_series_equal(har_features(shocked, ANN).iloc[t], base)


def test_features_use_today():
    v = daily_var()
    t = 60
    shocked = v.copy()
    shocked.iloc[t] *= 50
    assert (har_features(shocked, ANN).iloc[t] != har_features(v, ANN).iloc[t]).all()
