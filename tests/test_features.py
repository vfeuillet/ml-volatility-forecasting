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


def ohlcv(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.uniform(1e6, 2e6, n)}, index=idx)
    vix = pd.Series(rng.uniform(12, 30, n), index=idx)
    return df, vix


def all_features(df, vix):
    from volforecast.features import gbm_features
    from volforecast.targets import daily_variance
    return gbm_features(df, daily_variance(df, 1e-8), vix, ANN, 1e-8)


@pytest.mark.parametrize("offset", [1, 3, 22])
def test_gbm_features_ignore_future_prices_and_vix(offset):
    df, vix = ohlcv()
    t = 70
    base = all_features(df, vix).iloc[t]
    df2, vix2 = df.copy(), vix.copy()
    df2.iloc[t + offset] *= 1.3
    vix2.iloc[t + offset] *= 2
    pd.testing.assert_series_equal(all_features(df2, vix2).iloc[t], base)


def test_vix_enters_with_one_day_lag():
    df, vix = ohlcv()
    t = 70
    vix2 = vix.copy()
    vix2.iloc[t] *= 2
    a, b = all_features(df, vix), all_features(df, vix2)
    vix_cols = ["log_vix_lag", "vix_chg_5_lag"]
    pd.testing.assert_series_equal(a.iloc[t], b.iloc[t])           # VIX_t not used at t
    assert (a.iloc[t + 1][vix_cols] != b.iloc[t + 1][vix_cols]).all()  # used from t+1


def test_gbm_features_complete_after_warmup():
    df, vix = ohlcv()
    X = all_features(df, vix)
    assert X.iloc[23:].notna().all().all()
