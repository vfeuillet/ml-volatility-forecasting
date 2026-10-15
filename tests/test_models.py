"""Benchmark models: correctness on synthetic data and absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from volforecast.models.garch import garch_forecast
from volforecast.models.har import HAR
from volforecast.walkforward import make_splits


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 50_000
    X = pd.DataFrame(np.exp(rng.normal(-1.6, 0.4, (n, 3))), columns=["rv_d", "rv_w", "rv_m"])
    beta = np.array([-0.3, 0.2, 0.4, 0.3])
    s = 0.25
    y = pd.Series(beta[0] + np.log(X.to_numpy()) @ beta[1:] + rng.normal(0, s, n))
    model = HAR().fit(X, y)
    np.testing.assert_allclose(model.coef_, beta, atol=0.03)
    assert model.resid_var_ == pytest.approx(s**2, rel=0.05)
    # Bias-corrected variance forecast matches the empirical mean of exp(2y).
    pred = model.predict(X)
    assert np.mean(np.exp(2 * y)) / np.mean(pred) == pytest.approx(1.0, abs=0.03)


def _prices(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_t(5, n) * 0.01)), index=idx)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_garch_forecast_ignores_the_future(asymmetric):
    close = _prices()
    splits = make_splits(close.index, "2014-01-01", 21, 5)
    base, _ = garch_forecast(close, splits, asymmetric, 5, 252)

    origin = splits[3].test_dates[10]
    shocked = close.copy()
    after = shocked.index > origin
    shocked[after] = shocked[after] * np.exp(np.cumsum(np.full(after.sum(), 0.03)))
    new, _ = garch_forecast(shocked, splits, asymmetric, 5, 252)

    np.testing.assert_allclose(new[:origin], base[:origin])
    assert not np.allclose(new[new.index > origin], base[base.index > origin])


def test_garch_covers_all_test_dates():
    close = _prices()
    splits = make_splits(close.index, "2014-01-01", 21, 5)
    fc, n_failed = garch_forecast(close, splits, False, 5, 252)
    expected = pd.DatetimeIndex(np.concatenate([s.test_dates.values for s in splits]))
    assert fc.index.equals(expected)
    assert (fc > 0).all() and n_failed == 0
