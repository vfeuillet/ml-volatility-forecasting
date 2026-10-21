"""Backtest engine: accounting identities, costs, timing (no look-ahead), leverage cap."""

import numpy as np
import pandas as pd
import pytest

from volforecast.backtest import run_backtest, target_weights, weekly_rebalance_dates

CAL = pd.bdate_range("2020-01-06", periods=60)  # starts on a Monday


def rets(seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, 0.01, len(CAL)), index=CAL)


def test_fully_invested_equals_asset_without_costs():
    r, rf = rets(), pd.Series(0.0, index=CAL)
    bt = run_backtest(r, rf, pd.Series(1.0, index=[CAL[0]]), cost_bps=0)
    expected = (1 + r.iloc[1:]).cumprod()
    np.testing.assert_allclose(bt["nav"].iloc[1:], expected)


def test_all_cash_earns_risk_free():
    rf = pd.Series(1e-4, index=CAL)
    bt = run_backtest(rets(), rf, pd.Series(0.0, index=[CAL[0]]), cost_bps=0)
    assert bt["nav"].iloc[-1] == pytest.approx((1 + 1e-4) ** (len(CAL) - 1))


def test_costs_charged_on_turnover():
    rf = pd.Series(0.0, index=CAL)
    r = pd.Series(0.0, index=CAL)
    w = pd.Series([1.0, 0.5], index=[CAL[0], CAL[5]])
    bt = run_backtest(r, rf, w, cost_bps=5)
    assert bt["turnover"].iloc[0] == pytest.approx(1.0)
    assert bt["turnover"].iloc[5] == pytest.approx(0.5)
    assert bt["nav"].iloc[-1] == pytest.approx((1 - 5e-4) * (1 - 2.5e-4))


def test_weight_drifts_between_rebalances():
    r = pd.Series(0.0, index=CAL)
    r.iloc[1] = 0.10
    bt = run_backtest(r, pd.Series(0.0, index=CAL), pd.Series(0.5, index=[CAL[0]]), cost_bps=0)
    assert bt["weight"].iloc[1] == pytest.approx(0.55 / 1.05)


def test_return_of_day_d_uses_position_set_before_d():
    """Changing the weight decided at the close of day k must not change NAV up to k."""
    r, rf = rets(), pd.Series(0.0, index=CAL)
    w1 = pd.Series([1.0, 1.0], index=[CAL[0], CAL[10]])
    w2 = pd.Series([1.0, 2.0], index=[CAL[0], CAL[10]])
    a, b = run_backtest(r, rf, w1, 0), run_backtest(r, rf, w2, 0)
    np.testing.assert_allclose(a["nav"].iloc[:11], b["nav"].iloc[:11])
    assert not np.isclose(a["nav"].iloc[11], b["nav"].iloc[11])


def test_rebalance_uses_previous_day_forecast_and_caps_leverage():
    rebal = weekly_rebalance_dates(CAL)
    assert (rebal.dayofweek == 4).all()  # Fridays for a calendar without holidays
    vol = pd.Series(0.20, index=CAL)
    R = rebal[2]
    prev = CAL[CAL.get_loc(R) - 1]
    vol[R] = 0.01            # known only at the close of R: must be ignored at R
    vol[prev] = 0.02         # known at R-1: used at R, and capped at 2
    w = target_weights(vol, rebal, CAL, 0.10, 2.0)
    assert w[R] == 2.0
    assert w[rebal[1]] == pytest.approx(0.5)
    assert w.max() <= 2.0
