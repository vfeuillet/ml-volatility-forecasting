"""Tests for the variance proxy and the forward target (no look-ahead, no look-back)."""

import numpy as np
import pandas as pd
import pytest

from volforecast.targets import daily_variance, forward_realized_vol, variance_components

H, ANN, FLOOR = 5, 252, 1e-8


def make_ohlc(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Random but internally consistent OHLC bars."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1e6}, index=idx)


def test_garman_klass_hand_computed():
    df = pd.DataFrame({"open": [100.0, 101.0], "high": [102.0, 104.0],
                       "low": [99.0, 100.0], "close": [100.0, 103.0]})
    comp = variance_components(df)
    expected_overnight = np.log(101 / 100) ** 2
    expected_intraday = 0.5 * np.log(104 / 100) ** 2 - (2 * np.log(2) - 1) * np.log(103 / 101) ** 2
    assert comp["overnight"].iloc[1] == pytest.approx(expected_overnight)
    assert comp["intraday"].iloc[1] == pytest.approx(expected_intraday)
    # No previous close on the first day: overnight term undefined.
    assert np.isnan(comp["overnight"].iloc[0])


def test_intraday_term_non_negative_on_valid_bars():
    comp = variance_components(make_ohlc(500))
    assert (comp["intraday"] >= 0).all()


def test_floor_applied():
    df = pd.DataFrame({"open": [100.0, 100.0], "high": [100.0, 100.0],
                       "low": [100.0, 100.0], "close": [100.0, 100.0]})
    assert daily_variance(df, FLOOR).iloc[1] == FLOOR


def test_target_formula():
    v = pd.Series(np.arange(1, 21, dtype=float) * 1e-4)
    rv = forward_realized_vol(v, H, ANN)
    t = 3
    expected = np.sqrt(ANN / H * v.iloc[t + 1:t + 1 + H].sum())
    assert rv.iloc[t] == pytest.approx(expected)


def test_last_h_targets_are_nan():
    rv = forward_realized_vol(daily_variance(make_ohlc(), FLOOR), H, ANN)
    assert rv.iloc[-H:].isna().all()
    assert rv.iloc[1:-H].notna().all()


def test_missing_day_in_window_gives_nan():
    v = pd.Series(np.full(20, 1e-4))
    v.iloc[10] = np.nan
    rv = forward_realized_vol(v, H, ANN)
    # Windows t+1..t+5 containing index 10 are t = 5..9.
    assert rv.iloc[5:10].isna().all()
    assert rv.iloc[4] == pytest.approx(rv.iloc[10])


@pytest.mark.parametrize("offset, should_change", [
    (-3, False), (-1, False),         # past bars
    (1, True), (3, True), (5, True),  # the target window t+1..t+5
    (6, False), (10, False),          # beyond the window
])
def test_target_depends_only_on_t_plus_1_to_t_plus_h(offset, should_change):
    """Perturbation test on raw prices: the target at t reacts to the bars of
    t+1..t+5 and to nothing after t+5 nor before t."""
    df = make_ohlc(40)
    t = 15
    base = forward_realized_vol(daily_variance(df, FLOOR), H, ANN).iloc[t]

    shocked = df.copy()
    shocked.iloc[t + offset, :4] *= np.array([1.05, 1.10, 0.90, 0.97])
    new = forward_realized_vol(daily_variance(shocked, FLOOR), H, ANN).iloc[t]

    assert (not np.isclose(new, base)) == should_change


def test_open_high_low_at_t_do_not_enter_target():
    df = make_ohlc(40)
    t = 15
    base = forward_realized_vol(daily_variance(df, FLOOR), H, ANN).iloc[t]
    shocked = df.copy()
    shocked.iloc[t, :3] *= np.array([1.01, 1.10, 0.90])  # open, high, low only
    new = forward_realized_vol(daily_variance(shocked, FLOOR), H, ANN).iloc[t]
    assert np.isclose(new, base)


def test_shock_to_close_at_t_changes_overnight_of_t_plus_1():
    """C_t is known at t but enters v_{t+1} (overnight return). The target at t uses
    it only as a reference price, which is legitimate: no future information."""
    df = make_ohlc(40)
    t = 15
    base = forward_realized_vol(daily_variance(df, FLOOR), H, ANN).iloc[t]
    shocked = df.copy()
    shocked.iloc[t, shocked.columns.get_loc("close")] *= 1.02
    new = forward_realized_vol(daily_variance(shocked, FLOOR), H, ANN).iloc[t]
    assert not np.isclose(new, base)
