"""Tests for OHLCV cleaning."""

import numpy as np
import pandas as pd

from volforecast.data import OHLCV, clean


def bars(rows):
    idx = pd.bdate_range("2020-01-01", periods=len(rows), name="date")
    return pd.DataFrame(rows, columns=OHLCV, index=idx, dtype=float)


def test_valid_bars_untouched():
    df = bars([[100, 102, 99, 101, 1e6], [101, 103, 100, 102, 1e6]])
    out, n = clean(df)
    assert n == 0
    pd.testing.assert_frame_equal(out, df)


def test_inconsistent_bars_set_to_nan():
    df = bars([
        [100, 102, 99, 101, 1e6],   # valid
        [100, 99, 98, 101, 1e6],    # high below close
        [100, 102, 100.5, 101, 1e6],  # low above open
        [100, 100, 100, 100, 0],    # flat bar with zero volume
        [0, 102, 99, 101, 1e6],     # non-positive price
        [np.nan, 102, 99, 101, 1e6],  # missing price
        [100, 100, 100, 100, 5e5],  # flat bar with volume: rare but real, kept
    ])
    out, n = clean(df)
    assert n == 5
    assert out.iloc[1:6].isna().all().all()
    assert out.iloc[[0, 6]].notna().all().all()


def test_rounding_noise_tolerated():
    df = bars([[100, 101.0 * (1 - 1e-12), 99, 101.0, 1e6]])
    _, n = clean(df)
    assert n == 0
