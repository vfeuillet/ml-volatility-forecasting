"""Predictors built from information available at the close of day t.

Every feature at date t uses data up to and including t only (tested by
perturbation in tests/test_features.py).
"""

import numpy as np
import pandas as pd

HAR_LAGS = {"rv_d": 1, "rv_w": 5, "rv_m": 22}


def backward_realized_vol(daily_var: pd.Series, window: int, annualization: int) -> pd.Series:
    """Annualized realized vol over the last `window` days t-window+1 .. t."""
    past_sum = daily_var.rolling(window, min_periods=window).sum()
    return np.sqrt(annualization / window * past_sum)


def har_features(daily_var: pd.Series, annualization: int) -> pd.DataFrame:
    """Daily, weekly and monthly realized vol (Corsi, 2009), annualized."""
    return pd.DataFrame({
        name: backward_realized_vol(daily_var, w, annualization) for name, w in HAR_LAGS.items()
    })
