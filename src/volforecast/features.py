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


def _log_vol(daily_var: pd.Series, window: int, annualization: int, floor: float) -> pd.Series:
    """Log of the annualized vol of a daily variance series averaged over `window` days."""
    mean_var = daily_var.rolling(window, min_periods=window).mean().clip(lower=floor)
    return 0.5 * np.log(annualization * mean_var)


def gbm_features(df: pd.DataFrame, daily_var: pd.Series, vix: pd.Series,
                 annualization: int, floor: float) -> pd.DataFrame:
    """Full predictor set for the ML models (one ticker, benchmark calendar).

    df: OHLCV; daily_var: GK + overnight variance; vix: VIX close on the same calendar.
    """
    r = np.log(df["close"]).diff()
    overnight_var = np.log(df["open"] / df["close"].shift(1)) ** 2
    parkinson_var = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
    log_vix_lag = np.log(vix).shift(1)  # VIX closes after the stock market: use t-1
    volume = df["volume"].where(df["volume"] > 0)

    X = np.log(har_features(daily_var, annualization)).add_prefix("log_")
    X["ret_1"] = r
    X["ret_5"] = r.rolling(5).mean()
    X["ret_22"] = r.rolling(22).mean()
    X["abs_ret_1"] = r.abs()
    X["abs_ret_5"] = r.abs().rolling(5).mean()
    X["neg_ret_1"] = r.clip(upper=0)
    X["log_park_1"] = _log_vol(parkinson_var, 1, annualization, floor)
    X["log_park_5"] = _log_vol(parkinson_var, 5, annualization, floor)
    X["log_vix_lag"] = log_vix_lag
    X["vix_chg_5_lag"] = log_vix_lag - log_vix_lag.shift(5)
    X["log_volume_ratio"] = np.log(volume / volume.rolling(22).mean())
    X["log_overnight_5"] = _log_vol(overnight_var, 5, annualization, floor)
    return X


def pooled_dataset(panel: dict[str, pd.DataFrame], targets: pd.DataFrame, vix: pd.Series,
                   cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Stack every ticker's features and log-vol target into one (date, ticker) panel."""
    t = cfg["target"]
    frames = []
    for ticker, df in panel.items():
        tgt = targets.xs(ticker, level="ticker")
        X = gbm_features(df, tgt["daily_var"], vix, t["annualization"], t["variance_floor"])
        X["target_log_rv"] = tgt["target_log_rv"]
        X["ticker"] = ticker
        frames.append(X)
    data = pd.concat(frames).set_index("ticker", append=True).sort_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.drop(columns="target_log_rv"), data["target_log_rv"]
