"""Daily variance proxy and the 5-day forward realized volatility target.

Daily variance (Garman-Klass range estimator plus the squared overnight return):

    v_t = ln(O_t / C_{t-1})^2                         overnight jump
        + 0.5 * ln(H_t / L_t)^2                       range term
        - (2 ln 2 - 1) * ln(C_t / O_t)^2              drift correction

This is only a proxy: true realized variance needs intraday (e.g. 5-minute) data.

Target at date t (known only after the close of t + h):

    RV_t = sqrt( (252 / h) * sum_{i=1..h} v_{t+i} ),   y_t = ln RV_t
"""

import numpy as np
import pandas as pd

_GK_DRIFT = 2 * np.log(2) - 1


def variance_components(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight and intraday (Garman-Klass) daily variance, without floor.

    For a valid bar (O and C inside [L, H]) the GK term is >= 0, because
    |ln C/O| <= ln H/L and 2 ln 2 - 1 < 0.5.
    """
    overnight = np.log(df["open"] / df["close"].shift(1)) ** 2
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    intraday = 0.5 * hl**2 - _GK_DRIFT * co**2
    return pd.DataFrame({"overnight": overnight, "intraday": intraday})


def daily_variance(df: pd.DataFrame, floor: float) -> pd.Series:
    """Total daily variance v_t, floored so that its log is always defined."""
    comp = variance_components(df)
    total = comp["overnight"] + comp["intraday"]
    return total.clip(lower=floor).rename("daily_var")


def forward_realized_vol(daily_var: pd.Series, horizon: int, annualization: int) -> pd.Series:
    """Annualized realized volatility over t+1 .. t+horizon, stored at date t.

    NaN if any of the `horizon` future days is missing (we never sum fewer days).
    """
    # rolling(h).sum() at t+h covers t+1..t+h; shifting by -h stores it at t.
    future_sum = daily_var.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.sqrt(annualization / horizon * future_sum).rename("target_rv")


def build_targets(panel: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Long frame indexed by (date, ticker) with daily variance and the target."""
    t = cfg["target"]
    frames = []
    for ticker, df in panel.items():
        v = daily_variance(df, t["variance_floor"])
        rv = forward_realized_vol(v, t["horizon"], t["annualization"])
        out = pd.DataFrame({"daily_var": v, "target_rv": rv, "target_log_rv": np.log(rv)})
        out["ticker"] = ticker
        frames.append(out)
    return pd.concat(frames).set_index("ticker", append=True).sort_index()
