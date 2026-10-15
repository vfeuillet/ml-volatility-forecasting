"""Forecast evaluation on the variance scale.

`realized` is the realized annualized 5-day variance RV^2 and `forecast` the
model's forecast of it. QLIKE and MSE are robust to noise in the volatility
proxy (Patton, 2011): ranking models with them on a noisy proxy gives the same
ranking as with the true variance in expectation.
"""

import numpy as np
import pandas as pd


def qlike(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss, zero iff forecast == realized.

    Penalizes under-prediction more than over-prediction of the same size.
    """
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1


def squared_error(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return (realized - forecast) ** 2


def mincer_zarnowitz(realized: np.ndarray, forecast: np.ndarray) -> dict:
    """OLS of realized on forecast: unbiased forecasts have alpha = 0 and beta = 1."""
    Z = np.column_stack([np.ones_like(forecast), forecast])
    (alpha, beta), *_ = np.linalg.lstsq(Z, realized, rcond=None)
    resid = realized - Z @ np.array([alpha, beta])
    r2 = 1 - resid.var() / realized.var()
    return {"mz_alpha": alpha, "mz_beta": beta, "mz_r2": r2}


def metrics(realized: np.ndarray, forecast: np.ndarray) -> dict:
    realized, forecast = np.asarray(realized, float), np.asarray(forecast, float)
    return {
        "qlike": qlike(realized, forecast).mean(),
        "mse_var": squared_error(realized, forecast).mean(),
        "mae_vol": np.abs(np.sqrt(realized) - np.sqrt(forecast)).mean(),
        **mincer_zarnowitz(realized, forecast),
        "n_obs": len(realized),
    }


def common_sample(forecasts: pd.DataFrame, realized: pd.Series) -> pd.DataFrame:
    """Rows where the target and every model's forecast are available."""
    return forecasts.join(realized.rename("realized"), how="inner").dropna()


def metrics_by_ticker(sample: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per (ticker, model) on a (date, ticker)-indexed common sample."""
    rows = []
    for ticker, g in sample.groupby(level="ticker"):
        for m in models:
            rows.append({"ticker": ticker, "model": m, **metrics(g["realized"], g[m])})
    return pd.DataFrame(rows).set_index(["ticker", "model"])


def universe_summary(by_ticker: pd.DataFrame, reference: str = "har") -> pd.DataFrame:
    """Cross-sectional average of each metric, plus loss ratios relative to `reference`."""
    avg = by_ticker.groupby(level="model").mean()
    for loss in ("qlike", "mse_var", "mae_vol"):
        avg[f"{loss}_vs_{reference}"] = avg[loss] / avg.loc[reference, loss]
    # Share of tickers on which the model has a lower QLIKE than the reference.
    q = by_ticker["qlike"].unstack("model")
    avg[f"pct_tickers_beat_{reference}"] = 100 * (q.lt(q[reference], axis=0)).mean()
    return avg.drop(columns="n_obs").sort_values("qlike")
