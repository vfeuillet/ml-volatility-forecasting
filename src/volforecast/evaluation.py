"""Forecast evaluation on the variance scale.

`realized` is the realized annualized 5-day variance RV^2 and `forecast` the
model's forecast of it. QLIKE and MSE are robust to noise in the volatility
proxy (Patton, 2011): ranking models with them on a noisy proxy gives the same
ranking as with the true variance in expectation.
"""

import numpy as np
import pandas as pd
from scipy import stats


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


# --- Diebold-Mariano -------------------------------------------------------------

def newey_west_lags(n: int, horizon: int) -> int:
    """Bartlett bandwidth floor(sqrt(n)), and at least h-1.

    Overlapping h-day targets make the loss differential at least MA(h-1), and
    volatility clustering adds persistence on top. In simulations (MA(4) plus a
    persistent AR(1) component, see tests/test_evaluation.py) the textbook rule
    4 (n/100)^(2/9) rejects a true null 20-30% of the time at the 5% level, while
    sqrt(n) keeps the size around 5-8% for n >= 1000. With n ~ 250 (one-year
    regimes) no bandwidth is reliable: those p-values are indicative only.
    """
    return max(horizon - 1, int(np.floor(np.sqrt(n))))


def long_run_variance(d: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) estimate of the long-run variance of d."""
    d = d - d.mean()
    n = len(d)
    lrv = d @ d / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return lrv


def diebold_mariano(loss_model: np.ndarray, loss_ref: np.ndarray, horizon: int) -> dict:
    """DM test of equal predictive accuracy, with the Harvey-Leybourne-Newbold (1997)
    small-sample correction and Student-t(n-1) p-values (two-sided).

    d = loss_model - loss_ref: a negative statistic means the model beats the reference.
    """
    d = np.asarray(loss_model, float) - np.asarray(loss_ref, float)
    n = len(d)
    lags = newey_west_lags(n, horizon)
    dm = d.mean() / np.sqrt(long_run_variance(d, lags) / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * hln
    return {"mean_diff": d.mean(), "dm_stat": stat,
            "p_value": 2 * stats.t.sf(abs(stat), df=n - 1), "n_obs": n, "lags": lags}


LOSSES = {"qlike": qlike, "mse": squared_error}


def dm_table(sample: pd.DataFrame, models: list[str], references: list[str], horizon: int,
             loss: str = "qlike", alpha: float = 0.05) -> pd.DataFrame:
    """DM tests of each model against each reference.

    - universe: one test on the cross-sectional average loss differential per date,
      which accounts for the strong cross-correlation between tickers;
    - per ticker: number of tickers where the model is significantly better / worse.
    """
    f = LOSSES[loss]
    realized = sample["realized"].to_numpy()
    losses = pd.DataFrame({m: f(realized, sample[m].to_numpy()) for m in set(models) | set(references)},
                          index=sample.index)
    rows = []
    for ref in references:
        for m in models:
            if m == ref:
                continue
            diff = (losses[m] - losses[ref])
            daily = diff.groupby(level="date").mean().to_numpy()
            uni = diebold_mariano(daily, np.zeros_like(daily), horizon)
            better = worse = 0
            for _, g in diff.groupby(level="ticker"):
                r = diebold_mariano(g.to_numpy(), np.zeros(len(g)), horizon)
                better += (r["p_value"] < alpha) and r["dm_stat"] < 0
                worse += (r["p_value"] < alpha) and r["dm_stat"] > 0
            rows.append({"model": m, "reference": ref, "loss": loss,
                         "universe_dm_stat": uni["dm_stat"], "universe_p_value": uni["p_value"],
                         "n_tickers_better": better, "n_tickers_worse": worse,
                         "n_tickers": diff.index.get_level_values("ticker").nunique()})
    return pd.DataFrame(rows)


# --- Regimes and reporting --------------------------------------------------------

def regime_table(sample: pd.DataFrame, models: list[str], regimes: dict, horizon: int,
                 reference: str = "har") -> pd.DataFrame:
    """Ticker-averaged QLIKE per regime, with the universe DM test against `reference`."""
    dates = sample.index.get_level_values("date")
    rows = []
    for name, (start, end) in regimes.items():
        sub = sample[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        q = metrics_by_ticker(sub, models)["qlike"].unstack("model").mean()
        dm = dm_table(sub, models, [reference], horizon).set_index("model")
        for m in models:
            rows.append({"regime": name, "model": m, "qlike": q[m],
                         f"qlike_vs_{reference}": q[m] / q[reference],
                         "dm_p_value": dm["universe_p_value"].get(m, np.nan),
                         "n_days": sub.index.get_level_values("date").nunique()})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal GitHub-flavoured markdown table (avoids a `tabulate` dependency)."""
    df = df.reset_index()
    cells = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in cells])
