"""Volatility-targeting backtest on SPY.

Timing (no look-ahead):
- the forecast made at the close of day t uses information up to t;
- on each rebalancing day R (last trading day of the week) the target weight is
  w = min(target_vol / forecast_vol_{R-1}, max_leverage), i.e. it uses the forecast
  from the previous close, and the position is set at the close of R;
- the position earns the SPY total return from R+1 on and drifts with prices until
  the next rebalancing; the rest (or the borrowed amount) earns / pays the
  previous day's T-bill rate;
- trading costs: `cost_bps` per unit of turnover |w_target - w_drifted|.
"""

import numpy as np
import pandas as pd


def weekly_rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("W-FRI")).max().to_numpy())


def target_weights(forecast_vol: pd.Series, rebalance: pd.DatetimeIndex, calendar: pd.DatetimeIndex,
                   target_vol: float, max_leverage: float) -> pd.Series:
    """Weight decided at each rebalancing date from the forecast of the previous trading day."""
    prev_day = pd.Series(calendar[:-1], index=calendar[1:])
    rebalance = rebalance[rebalance.isin(prev_day.index)]
    known = forecast_vol.reindex(prev_day[rebalance].to_numpy()).to_numpy()
    w = np.minimum(target_vol / known, max_leverage)
    return pd.Series(w, index=rebalance).dropna()


def run_backtest(asset_ret: pd.Series, rf_daily: pd.Series, weights: pd.Series,
                 cost_bps: float) -> pd.DataFrame:
    """Daily NAV of a portfolio rebalanced to `weights` on their dates (close to close).

    Starts at the first weight date with NAV = 1 and the portfolio in cash.
    """
    days = asset_ret.index[asset_ret.index >= weights.index[0]]
    ret, rf = asset_ret.reindex(days).to_numpy(), rf_daily.reindex(days).to_numpy()
    targets = weights.reindex(days).to_numpy()

    risky, cash = 0.0, 1.0
    rows = np.zeros((len(days), 4))  # nav, weight after close, turnover, cost (fraction of NAV)
    for i in range(len(days)):
        if i > 0:  # returns of day i on the position held since the close of i-1
            risky *= 1 + ret[i]
            cash *= 1 + rf[i]
        nav = risky + cash
        turnover = cost = 0.0
        if not np.isnan(targets[i]):
            turnover = abs(targets[i] - risky / nav)
            cost = turnover * cost_bps / 1e4
            nav *= 1 - cost
            risky, cash = targets[i] * nav, (1 - targets[i]) * nav
        rows[i] = nav, risky / nav, turnover, cost
    return pd.DataFrame(rows, index=days, columns=["nav", "weight", "turnover", "cost"])


def performance(bt: pd.DataFrame, rf_daily: pd.Series, annualization: int = 252) -> dict:
    """Annualized return / vol, Sharpe on excess returns with its standard error
    (Lo, 2002, iid approximation), max drawdown, turnover and average leverage."""
    r = bt["nav"].pct_change().dropna()
    excess = r - rf_daily.reindex(r.index)
    years = len(r) / annualization
    sharpe = excess.mean() / excess.std() * np.sqrt(annualization)
    drawdown = bt["nav"] / bt["nav"].cummax() - 1
    return {
        "ann_return": bt["nav"].iloc[-1] ** (1 / years) - 1,
        "ann_vol": r.std() * np.sqrt(annualization),
        "sharpe": sharpe,
        "sharpe_se": np.sqrt((1 + 0.5 * sharpe**2) / years),
        "max_drawdown": drawdown.min(),
        "turnover_per_year": bt["turnover"].sum() / years,
        "avg_weight": bt["weight"].mean(),
        "cost_per_year": bt["cost"].sum() / years,
    }
