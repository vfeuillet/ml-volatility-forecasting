"""Download, cache and clean daily OHLCV data from Yahoo Finance.

Raw downloads are cached untouched in parquet so that cleaning rules can change
without re-downloading. All series are aligned on the trading calendar of the
benchmark (SPY). Prices are never forward-filled: a missing or invalid session
stays NaN and propagates to anything that depends on it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OHLCV = ["open", "high", "low", "close", "volume"]

# Relative tolerance for OHLC consistency checks. Adjusted prices are multiplied
# by a common factor, which can create rounding violations of ~1e-12.
_TOL = 1e-6


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{symbol.replace('^', 'IDX_')}.parquet"


def download(symbol: str, start: str, end: str, cache_dir: Path, force: bool = False) -> pd.DataFrame:
    """Return raw daily OHLCV for `symbol` between `start` and `end` (both inclusive).

    Prices are split- and dividend-adjusted (auto_adjust=True). The adjustment is a
    multiplicative factor applied to O, H, L and C alike, so intraday ratios such as
    H/L and C/O are unchanged.
    """
    path = _cache_path(symbol, cache_dir)
    if path.exists() and not force:
        return pd.read_parquet(path)

    # yfinance treats `end` as exclusive.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(symbol).history(start=start, end=end_exclusive, auto_adjust=True, actions=False)
    if raw.empty:
        raise RuntimeError(f"No data returned by Yahoo Finance for {symbol}")

    df = raw.rename(columns=str.lower)[OHLCV].astype("float64")
    # Exchange-local timestamps -> plain calendar dates.
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def invalid_sessions(df: pd.DataFrame) -> pd.Series:
    """Flag sessions whose OHLC values are missing or mutually inconsistent."""
    o, h, l, c, v = (df[k] for k in OHLCV)
    missing = df[["open", "high", "low", "close"]].isna().any(axis=1)
    non_positive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    high_too_low = h < np.maximum(o, c) * (1 - _TOL)
    low_too_high = l > np.minimum(o, c) * (1 + _TOL)
    # A flat bar with zero volume is a placeholder row, not a real session.
    stale = (h == l) & (v == 0)
    return missing | non_positive | high_too_low | low_too_high | stale


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set invalid sessions to NaN (we do not try to repair them).

    Returns the cleaned frame and the number of sessions invalidated.
    """
    bad = invalid_sessions(df)
    out = df.copy()
    out.loc[bad, OHLCV] = np.nan
    return out, int(bad.sum())


def load_universe(cfg: dict, force: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download (or read from cache), clean and align every ticker of the universe.

    Returns a dict {ticker: OHLCV frame on the benchmark calendar} and a cleaning
    report with one row per ticker.
    """
    d = cfg["data"]
    calendar = download(d["benchmark"], d["start"], d["end"], d["cache_dir"], force).index

    panel, report = {}, []
    for ticker in d["tickers"]:
        raw = download(ticker, d["start"], d["end"], d["cache_dir"], force)
        cleaned, n_invalid = clean(raw)
        n_off_calendar = int((~raw.index.isin(calendar)).sum())
        aligned = cleaned.reindex(calendar)
        panel[ticker] = aligned
        report.append({
            "ticker": ticker,
            "first_date": raw.index.min().date(),
            "n_sessions": len(raw),
            "n_invalid": n_invalid,
            "n_off_calendar": n_off_calendar,
            "n_missing_after_align": int(aligned["close"].isna().sum()),
        })
    return panel, pd.DataFrame(report).set_index("ticker")


def load_close(symbol: str, cfg: dict, calendar: pd.DatetimeIndex, force: bool = False) -> pd.Series:
    """Closing level of an index (e.g. ^VIX, ^IRX) on the benchmark calendar, not filled."""
    d = cfg["data"]
    close = download(symbol, d["start"], d["end"], d["cache_dir"], force)["close"]
    return close.reindex(calendar).rename(symbol)
