"""Naive benchmark: next week's volatility equals last week's."""

import pandas as pd


def naive_forecast(features: pd.DataFrame) -> pd.Series:
    """Forecast variance = squared realized vol of the last 5 days (rv_w^2). No fitting."""
    return (features["rv_w"] ** 2).rename("forecast_var")
