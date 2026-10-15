"""GARCH(1,1) and GJR-GARCH(1,1) with Student-t innovations (package `arch`).

Estimated on close-to-close log returns (x100 for numerical stability), which
include the overnight move and so measure the same daily variance as the target.
Parameters are re-estimated at every refit date on all returns up to T; between
refits they are frozen while the conditional variance keeps being filtered with
each new return, as a desk would do in practice.
"""

import numpy as np
import pandas as pd
from arch import arch_model

from volforecast.walkforward import Split


def garch_forecast(close: pd.Series, splits: list[Split], asymmetric: bool,
                   horizon: int, annualization: int) -> tuple[pd.Series, int]:
    """Walk-forward variance forecasts for one ticker.

    Returns the annualized h-day variance forecast indexed by origin date, and the
    number of refits where the optimizer did not converge (the previous parameters
    are then kept).
    """
    returns = (100 * np.log(close).diff()).dropna()
    model = arch_model(returns, mean="Constant", vol="GARCH", p=1, o=int(asymmetric), q=1, dist="t")

    params, n_failed, preds = None, 0, []
    for s in splits:
        # `last_obs` is exclusive: this uses every return up to and including T.
        last_obs = int(returns.index.searchsorted(s.refit_date, side="right"))
        res = model.fit(last_obs=last_obs, disp="off", show_warning=False)
        if res.convergence_flag != 0:
            n_failed += 1
        if res.convergence_flag == 0 or params is None:
            params = res.params
        # Forecast made at origin t uses returns up to t only.
        fc = res.forecast(params=params, horizon=horizon, start=s.refit_date, reindex=False).variance
        fc = fc.loc[fc.index.isin(s.test_dates)]
        preds.append(fc.sum(axis=1) / 1e4 * annualization / horizon)

    return pd.concat(preds).rename("forecast_var"), n_failed
