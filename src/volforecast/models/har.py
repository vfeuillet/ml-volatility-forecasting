"""HAR-RV in logs (Corsi, 2009), direct 5-day forecast.

    ln RV_{t+1:t+5} = b0 + bd ln rv_d,t + bw ln rv_w,t + bm ln rv_m,t + e_t

Variance forecast: if e ~ N(0, s^2) then E[RV^2 | X] = exp(2 m + 2 s^2), where m is
the fitted log vol. Without this term exp(2 m) is biased downwards (Jensen).
"""

import numpy as np
import pandas as pd

FEATURES = ["rv_d", "rv_w", "rv_m"]


class HAR:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        Z = self._design(X)
        self.coef_, *_ = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - Z @ self.coef_
        self.resid_var_ = resid.var(ddof=Z.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m = self._design(X) @ self.coef_
        return np.exp(2 * m + 2 * self.resid_var_)

    @staticmethod
    def _design(X: pd.DataFrame) -> np.ndarray:
        logs = np.log(X[FEATURES].to_numpy())
        return np.column_stack([np.ones(len(X)), logs])
