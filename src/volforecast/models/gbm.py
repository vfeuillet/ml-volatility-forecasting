"""Pooled gradient boosting (scikit-learn HistGradientBoostingRegressor) on log RV.

scikit-learn's built-in early stopping draws its validation set at random, which
would leak information through overlapping 5-day targets. Instead, at every refit:

1. the last `val_days` dates of the training window form a validation set, purged
   by `horizon` days from the fitting part;
2. a first model is fitted with early stopping on that time-ordered validation set;
   its validation residuals give the bias-correction variance s^2 and the
   permutation importances;
3. the final model is refitted on the whole window with the number of trees found
   in step 2, so that the most recent year is not thrown away.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from volforecast.walkforward import dates_of


class GBM:
    def __init__(self, params: dict, features: list[str], val_days: int, horizon: int,
                 max_iter: int, n_iter_no_change: int, seed: int, importance: bool = False):
        self.params, self.features = params, features
        self.val_days, self.horizon = val_days, horizon
        self.max_iter, self.n_iter_no_change = max_iter, n_iter_no_change
        self.seed, self.importance = seed, importance

    def _regressor(self, **kw) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(loss="squared_error", random_state=self.seed,
                                             **self.params, **kw)

    def split(self, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks (fit, validation): the last `val_days` dates validate, and the
        fitting part stops `horizon` days earlier so that no fitting target, which
        spans t+1..t+h, overlaps a validation target (purge)."""
        unique = dates.unique().sort_values()
        val_start = unique[-self.val_days]
        fit_end = unique[len(unique) - self.val_days - self.horizon]
        return np.asarray(dates <= fit_end), np.asarray(dates >= val_start)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBM":
        X = X[self.features]
        is_fit, is_val = self.split(dates_of(X.index))

        first = self._regressor(max_iter=self.max_iter, early_stopping=True,
                                n_iter_no_change=self.n_iter_no_change, validation_fraction=None)
        first.fit(X[is_fit], y[is_fit], X_val=X[is_val], y_val=y[is_val])
        # Stopping happens `n_iter_no_change` rounds after the best validation score.
        val_scores = first.validation_score_[1:]  # index 0 is the score before any tree
        self.n_trees_ = int(np.argmax(val_scores)) + 1
        val_pred = list(first.staged_predict(X[is_val]))[self.n_trees_ - 1]
        self.resid_var_ = float(np.var(y[is_val].to_numpy() - val_pred))
        if self.importance:
            # Model with the selected number of trees, scored on the validation year only.
            first = self._regressor(max_iter=self.n_trees_, early_stopping=False).fit(X[is_fit], y[is_fit])
            imp = permutation_importance(first, X[is_val], y[is_val], n_repeats=3,
                                         random_state=self.seed, scoring="neg_mean_squared_error")
            self.importances_ = pd.Series(imp.importances_mean, index=self.features)

        self.model_ = self._regressor(max_iter=self.n_trees_, early_stopping=False).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        m = self.model_.predict(X[self.features])
        return np.exp(2 * m + 2 * self.resid_var_)
