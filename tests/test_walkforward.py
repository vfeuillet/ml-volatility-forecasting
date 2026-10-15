"""Walk-forward splits: purging, ordering and coverage."""

import numpy as np
import pandas as pd

from volforecast.walkforward import make_splits, run_supervised

H, EVERY = 5, 21
CAL = pd.bdate_range("2010-01-01", "2014-12-31")


def test_training_targets_fully_observed_at_refit():
    for s in make_splits(CAL, "2012-01-01", EVERY, H):
        pos_T = CAL.get_loc(s.refit_date)
        pos_end = CAL.get_loc(s.train_end)
        # The last training target spans pos_end+1 .. pos_end+H, which must be <= T.
        assert pos_end + H <= pos_T
        assert s.train_end < s.test_dates[0]
        assert s.test_dates[0] == s.refit_date


def test_test_blocks_cover_the_period_without_overlap():
    splits = make_splits(CAL, "2012-01-01", EVERY, H)
    all_test = np.concatenate([s.test_dates.values for s in splits])
    assert len(all_test) == len(np.unique(all_test))
    assert pd.Timestamp(all_test[0]) == CAL[CAL >= "2012-01-01"][0]
    assert pd.Timestamp(all_test[-1]) == CAL[-1]


class RecordingModel:
    """Fake model that records the dates it was trained on."""
    seen: list = []

    def fit(self, X, y):
        RecordingModel.seen.append((X.index.max(), y.index.max()))
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_run_supervised_trains_only_up_to_train_end():
    X = pd.DataFrame({"a": np.arange(len(CAL), dtype=float)}, index=CAL)
    y = pd.Series(1.0, index=CAL)
    splits = make_splits(CAL, "2012-01-01", EVERY, H)
    RecordingModel.seen = []
    preds = run_supervised(RecordingModel, X, y, splits)
    for (x_max, y_max), s in zip(RecordingModel.seen, splits):
        assert x_max == s.train_end and y_max == s.train_end
    assert preds.index.min() == splits[0].refit_date
