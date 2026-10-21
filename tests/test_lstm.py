"""LSTM: sequences never look ahead nor cross tickers; scalers and training use the
training window only; results are reproducible for a fixed seed."""

import numpy as np
import pandas as pd
import pytest
import torch

from volforecast.models.lstm import LSTMConfig, SequencePanel, fit_predict

L, H = 5, 5
CFG = LSTMConfig(seq_len=L, hidden=4, dropout=0.0, lr=0.01, batch_size=64, max_epochs=3,
                 patience=2, val_days=40, horizon=H, num_threads=1)


def pooled(n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 2)), index=idx, columns=["f1", "f2"])
    y = pd.Series(-1.6 + 0.3 * X["f1"] + rng.normal(0, 0.1, len(idx)), index=idx)
    return X, y


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def test_window_is_the_last_L_days_of_the_same_ticker():
    X, y = pooled()
    # Encode (ticker, date) in the features to read windows back.
    day = X.index.get_level_values("date").map(pd.Timestamp.toordinal).to_numpy(float)
    X["f1"] = day
    X["f2"] = (X.index.get_level_values("ticker") == "B").astype(float)
    panel = SequencePanel(X, y, L)
    row = int(np.flatnonzero(panel.window_ok)[50])
    w = panel.windows(np.array([row]), np.zeros(2, np.float32), np.ones(2, np.float32))[0].numpy()
    end_date = panel.dates[row]
    assert w[-1, 0] == end_date.toordinal()                       # ends at t
    assert (np.diff(w[:, 0]) > 0).all() and w[0, 0] < end_date.toordinal()  # strictly past days
    assert len(set(w[:, 1])) == 1                                  # a single ticker


def test_first_rows_of_each_ticker_have_no_window():
    X, y = pooled()
    panel = SequencePanel(X, y, L)
    tickers = panel.index.get_level_values("ticker")
    for tk in ("A", "B"):
        ok = panel.window_ok[tickers == tk]
        assert not ok[:L - 1].any() and ok[L - 1:].all()


def test_training_ignores_everything_after_train_end():
    X, y = pooled()
    dates = X.index.get_level_values("date").unique()
    train_end, test_dates = dates[200], dates[206:230]
    far_future = X.index.get_level_values("date") > dates[260]

    def run(X, y):
        panel = SequencePanel(X, y, L)
        rows = np.flatnonzero(panel.window_ok & np.asarray(panel.dates.isin(test_dates)))
        return fit_predict(panel, train_end, rows, CFG, seed=0)[0]

    base = run(X, y)
    X2, y2 = X.copy(), y.copy()
    X2.loc[far_future] *= 100     # would move the scalers if they used the future
    y2.loc[far_future] += 5       # would move the target scaler / training
    np.testing.assert_array_equal(run(X2, y2), base)


def test_reproducible_for_a_given_seed():
    X, y = pooled()
    panel = SequencePanel(X, y, L)
    end = panel.dates.unique()[200]
    rows = np.flatnonzero(panel.window_ok & np.asarray(panel.dates > end))
    a, _ = fit_predict(panel, end, rows, CFG, seed=0)
    b, _ = fit_predict(panel, end, rows, CFG, seed=0)
    c, _ = fit_predict(panel, end, rows, CFG, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
