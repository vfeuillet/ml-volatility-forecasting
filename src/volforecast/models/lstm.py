"""Pooled LSTM (PyTorch, CPU) on sequences of the last L days of features.

A sequence for (ticker, t) stacks that ticker's features over t-L+1 .. t; it never
crosses tickers. Sequences are built lazily by fancy indexing into one stacked
array, so memory stays O(rows x features) instead of O(rows x L x features).

At every refit date T:
1. candidate training rows are those with a full feature window and t <= T - h;
2. the last `val_days` dates are a purged validation set for early stopping;
3. feature and target scalers are fitted on the fitting rows only;
4. the weights of the best validation epoch are kept (no refit on the full window).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from volforecast.walkforward import Split, purged_holdout


@dataclass
class LSTMConfig:
    seq_len: int
    hidden: int
    dropout: float
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    val_days: int
    horizon: int
    num_threads: int

    @classmethod
    def from_cfg(cls, cfg: dict, **overrides) -> "LSTMConfig":
        c = cfg["lstm"]
        fields = {k: c[k] for k in ("seq_len", "hidden", "dropout", "lr", "batch_size",
                                    "max_epochs", "patience", "val_days", "num_threads")}
        return cls(horizon=cfg["target"]["horizon"], **{**fields, **overrides})


class SequencePanel:
    """Features stacked ticker by ticker, with the rows that can end a full window."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int):
        order = X.index.to_frame(index=False).sort_values(["ticker", "date"]).index.to_numpy()
        self.index = X.index[order]
        self.dates = self.index.get_level_values("date")
        self.feats = X.to_numpy(dtype=np.float32)[order]
        self.target = y.to_numpy(dtype=np.float64)[order]
        self.seq_len = seq_len

        tickers = self.index.get_level_values("ticker").to_numpy()
        row_ok = ~np.isnan(self.feats).any(axis=1)
        # A window ending at row i is valid if its L rows are complete and belong
        # to the same ticker as row i.
        complete = pd.Series(row_ok).rolling(seq_len, min_periods=seq_len).sum().to_numpy() == seq_len
        start = np.arange(len(tickers)) - seq_len + 1
        same_ticker = (start >= 0) & (tickers[np.clip(start, 0, None)] == tickers)
        self.window_ok = complete & same_ticker
        self._offsets = np.arange(-seq_len + 1, 1)

    def windows(self, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        """(len(rows), L, n_features) tensor of standardized sequences ending at `rows`."""
        x = self.feats[rows[:, None] + self._offsets]
        return torch.from_numpy((x - mean) / std)


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


def _predict(net: _Net, panel: SequencePanel, rows: np.ndarray, mean, std, batch: int) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            out.append(net(panel.windows(rows[i:i + batch], mean, std)).numpy())
    return np.concatenate(out) if out else np.empty(0)


def fit_predict(panel: SequencePanel, train_end: pd.Timestamp, test_rows: np.ndarray,
                cfg: LSTMConfig, seed: int) -> tuple[np.ndarray, dict]:
    """Train on rows dated <= train_end, return variance forecasts for `test_rows`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    candidates = np.flatnonzero(panel.window_ok & ~np.isnan(panel.target)
                                & np.asarray(panel.dates <= train_end))
    is_fit, is_val = purged_holdout(panel.dates[candidates], cfg.val_days, cfg.horizon)
    fit_rows, val_rows = candidates[is_fit], candidates[is_val]

    # Scalers on the fitting period only (every complete row up to the last fit date).
    fit_end = panel.dates[fit_rows].max()
    scaler_rows = np.asarray(panel.dates <= fit_end) & ~np.isnan(panel.feats).any(axis=1)
    mean = panel.feats[scaler_rows].mean(axis=0)
    std = panel.feats[scaler_rows].std(axis=0) + 1e-8
    y_mean, y_std = panel.target[fit_rows].mean(), panel.target[fit_rows].std()
    y_fit = torch.from_numpy(((panel.target[fit_rows] - y_mean) / y_std).astype(np.float32))

    net = _Net(panel.feats.shape[1], cfg.hidden, cfg.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
    y_val = panel.target[val_rows]

    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        perm = rng.permutation(len(fit_rows))  # shuffling inside the training set only
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            loss = ((net(panel.windows(fit_rows[b], mean, std)) - y_fit[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_pred = _predict(net, panel, val_rows, mean, std, 4096) * y_std + y_mean
        val_loss = float(np.mean((y_val - val_pred) ** 2))
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_resid_var = float(np.var(y_val - val_pred))
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    net.load_state_dict(best_state)
    m = _predict(net, panel, test_rows, mean, std, 4096) * y_std + y_mean
    # Log-normal bias correction, with the residual variance of the validation year.
    forecast = np.exp(2 * m + 2 * best_resid_var)
    return forecast, {"best_epoch": best_epoch, "val_mse_log": best_loss, "resid_var": best_resid_var}


def lstm_walkforward(X: pd.DataFrame, y: pd.Series, splits: list[Split], cfg: LSTMConfig,
                     seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for every seed (one column each) and per-refit training diagnostics."""
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    panel = SequencePanel(X, y, cfg.seq_len)
    preds = {seed: [] for seed in seeds}
    logs = []
    for s in splits:
        test_rows = np.flatnonzero(panel.window_ok & np.asarray(panel.dates.isin(s.test_dates)))
        for seed in seeds:
            fc, info = fit_predict(panel, s.train_end, test_rows, cfg, seed)
            preds[seed].append(pd.Series(fc, index=panel.index[test_rows]))
            logs.append({"refit_date": s.refit_date, "seed": seed, **info})
    out = pd.DataFrame({f"seed_{k}": pd.concat(v) for k, v in preds.items()}).sort_index()
    return out, pd.DataFrame(logs)
