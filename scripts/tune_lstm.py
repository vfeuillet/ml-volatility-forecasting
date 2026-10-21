"""One-off selection of the LSTM sequence length and width, strictly before the test.

Same protocol as tune_gbm.py: train on 2005-2008, QLIKE on 2009-2011, averaged over
the configured seeds. Copy the result into configs/default.yaml (lstm.seq_len, hidden).
"""

import itertools

import numpy as np
import pandas as pd

from volforecast.config import PROJECT_ROOT, load_config
from volforecast.data import load_close, load_universe
from volforecast.evaluation import qlike
from volforecast.features import pooled_dataset
from volforecast.models.lstm import LSTMConfig, SequencePanel, fit_predict
from volforecast.targets import build_targets
from volforecast.walkforward import tuning_masks


def main() -> None:
    import torch

    cfg = load_config()
    c, tun, h = cfg["lstm"], cfg["lstm"]["tuning"], cfg["target"]["horizon"]
    panel_data, _ = load_universe(cfg)
    calendar = panel_data[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar)
    X, y = pooled_dataset(panel_data, build_targets(panel_data, cfg), vix, cfg)
    X = X[c["features"]]
    torch.set_num_threads(c["num_threads"])
    torch.use_deterministic_algorithms(True)

    grid = tun["grid"]
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        lc = LSTMConfig.from_cfg(cfg, **params)
        panel = SequencePanel(X, y, lc.seq_len)
        train, val = tuning_masks(panel.dates, calendar, h, tun["train_end"], tun["val_end"])
        train_end = panel.dates[train].max()
        val_rows = np.flatnonzero(val & panel.window_ok & ~np.isnan(panel.target))
        realized = np.exp(2 * panel.target[val_rows])
        scores = []
        for seed in c["seeds"]:
            fc, info = fit_predict(panel, train_end, val_rows, lc, seed)
            scores.append(qlike(realized, fc).mean())
        rows.append({**params, "val_qlike": np.mean(scores), "val_qlike_std": np.std(scores)})
        print(rows[-1])

    res = pd.DataFrame(rows).sort_values("val_qlike")
    res.round(5).to_csv(PROJECT_ROOT / "reports" / "tables" / "lstm_tuning.csv", index=False)
    print(f"\nBest (copy into configs/default.yaml -> lstm):\n{res.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
