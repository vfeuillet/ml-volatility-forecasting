"""One-off hyperparameter selection for the GBM, strictly before the test period.

Train on 2005-2008, evaluate QLIKE on 2009-2011. The last 2011 targets end before
the first test origin (2012), so the test period is never seen. The selected
parameters must then be copied into configs/default.yaml (gbm.params).
"""

import itertools

import numpy as np
import pandas as pd

from volforecast.config import PROJECT_ROOT, load_config
from volforecast.data import load_close, load_universe
from volforecast.evaluation import qlike
from volforecast.features import pooled_dataset
from volforecast.models.gbm import GBM
from volforecast.targets import build_targets
from volforecast.walkforward import dates_of


def main() -> None:
    cfg = load_config()
    g, tun, h = cfg["gbm"], cfg["gbm"]["tuning"], cfg["target"]["horizon"]
    panel, _ = load_universe(cfg)
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar)
    targets = build_targets(panel, cfg)
    X, y = pooled_dataset(panel, targets, vix, cfg)
    ok = X[g["features"]].notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]

    dates = dates_of(X.index)
    cal = calendar[calendar <= tun["val_end"]]
    # Purged split: training targets end by the first validation origin; validation
    # targets end by the last trading day of 2011.
    val_first = cal[cal.searchsorted(pd.Timestamp(tun["train_end"]), side="right")]
    train = dates <= cal[cal.get_loc(val_first) - h]
    val = (dates >= val_first) & (dates <= cal[-1 - h])
    realized = np.exp(2 * y[val])  # y is log RV, so this is RV^2

    grid = tun["grid"]
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        model = GBM(params, g["features"], g["val_days"], h, g["max_iter"],
                    g["n_iter_no_change"], cfg["seed"]).fit(X[train], y[train])
        score = qlike(realized.to_numpy(), model.predict(X[val])).mean()
        rows.append({**params, "n_trees": model.n_trees_, "val_qlike": score})
        print(rows[-1])

    res = pd.DataFrame(rows).sort_values("val_qlike")
    out = PROJECT_ROOT / "reports" / "tables" / "gbm_tuning.csv"
    res.round(5).to_csv(out, index=False)
    print(f"\nBest (copy into configs/default.yaml -> gbm.params):\n{res.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
