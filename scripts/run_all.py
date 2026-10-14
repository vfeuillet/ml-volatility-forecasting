"""End-to-end pipeline: `python scripts/run_all.py` (add --force to re-download)."""

import argparse

import numpy as np
import pandas as pd

from volforecast.config import PROJECT_ROOT, load_config
from volforecast.data import load_close, load_universe
from volforecast.plots import target_vs_vix
from volforecast.targets import build_targets, variance_components

PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"


def data_stage(cfg: dict, force: bool) -> pd.DataFrame:
    """Download and clean the data, build the target, write a sanity-check report."""
    panel, cleaning = load_universe(cfg, force)
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar, force)
    load_close(cfg["data"]["risk_free"], cfg, calendar, force)  # cached for the backtest

    targets = build_targets(panel, cfg)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(PROCESSED / "targets.parquet")

    ann = cfg["target"]["annualization"]
    rows = []
    for ticker, df in panel.items():
        comp = variance_components(df).dropna()
        tgt = targets.xs(ticker, level="ticker")
        rows.append({
            "ticker": ticker,
            "mean_vol_pct": 100 * np.sqrt(ann * tgt["daily_var"].mean()),
            "overnight_share_pct": 100 * comp["overnight"].sum() / comp.sum().sum(),
            "n_targets": int(tgt["target_rv"].notna().sum()),
        })
    summary = cleaning.join(pd.DataFrame(rows).set_index("ticker")).round(2)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "data_summary.csv")
    print(summary.to_string())
    print(f"\nVIX missing on benchmark calendar: {int(vix.isna().sum())} days")

    spy = targets.xs(cfg["data"]["benchmark"], level="ticker")["target_rv"]
    target_vs_vix(spy, vix, FIGURES / "spy_target_vs_vix.png")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download raw data")
    args = parser.parse_args()
    cfg = load_config()
    data_stage(cfg, args.force)


if __name__ == "__main__":
    main()
